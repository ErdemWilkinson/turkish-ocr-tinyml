"""Train the compact Turkish line OCR CNN-CTC model."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("OCR_ARTIFACTS", ROOT / "artifacts"))
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 32
MAX_TEXT_LENGTH = 24
TIME_STEPS = IMAGE_WIDTH // 4  # matches the two stride-2 conv layers below

# IDs start at one so zero can safely be used for padded labels.
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ çğıİöşüÇĞÖŞÜ.,:;!?-_/()'%"
CHAR_TO_ID = {character: index + 1 for index, character in enumerate(ALPHABET)}
BLANK_ID = len(CHAR_TO_ID) + 1
NUM_CLASSES = BLANK_ID + 1


def labels_csv_paths() -> list[Path]:
    raw = os.environ.get("OCR_LABELS_CSV", str(ROOT / "data" / "labels.csv"))
    return [Path(part) for part in raw.split(os.pathsep) if part]


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for labels_csv in labels_csv_paths():
        if not labels_csv.exists():
            raise SystemExit(f"Missing {labels_csv}. Create the dataset manifest first.")
        with labels_csv.open(encoding="utf-8", newline="") as handle:
            file_rows = list(csv.DictReader(handle))
        required = {"image", "text", "group"}
        if not file_rows or not required.issubset(file_rows[0]):
            raise SystemExit(f"{labels_csv} must contain image,text,group columns.")
        base_dir = labels_csv.parent
        for row in file_rows:
            row["_base"] = str(base_dir)
        rows.extend(file_rows)
    if not rows:
        raise SystemExit("No training rows found.")
    for row in rows:
        if not row["text"] or len(row["text"]) > MAX_TEXT_LENGTH:
            raise SystemExit(f"Text must contain 1-{MAX_TEXT_LENGTH} characters: {row['text']!r}")
        unsupported = sorted(set(row["text"]) - set(CHAR_TO_ID))
        if unsupported:
            raise SystemExit(f"Unsupported characters in {row['text']!r}: {unsupported}")
        if not (Path(row["_base"]) / row["image"]).exists():
            raise SystemExit(f"Missing image: {row['image']}")
    return rows


def load_image(base_dir: str, relative_path: str) -> np.ndarray:
    path = Path(base_dir) / relative_path
    image = tf.io.read_file(str(path))
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image.numpy()


def encode_text(text: str) -> list[int]:
    return [CHAR_TO_ID[character] for character in text]


def make_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images = np.stack([load_image(row["_base"], row["image"]) for row in rows]).astype(np.float32)
    labels = np.zeros((len(rows), MAX_TEXT_LENGTH), dtype=np.int32)
    lengths = np.zeros(len(rows), dtype=np.int32)
    for index, row in enumerate(rows):
        encoded = encode_text(row["text"])
        labels[index, :len(encoded)] = encoded
        lengths[index] = len(encoded)
    groups = np.array([f"{row['_base']}::{row['group']}" for row in rows])
    return images, labels, lengths, groups


L2 = tf.keras.regularizers.l2(1e-4)


def recognition_body(image_input: tf.Tensor) -> tf.Tensor:
    """The conv+BiLSTM+softmax stack shared by the training and recognition
    graphs. Pulled out into its own function (rather than inlined in
    build_models()) so export_tflite.py can rebuild an identical graph on a
    statically-shaped input and copy the trained weights across -- see that
    file's comment for why a second, static-shape build is required rather
    than exporting the model this function's caller already has.

    unroll=False (the default) is required on both LSTM layers, not just
    for speed: with unroll=True the TFLite converter has no recurrent-cell
    structure left to recognize at export time, so it flattens each LSTM
    into its constituent gate ops (ADD/MUL/LOGISTIC/TANH/FULLY_CONNECTED/
    SPLIT) repeated once per timestep -- 2 stacked BiLSTMs over
    TIME_STEPS=40 steps measured out to 2,308 ops in the exported int8
    .tflite, versus 28 total (4 of them fused UNIDIRECTIONAL_SEQUENCE_LSTM
    ops) once the converter can fuse each direction of each layer. See
    DECISION_NOTES.md.
    """
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=L2)(image_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 16 x 80

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 8 x 40
    x = tf.keras.layers.SpatialDropout2D(0.1)(x)

    x = tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu", kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)  # 4 x 40

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)  # 2 x 40
    x = tf.keras.layers.SpatialDropout2D(0.15)(x)

    x = tf.keras.layers.Permute((2, 1, 3))(x)  # width, height, channels
    x = tf.keras.layers.Reshape((TIME_STEPS, 2 * 128))(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=L2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(96, return_sequences=True, unroll=False,
                              kernel_regularizer=L2, recurrent_dropout=0.0)
    )(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(96, return_sequences=True, unroll=False,
                              kernel_regularizer=L2, recurrent_dropout=0.0)
    )(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    return tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="characters")(x)


def build_models() -> tuple[tf.keras.Model, tf.keras.Model]:
    image_input = tf.keras.Input((IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="image")
    labels_input = tf.keras.Input((MAX_TEXT_LENGTH,), dtype="int32", name="labels")
    input_length = tf.keras.Input((1,), dtype="int32", name="input_length")
    label_length = tf.keras.Input((1,), dtype="int32", name="label_length")

    logits = recognition_body(image_input)

    def ctc_loss(arguments: list[tf.Tensor]) -> tf.Tensor:
        predictions, labels, prediction_lengths, text_lengths = arguments
        return tf.keras.backend.ctc_batch_cost(labels, predictions, prediction_lengths, text_lengths)

    loss = tf.keras.layers.Lambda(ctc_loss, name="ctc_loss")(
        [logits, labels_input, input_length, label_length]
    )
    training_model = tf.keras.Model(
        [image_input, labels_input, input_length, label_length], loss
    )
    recognition_model = tf.keras.Model(image_input, logits)
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=5.0),
        loss=lambda _, value: value,
    )
    return training_model, recognition_model


def main() -> None:
    rows = load_rows()
    images, labels, lengths, groups = make_arrays(rows)
    if len(np.unique(groups)) < 2:
        raise SystemExit("At least two dataset groups are required for validation.")

    rng = np.random.default_rng(42)
    shuffle_idx = rng.permutation(len(rows))
    images, labels, lengths, groups = images[shuffle_idx], labels[shuffle_idx], lengths[shuffle_idx], groups[shuffle_idx]

    train_idx, validation_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(images, groups=groups)
    )
    input_lengths = np.full((len(rows), 1), TIME_STEPS, dtype=np.int32)
    label_lengths = lengths[:, np.newaxis]
    train_inputs = [images[train_idx], labels[train_idx], input_lengths[train_idx], label_lengths[train_idx]]
    validation_inputs = [images[validation_idx], labels[validation_idx], input_lengths[validation_idx], label_lengths[validation_idx]]

    print(f"Train samples: {len(train_idx)}, validation samples: {len(validation_idx)}")

    training_model, recognition_model = build_models()
    training_model.fit(
        train_inputs,
        np.zeros((len(train_idx), 1), dtype=np.float32),
        validation_data=(validation_inputs, np.zeros((len(validation_idx), 1), dtype=np.float32)),
        epochs=150,
        batch_size=64,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ],
        verbose=2,
    )

    ARTIFACTS.mkdir(exist_ok=True)
    recognition_model.save(ARTIFACTS / "turkish_line_ocr.keras")
    (ARTIFACTS / "alphabet.json").write_text(
        json.dumps({"alphabet": ALPHABET, "blank_id": BLANK_ID}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rep_rng = np.random.default_rng(7)
    real_train_idx = np.array([i for i in train_idx if rows[i]["image"].startswith("real_lines/")])
    synthetic_train_idx = np.array([i for i in train_idx if not rows[i]["image"].startswith("real_lines/")])
    real_sample = rep_rng.choice(real_train_idx, size=min(150, len(real_train_idx)), replace=False) if len(real_train_idx) else np.array([], dtype=int)
    synthetic_sample = rep_rng.choice(synthetic_train_idx, size=min(150, len(synthetic_train_idx)), replace=False) if len(synthetic_train_idx) else np.array([], dtype=int)
    representative_idx = np.concatenate([real_sample, synthetic_sample]).astype(int)
    np.save(ARTIFACTS / "representative_images.npy", images[representative_idx])
    print(f"Saved OCR model with {len(rows)} samples and {len(CHAR_TO_ID)} characters.")


if __name__ == "__main__":
    main()
