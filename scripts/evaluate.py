"""Evaluate a trained OCR recognizer: CER and exact-line accuracy on a held-out group split."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 32
MAX_TEXT_LENGTH = 24


def load_image(path: Path) -> np.ndarray:
    image = tf.io.read_file(str(path))
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image.numpy()


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1]


def greedy_ctc_decode(logits: np.ndarray, blank_id: int, id_to_char: dict[int, str]) -> str:
    best = np.argmax(logits, axis=-1)
    chars = []
    previous = -1
    for index in best:
        if index != previous and index != blank_id and index != 0:
            chars.append(id_to_char.get(int(index), ""))
        previous = index
    return "".join(chars)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--labels", action="append", required=True,
                         help="labels.csv path(s) relative to ocr/data; repeatable")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    alphabet_info = json.loads((artifacts / "alphabet.json").read_text(encoding="utf-8"))
    alphabet = alphabet_info["alphabet"]
    blank_id = alphabet_info["blank_id"]
    id_to_char = {index + 1: character for index, character in enumerate(alphabet)}

    rows = []
    for labels_path in args.labels:
        full_path = ROOT / "data" / labels_path
        with full_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["_base"] = full_path.parent
                rows.append(row)
    if not rows:
        raise SystemExit("No evaluation rows found.")

    groups = np.array([f"{row['_base']}::{row['group']}" for row in rows])
    if len(np.unique(groups)) >= 2:
        _, validation_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(rows, groups=groups)
        )
    else:
        validation_idx = np.arange(len(rows))

    model = tf.keras.models.load_model(artifacts / "turkish_line_ocr.keras")

    total_chars = 0
    total_edits = 0
    exact = 0
    n = 0
    examples = []
    for i in validation_idx:
        row = rows[i]
        image_path = row["_base"] / row["image"]
        if not image_path.exists():
            continue
        image = load_image(image_path)[np.newaxis, ...]
        logits = model.predict(image, verbose=0)[0]
        prediction = greedy_ctc_decode(logits, blank_id, id_to_char)
        truth = row["text"]
        edits = edit_distance(prediction, truth)
        total_edits += edits
        total_chars += max(1, len(truth))
        exact += int(prediction == truth)
        n += 1
        if len(examples) < 15:
            examples.append((truth, prediction))

    cer = total_edits / total_chars if total_chars else float("nan")
    accuracy = exact / n if n else float("nan")
    print(f"Artifacts: {artifacts}")
    print(f"Validation samples: {n}")
    print(f"Character error rate: {cer:.4f}")
    print(f"Exact-line accuracy: {accuracy:.4f}")
    print("\nSample predictions (truth -> prediction):")
    for truth, prediction in examples:
        print(f"  {truth!r:30s} -> {prediction!r}")


if __name__ == "__main__":
    main()
