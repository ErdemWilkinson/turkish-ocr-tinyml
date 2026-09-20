"""Export the trained OCR recognizer as a full-int8 TensorFlow Lite model.

Rebuilds the recognition graph on a statically-shaped (batch=1) input and
copies the trained weights across before converting, rather than exporting
turkish_line_ocr.keras directly. This is required, not just tidier: the
saved recognition_model has a flexible (None) batch dimension, and with
train.py's LSTM layers set to unroll=False (see recognition_body()'s
docstring for why unroll=True is worse), TFLite's converter cannot lower
the LSTM's internal TensorList ops unless every shape in the graph --
including the batch dimension -- is static:

    error: 'tf.TensorListReserve' op requires element_shape to be
    static during TF Lite transformation pass

A batch size of 1 matches how the device actually calls the model (one
cropped text line at a time), so this loses no real capability.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import IMAGE_HEIGHT, IMAGE_WIDTH, recognition_body

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("OCR_ARTIFACTS", ROOT / "artifacts"))


def representative_dataset():
    images = np.load(ARTIFACTS / "representative_images.npy")
    for image in images:
        yield [image[np.newaxis, ...].astype(np.float32)]


def build_static_batch_model(trained_model: tf.keras.Model) -> tf.keras.Model:
    static_input = tf.keras.Input(batch_shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="image")
    static_logits = recognition_body(static_input)
    static_model = tf.keras.Model(static_input, static_logits)
    static_model.set_weights(trained_model.get_weights())
    return static_model


def main() -> None:
    model_path = ARTIFACTS / "turkish_line_ocr.keras"
    if not model_path.exists():
        raise SystemExit("Train the OCR model before exporting it.")
    trained_model = tf.keras.models.load_model(model_path)
    export_model = build_static_batch_model(trained_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output = converter.convert()
    target = ARTIFACTS / "turkish_line_ocr_int8.tflite"
    target.write_bytes(output)
    print(f"Wrote {target} ({len(output):,} bytes)")


if __name__ == "__main__":
    main()
