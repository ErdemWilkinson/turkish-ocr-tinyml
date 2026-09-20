"""Export the trained OCR recognizer as a full-int8 TensorFlow Lite model."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("OCR_ARTIFACTS", ROOT / "artifacts"))


def representative_dataset():
    images = np.load(ARTIFACTS / "representative_images.npy")
    for image in images:
        yield [image[np.newaxis, ...].astype(np.float32)]


def main() -> None:
    model_path = ARTIFACTS / "turkish_line_ocr.keras"
    if not model_path.exists():
        raise SystemExit("Train the OCR model before exporting it.")
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(model_path))
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
