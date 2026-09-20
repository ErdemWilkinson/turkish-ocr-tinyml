"""Sanity-check the exported int8 TFLite model against a few validation samples."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 32
SAMPLE_COUNT = 40


def load_image(path: Path) -> np.ndarray:
    image = tf.io.read_file(str(path))
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image.numpy()


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
    alphabet_info = json.loads((ARTIFACTS / "alphabet.json").read_text(encoding="utf-8"))
    blank_id = alphabet_info["blank_id"]
    id_to_char = {index + 1: character for index, character in enumerate(alphabet_info["alphabet"])}

    model_bytes = (ARTIFACTS / "turkish_line_ocr_int8.tflite").read_bytes()
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]

    rows = []
    with (ROOT / "data" / "real_labels.csv").open(encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    rows = rows[:SAMPLE_COUNT]

    correct = 0
    for row in rows:
        image = load_image(ROOT / "data" / row["image"])
        quantized = np.round(image / in_scale + in_zero).astype(input_detail["dtype"])[np.newaxis, ...]
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_detail["index"])[0]
        dequantized = (raw_output.astype(np.float32) - out_zero) * out_scale
        prediction = greedy_ctc_decode(dequantized, blank_id, id_to_char)
        match = prediction == row["text"]
        correct += int(match)
        marker = "OK " if match else "-- "
        print(f"{marker}{row['text']!r:30s} -> {prediction!r}")

    print(f"\nint8 TFLite exact-match on {len(rows)} samples: {correct / len(rows):.3f}")


if __name__ == "__main__":
    main()
