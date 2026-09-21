"""Sanity-check the exported int8 TFLite model against a few validation samples."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 32


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


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


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

    split_path = ARTIFACTS / "split_manifest.json"
    if not split_path.exists():
        raise SystemExit("Missing split_manifest.json; retrain before evaluating the int8 model.")
    rows = json.loads(split_path.read_text(encoding="utf-8")).get("validation", [])
    if not rows:
        raise SystemExit("Split manifest contains no validation samples.")

    correct = total_edits = total_chars = 0
    for row in rows:
        image = load_image(Path(row["base"]) / row["image"])
        quantized = np.round(image / in_scale + in_zero).astype(input_detail["dtype"])[np.newaxis, ...]
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_detail["index"])[0]
        dequantized = (raw_output.astype(np.float32) - out_zero) * out_scale
        prediction = greedy_ctc_decode(dequantized, blank_id, id_to_char)
        match = prediction == row["text"]
        correct += int(match)
        total_edits += edit_distance(row["text"], prediction)
        total_chars += len(row["text"])
        marker = "OK " if match else "-- "
        print(f"{marker}{row['text']!r:30s} -> {prediction!r}")

    print(f"\nint8 TFLite held-out exact-match on {len(rows)} samples: {correct / len(rows):.3f}")
    print(f"int8 TFLite held-out CER: {total_edits / max(total_chars, 1):.3f}")


if __name__ == "__main__":
    main()
