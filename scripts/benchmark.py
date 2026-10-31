"""Score one or more trained models, float and int8, on the same test sets.

Every model is measured on identical rows so versions can be compared:
  tstr_test   -- TS-TR official test split (real scene text, never trained on)
  plates      -- Turkish License Plate 250 crops (real photos, never trained on);
                 plates are also scored ignoring spaces, since the file names
                 that give the labels space plates inconsistently
  heldout     -- the 2026-10-25 held-out manifest minus orkungedik rows, which
                 is the fixed validation set of every run since 2026-10-10

    python scripts/benchmark.py --model old_1009=artifacts_backup_2026-10-09 --model A=artifacts_A --out results.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from evaluate import edit_distance, greedy_ctc_decode, load_image

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_MANIFEST = ROOT / "artifacts_backup_2026-10-25" / "split_manifest.json"
EXCLUDED_SOURCES = {"orkungedik_lines"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [{"base": str(path.parent), "image": row["image"], "text": row["text"]} for row in csv.DictReader(handle)]


def test_sets() -> dict[str, list[dict]]:
    heldout = [row for row in json.loads(HELDOUT_MANIFEST.read_text(encoding="utf-8"))["validation"]
               if row["image"].split("/")[0] not in EXCLUDED_SOURCES]
    sets = {"tstr_test": read_csv(ROOT / "data" / "tstr_test_labels.csv"), "heldout": heldout}
    if (ROOT / "data" / "plates_test_labels.csv").exists():
        sets["plates"] = read_csv(ROOT / "data" / "plates_test_labels.csv")
    return sets


def score(predictions: list[str], rows: list[dict], ignore_spaces: bool = False) -> dict:
    exact = edits = chars = 0
    for prediction, row in zip(predictions, rows):
        truth = row["text"]
        if ignore_spaces:
            prediction, truth = prediction.replace(" ", ""), truth.replace(" ", "")
        exact += int(prediction == truth)
        edits += edit_distance(prediction, truth)
        chars += max(1, len(truth))
    return {"samples": len(rows), "exact": exact / len(rows), "cer": edits / chars}


def predict_float(model, images: np.ndarray, decode) -> list[str]:
    logits = model.predict(images, batch_size=256, verbose=0)
    return [decode(row) for row in logits]


def predict_int8(tflite_path: Path, images: np.ndarray, decode) -> list[str]:
    interpreter = tf.lite.Interpreter(model_content=Path(tflite_path).read_bytes())
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    predictions = []
    for image in images:
        interpreter.set_tensor(inp["index"], np.round(image / in_scale + in_zero).astype(inp["dtype"])[np.newaxis])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0]
        predictions.append(decode((raw.astype(np.float32) - out_zero) * out_scale))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name=artifacts_dir, repeatable")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sets", default="tstr_test,plates,heldout")
    args = parser.parse_args()

    wanted = args.sets.split(",")
    sets = {name: rows for name, rows in test_sets().items() if name in wanted}
    cache = {name: np.stack([load_image(Path(row["base"]) / row["image"]) for row in rows]) for name, rows in sets.items()}
    results = json.loads(Path(args.out).read_text(encoding="utf-8")) if Path(args.out).exists() else {}
    for spec in args.model:
        name, folder = spec.split("=", 1)
        artifacts = Path(folder).resolve()
        alphabet = json.loads((artifacts / "alphabet.json").read_text(encoding="utf-8"))
        id_to_char = {index + 1: character for index, character in enumerate(alphabet["alphabet"])}
        decode = lambda logits: greedy_ctc_decode(logits, alphabet["blank_id"], id_to_char)
        model = tf.keras.models.load_model(artifacts / "erdem_ocr.keras", compile=False)
        entry = {"params": model.count_params(),
                 "tflite_bytes": (artifacts / "erdem_ocr_int8.tflite").stat().st_size
                 if (artifacts / "erdem_ocr_int8.tflite").exists() else None}
        for set_name, rows in sets.items():
            variants = {"float": predict_float(model, cache[set_name], decode)}
            if entry["tflite_bytes"]:
                variants["int8"] = predict_int8(artifacts / "erdem_ocr_int8.tflite", cache[set_name], decode)
            for variant, predictions in variants.items():
                entry[f"{set_name}/{variant}"] = score(predictions, rows)
                if set_name == "plates":
                    entry[f"{set_name}_nospace/{variant}"] = score(predictions, rows, ignore_spaces=True)
                print(name, set_name, variant, entry[f"{set_name}/{variant}"], flush=True)
        results[name] = entry
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
