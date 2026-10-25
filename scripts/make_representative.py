"""Rebuild representative_images.npy, the int8 calibration set, without retraining.

Uses the same selection as train.py (representative_indices) over the training
rows only: rows that artifacts/split_manifest.json lists as held out are left
out, so calibration never sees validation data. Reads the manifests named by
OCR_LABELS_CSV, like train.py.
"""
from __future__ import annotations

import json

import numpy as np

from train import (ARTIFACTS, describe_sources, load_image, load_rows,
                   representative_indices)


def main() -> None:
    rows = load_rows()
    split_path = ARTIFACTS / "split_manifest.json"
    held_out: set[str] = set()
    if split_path.exists():
        held_out = {row["image"] for row in json.loads(split_path.read_text(encoding="utf-8"))["validation"]}
    else:
        print(f"Warning: {split_path} not found; calibrating on all rows, validation rows included.")
    candidates = np.array([i for i, row in enumerate(rows) if row["image"] not in held_out], dtype=int)
    if not len(candidates):
        raise SystemExit("No training rows left after excluding the held-out set; check OCR_LABELS_CSV.")

    chosen = representative_indices(rows, candidates)
    images = np.stack([load_image(rows[i]["_base"], rows[i]["image"]) for i in chosen]).astype(np.float32)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS / "representative_images.npy", images)
    print(f"Wrote {len(images)} calibration images: {describe_sources(rows, chosen)}")


if __name__ == "__main__":
    main()
