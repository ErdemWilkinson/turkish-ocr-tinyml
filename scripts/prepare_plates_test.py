"""Turkish License Plate 250 (Zenodo 23369685, CC BY 4.0) -> a real-photo test set.

Crops each plate box out of the photo and takes the plate text from the file
name ("01 DVZ 18.jpg" -> "01 DVZ 18"). The box files hold one normalised
"ymin, xmin, ymax, xmax" line. Evaluation only: never add this to training.

    python scripts/prepare_plates_test.py D:/datasets/raw/tr-plates-250/extracted/Turkish_Licence_Plates_250
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from PIL import Image

from train import CHAR_TO_ID, MAX_TEXT_LENGTH

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "plates_lines"
LABELS = ROOT / "data" / "plates_test_labels.csv"


def main() -> None:
    source = Path(sys.argv[1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for image_path in sorted((source / "images_with_texts").iterdir()):
        box_path = source / "yolo_labels_and_bbox" / f"{image_path.stem}.txt"
        text = re.sub(r"\s+", " ", re.sub(r"\(\d+\)$", "", image_path.stem)).strip().upper()
        if not box_path.exists() or not text or len(text) > MAX_TEXT_LENGTH or set(text) - set(CHAR_TO_ID):
            skipped += 1
            continue
        ymin, xmin, ymax, xmax = (float(value) for value in box_path.read_text().split(","))
        image = Image.open(image_path).convert("L")
        width, height = image.size
        crop = image.crop((round(xmin * width), round(ymin * height), round(xmax * width), round(ymax * height)))
        name = f"{len(rows):04d}.png"
        crop.save(OUT_DIR / name)
        rows.append({"image": f"plates_lines/{name}", "text": text, "group": "tr_plates_250"})
    with LABELS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "text", "group"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} plate crops to {OUT_DIR} ({skipped} skipped)")


if __name__ == "__main__":
    main()
