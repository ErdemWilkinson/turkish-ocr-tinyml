"""Import the Turkish Scene Text Recognition (TS-TR) dataset into our line-crop
dataset format.

Source: https://www.kaggle.com/datasets/serdaryildiz/turkish-scene-text-recognition-dataset
(CC BY-NC 4.0 -- non-commercial; Yildiz, "Turkish Scene Text Recognition:
Introducing Extensive Real and Synthetic Datasets and a Novel Recognition Model").

The archive holds cropped real-world word images plus train.txt/test.txt with
one "<file>\t<text>" row per crop. The official train split becomes training
data (data/tstr_labels.csv); the official test split is kept apart as an
evaluation-only manifest (data/tstr_test_labels.csv).
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "tstr_source" / "extracted" / "TS-TR" / "TS-TR"
MAX_TEXT_LENGTH = 24
ALPHABET = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ çğıİöşüÇĞÖŞÜ.,:;!?-_/()'%")
# Crops are numbered in the order they were cut from the source photos, so
# neighbours come from the same photo. Bucketing by index keeps one photo's
# crops in the same group, which GroupShuffleSplit then keeps in one split.
GROUP_SIZE = 50

OUTPUT_DIR = ROOT / "data" / "tstr_lines"


def is_usable(text: str) -> bool:
    return 1 <= len(text) <= MAX_TEXT_LENGTH and set(text).issubset(ALPHABET)


def import_split(source: Path, split: str, output_csv: Path) -> tuple[int, int]:
    rows: list[dict[str, str]] = []
    skipped = 0
    for line in (source / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        filename, _, text = line.partition("\t")
        text = text.strip()
        image = source / split / filename
        if not is_usable(text) or not image.is_file():
            skipped += 1
            continue
        target_name = f"{split}_{image.stem.zfill(5)}{image.suffix}"
        shutil.copyfile(image, OUTPUT_DIR / target_name)
        rows.append({
            "image": f"tstr_lines/{target_name}",
            "text": text,
            "group": f"tstr_{split}_{int(image.stem) // GROUP_SIZE:03d}",
        })
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "text", "group"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Folder holding train/, test/, train.txt and test.txt")
    args = parser.parse_args()
    if not (args.source / "train.txt").is_file():
        raise SystemExit(f"Missing extracted TS-TR archive: {args.source}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "tstr_labels.csv"), ("test", "tstr_test_labels.csv")):
        kept, skipped = import_split(args.source, split, ROOT / "data" / name)
        print(f"{split}: wrote {kept} crops to data/{name}, skipped {skipped} (too long, unsupported characters, or missing image)")


if __name__ == "__main__":
    main()
