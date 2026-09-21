"""Import the Zenodo 'Turkish OCR Text Image Dataset' (wiki_ocr printed subset + gold
standard printed photos) into our line-crop dataset format.

Source: https://zenodo.org/records/21923181 (CC BY 4.0)
"""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = Path(
    r"C:\Users\erdem\AppData\Local\Temp\claude\c--Users-erdem-OneDrive-Masa-st--Makeshift-flipper"
    r"\2c6a9ecc-ed54-4ec7-9386-a87d3fd23487\scratchpad\zenodo_ds\dataset.zip"
)
MAX_TEXT_LENGTH = 24
ALPHABET = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ çğıİöşüÇĞÖŞÜ.,:;!?-_/()'%")

WIKI_PREFIX = "syntetic-turkish-dataset/wiki_ocr/wiki_ocr_zenodo/"
WIKI_GT = WIKI_PREFIX + "ground_truth.json"
GOLD_PREFIX = "syntetic-turkish-dataset/gold_standard_dataset/gold_standard_dataset/"
GOLD_GT = GOLD_PREFIX + "ground_truth.json"

OUTPUT_DIR = ROOT / "data" / "zenodo_lines"
OUTPUT_CSV = ROOT / "data" / "zenodo_labels.csv"


def is_usable(text: str) -> bool:
    text = text.strip()
    if not (1 <= len(text) <= MAX_TEXT_LENGTH):
        return False
    return set(text).issubset(ALPHABET)


def main() -> None:
    if not ZIP_PATH.exists():
        raise SystemExit(f"Missing downloaded archive: {ZIP_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())

        # --- wiki_ocr printed subset (synthetic, modern Turkish, short entries) ---
        wiki_gt = json.loads(archive.read(WIKI_GT))
        wiki_printed_paths = {
            Path(name).name: name
            for name in names
            if name.startswith(WIKI_PREFIX + "printed/") and name.lower().endswith((".png", ".jpg", ".jpeg"))
        }
        index = 0
        for filename, meta in wiki_gt.items():
            if meta.get("type") != "printed":
                continue
            text = meta.get("ground_truth", "")
            if not is_usable(text):
                continue
            zip_path = wiki_printed_paths.get(filename)
            if zip_path is None:
                continue
            target_name = f"wiki_{index:05d}{Path(filename).suffix}"
            target_path = OUTPUT_DIR / target_name
            target_path.write_bytes(archive.read(zip_path))
            rows.append({
                "image": f"zenodo_lines/{target_name}",
                "text": text.strip(),
                "group": f"zenodo_wiki_{meta.get('subcategory', 'unknown')}",
            })
            index += 1

        # --- gold standard printed subset (real photos) ---
        gold_gt = json.loads(archive.read(GOLD_GT))
        gold_printed_paths = {
            Path(name).name: name
            for name in names
            if name.startswith(GOLD_PREFIX + "printed/") and name.lower().endswith((".png", ".jpg", ".jpeg"))
        }
        gold_index = 0
        for filename, meta in gold_gt.items():
            if meta.get("type") != "printed":
                continue
            text = meta.get("ground_truth", "")
            if not is_usable(text):
                continue
            zip_path = gold_printed_paths.get(filename)
            if zip_path is None:
                continue
            target_name = f"gold_{gold_index:05d}{Path(filename).suffix}"
            target_path = OUTPUT_DIR / target_name
            target_path.write_bytes(archive.read(zip_path))
            rows.append({
                "image": f"zenodo_lines/{target_name}",
                "text": text.strip(),
                "group": f"zenodo_gold_{meta.get('subcategory', 'unknown')}",
            })
            gold_index += 1

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "text", "group"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} lines ({index} synthetic wiki + {gold_index} real gold) to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
