"""Create reviewable line crops and labels from document & code photos.

Experimental, and kept apart from prepare_real_lines.py on purpose: its crops
go to data/code_lines/ and its manifest to data/code_real_labels.csv. An
earlier version wrote into data/real_lines/ and data/real_labels.csv, which
silently replaced the crops the real-photo labels and test_labels.csv refer to.

The manifest is NOT loadable by train.py as is: lines go up to 60 characters
and the alphabet includes code symbols, while train.py takes 1-24 characters
from its 88-character alphabet. Labels are Tesseract's own output
(confidence >= MIN_CONFIDENCE), not human-verified.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [
    ROOT / "eğitim verisi" / "kullanılabilir",
    ROOT / "eğitim verisi" / "şüpheli_parlamış",
    ROOT / "eğitim verisi",
]
OUTPUT = ROOT / "data" / "code_lines"
MANIFEST = ROOT / "data" / "code_real_labels.csv"
TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
MAX_TEXT_LENGTH = 60
MIN_CONFIDENCE = 40.0
ALPHABET = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ çğıİöşüÇĞÖŞÜ.,:;!?-_/()'\"{}[]=+*<>#@$&|\\~`")


def tsv_lines(image_path: Path) -> list[tuple[str, tuple[int, int, int, int], float]]:
    result = subprocess.run(
        [str(TESSERACT), str(image_path), "stdout", "-l", "tur+eng", "--psm", "6", "tsv"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    reader = csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
    for row in reader:
        if not row.get("level") or row["level"] != "5" or not row.get("text") or not row["text"].strip():
            continue
        try:
            confidence = float(row["conf"])
        except ValueError:
            continue
        if confidence < MIN_CONFIDENCE:
            continue
        key = (row.get("block_num", "1"), row.get("par_num", "1"), row.get("line_num", "1"), row.get("page_num", "1"))
        grouped[key].append(row)

    lines = []
    for words in grouped.values():
        words.sort(key=lambda word: int(word["left"]))
        text = " ".join(word["text"].strip() for word in words).strip()
        if not 1 <= len(text) <= MAX_TEXT_LENGTH or not set(text).issubset(ALPHABET):
            continue
        left = min(int(word["left"]) for word in words)
        top = min(int(word["top"]) for word in words)
        right = max(int(word["left"]) + int(word["width"]) for word in words)
        bottom = max(int(word["top"]) + int(word["height"]) for word in words)
        confidence = sum(float(word["conf"]) for word in words) / len(words)
        lines.append((text, (left, top, right, bottom), confidence))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help=f"overwrite an existing {MANIFEST.name}")
    args = parser.parse_args()
    if MANIFEST.exists() and not args.force:
        raise SystemExit(f"{MANIFEST} already exists; pass --force to regenerate it and the crops in {OUTPUT}.")
    if not TESSERACT.exists():
        raise SystemExit(f"Tesseract not found: {TESSERACT}")

    images = []
    for sdir in SOURCE_DIRS:
        if sdir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
                images.extend(sdir.glob(ext))
    
    # Remove duplicates
    images = list(dict.fromkeys(images))
    if not images:
        raise SystemExit("No images found for processing.")

    print(f"Toplam {len(images)} adet görsel taranacak ve satırlarına (line crop) ayrıştırılacak...")

    rows: list[dict[str, str]] = []
    output_index = 0
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(images):
        try:
            image = Image.open(image_path).convert("L")
            detected = tsv_lines(image_path)
            for text, (left, top, right, bottom), confidence in detected:
                pad = 4
                crop = image.crop((max(0, left - pad), max(0, top - pad), min(image.width, right + pad), min(image.height, bottom + pad)))
                relative = Path("code_lines") / f"{output_index:06d}.png"
                target = ROOT / "data" / relative
                crop.save(target)
                rows.append({
                    "image": str(relative).replace("\\", "/"),
                    "text": text,
                    "group": image_path.stem,
                    "source": image_path.name,
                    "confidence": f"{confidence:.1f}",
                })
                output_index += 1
        except Exception as e:
            print(f"Hata ({image_path.name}): {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(f"[{i + 1}/{len(images)}] görsel işlendi. Toplam çıkarılan satır: {output_index}")

    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "text", "group", "source", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ İşlem tamamlandı! {len(images)} fotoğraf üzerinden toplam {len(rows)} satır kırpma görseli (`data/code_lines/`) ve etiket dosyası (`data/code_real_labels.csv`) başarıyla oluşturuldu.")


if __name__ == "__main__":
    main()
