"""Create reviewable line crops and preliminary Turkish labels from document photos."""
from __future__ import annotations

import csv
import io
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eğitim verisi"
OUTPUT = ROOT / "data" / "real_lines"
MANIFEST = ROOT / "data" / "real_labels.csv"
TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
MAX_TEXT_LENGTH = 24
MIN_CONFIDENCE = 55.0
ALPHABET = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ çğıİöşüÇĞÖŞÜ.,:;!?-_/()'")


def tsv_lines(image_path: Path) -> list[tuple[str, tuple[int, int, int, int], float]]:
    result = subprocess.run(
        [str(TESSERACT), str(image_path), "stdout", "-l", "tur", "--psm", "6", "tsv"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    reader = csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
    for row in reader:
        if row["level"] != "5" or not row["text"].strip():
            continue
        try:
            confidence = float(row["conf"])
        except ValueError:
            continue
        if confidence < MIN_CONFIDENCE:
            continue
        key = (row["block_num"], row["par_num"], row["line_num"], row["page_num"])
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
    if not TESSERACT.exists():
        raise SystemExit(f"Tesseract not found: {TESSERACT}")
    images = sorted(SOURCE.glob("*.jpeg"))
    if not images:
        raise SystemExit(f"No JPEG images found in {SOURCE}")

    rows: list[dict[str, str]] = []
    output_index = 0
    for image_path in images:
        image = Image.open(image_path).convert("L")
        for text, (left, top, right, bottom), confidence in tsv_lines(image_path):
            pad = 4
            crop = image.crop((max(0, left - pad), max(0, top - pad), min(image.width, right + pad), min(image.height, bottom + pad)))
            relative = Path("real_lines") / f"{output_index:06d}.png"
            target = ROOT / "data" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)
            rows.append({
                "image": str(relative).replace("\\", "/"),
                "text": text,
                "group": image_path.stem,
                "source": image_path.name,
                "confidence": f"{confidence:.1f}",
            })
            output_index += 1

    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "text", "group", "source", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(rows)} preliminary line labels from {len(images)} document photos.")
    print(f"Review {MANIFEST} before using these labels for final training.")


if __name__ == "__main__":
    main()
