"""Gradio demo: upload a document photo, auto-split it into lines, and OCR each line."""
from __future__ import annotations

import csv
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 32
TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

alphabet_info = json.loads((ARTIFACTS / "alphabet.json").read_text(encoding="utf-8"))
BLANK_ID = alphabet_info["blank_id"]
ID_TO_CHAR = {index + 1: character for index, character in enumerate(alphabet_info["alphabet"])}

model = tf.keras.models.load_model(ARTIFACTS / "turkish_line_ocr.keras")


def greedy_ctc_decode(logits: np.ndarray) -> str:
    best = np.argmax(logits, axis=-1)
    chars = []
    previous = -1
    for index in best:
        if index != previous and index != BLANK_ID and index != 0:
            chars.append(ID_TO_CHAR.get(int(index), ""))
        previous = index
    return "".join(chars)


def preprocess(pil_image: Image.Image) -> np.ndarray:
    image = tf.convert_to_tensor(np.array(pil_image.convert("L")))
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image.numpy()


def recognize_line(pil_image: Image.Image) -> str:
    image = preprocess(pil_image)[np.newaxis, ...]
    logits = model.predict(image, verbose=0)[0]
    return greedy_ctc_decode(logits)


def detect_line_boxes(pil_image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Use Tesseract only to find line bounding boxes (classical layout detection),
    not to read the text itself — our own model does the recognition."""
    if not TESSERACT.exists():
        return []
    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="PNG")
    result = subprocess.run(
        [str(TESSERACT), "stdin", "stdout", "-l", "tur", "--psm", "6", "tsv"],
        input=buffer.getvalue(),
        capture_output=True,
        check=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        key = (row["block_num"], row["par_num"], row["line_num"], row["page_num"])
        grouped[key].append(row)

    boxes = []
    for words in grouped.values():
        left = min(int(w["left"]) for w in words)
        top = min(int(w["top"]) for w in words)
        right = max(int(w["left"]) + int(w["width"]) for w in words)
        bottom = max(int(w["top"]) + int(w["height"]) for w in words)
        boxes.append((left, top, right, bottom))
    boxes.sort(key=lambda box: box[1])
    return boxes


def recognize(pil_image: Image.Image | None) -> str:
    if pil_image is None:
        return ""
    boxes = detect_line_boxes(pil_image)
    if not boxes:
        return recognize_line(pil_image)
    pad = 4
    results = []
    for left, top, right, bottom in boxes:
        crop = pil_image.crop((
            max(0, left - pad), max(0, top - pad),
            min(pil_image.width, right + pad), min(pil_image.height, bottom + pad),
        ))
        results.append(recognize_line(crop))
    return "\n".join(results)


demo = gr.Interface(
    fn=recognize,
    inputs=gr.Image(type="pil", label="Bir belge fotoğrafı yükle (tek satır veya çok satırlı)"),
    outputs=gr.Textbox(label="Model tahmini (her satır ayrı satırda)", lines=10),
    title="Türkçe Satır OCR Demo",
    description=(
        "Bir görüntü yükleyin. Satır konumları Tesseract'ın klasik metin-tespit (layout "
        "detection) özelliğiyle bulunur, ancak **metnin okunması tamamen bizim kendi "
        "modelimiz** tarafından yapılır — Tesseract burada sadece 'nerede bir satır var' "
        "sorusuna cevap veriyor. Model yalnızca **basılı/yazdırılmış** metin üzerinde "
        "eğitildi, el yazısı desteklenmiyor."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
