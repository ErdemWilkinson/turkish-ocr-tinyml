"""Generate a varied synthetic Turkish line-OCR dataset."""
from __future__ import annotations

import csv
import io
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "lines"
LABELS = ROOT / "data" / "labels.csv"
WORDLIST = ROOT / "data" / "wordlists" / "tr_50k.txt"
COUNT = 18000
RANDOM = random.Random(42)

# Domain-specific vocabulary for the device's expected use case (menus, labels, documents).
DOMAIN_WORDS = [
    "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Konya", "Adana", "Kayseri",
    "Çalışma", "şifre", "Giriş", "Çıkış", "yap", "Kart", "oku", "WiFi", "tara",
    "Güvenli", "bağlantı", "ölçüm", "Kızılötesi", "Bluetooth", "menü", "aç", "kapat",
    "geri", "dön", "iptal", "onayla", "HATA", "AĞ", "YOK", "VAR", "Ürün", "adı",
    "çağrı", "başlat", "durdur", "kayıt", "ayarlar", "bilgi", "uyarı", "başarılı",
    "başarısız", "bağlan", "kes", "tekrar", "dene", "devam", "et", "bekle",
    "Belge", "No", "Tarih", "Saat", "Ad", "Soyad", "Adres", "Telefon", "Fatura",
    "Ödeme", "Tutar", "TL", "Miktar", "Toplam", "Vergi", "Numarası", "Şirket",
    "Müşteri", "Hesap", "Bakiye", "İşlem", "Onay", "Red", "Beklemede", "Tamamlandı",
    "su", "elektrik", "doğalgaz", "internet", "abone", "tüketim", "okuma",
    "cihaz", "sensör", "pil", "şarj", "sinyal", "mesafe", "hız", "sıcaklık",
    "nem", "basınç", "voltaj", "akım", "direnç", "frekans", "kanal", "mod",
    "profil", "kullanıcı", "parola", "e-posta", "kod", "doğrulama", "erişim",
    "izin", "reddedildi", "sistem", "güncelleme", "sürüm", "hafıza", "depolama",
    "dosya", "klasör", "yazıcı", "tarayıcı", "kamera", "mikrofon", "hoparlör",
    "ekran", "parlaklık", "kontrast", "renk", "boyut", "dil", "Türkçe", "İngilizce",
]

# Characters the recognizer alphabet supports (train.py ALPHABET, minus digits/punctuation
# that are handled separately by SUFFIXES).
_ALLOWED_LETTERS = set("abcdefghijklmnopqrstuvwxyzçğıöşü")


def load_frequency_words(limit: int = 4000) -> list[str]:
    if not WORDLIST.exists():
        return []
    words: list[str] = []
    with WORDLIST.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(" ")
            if not parts or not parts[0]:
                continue
            word = parts[0]
            if len(word) < 2 or len(word) > 15:
                continue
            if not set(word.lower()).issubset(_ALLOWED_LETTERS):
                continue
            words.append(word)
            if len(words) >= limit:
                break
    return words


WORDS = DOMAIN_WORDS + load_frequency_words()

SUFFIXES = ["", ":", ".", ",", "!", "?", "-", "%", "01", "12", "2026", "42", "123", "(1)", "/A"]

FONT_NAMES = [
    "segoeui.ttf", "arial.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf",
    "consola.ttf", "cour.ttf", "georgia.ttf", "trebuc.ttf", "corbel.ttf",
    "candara.ttf", "constan.ttf", "framd.ttf", "micross.ttf", "segoeuib.ttf",
    "arialbd.ttf", "calibrib.ttf", "timesbd.ttf", "times.ttf",
]

FONT_DIR = Path("C:/Windows/Fonts")
# Pixel size the line is *rendered* at before being cropped and resized to
# 160x32. The model never sees a point size: only how much detail survives the
# resize, so small values simply give blurrier, low-resolution crops.
RENDER_SIZES = (10, 12, 14, 16, 20, 24, 32, 40)
CANVAS_W, CANVAS_H = 160, 32
NP_RNG = np.random.default_rng(42)
_PROBE = ImageDraw.Draw(Image.new("L", (1, 1)))


@lru_cache(maxsize=None)
def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def available_fonts() -> list[tuple[str, Path]]:
    fonts = []
    for name in FONT_NAMES:
        path = FONT_DIR / name
        if path.exists():
            fonts.append((path.stem, path))
    if not fonts:
        raise SystemExit("No usable TrueType fonts found under C:/Windows/Fonts.")
    return fonts


def make_text() -> str:
    n = RANDOM.choice([1, 1, 2, 2, 3, 3, 4])
    parts = [RANDOM.choice(WORDS) for _ in range(n)]
    text = " ".join(parts)
    suffix = RANDOM.choice(SUFFIXES)
    text = f"{text}{suffix}" if not suffix or suffix[0] in ":.!?%" else f"{text} {suffix}"
    return text[:24].rstrip()


def draw_native(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Draw the whole text at its natural size on a tightly cropped canvas.

    Nothing is clipped, however long the text is: the crop is later resized to
    160x32 exactly like a real line crop, so long lines are squeezed and short
    ones stretched. The box is either the ink bounds (what a text detector
    returns) or the full ascender-to-descender line box, half of the time each.
    """
    if RANDOM.random() < 0.2:  # faded / low-contrast print
        background, foreground = RANDOM.randint(170, 230), RANDOM.randint(60, 130)
    else:
        background, foreground = RANDOM.randint(210, 255), RANDOM.randint(0, 60)
    left, top, right, bottom = _PROBE.textbbox((0, 0), text, font=font)
    if RANDOM.random() < 0.5:
        box_top, box_bottom = top, bottom
    else:
        ascent, descent = font.getmetrics()
        box_top, box_bottom = 0, ascent + descent
    width, height = right - left, box_bottom - box_top
    pad_x = RANDOM.randint(1, max(2, height // 2))
    pad_y = RANDOM.randint(1, max(2, height // 4))
    image = Image.new("L", (width + 2 * pad_x, height + 2 * pad_y), color=background)
    ImageDraw.Draw(image).text((pad_x - left, pad_y - box_top), text, fill=foreground, font=font)
    return image


def add_illumination(image: Image.Image) -> Image.Image:
    """Uneven lighting: a linear brightness ramp across a random direction."""
    width, height = image.size
    angle = RANDOM.uniform(0, 2 * math.pi)
    ramp = math.cos(angle) * np.linspace(-1, 1, width)[None, :] + math.sin(angle) * np.linspace(-1, 1, height)[:, None]
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-6)
    factor = 1.0 - RANDOM.uniform(0.1, 0.4) * ramp
    pixels = np.asarray(image, dtype=np.float32) * factor
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def camera_degrade(image: Image.Image) -> Image.Image:
    """What a phone camera does to a crop before it reaches the recognizer.

    Applied at the crop's native resolution, in capture order (geometry,
    lighting, optics, sensor, compression), and only then is the crop resized
    to the model's input size, as in the real pipeline.
    """
    height = image.height
    if RANDOM.random() < 0.3:  # slight camera tilt
        image = image.rotate(RANDOM.uniform(-2.5, 2.5), resample=Image.BILINEAR, expand=True,
                             fillcolor=int(np.asarray(image)[0, 0]))
    if RANDOM.random() < 0.3:
        image = add_illumination(image)
    # Softness is chosen once, never stacked: a defocus blur on top of a
    # low-resolution capture leaves nothing a human could read, and a label
    # without legible ink only teaches the model to hallucinate. The radius is
    # tied to the crop height because the final resize stretches short crops
    # a lot horizontally; a fixed native radius would be multiplied by that.
    softness = RANDOM.random()
    if softness < 0.45:  # defocus / motion softness
        image = image.filter(ImageFilter.GaussianBlur(radius=RANDOM.uniform(0.4, 1.4) * height / 32))
    elif softness < 0.7 and height > 14:  # low-resolution sensor
        scale = max(RANDOM.uniform(0.4, 0.75), 10 / height)  # keep at least ~10 px of text height
        small = (max(4, int(image.width * scale)), max(4, int(image.height * scale)))
        image = image.resize(small, Image.BILINEAR)
    if RANDOM.random() < 0.35:  # sensor noise
        pixels = np.asarray(image, dtype=np.float32)
        pixels = pixels + NP_RNG.normal(0.0, RANDOM.uniform(2, 12), pixels.shape)
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    if RANDOM.random() < 0.4:  # JPEG compression artifacts
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=RANDOM.randint(35, 80))
        image = Image.open(buffer).convert("L")
    return image


def render_line(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    image = camera_degrade(draw_native(text, font))
    return image.resize((CANVAS_W, CANVAS_H), Image.BILINEAR)


def main() -> None:
    fonts = available_fonts()
    rows: list[dict[str, str]] = []
    skipped = 0
    for index in range(COUNT):
        font_name, font_path = RANDOM.choice(fonts)
        font = load_font(font_path, RANDOM.choice(RENDER_SIZES))
        text = make_text()
        if not text:
            skipped += 1
            continue
        image = render_line(text, font)
        relative = Path("lines") / "synthetic" / f"{index:06d}.png"
        target = ROOT / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
        rows.append({
            "image": str(relative).replace("\\", "/"),
            "text": text,
            "group": font_name,  # size no longer exists as a property of the crop
        })

    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "text", "group"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} samples at {OUTPUT} using {len(fonts)} fonts ({skipped} skipped).")


if __name__ == "__main__":
    main()
