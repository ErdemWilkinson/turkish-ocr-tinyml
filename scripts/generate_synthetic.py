"""Generate a varied synthetic Turkish line-OCR dataset."""
from __future__ import annotations

import csv
import random
from pathlib import Path

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
SIZES = (14, 16, 18, 20, 22, 24, 26)


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


def render_line(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    background = RANDOM.randint(210, 255)
    foreground = RANDOM.randint(0, 60)
    image = Image.new("L", (160, 32), color=background)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(1, RANDOM.randint(1, max(1, 156 - text_w)))
    y = max(0, (32 - text_h) // 2 - bbox[1] + RANDOM.randint(-2, 2))
    draw.text((x, y), text, fill=foreground, font=font)

    if RANDOM.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=RANDOM.uniform(0.3, 0.9)))
    if RANDOM.random() < 0.25:
        noise = Image.effect_noise((160, 32), RANDOM.uniform(8, 25))
        image = Image.blend(image, noise.convert("L"), alpha=RANDOM.uniform(0.05, 0.15))
    if RANDOM.random() < 0.2:
        shift = RANDOM.uniform(0.85, 1.15)
        image = image.point(lambda p: max(0, min(255, int(p * shift))))
    return image


def main() -> None:
    fonts = available_fonts()
    rows: list[dict[str, str]] = []
    for index in range(COUNT):
        font_name, font_path = RANDOM.choice(fonts)
        size = RANDOM.choice(SIZES)
        font = ImageFont.truetype(str(font_path), size)
        text = make_text()
        if not text:
            continue
        image = render_line(text, font)
        relative = Path("lines") / "synthetic" / f"{index:06d}.png"
        target = ROOT / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
        rows.append({
            "image": str(relative).replace("\\", "/"),
            "text": text,
            "group": f"{font_name}_{size}",
        })

    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "text", "group"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} samples at {OUTPUT} using {len(fonts)} fonts.")


if __name__ == "__main__":
    main()
