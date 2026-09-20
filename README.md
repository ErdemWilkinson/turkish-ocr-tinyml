# Turkish TinyML Line OCR

A compact, full-int8 CNN-CTC recognizer for a single horizontal line of
printed Turkish text, sized to eventually run on an embedded target (the
[makeshift-flipper](https://github.com/ErdemWilkinson/makeshift-flipper)
ESP32-P4 device). This repository was split out of that firmware repo
because it's a separate concern (Python/TensorFlow training pipeline vs.
ESP-IDF C firmware) with its own, much larger dataset.

**Status: training pipeline works end-to-end; not yet integrated into any
firmware.** See [Hardware integration status](#hardware-integration-status)
below for exactly what's blocking that and what's already decided.

## Why a custom model instead of an existing OCR package

This was a deliberate choice, made before any training code was written —
see [DECISION_NOTES.md](DECISION_NOTES.md) for the full reasoning. Short
version: PaddleOCR/PP-OCR and similar general OCR stacks assume a phone or
a real application processor, not a memory-constrained MCU; Tesseract isn't
a TinyML model at all; and per-character classification breaks on kerning,
accents, and variable spacing. A small grayscale CNN + CTC sequence
recognizer for one pre-cropped text line is the smallest model that still
handles variable-length Turkish text correctly.

## Model

- Input: `160 x 32 x 1` grayscale image (one cropped, mostly-horizontal
  text line)
- Architecture: 4 conv+batchnorm+pool blocks → 2-layer BiLSTM → dense
  softmax over the character set (see `scripts/train.py`'s `build_models()`)
- Output: per-timestep character logits (40 timesteps — `IMAGE_WIDTH // 4`,
  matching the two width-reducing stride-2 pools), including a CTC blank
  class
- Character set (`ALPHABET` in `scripts/train.py`): digits, ASCII
  upper/lowercase, the Turkish letters `ç ğ ı İ ö ş ü Ç Ğ Ö Ş Ü`, a space,
  and `.,:;!?-_/()'%`. Max decoded text length: 24 characters.
- Decoder: greedy CTC (argmax per timestep + collapse repeats/blanks) —
  no beam search in this version, deliberately, to keep post-processing
  trivial on-device
- Quantization: full int8 (weights, activations, input, output) via
  `scripts/export_tflite.py`
- Text line detection/cropping is **not** part of this model — classical
  preprocessing (thresholding, morphology, connected components) is
  expected to produce the line crop before this model ever runs. No
  neural detector in this version; see `DECISION_NOTES.md` §4 for why.

## Repository layout

```text
ocr/
├── scripts/
│   ├── generate_synthetic.py    # builds data/lines/synthetic/ + labels.csv rows
│   ├── prepare_real_lines.py    # crops real document photos into labeled lines
│   ├── train.py                 # trains the CNN-CTC model, writes artifacts/
│   ├── evaluate.py               # CER + exact-line accuracy on held-out groups
│   ├── make_representative.py   # rebuilds the int8 calibration sample set
│   ├── export_tflite.py         # float .keras -> full-int8 .tflite
│   └── evaluate_tflite.py       # sanity-checks the exported .tflite
├── data/
│   ├── wordlists/tr_50k.txt      # tracked: small (~700KB) input corpus
│   ├── lines/, real_lines/       # NOT tracked: generated/derived images
│   └── labels.csv                # NOT tracked: dataset manifest (image,text,group)
├── artifacts/
│   ├── alphabet.json             # tracked: the character set, hand-decided
│   └── *.keras, *.tflite, *.npy  # NOT tracked: trained model outputs
├── DECISION_NOTES.md             # the model-selection writeup, moved here from
│                                  # the firmware repo's OCR_TINYML_NOTES.md
└── requirements-train.txt
```

**What's tracked in git and what isn't:** only code, the wordlist corpus,
and the alphabet config are tracked. All generated/derived data (synthetic
line images, real-photo crops, the training manifest, and every trained
model artifact) is gitignored — regenerate it locally with the commands
below. This keeps the repo small and avoids committing personal training
photos.

## Dataset layout

```text
ocr/data/lines/<split>/<id>.png
ocr/data/labels.csv
```

`labels.csv` columns: `image,text,group`.

```csv
image,text,group
lines/train/000001.png,İstanbul,generated_font_01
lines/train/000002.png,şifre 123,phone_photo_01
```

`group` drives the train/validation split (`GroupShuffleSplit` in
`train.py`, 15% held out): images from the same font/background/source
group must never appear in both splits, or validation numbers become
optimistic. `train.py` enforces at least 2 distinct groups and rejects any
row whose text is empty, over 24 characters, or contains a character
outside `ALPHABET`.

Images should contain one mostly horizontal printed text line, and the
dataset overall should cover Turkish diacritics, upper/lowercase, punctuation,
blur, perspective distortion, uneven lighting, sensor noise, and a few
invalid/empty crops (so the model learns what "no usable text" looks like,
not just clean lines).

### Building the dataset

Two sources feed `data/labels.csv`, and both are scripted (nothing here is
committed pre-built):

1. **Synthetic** (`scripts/generate_synthetic.py`): renders 18,000 lines
   from `data/wordlists/tr_50k.txt` using PIL, with randomized fonts,
   backgrounds, blur, and perspective (fixed `random.Random(42)` seed for
   reproducibility). Writes to `data/lines/synthetic/` and appends to
   `labels.csv`.
2. **Real** (`scripts/prepare_real_lines.py`): takes photographed document
   pages (not tracked in this repo — keep them locally, e.g. in a personal
   `eğitim verisi/`-style folder outside version control) and crops them
   into individual line images plus a preliminary transcription pass under
   `data/real_lines/`, merged into `labels.csv` as `real_lines/...` rows.
   `train.py`'s representative-sample logic specifically balances real vs.
   synthetic samples (`row["image"].startswith("real_lines/")`) so the
   int8 calibration set isn't all-synthetic.

## Training pipeline

From this directory, in a Python 3.11 environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt

python scripts\generate_synthetic.py     # builds the synthetic split
python scripts\prepare_real_lines.py     # optional: only if you have real photos locally
python scripts\train.py                  # trains, writes artifacts/turkish_line_ocr.keras
python scripts\evaluate.py               # reports CER + exact-line accuracy
python scripts\make_representative.py    # rebuilds the int8 calibration sample set
python scripts\export_tflite.py          # writes artifacts/turkish_line_ocr_int8.tflite
python scripts\evaluate_tflite.py        # sanity-checks the quantized model
```

`train.py` respects two environment variables if you want to point it at a
different manifest/output location without editing the script:
`OCR_LABELS_CSV` (colon/semicolon-separated list of `labels.csv` paths,
useful for combining datasets) and `OCR_ARTIFACTS` (output directory,
defaults to `artifacts/`).

Training uses `Adam(1e-3)` with gradient clipping, early stopping on
validation loss (patience 6), and LR reduction on plateau — up to 150
epochs, whichever early-stopping hits first.

## Acceptance targets

Before any of this is treated as ready to integrate into firmware, on a
source/font/background-independent held-out set:

- Turkish character error rate (CER): ≤ 5%
- Exact-line accuracy: ≥ 80% for short controlled lines
- No unsupported operators in the target embedded TFLite runtime
- Model + tensor arena size measured on real target hardware, not guessed
- First-pass latency target: ≤ 1 second per cropped line

`scripts/evaluate.py` reports CER and exact-line accuracy against the same
group-based held-out split `train.py` used.

### Current measured results (2026-09-20, `data/labels.csv`, 3681 held-out samples)

```
python scripts/evaluate.py --labels labels.csv
```

- Character error rate: **13.09%** (target: ≤ 5%)
- Exact-line accuracy: **57.35%** (target: ≥ 80%)

**Not yet at the acceptance bar, but not unusable either.** Most errors in
the sample predictions are small — a dropped trailing word/character or a
digit substitution (`"akl'ma buraday'm yapmas'"` → `"akl'ma buraday' 2"`),
not wholesale garbage. This reads as an undertrained/underfit baseline
(more epochs, more synthetic data variety, or real-photo fine-tuning are
the obvious next levers) rather than an architecture that doesn't work at
all — contrast this with `voice/`'s CTC baseline (Round 17 in
`KNOWN_ISSUES.md`), which produces character soup, not near-misses. This
number has not been re-measured against a real device-representative test
set (only the training pipeline's own synthetic held-out split, generated
by the same process as the training data — see `DECISION_NOTES.md` for
why that's a weaker signal than font/background/source-independent data
would be).

## Hardware integration status

This pipeline was deliberately built **before** committing to firmware
integration, because on-device OCR has a hardware dependency this repo
alone can't resolve: **the current makeshift-flipper firmware has no
camera driver, camera pin assignment, frame buffer, or TinyML runtime
integration at all.** The device's only current image-adjacent hardware is
a 128x64 SSD1306 OLED, which is an output-only display and cannot supply
an image to this model.

Before writing any firmware integration code, these need to be confirmed
against the real board:

- Exact ESP32-P4-Pico revision and installed PSRAM
- Camera module + sensor interface (DVP/MIPI) and actual free pin budget
- Whether the camera can deliver grayscale or only RGB565 frames
- Flash/PSRAM headroom left after the existing firmware links
- Target embedded runtime for this ESP-IDF version: LiteRT/TFLite Micro vs.
  ESP-DL
- The real use case this needs to serve: printed labels/cards vs. general
  scene text (this model is scoped for the former — see
  `DECISION_NOTES.md` §5 for what it explicitly does *not* promise)

See `DECISION_NOTES.md` for the full writeup this scoping came from,
including why a scene-text detector + recognizer pipeline was rejected as
the *first* version (too much memory/latency/data for an unproven need).

## Related repositories

- [makeshift-flipper](https://github.com/ErdemWilkinson/makeshift-flipper) —
  the ESP32-P4/C6 firmware this model is ultimately meant to run on
- [turkish-asr-whisper](https://github.com/ErdemWilkinson/turkish-asr-whisper) —
  the sibling TinyML pipeline (offline Turkish whisper-command recognition),
  split out for the same reason (separate concern, separate dataset)
