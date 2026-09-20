# Turkish OCR TinyML Decision Notes

> Moved here from `OCR_TINYML_NOTES.md` in the makeshift-flipper firmware
> repo, where this analysis originated (the model was scoped against that
> device's actual hardware before this training pipeline existed). See
> that repo's [README](https://github.com/ErdemWilkinson/makeshift-flipper)
> for the current firmware/peripheral state.

## 1. Project understanding

- Main firmware: ESP-IDF application for an ESP32-P4-Pico.
- Current peripherals: 128x64 SSD1306 OLED, analog joystick, buttons, RFID/NFC readers, IR receiver/transmitter, vibration motor.
- Companion firmware: ESP32-C6 for Wi-Fi and BLE scanning, connected to the P4 through UART.
- The existing `voice/` (now `turkish-asr-whisper`) pipeline trains a compact Keras model and exports full-int8 TensorFlow Lite (`.tflite`) for an embedded target.
- The current firmware has no camera driver, camera pin assignment, image buffer, OCR code, or TinyML runtime integration.
- The OLED is an output display only; it cannot provide an image to OCR.

## 2. Important hardware conclusion

OCR cannot be added as a model-only feature. The device first needs an image source:

1. A supported camera connected to the ESP32-P4, or
2. A phone/PC/C6-side image sender over Wi-Fi/UART.

For true on-device OCR, option 1 is the intended route. The camera choice, pixel format, resolution, frame buffer location, and available PSRAM must be confirmed before firmware integration.

## 3. Selected TinyML approach

### Selected model: custom constrained Turkish line OCR

Use a small grayscale CNN + CTC sequence recognizer, exported as a full-int8 TensorFlow Lite model and executed with LiteRT/TensorFlow Lite Micro or the ESP32-P4's supported embedded inference stack.

Recommended first input contract:

- Input: one cropped text line, grayscale, fixed size approximately `160x32` pixels.
- Output: character logits over image columns plus a CTC blank class.
- Alphabet: digits, ASCII letters, and Turkish letters `ç ğ ı İ ö ş ü` plus punctuation required by the use case.
- Decoder: greedy CTC decoder first; beam search is optional later and should run outside the model.
- Quantization: full int8 for weights, activations, input, and output.
- Preprocessing: grayscale, contrast normalization, resize with aspect-ratio preservation, and padding.
- Text detection: classical image processing first (thresholding, morphology, connected components); do not add a second neural detector in version 1.

This is the best first TinyML target for this hardware because it avoids a large general-purpose OCR stack while still supporting variable-length Turkish text within a known line crop.

## 4. Why other common choices are not selected first

- PaddleOCR/PP-OCR: useful for phones, Linux, or a stronger application processor, but the normal detection + recognition pipeline is too large and operationally complex for the first P4 firmware milestone.
- Tesseract: not a TinyML neural model; memory, font data, and runtime behavior are not a good fit for this first embedded integration.
- Character-by-character classification: very small, but requires reliable character segmentation and fails on touching/kerning characters, accents, and variable spacing.
- Whisper or speech models: unrelated to image OCR; the sibling `turkish-asr-whisper` model cannot be reused for text recognition.
- A scene-text detector plus recognizer: more general, but it multiplies memory, latency, and data requirements. Add it only if line-crop OCR is proven insufficient.

## 5. Scope of the first usable version

The first model should read deliberately captured, reasonably high-contrast text lines such as:

- RFID/NFC labels or printed identifiers,
- short device labels,
- menu or status text from a controlled distance,
- Turkish words and short phrases on a mostly horizontal baseline.

It should not initially promise arbitrary photographs, full pages, handwriting, curved text, or low-light distant signs.

## 6. Training plan

Build an OCR training pipeline as its own repository rather than changing the main firmware first:

1. Define a versioned Turkish alphabet and maximum line length.
2. Collect or generate line-level images with speaker-independent concerns replaced by source/font/background splits.
3. Include Turkish diacritics, uppercase/lowercase, punctuation, blur, perspective, illumination changes, sensor noise, and empty/invalid crops.
4. Split by source/font/background generation group so near-duplicate images cannot leak into validation.
5. Train the float model with CTC loss.
6. Export representative image tensors and convert to full-int8 `.tflite`.
7. Measure model byte size, tensor arena requirement, latency, character error rate (CER), exact-line accuracy, and Turkish-character accuracy.
8. Run a small host-side int8 inference test before touching firmware.
9. Add camera capture and inference to firmware only after the model passes the host test.

Suggested initial acceptance targets:

- Turkish character error rate: <= 5% on a speaker/source-independent test equivalent.
- Exact-line accuracy: >= 80% for short controlled lines.
- No unsupported TFLite operators in the embedded runtime.
- Model plus tensor arena sized from measured P4 memory, not guessed values.
- A practical first-pass latency target of <= 1 second per cropped line.

## 7. Decision boundary before implementation

Before writing firmware integration code, confirm these hardware facts:

- Exact ESP32-P4-Pico board revision and installed PSRAM.
- Camera module and sensor interface (DVP/MIPI), including actual pin availability.
- Whether the camera can provide grayscale or RGB565 frames.
- Available flash and PSRAM after the current firmware is linked.
- Whether the chosen embedded runtime is LiteRT/TFLite Micro or ESP-DL for this ESP-IDF version.
- OCR use case: printed text, card text, screen text, labels, or general scene text.

## Final selection

Start with **a custom full-int8 CNN-CTC Turkish line OCR model**, not a general-purpose OCR package. Keep detection classical and restrict version 1 to one cropped horizontal text line. This gives the project a realistic path to train, measure, and deploy on the ESP32-P4 while leaving room for a larger detector/recognizer pipeline later if the real use case requires it.

## Status update (this repository)

The training pipeline described above is now implemented — see the main
[README.md](README.md) for the current dataset/scripts/acceptance-target
state. The hardware decision boundary in §7 is still **unresolved**: no
camera has been selected or wired to the P4 yet, so nothing here has been
integrated into the firmware repo.

## 8. Export-time LSTM graph-explosion bug (found and fixed)

A real device-integration audit against the trained `turkish_line_ocr_int8.tflite`
found that the exported graph had 2,308 ops instead of the low dozens
expected for this architecture — not a size problem (the file was a
reasonable 1.45MB, peak activation RAM ~320KB) but a per-op interpreter
overhead problem: TFLite Micro pays a fixed cost per op, and 2,308 of them
would make inference either very slow or the firmware image too large to
build comfortably.

**Root cause:** `scripts/train.py`'s two `Bidirectional(LSTM(...))` layers
were built with `unroll=True`. That flag tells Keras to statically unroll
the recurrent loop into individual dense ops at graph-build time (useful
for training-time XLA fusion on GPU), which leaves the TFLite converter
with no cell-level structure to recognize and fuse — it exports each of
the 40 timesteps × 2 layers × 2 directions worth of gate math
(ADD/MUL/LOGISTIC/TANH/FULLY_CONNECTED/SPLIT) as separate ops.

**Fix (both changes required together — either alone still fails):**

1. `unroll=False` (the default) on both LSTM layers in `recognition_body()`
   (`scripts/train.py`).
2. `scripts/export_tflite.py` no longer converts `turkish_line_ocr.keras`
   directly. It rebuilds the recognition graph on a **statically-shaped**
   `batch_shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)` input via the same
   `recognition_body()` function, copies the trained weights across with
   `set_weights()`, and converts *that* model. This second step is not
   optional: with `unroll=False` alone, on the original flexible-batch
   (`None, ...`) input, the converter fails outright —
   `'tf.TensorListReserve' op requires element_shape to be static during
   TF Lite transformation pass` — because the LSTM's internal
   `TensorList` ops need every shape in the graph, including the batch
   dimension, to be static before they can be lowered. Batch size 1
   matches how the device actually calls the model (one cropped line at
   a time), so nothing is lost by fixing it.

**Verified end-to-end** (fake/random weights, since this only tests the
export path's correctness, not trained accuracy — a real retrain is still
needed to produce a usable model): op count dropped from 2,308 to **28**
(4 of them genuine fused `UNIDIRECTIONAL_SEQUENCE_LSTM` ops, one per
direction per layer, plus `REVERSE_V2` for the backward direction), file
size dropped from 1.45MB to **641KB**, weight transfer confirmed exact,
input/output tensor shapes unchanged (`1×32×160×1` → `1×40×90`), and peak
activation RAM stayed ~320KB (expected — the actual per-timestep
computation is identical, only its graph *representation* shrank).

**Consequence: the existing `artifacts/turkish_line_ocr.keras` and
`artifacts/turkish_line_ocr_int8.tflite` were trained/exported before this
fix and still have the 2,308-op graph baked in.** A full retrain is
required — this is a code fix, not a re-export; the saved `.keras` file's
graph already has `unroll=True` compiled into its structure. Re-run
`scripts/train.py` end-to-end once real training data is available, then
`scripts/export_tflite.py`.
