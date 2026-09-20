"""Rebuild representative_images.npy with a balanced mix of synthetic and real samples."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 32


def load_image(path: Path) -> np.ndarray:
    image = tf.io.read_file(str(path))
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image.numpy()


def main() -> None:
    rng = np.random.default_rng(7)

    with (ROOT / "data" / "real_labels.csv").open(encoding="utf-8", newline="") as handle:
        real_rows = list(csv.DictReader(handle))
    with (ROOT / "data" / "labels.csv").open(encoding="utf-8", newline="") as handle:
        synthetic_rows = list(csv.DictReader(handle))

    real_sample = rng.choice(len(real_rows), size=min(150, len(real_rows)), replace=False)
    synthetic_sample = rng.choice(len(synthetic_rows), size=min(150, len(synthetic_rows)), replace=False)

    images = []
    for i in real_sample:
        images.append(load_image(ROOT / "data" / real_rows[i]["image"]))
    for i in synthetic_sample:
        images.append(load_image(ROOT / "data" / synthetic_rows[i]["image"]))

    array = np.stack(images).astype(np.float32)
    np.save(ARTIFACTS / "representative_images.npy", array)
    print(f"Wrote {array.shape[0]} representative images ({len(real_sample)} real, {len(synthetic_sample)} synthetic).")


if __name__ == "__main__":
    main()
