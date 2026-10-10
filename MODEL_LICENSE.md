# License for trained models

The [MIT License](LICENSE) in this repository covers the source code only.
It does not cover trained model files (`*.keras`, `*.tflite`) produced by
that code, whether or not they are distributed with this repository.

## Models trained with TS-TR

From 2026-10-09 the training set includes the Turkish Scene Text Recognition
(TS-TR) dataset by Serdar Yıldız, which is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Any model trained with TS-TR is released under
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**:

- You may use, share and adapt the model for non-commercial purposes.
- You must give credit to this project and to the datasets listed below.
- You may not use the model for commercial purposes.

## Models trained with the 2026-10-09 evening data

Models trained on the additional word crops are bound by these terms as well:

- Synthetic Turkish Scene Text (STS-TR), CC BY-NC 4.0: non-commercial.
- esengul3 Turkish word OCR, CC BY-SA 4.0: attribution and share-alike.
- orkungedik OCR Turkish word dataset: **no license is stated**. Do not
  redistribute a model trained on it until the author clarifies.

The model in `artifacts/` from 2026-10-10 onward was trained with all three,
so all three points apply to it. The last model trained without them is the
2026-10-09 morning one.

## Attribution

- Synthetic Turkish Scene Text Recognition (STS-TR), Serdar Yıldız, CC BY-NC 4.0.
  <https://www.kaggle.com/datasets/serdaryildiz/synthetic-turkish-scene-text-recognition-dataset>
- Turkish Word OCR, esengul3, CC BY-SA 4.0.
  <https://huggingface.co/datasets/esengul3/turkish-word-ocr>
- OCR Turkish Word Dataset, orkungedik (no license listed).
  <https://huggingface.co/datasets/orkungedik/ocr_turkish_word_dataset>
- Turkish Scene Text Recognition (TS-TR) dataset, Serdar Yıldız, CC BY-NC 4.0.
  <https://www.kaggle.com/datasets/serdaryildiz/turkish-scene-text-recognition-dataset>
- Turkish OCR Text Image Dataset, CC BY 4.0.
  <https://zenodo.org/records/21923181>

## Models trained without TS-TR

A model trained only on the synthetic lines, your own photos and the Zenodo
dataset is not bound by the non-commercial term. Train it by leaving
`data/tstr_labels.csv` out of `OCR_LABELS_CSV`.
