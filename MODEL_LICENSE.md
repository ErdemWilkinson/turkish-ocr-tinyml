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

## Attribution

- Turkish Scene Text Recognition (TS-TR) dataset, Serdar Yıldız, CC BY-NC 4.0.
  <https://www.kaggle.com/datasets/serdaryildiz/turkish-scene-text-recognition-dataset>
- Turkish OCR Text Image Dataset, CC BY 4.0.
  <https://zenodo.org/records/21923181>

## Models trained without TS-TR

A model trained only on the synthetic lines, your own photos and the Zenodo
dataset is not bound by the non-commercial term. Train it by leaving
`data/tstr_labels.csv` out of `OCR_LABELS_CSV`.
