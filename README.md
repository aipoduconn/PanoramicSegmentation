# Tooth Segmentation and Numbering on Panoramic Radiographs

Code for the study **"Deep learning–based tooth segmentation and numbering on panoramic radiographs: development and performance in patients with impacted maxillary canines."**

A DeepLabV3 model with a ResNet-50 backbone labels every pixel of a panoramic radiograph as background or as one of 32 permanent teeth, so each tooth is outlined and numbered in a single pass.

---

## Overview

| | |
|---|---|
| **Task** | Semantic segmentation (33 classes: background + 32 teeth) |
| **Model** | `torchvision` DeepLabV3-ResNet50, ImageNet-pretrained, with an auxiliary head |
| **Dataset** | 1,543 de-identified multicenter panoramic radiographs (1,080 train / 120 validation / 343 test) |
| **Annotation** | Pixel-wise tooth outlines in [COCO Annotator](https://github.com/jsbroks/coco-annotator), labeled by quadrant (UR, UL, LL, LR) and tooth number (1–8) |
| **Metric** | Intersection over union (IoU), per tooth and mean over the 32 tooth classes |

### Model development

The model was developed in four sequential steps:

| Step | Change | Training images | Input size | Mean IoU |
|---|---|---|---|---|
| 1 | Baseline, no augmentation | 467 | 256 × 256 | 0.401 |
| 2 | + Data augmentation | 4,540 | 256 × 256 | 0.554 |
| 3 | + Full training set | 12,098 | 256 × 256 | 0.594 |
| 4 | + Higher resolution | 12,098 | 512 × 512 | **0.678** |

Step 1 was evaluated on 320 radiographs; steps 2–4 on the 343-radiograph test set. Excluding third molars, the final model reached a mean IoU of 0.710. See the paper for the per-tooth results and the comparison with human annotation in 41 radiographs with impacted maxillary canines.

> **Data availability:** The radiographs and annotations are not included in this repository because they are patient data. This repository contains code only.

---

## Repository structure

```
.
├── config/
│   └── train-config.yaml          # Hydra training configuration
├── data/                          # Empty placeholders; see data/README.md
│   ├── annots/                    # COCO-format annotation JSON files
│   ├── mask/                      # Generated label masks
│   ├── metadata/                  # Train/test CSV indexes
│   └── raw_imgs/                  # Panoramic radiographs
├── models/                        # Local checkpoints (if MLflow logging is off)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_evaluation.ipynb
│   ├── 03_impacted_canine_evaluation.ipynb
│   └── 04_augmentation_examples.ipynb
├── src/
│   ├── data/
│   │   ├── augmentation.py        # Rotation, translation, gamma/brightness/contrast/sharpness
│   │   ├── preprocess.py          # COCO → masks, train/test split, augmentation
│   │   └── utils.py               # Tooth class map, mask generation, visualization
│   └── model/
│       ├── models.py              # DeepLabV3 model definition
│       ├── train.py               # Training loop (Accelerate + Hydra + MLflow)
│       └── inference.py           # Test-set evaluation (per-class IoU, accuracy, F1)
└── requirements.txt
```

---

## Installation

Python 3.9 or 3.10 is recommended.

```bash
git clone https://github.com/aipoduconn/PanoramicSegmentation.git
cd PanoramicSegmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended. The published model was trained on an NVIDIA T4.

---

## Usage

### 1. Prepare the data

Put the radiographs and COCO annotation files in `data/` as described in [`data/README.md`](data/README.md), then run:

```bash
python src/data/preprocess.py --test_size 0.2
```

This script:
- converts the COCO polygons into single-channel label masks (pixel value = class index),
- splits the radiographs into training and test sets *before* augmentation, so that augmented copies never leak into the test set,
- generates augmented copies of the training images,
- writes `data/metadata/train_data.csv` and `data/metadata/test_data.csv`.

Run `python src/data/preprocess.py --help` to see all augmentation options.

### 2. Train

Training metrics and checkpoints are logged to [MLflow](https://mlflow.org/). Start a local tracking server, or point to your own:

```bash
mlflow server --host 127.0.0.1 --port 5000 &
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Train with the default configuration from `config/train-config.yaml`:

```bash
python src/model/train.py
```

You can override any parameter from the command line with [Hydra](https://hydra.cc/) syntax:

```bash
python src/model/train.py params.epochs=80 params.img_size.height=256 params.img_size.width=256
```

For multi-GPU training, run the same script with `accelerate launch src/model/train.py`.

### 3. Evaluate

Evaluate one or more logged checkpoints on the test set:

```bash
python src/model/inference.py "s3://<bucket>/artifacts/<experiment_id>/<run_id>/artifacts/models/model_epoch_74"
```

This prints the per-class and mean IoU, accuracy, and F1 score, both with and without the background class.

---

## Training configuration

The main settings are in `config/train-config.yaml`:

| Parameter | Value |
|---|---|
| Optimizer | AdamW, learning rate 5 × 10⁻⁴ |
| LR schedule | Cosine with 50 warm-up steps |
| Batch size | 24 (train), 8 (validation) |
| Input size | 512 × 512 |
| Loss | Class-weighted cross-entropy (background weight 0.005, tooth classes 0.1) + 0.5 × auxiliary-head loss |
| Training data | Original + augmented images (`params.data.use_augmented=false` for the step 1 baseline) |
| Validation | 10% of the training radiographs, split by radiograph so that augmented copies stay in the same subset as their original |
| Checkpointing | Every 2 epochs |

---

## Tooth classes

Teeth are labeled by quadrant and position, from 1 (central incisor) to 8 (third molar):

| Index | Teeth |
|---|---|
| 0 | Background |
| 1–8 | Upper right: UR8 → UR1 |
| 9–16 | Upper left: UL1 → UL8 |
| 17–24 | Lower left: LL8 → LL1 |
| 25–32 | Lower right: LR1 → LR8 |

The full mapping and the visualization colors are defined in `src/data/utils.py`.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{panoramic_tooth_segmentation,
  title   = {Deep learning--based tooth segmentation and numbering on panoramic radiographs:
             development and performance in patients with impacted maxillary canines},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD}
}
```

Citation details will be updated once the paper is published.
