<div align="center">

# UAVid++: Higher-Quality Labels and Expanded Semantic Taxonomy for Aerial Semantic Segmentation

**[Vivian Chiciudean](https://users.utcluj.ro/~vivianc/), [Sergiu Nedevschi](https://users.utcluj.ro/~nedevschi/), [Florin Oniga](https://users.utcluj.ro/~onigaf/)**

Department of Computer Science, Technical University of Cluj-Napoca, Romania


[![Project Page](https://img.shields.io/badge/🌐_UAVid%2B%2B-Project_Page-1a73e8?style=flat&logoColor=white)](https://vivichiciudean.github.io/uavidplusplus/)
[![Paper](https://img.shields.io/badge/TGRS-Paper-00629b?style=flat&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11614898)
[![Preprint](https://img.shields.io/badge/📄_PDF-Preprint-b31b1b?style=flat&logo=adobeacrobatreader&logoColor=white)](https://users.utcluj.ro/~vivianc/papers/TGRS_2026.pdf)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21F?style=flat&logo=huggingface&logoColor=yellow)](https://huggingface.co/datasets/vivianchiciudean/uavidplusplus)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-FFD21F?style=flat&logo=huggingface&logoColor=yellow)](https://huggingface.co/datasets/vivianchiciudean/uavidplusplus)
[![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey?style=flat&logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


</div>

Official code for the IEEE Transactions on Geoscience and Remote Sensing (TGRS) 2026 paper **UAVid++**. This repository contains the training and inference protocol for the proposed segmentation approach: **frozen DINO-pretrained ViT backbones adapted with trainable task-specific heads**.

> ⭐ If you find UAVid++ useful, please consider giving the repo a star and citing our work.

---
## Overview
We employ an adaptation strategy that combines frozen DINO-pretrained ViT backbones with trainable task-specific heads for aerial scenarios with limited labeled data. This design preserves the representational power of Vision Transformers while quantifying the gains from improved labels relative to pretrained backbone capacity. Relative to the best-performing state-of-the-art method, our configurations achieve a **2.7% mIoU** improvement on UAVid++, up to **12.4% mIoU** gains on out-of-distribution UAV datasets, and a **38.85% mIoU** gain on a cross-domain dataset.

## Method
A **frozen** DINO-pretrained ViT backbone extracts features that are decoded by a lightweight **trainable** head. Three head variants are provided:
| Head | Description | Params (w/ ViT-H+) |
|---|---|---:|
| **Linear** | Single linear projection over patch features | ~15.3K |
| **Conv** | 2× (Conv–BN–ReLU) + 1×1 projection | ~22M |
| **UNet-inspired** | 3 downsample + `C/16` bottleneck + 3 upsample with skip connections + 1×1 projection — **best** | ~26M |


## Repository Structure
After completing the setup, your project directory should look like this:
```
uavidplusplus-code/
├── code
│   ├── dataloader.py
│   ├── dinov3_wrapper.py
│   ├── inference_strategy.py
│   ├── __init__.py
│   ├── loss.py
│   ├── metrics.py
│   ├── model_inference.py
│   ├── model_train.py
│   ├── __pycache__
│   ├── segmentation_heads.py
│   └── utils.py
├── commands.txt
├── README.md
├── data
│   ├── test
│   ├── train
│   └── val
├── data_processing
│   ├── 1_flatten.py
│   └── 2_preprocess_images.py
├── env.yml
├── models
│   ├── dinov3
│   ├── dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
│   ├── dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
│   └── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
└── output 
    ├── 1_9_dinov3_vits_unet_inv_uavid++_mask.png
    ├── 2_9_dinov3_vits_unet_inv_uavid++_colormask.png
    ├── 3_9_dinov3_vits_unet_inv_uavid++_img.png
    ├── 4_9_dinov3_vits_unet_inv_uavid++_overlayed.png
    ├── 5_9_dinov3_vits_unet_inv_uavid++_loss.png
    ├── 6_9_dinov3_vits_unet_inv_uavid++_iou.png
    ├── 7_9_dinov3_vits_unet_inv_uavid++_accuracy.png
    ├── best_model.pth
    ├── test_predictions_9_dinov3_vits_unet_inv_uavid++
    └── training_log_20260718_010918.txt
```
- `output/` – Created automatically when training from scratch. If you are using pretrained models instead, download the weights from [🤗 Hugging Face](https://huggingface.co/datasets/vivianchiciudean/uavidplusplus) and place them in this directory.
- `models/` – Created after completing [Step 1](#step-1---dinov3).
- `data/` – Created after completing [Step 2](#step-2---dataset-and-preprocessing).

## Installation and Running

```bash
git clone https://github.com/vivichiciudean/uavidplusplus-code.git
cd uavidplusplus-code

# Python 3.10+ recommended
conda env create -f env.yml
conda activate uavidpp

# Complete Step 1: Download the DINOv3 models
cd ./models
cd ..

# Complete Step 2: Download and prepare the dataset
cd ./data
cd ..

cd code

# Train from scratch
python model_train.py

# Or run inference with pretrained weights
# (see Step 3)
# python model_inference.py
```

## Step 1 – Set up DINOv3

Clone the official DINOv3 repository into the `models/` directory:
```bash
git clone https://github.com/facebookresearch/dinov3.git
```
Next, download the DINOv3 checkpoint(s) you want to use from the official Hugging Face collection:

> https://huggingface.co/collections/facebook/dinov3

Place the downloaded checkpoint files directly in the `models/` directory (alongside the `dinov3/` repository).


## Step 2 – Download and Preprocess the Dataset

Download the UAVid++ dataset and review the dataset documentation on 🤗 **[Hugging Face](https://huggingface.co/datasets/vivianchiciudean/uavidplusplus)**.

This guide assumes you are using the **full UAVid++ dataset**. If you choose a different dataset variant, you may need to adjust the preprocessing steps (and the dataloader) accordingly.

After downloading the dataset, preprocess it in the following order:

1. Run `1_flatten.py` separately for each split:
   - `uavid_train/`
   - `uavid_val/`
   - `uavid_test/`

2. Run `2_preprocess_images.py` to generate image crops for the:
   - training split (`uavid_train`)
   - validation split (`uavid_val`)

Before running either script, ensure your current working directory is correct. Also review the `TODO` comments in the preprocessing scripts and update any required paths or configuration before execution.



## Step 3 (Optional) – Use Pretrained Weights

If you do not want to train the model from scratch, you can use one of the provided pretrained checkpoints instead.

Pretrained checkpoints are available on 🤗 **[Hugging Face](https://huggingface.co/datasets/vivianchiciudean/uavidplusplus)** alongside the dataset. We currently provide checkpoints for the following configurations:

- **ViT-S UNet trained on UAVid++**
- **ViT-H+ UNet trained on UAVid++**
- **ViT-7B UNet trained on UAVid++**

Download the desired checkpoint and place it in the `output/` directory. You can then run inference directly using `model_inference.py`.

If you use a different DINOv3 backbone, you will need to train the model from scratch.


## Training and Inference

Before training or running inference, complete the setup described in [Step 1](#step-1--set-up-dinov3) and [Step 2](#step-2--download-and-preprocess-the-dataset). If you plan to use pretrained segmentation models, also complete [Step 3](#step-3-optional--use-pretrained-weights).

### Training Settings

The experiments reported in the paper use the following configuration:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Base learning rate | 3 × 10⁻⁵ |
| Weight decay | 0.001 |
| Batch size | 8 |
| Epochs | 40 |
| LR schedule | Linear warmup (5 epochs), then cosine annealing |
| Augmentations | Geometric + photometric + coarse dropout |
| Input tiling | Non-overlapping 1088 × 1088 tiles; 8 tiles per 4K frame (1,600 training + 560 validation tiles) |
| Hardware | 1 × NVIDIA A100 40 GB |

### Training

After selecting the DINOv3 backbone (e.g., ViT-H+) and the corresponding experiment configuration. The training script also performs inference.

```bash
python model_train.py
```

### Inference

To run only inference using a trained or pretrained checkpoint:

```bash
python model_inference.py
```

Inference uses the same **1088 × 1088** non-overlapping tiling strategy as training. Predictions from individual tiles are stitched together into a full-resolution segmentation map. Performance is reported using **per-class IoU** and **mean IoU (mIoU)**.

## Model Zoo & Main Results
We provide pretrained checkpoints for the following configurations:

- **DINOv3 ViT-7B + UNet** (best-performing model)
- **DINOv3 ViT-H+ + UNet**
- **DINOv3 ViT-S + UNet** (smoke test)

Proposed approach on **UAVid++** (mIoU %, train/test on UAVid++):

| Backbone | Head | mIoU |
|---|---|---:|
| DINOv3 ViT-L | UNet | 80.82 |
| DINOv3 ViT-H+ | Linear | 70.13 |
| DINOv3 ViT-H+ | Conv | 80.86 |
| DINOv3 ViT-H+ | UNet | 81.42 |
| **DINOv3 ViT-7B** | **UNet** | **82.04** |

**Head ablation** (mIoU %, Linear / Conv / UNet):

| Backbone | Linear | Conv | UNet |
|---|---:|---:|---:|
| DINOv3 ViT-L | 72.42 | 80.41 | 80.82 |
| DINOv3 ViT-H+ | 70.13 | 80.86 | 81.42 |
| DINOv3 ViT-7B | 74.95 | 81.99 | 82.04 |

## Generalization

**Out-of-distribution UAV datasets** (trained on UAVid++, tested on UDD and VDD - using IDD labels), mIoU %:

| Test set | D2LS | Ours (ViT-H+ UNet) |
|---|---:|---:|
| UDD | 55.46 | **67.89 (+12.4)** |
| VDD | 60.14 | **69.00 (+8.9)** |

**Cross-domain** (trained on UAVid++, tested on OpenEarthMap), mIoU %:

| Method | mIoU |
|---|---:|
| D2LS | 24.10 |
| **Ours (ViT-H+ UNet)** | **62.95 (+38.85)** |

## License

This code and the UAVid++ annotations are released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)** — https://creativecommons.org/licenses/by-nc-sa/4.0/.

## Citation

```bibtex
@article{chiciudean2026uavidplusplus,
  author={Chiciudean, Vivian and Nedevschi, Sergiu and Oniga, Florin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={UAVid++: Higher-Quality Labels and Expanded Semantic Taxonomy for Aerial Semantic Segmentation}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2026.3715191}}
```

## Acknowledgments
This work was supported by the "Romanian Hub for Artificial Intelligence – HRIA" project, Smart Growth, Digitization and Financial Instruments Program, MySMIS no. 351416, Ministry of Investments and European Projects, Romanian Government.
