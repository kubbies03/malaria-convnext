# Malaria Cell Image Classification with ConvNeXt-Tiny

## Overview

This project performs binary classification of red blood cell microscopy images to detect malaria infection. Each image is classified as either **Parasitized** (infected with Plasmodium) or **Uninfected**. The primary objective is to achieve high accuracy while keeping training time reasonable by applying a 2-stage progressive resizing strategy.

Model: `convnext_tiny.in12k_ft_in1k` (ConvNeXt-Tiny pre-trained on ImageNet-12k, fine-tuned on ImageNet-1k)  
Platform: Kaggle (Tesla T4 x2)  
Best validation accuracy: **97.59%** (with Test-Time Augmentation)

---

## Dataset

Source: [Kaggle - Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

| Split      | Parasitized | Uninfected | Total  |
|------------|------------|-----------|--------|
| Train (80%) | 11,024     | 11,022    | 22,046 |
| Val (20%)  | 2,756      | 2,756     | 5,512  |
| **Total**  | **13,780** | **13,780**| **27,558** |

The dataset is perfectly balanced. The split is stratified by label with a fixed random seed (42) for reproducibility.

---

## Model Architecture

- **Backbone:** `convnext_tiny.in12k_ft_in1k` loaded via `timm`
- **Head:** Linear classifier with 2 output classes, dropout 0.2
- **Gradient checkpointing:** Enabled to reduce GPU memory usage at 384px resolution

---

## Training Strategy

### 2-Stage Progressive Resizing

The model is trained in two sequential stages on the same set of weights. Stage 1 uses a smaller resolution for fast warm-up; Stage 2 fine-tunes at a higher resolution to recover detail-sensitive accuracy.

| Stage   | Image size    | Batch size | Max epochs | Learning rate |
|---------|--------------|-----------|-----------|---------------|
| Stage 1 | 224 x 224 px | 128       | 5         | 5e-4          |
| Stage 2 | 384 x 384 px | 32        | 4         | 2e-4          |

### Optimizer and Scheduler

- **Optimizer:** AdamW (weight decay = 1e-4)
- **Scheduler:** OneCycleLR per stage (pct_start=0.2, div_factor=10, final_div_factor=200)
- **Loss:** CrossEntropyLoss with label smoothing = 0.02
- **Mixed Precision:** `torch.amp.autocast("cuda")` + `GradScaler("cuda")`
- **Early stopping:** patience = 2 epochs per stage (monitors val accuracy)
- **Checkpoint:** Best model weights saved to `best_convnext_224to384.pth`

### Augmentation Pipeline

| Phase      | Transforms |
|------------|-----------|
| Train      | RandomResizedCrop (scale 0.80-1.0) -> HorizontalFlip -> VerticalFlip -> Rotate (+-15 deg) -> ShiftScaleRotate -> RandomBrightnessContrast -> GaussNoise -> CoarseDropout -> Normalize |
| Val / Test | Resize -> Normalize |

Normalization uses ImageNet statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).

---

## Results

### Accuracy

| Evaluation method       | Val Accuracy | Val Loss |
|------------------------|-------------|---------|
| No TTA at 384 px       | 97.42%      | 0.1214  |
| TTA (4 ops) at 384 px  | 97.59%      | -       |

TTA applies four transforms to each image (identity, horizontal flip, vertical flip, 90-degree rotation) and averages the logits before taking the final prediction.

### Confusion Matrix (No TTA, 5,512 validation images)

```
                       Predicted: Uninfected   Predicted: Parasitized
Actual: Uninfected           2690                     66
Actual: Parasitized            76                   2680
```

### Classification Report (TTA)

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Uninfected  | 0.9743    | 0.9775 | 0.9759   | 2,756   |
| Parasitized | 0.9774    | 0.9742 | 0.9758   | 2,756   |
| **Accuracy**|           |        | **0.9759**| 5,512  |
| Macro avg   | 0.9759    | 0.9759 | 0.9759   | 5,512   |

---

## Environment

| Component     | Version               |
|---------------|-----------------------|
| Python        | 3.12.12               |
| PyTorch       | 2.8.0+cu126           |
| timm          | 1.0.20                |
| Albumentations| 2.0.8                 |
| OpenCV        | 4.12.0                |
| GPU           | Tesla T4 x2 (Kaggle)  |

Multi-GPU training via `nn.DataParallel` is enabled automatically when more than one GPU is available.

---

## Usage

1. On Kaggle, open a new notebook and add the dataset `iarunava/cell-images-for-detecting-malaria`.
2. Upload or paste `ConvNeXt.ipynb` and run all cells sequentially.
3. Dataset paths are discovered automatically via glob on `/kaggle/input/**/Parasitized` and `/kaggle/input/**/Uninfected` — no manual path configuration required.
4. The best checkpoint is written to `/kaggle/working/best_convnext_224to384.pth` at the end of training.
5. The final two evaluation cells report accuracy without and with TTA, along with the confusion matrix and per-class classification report.

If Stage 2 runs out of GPU memory, reduce `CFG.BS2` from 32 to 16 in the config cell.
