# GoogLeNet Blood Cell Classification

> Deep CNN with Inception Modules (~6.8M params) for 4-class Blood Cell Classification

## Overview

This project implements a custom **GoogLeNet (Inception V1)** architecture for classifying microscopic blood cell images into 4 white blood cell types: **Eosinophil, Lymphocyte, Monocyte, Neutrophil**.

### Architecture Highlights
- **9 Inception modules** with parallel 1x1, 3x3, 5x5 convolutions + max pooling
- **2 auxiliary classifiers** for gradient injection during training
- **Global Average Pooling** instead of FC layers
- **BatchNorm** after every convolution
- **~6.8M parameters**

## Dataset

- **Name**: Blood Cell Images (BCCD)
- **Source**: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
- **Size**: ~12,500 augmented microscopic JPEG images
- **Classes**: Eosinophil, Lymphocyte, Monocyte, Neutrophil

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (train + evaluate)
python main.py

# Train only
python main.py --mode train

# Evaluate only (requires trained checkpoint)
python main.py --mode evaluate
```

## Project Structure

```
google_net/
├── configs/config.yaml      # Hyperparameters
├── src/
│   ├── model.py              # GoogLeNet + InceptionModule + AuxClassifier
│   ├── dataset.py            # Kaggle download, augmentation, loaders
│   ├── train.py              # Training loop with auxiliary loss
│   ├── evaluate.py           # Metrics & visualization
│   └── utils.py              # Plotting, Grad-CAM, checkpoints
├── report/
│   ├── report.tex            # IEEE conference paper
│   └── references.bib        # Bibliography
├── main.py                   # Entry point
├── requirements.txt          # Dependencies
└── README.md
```

## Outputs

After training, the following artifacts are generated:

```
outputs/
├── checkpoints/best_model.pth    # Best model weights
├── plots/
│   ├── training_curves.png       # Loss & accuracy curves
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── roc_curves.png            # Per-class ROC curves
│   └── f1_scores.png             # Per-class F1 bar chart
├── gradcam/gradcam_grid.png      # Grad-CAM visualizations
├── metrics/test_metrics.json     # All metrics in JSON
└── logs/training_log.csv         # Per-epoch training log
```

## Evaluation Metrics

| Metric | 
|--------|
| Accuracy |	85.12% |
| Weighted F1 |	0.8547 |
| Macro F1 |	0.8548 |
| Macro AUC-ROC | 0.9625 |
| Precision |	0.8714 |
| Recall |	0.8512 |


## References

- Szegedy et al., "Going Deeper with Convolutions", CVPR 2015
- Blood Cell Images Dataset, Kaggle (Paul Mooney)
