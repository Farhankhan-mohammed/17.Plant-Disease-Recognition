# Plant Disease Recognition Using MobileNet

**MSc Data Science - Deep Learning Applications (CMP-L016)**

Project 17: Plant Disease Recognition Using MobileNet Variants

## Project Overview

This project implements an automated plant disease classification system using transfer learning with MobileNetV2. The model can identify 23 different disease categories across 5 crop types (Apple, Corn, Pepper, Potato, and Tomato) from leaf images. Trained on the PlantVillage dataset containing 35,725 images, the model achieves **93.69% validation accuracy**.

## Results Summary

| Metric              | Value                 |
| ------------------- | --------------------- |
| Validation Accuracy | **93.69%**      |
| Validation Loss     | 0.1899                |
| Total Images        | 35,725                |
| Classes             | 23                    |
| Training Samples    | 28,589                |
| Validation Samples  | 7,136                 |
| Model Parameters    | 2.4M (167K trainable) |

## Project Structure

```
plant-disease-recognition/
├── notebooks/
│   └── plant_disease_classification.ipynb    # Main training notebook
├── figures/
│   ├── class_distribution.png               # Dataset class distribution
│   ├── sample_images.png                    # Sample images per class
│   ├── augmented_samples.png                # Data augmentation examples
│   ├── mobilenet_accuracy.png               # Training accuracy curve
│   ├── mobilenet_loss.png                   # Training loss curve
│   ├── confusion_matrix.png                 # Model confusion matrix
│   ├── per_class_accuracy.png               # Per-class accuracy
│   └── sample_predictions.png               # Sample predictions
├── models/
│   └── plant_disease_mobilenetv2.h5         # Trained model (11MB)
├── report/
│   ├── main.pdf                     
├── class_labels.json                         # Class label mapping
├── classification_report.txt                # Detailed metrics
├── requirements.txt                          # Dependencies
└── README.md                                 # This file
```

## Quick Start

### Option 1: Run on Kaggle (Recommended)

1. Go to [Kaggle](https://www.kaggle.com/)
2. Create new notebook
3. Upload `plant_disease_classification.ipynb`
4. Enable GPU: Settings → Accelerator → GPU P100
5. Run all cells

### Option 2: Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run all cells

## Dataset

**Source:** [Plant Disease Detection on Kaggle](https://www.kaggle.com/datasets/karagwaanntreasure/plant-disease-detection)

### Classes (23 total):

- Apple: Black rot, Cedar rust, Scab, Healthy
- Corn: Cercospora, Common rust, Northern blight, Healthy
- Pepper: Bacterial spot, Healthy
- Potato: Early blight, Late blight, Healthy
- Tomato: Multiple diseases and Healthy

## Model Architecture

```
MobileNetV2 (ImageNet weights, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(23, Softmax)
```

### Training Configuration

- Optimizer: Adam (lr=0.0001)
- Loss: Categorical Cross-Entropy
- Epochs: 15
- Batch Size: 32
- Image Size: 224×224

## Generated Figures

| Figure                     | Description                         |
| -------------------------- | ----------------------------------- |
| `class_distribution.png` | Bar chart of images per class       |
| `sample_images.png`      | Sample leaf images from each class  |
| `mobilenet_accuracy.png` | Training/validation accuracy curves |
| `mobilenet_loss.png`     | Training/validation loss curves     |
| `confusion_matrix.png`   | 23×23 confusion matrix             |
| `per_class_accuracy.png` | Accuracy breakdown by disease       |
| `sample_predictions.png` | Example predictions with confidence |

## Requirements

```
tensorflow>=2.10.0
numpy
pandas
matplotlib
seaborn
scikit-learn
kagglehub
```

## References

1. Mohanty et al. (2016) - Deep learning for plant disease detection
2. Howard et al. (2017) - MobileNets architecture
3. Sandler et al. (2018) - MobileNetV2

## Author

Farhan Khan Mohammed (A00051779)
MSc Data Science
University of Roehampton

---

**Submission:** December 2025
