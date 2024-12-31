# MVTec AD Anomaly Detection Benchmark 🔍

This project implements and compares different approaches for industrial anomaly detection using the MVTec AD dataset. The benchmark includes both supervised and unsupervised methods across three levels of detection complexity.

## Project Overview

The project aims to benchmark six different anomaly detection methods:

### 1. Image-level Detection
- **Supervised**: ResNet-18 with fine-tuning
- **Unsupervised**: One-Class SVM with ResNet features

### 2. Anomaly Localization
- **Supervised**: Teacher-Student Network
- **Unsupervised**: PaDiM (Patch Distribution Modeling)

### 3. Anomaly Segmentation
- **Supervised**: U-Net
- **Unsupervised**: Autoencoder with skip connections

## Features

- Comprehensive evaluation of different anomaly detection approaches
- Implementation of both supervised and unsupervised methods
- Performance comparison across different detection levels
- Visualization tools for results analysis
- Standardized evaluation metrics for each detection level

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mvtec-anomaly-benchmark.git
cd mvtec-anomaly-benchmark
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Unix/macOS
# OR
venv\Scripts\activate  # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download MVTec AD dataset:
```bash
python scripts/download_dataset.py
```

## Project Structure

```
mvtec-anomaly-benchmark/
│
├── models/
│   ├── image_level/
│   │   ├── resnet.py
│   │   └── ocsvm.py
│   ├── localization/
│   │   ├── teacher_student.py
│   │   └── padim.py
│   └── segmentation/
│       ├── unet.py
│       └── autoencoder.py
│
├── utils/
│   ├── metrics.py
│   ├── visualization.py
│   └── data_loader.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── download_dataset.py
│
├── configs/
│   └── model_configs.yaml
│
└── notebooks/
    └── results_analysis.ipynb
```

## Usage

1. Train models:
```bash
python scripts/train.py --model [model_name] --type [supervised/unsupervised]
```

2. Evaluate models:
```bash
python scripts/evaluate.py --model [model_name]
```

3. View results:
```bash
jupyter notebook notebooks/results_analysis.ipynb
```

## Evaluation Metrics

### Image-level Detection
- Accuracy
- ROC-AUC
- F1 Score

### Anomaly Localization
- AUROC
- AUPRO (Area Under the Per-Region Overlap)

### Segmentation
- IoU (Intersection over Union)
- Dice Coefficient

## Results

Results will be presented in a comparative format:

| Model | Approach | Accuracy | ROC-AUC | F1 Score | Training Time |
|-------|----------|----------|----------|-----------|---------------|
| ResNet | Supervised | - | - | - | - |
| OCSVM | Unsupervised | - | - | - | - |
| ... | ... | ... | ... | ... | ... |

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
