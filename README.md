# WildCareAI2 🐾

A comprehensive, end-to-end computer vision pipeline for wildlife object detection. This repository automates the process of dataset preparation, class mapping, balancing, and multi-model training using state-of-the-art deep learning architectures. 

It is designed to be highly reproducible and includes memory optimizations (like mixed precision training and optimized batch sizes) to run efficiently on consumer GPUs (e.g., RTX 40-series).

## ✨ Features

- **Automated Data Preparation Pipeline**: Handles downloading, class mapping, cleaning, balancing, and dataset splitting (Train/Val/Test).
- **Multi-Model Support**: Standardized training scripts for:
  - **YOLOv8** (Nano & Medium)
  - **Faster R-CNN** (ResNet-50 FPN)
  - **EfficientDet** (D2)
  - **DETR** (ResNet-50)
- **Consumer GPU Optimized**: Built-in AMP (Automatic Mixed Precision) support, adjusted batch sizes, and optimized hyperparameter settings to fit within 4-6GB VRAM limits (perfect for RTX 4050 and similar cards).
- **Unified Evaluation**: Evaluate all trained models consistently with `evaluate_all.py` and visualize results.

## 🛠️ Requirements

- **Python**: 3.9+ recommended
- **CUDA**: 11.8+ (for GPU acceleration)
- **PyTorch**: Compatible with your CUDA version

## 📊 Dataset

The pipeline uses a custom curated dataset focusing on 4 wildlife classes:
- `buffalo`
- `elephant`
- `rhino`
- `zebra`

The data preparation steps (`step1` through `step5`) handle downloading raw annotations, mapping synonymous labels (e.g., "african elephant" -> "elephant"), filtering out minority classes, balancing the class distribution, and creating a standard Train/Val/Test split in YOLO format.

## 📈 Results (Evaluation)

The models were evaluated on the test set using standard COCO metrics. Here is a summary of the **mAP@50** (Mean Average Precision at IoU 0.50) scores achieved:

| Model | Overall mAP@50 | Buffalo | Elephant | Rhino | Zebra |
|-------|----------------|---------|----------|-------|-------|
| **YOLOv8m** | **90.4%** | 80.5% | 95.4% | 97.9% | 87.8% |
| **Faster R-CNN** | **90.0%** | 79.5% | 89.2% | **99.1%** | **92.2%** |
| **YOLOv8n** | **88.5%** | 78.3% | 91.6% | 95.9% | 88.2% |
| **DETR** | **85.8%** | 76.5% | 86.3% | 94.2% | 86.0% |
| **EfficientDet** | **84.9%** | 76.7% | 86.2% | 96.7% | 79.8% |

*(Detailed metrics including AP at different scales and AR can be found in the `results/` directory post-evaluation.)*

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AaroAarav/WildCareAI2.git
   cd WildCareAI2
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r req.txt
   ```

## 📋 Usage

The project is structured into three main phases:

### 1. Data Preparation
Run the complete data preprocessing pipeline (steps 2 through 7):
```bash
python prepare_all.py
```
*This will execute class mapping, data cleaning, balancing, splitting, and config generation sequentially.*

### 2. Model Training
Train all configured models (YOLOv8m, YOLOv8n, Faster R-CNN, EfficientDet, DETR):
```bash
python train_all.py
```
*Checkpoints and hyperparameters will be saved in the `checkpoints/` and `results/` directories.*

### 3. Evaluation
Evaluate the models to generate performance metrics and comparisons:
```bash
python evaluate_all.py
```

## 📂 Project Structure

```text
WildCareAI2/
├── req.txt                      # Project dependencies
├── prepare_all.py               # Master script for dataset preparation
├── train_all.py                 # Master script to train all models
├── evaluate_all.py              # Master script for evaluation
├── step1_download_inspect.py    # Download and inspect dataset
├── step2_class_mapping.py       # Map original classes to target classes
├── step3_convert_and_clean.py   # Format conversion and cleaning
├── step4_balance.py             # Dataset balancing
├── step5_split.py               # Train/Val/Test splitting
├── step6_generate_configs.py    # YAML config generation for YOLO
├── step7_validate_and_report.py # Final validation report
├── train_*.py                   # Individual model training scripts
└── utils.py                     # Helper functions
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
