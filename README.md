# A Reproducible Framework for Vision Model Benchmarking

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An integrated, configuration-driven framework for fine-tuning and exhaustively benchmarking computer vision models, with built-in support for **RF-DETR**, **YOLO**, and **DINOv3**. This repository provides a clean, reproducible workflow to train multiple model architectures and generate unified, comprehensive performance reports.

---

## Key Features

* **Multi-Model Support**: Natively handles **RF-DETR**, **YOLO**, and **DINOv3** models within the same structured environment.
* **YAML-Driven Experiments**: Easily define and manage all training hyperparameters for any model in simple `.yaml` configuration files.
* **Automated Workflows**: Run all training and benchmarking jobs sequentially with single commands.
* **Comprehensive Benchmarking**: Automatically evaluate all trained models across multiple confidence thresholds.
* **Rich Performance Metrics**: Generates detailed CSV reports with metrics including mAP, Mask IoU (with SAM), inference time, and false positive rates.
* **Advanced Visualization**: Automatically saves visual outputs for both correct detections (mask + bounding box) and false positives (bounding box) in neatly organized directories specific to each model run.
* **Guaranteed Reproducibility**: With configuration-based runs and a fixed random seed, your results are easily reproducible.

---

## Repository Structure

To ensure a clean and scalable workflow, the project is organized by model type into three main directories, `rf-detr`, `yolo`, and `dinov3`.

```
model-benchmark-suite/
├── rf-detr/
│   ├── configs/
│   │   └── (RF-DETR .yaml files...)
│   ├── finetune.py
│   ├── benchmark.py
│   ├── output/
│   │   └── (RF-DETR training outputs...)
│   └── results/
│       ├── visualizations/
│       ├── false_positives/
│       └── rfdetr_benchmark_results.csv
│
├── yolo/
│   ├── configs/
│   │   └── (YOLO .yaml files...)
│   ├── finetune.py
│   ├── benchmark.py
│   ├── output/
│   │   └── (YOLO training outputs...)
│   └── results/
│       ├── visualizations/
│       ├── false_positives/
│       └── yolo_benchmark_results.csv
│
├── dinov3/
│   ├── configs/
│   │   └── (DINOv3 .yaml files...)
│   ├── model/
│   │   └── (DINOv3 model weights...)
│   ├── train.py
│   ├── benchmark.py
│   ├── output/
│   │   └── (DINOv3 training outputs...)
│   └── results/
│       ├── visualizations/
│       ├── false_positives/
│       └── dinov3_benchmark_results.csv
│
├── data/
│   ├── coco_dataset_bbox/
│   ├── test_images/
│   ├── test_gt_masks/
│   └── reject/
│
├── sam_vit_h_4b8939.pth
└── requirements.txt
```

---

## Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd model-benchmark-suite
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data**
    * Place your datasets in the appropriate subdirectories within the `data/` folder.
    * Download the SAM checkpoint (`vit_h` model) from the [official repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and place the `sam_vit_h_4b8939.pth` file in the project's root directory.

---

## Workflow

The project uses a simple, two-step workflow: **Train**, then **Benchmark**. These steps are performed independently for each model architecture from within their respective directories.

### For RF-DETR

1.  **Fine-Tune All Models**
    The script automatically finds and runs every `.yaml` file in the `rf-detr/configs/` directory.
    ```bash
    cd rf-detr
    python finetune.py
    ```
    Trained models and config copies are saved to `rf-detr/output/`.

2.  **Run Comprehensive Benchmarking**
    This script finds all trained RF-DETR models and evaluates them.
    ```bash
    python benchmark.py
    ```
    Results are saved to `rf-detr/results/rfdetr_benchmark_results.csv` and the corresponding visualization folders.

### For YOLO

1.  **Fine-Tune All Models**
    The training script finds and runs every `.yaml` file in `yolo/configs/`.
    ```bash
    cd yolo
    python finetune.py
    ```
    Trained models and config copies are saved to `yolo/output/`.

2.  **Run Comprehensive Benchmarking**
    This script finds all trained YOLO models and evaluates them.
    ```bash
    python benchmark.py
    ```
    Results are saved to `yolo/results/yolo_benchmark_results.csv` and the corresponding visualization folders.

### For DINOv3

1.  **Clone the DINOv3 repository and download model weights**
    ```bash
    cd dinov3
    git clone https://github.com/facebookresearch/dinov3
    # Follow instructions on the DINOv3 GitHub page to download pre-trained model weights.
    # Place the downloaded weights in the `dinov3/model` directory or update the finetune script path accordingly.
    ```

2.  **Fine-Tune All Models**
    The training script finds and runs every `.yaml` file in `dinov3/configs/`.
    ```bash
    python train.py
    ```
    Trained models and config copies are saved to `dinov3/output/`.

3.  **Run Comprehensive Benchmarking**
    This script finds all trained DINOv3 models and evaluates them.
    ```bash
    python benchmark.py
    ```
    Results are saved to `dinov3/results/dinov3_benchmark_results.csv` and the corresponding visualization folders.

## Future Work

* **Better Datasets:** Use datasets with more images to improve performance
* **Change Segmentation Head:** Swap out the simple convolutional segmentation head for a more sophisticated architecture.
* **Improve Dataset Ingestion:** Develop a system for managing datasets across multiple formats (e.g. YOLO <-> COCO)
