# multimodal-plan-analysis
Fine-tuning LayoutLMv3 on CubiCasa5k for multi-modal architectural plan understanding and room classification.

# Multi-Modal Document Understanding for Architectural Plans 🏗️ 📄

![Status](https://img.shields.io/badge/Status-Active_Development-green)
![Tech](https://img.shields.io/badge/Model-LayoutLMv3-blue)
![Data](https://img.shields.io/badge/Dataset-CubiCasa5k-orange)

## 📌 Project Overview
This project implements a **multi-modal deep learning pipeline** to analyze unstructured architectural floor plans. By fine-tuning **LayoutLMv3** on the **CubiCasa5k** dataset, the system fuses visual features (document layout) with textual semantics (room labels) to accurately classify and segment building components (e.g., Kitchen, Bedroom, Entryway).

Unlike standard object detection (YOLO), this approach leverages **Geometric Transformations** and **Transformer-based multi-modal fusion** to resolve ambiguities in structurally similar rooms.

## 🚀 Key Features (Implemented & Planned)
- **SVG Parsing Pipeline:** Custom preprocessing script to parse raw `model.svg` polygons into normalized bounding boxes (0-1000 scale).
- **Multi-Modal Data Loader:** A custom PyTorch `Dataset` class that handles image rasterization, text tokenization, and bbox alignment simultaneously.
- **LayoutLMv3 Fine-Tuning:** Implementation of the Hugging Face `LayoutLMv3ForTokenClassification` architecture.
- **Benchmarking:** Comparative analysis against a vision-only baseline (YOLOv8) to demonstrate the lift from adding semantic text features.

## 🛠️ Tech Stack
- **Core:** Python, PyTorch, Hugging Face Transformers
- **Data Processing:** NumPy, Pandas, PIL, xml.etree (SVG parsing)
- **Model:** Microsoft LayoutLMv3 (Pre-trained on IIT-CDIP)
- **Dataset:** [CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k)

## 📂 Repository Structure
```bash
├── data/
│   ├── raw/              # Raw CubiCasa SVG and PNG files
│   └── processed/        # Normalized JSON annotations for LayoutLM
├── notebooks/
│   ├── 01_svg_parsing.ipynb    # Exploratory Data Analysis & SVG Visualization
│   └── 02_model_training.ipynb # Fine-tuning loop (Colab compatible)
├── src/
│   ├── dataset.py        # Custom PyTorch Dataset class
│   └── utils.py          # Geometric transformation helper functions
└── results/              # Confusion matrices and inference examples
