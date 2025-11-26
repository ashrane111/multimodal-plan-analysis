# Multi-Modal Floor Plan Analysis with LayoutLMv3

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers%20%7C%20Datasets-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This project implements a **Multi-Modal Document Understanding** pipeline to automate the semantic analysis of complex architectural floor plans. Using the **CubiCasa5k** dataset (60,000+ annotated samples), it fine-tunes a **LayoutLMv3** Transformer to fuse visual features, text semantics, and spatial layout data.

Unlike standard object detectors (YOLO/ResNet) which treat plans as static images, this architecture interprets them as structured documents, allowing for precise classification of room types even when visual cues are ambiguous. The project features a high-throughput training pipeline optimized for **NVIDIA A100/V100 GPUs** using **Apache Arrow** for zero-copy data loading.

## 🚀 Key Features

### 🏗️ Data Engineering & Ontology Mapping
* **Custom SVG Parser:** Recursive XML parsing logic to extract polygon geometry and labels from raw vector files.
* **Semantic Normalization:** Implemented a regex-based **Ontology Mapping Layer** to consolidate over 1,000 messy/multilingual labels (e.g., *'MH1'*, *'Keittiö'*, *'Sleeping Qtrs'*) into **14 standardized classes** (e.g., *Bedroom*, *Kitchen*), resolving severe class imbalance.
* **Geometric Normalization:** Maps arbitrary coordinate systems to a fixed `[0, 1000]` integer space to ensure resolution invariance.

### ⚡ MLOps & High-Performance Computing
* **I/O Optimization:** Converted the raw dataset into **Apache Arrow** format (Hugging Face Datasets). This enabled memory mapping, eliminating small-file I/O bottlenecks and increasing GPU throughput by **~20x**.
* **HPC Orchestration:** Managed training via **Slurm Batch Jobs** on the Northeastern Explorer Cluster, utilizing **Offline Mode** to handle compute nodes without internet access.
* **Robust Inference:** Implemented token-to-word alignment logic to correct sub-word token shifts during inference, ensuring precise bounding box labeling.

## 📊 Performance Results
The model was fine-tuned on ~45,000 samples and validated on ~5,000 unseen plans.

| Metric | Score |
| :--- | :--- |
| **Macro F1-Score** | **99.27%** |
| **Recall** | **99.32%** |
| **Accuracy** | **99.43%** |
| **Training Time** | ~7 mins / epoch (V100) |

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone [https://github.com/ashrane111/multimodal-plan-analysis.git](https://github.com/ashrane111/multimodal-plan-analysis.git)
cd multimodal-plan-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install -U "accelerate>=0.26.0" "evaluate"
```

## 📂 Project Structure
```text
.
├── data/                   # Raw CubiCasa5k and Arrow binaries (Ignored)
├── models_output/          # Trained weights and visual outputs
├── logs/                   # Slurm execution logs
├── src/
│   ├── data/
│   │   ├── convert_to_arrow.py  # ETL script: SVG -> DataFrame -> Arrow
│   │   ├── svg_parser.py        # Parsing & Geometric Normalization logic
│   │   └── dataset.py           # PyTorch Dataset with Ontology Mapping
│   ├── models/
│   │   └── layoutlm_v3.py       # LayoutLMv3ForTokenClassification Wrapper
│   ├── train.py                 # Hugging Face Trainer loop
│   └── inference.py             # Visual Inference Engine
├── run_train.sh            # Slurm Batch Script for HPC
└── requirements.txt
```

## ⚡ Usage

### 1. Data Preparation (ETL)
Parse the raw SVGs and convert them to a memory-mapped Arrow dataset for speed:
```bash
python src/data/convert_to_arrow.py
```

### 2. Training (Slurm / HPC)
Submit the batch job to the cluster. This script handles module loading, environment activation, and multi-core data fetching.
```bash
sbatch run_train.sh
```
*Note: The script uses `eval_accumulation_steps=1` to prevent OOM on large validation sets.*

### 3. Inference & Visualization
Run the model on a random sample from the dataset to generate annotated visualizations:
```bash
python -m src.inference
```
*Output visualization is saved to `models_output/real_prediction_vis.png`*

## 🔬 Technical Approach

### Multi-Modal Fusion
The model processes three synchronized inputs to disambiguate rooms:
1.  **Visual Embedding:** A Vision Transformer (ViT) backbone analyzes image patches (e.g., recognizing a bed icon).
2.  **Text Embedding:** RoBERTa analyzes the text labels (e.g., recognizing "MBR" implies Bedroom).
3.  **Layout Embedding:** 2D positional embeddings represent the spatial bounding box of the room.

### Ontology & Class Mapping
To make the output useful for downstream code compliance (CivCheck), raw labels are mapped to a fixed schema:
* **Standard:** `Kitchen`, `Bathroom`, `Bedroom`, `LivingRoom`, `Dining`
* **Service:** `Utility`, `Storage`, `Garage`
* **Structural:** `Entry`, `Hallway`, `Stairs`, `Outdoor`
* **Ignored:** Dimension strings (e.g., `11' x 12'`) are mapped to `Background` and ignored during training via `label_id=-100`.

## 📜 License
Distributed under the MIT License.
