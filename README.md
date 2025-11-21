# Multi-Modal Floor Plan Analysis with LayoutLMv3

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This project implements a **Multi-Modal Document Understanding** pipeline to analyze complex architectural floor plans (CubiCasa5k format). It leverages **LayoutLMv3** to fuse visual features (images) with semantic layout data (text labels and spatial coordinates) to classify room types and structural elements.

This repository serves as the reference implementation for an independent research initiative, focusing on replacing traditional heuristic rules with a learnable Transformer-based approach.

## 🚀 Key Features
* **Geometric Transformations:** Custom SVG parser (`src/data/svg_parser.py`) that converts raw polygon data into normalized bounding boxes (0-1000 scale) compatible with Vision Transformers.
* **Multi-Modal Data Fusion:** A custom PyTorch `Dataset` (`src/data/dataset.py`) that aligns pixel values with text tokens and spatial coordinates.
* **End-to-End Training Pipeline:** Automated training loop using Hugging Face `Trainer` with `seqeval` metrics for token classification (Precision, Recall, F1).
* **Inference Engine:** Standalone script that loads fine-tuned models to visualize predictions on raw floor plans.

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
├── data/                   # Raw and Processed Data (Ignored by Git)
├── models_output/          # Checkpoints and Inference Visualizations
├── src/
│   ├── data/
│   │   ├── get_data.py     # Generates mock CubiCasa5k samples for testing
│   │   ├── svg_parser.py   # Parses SVGs -> Bounding Boxes
│   │   └── dataset.py      # PyTorch Dataset & LayoutLMv3Processor
│   ├── models/
│   │   └── layoutlm_v3.py  # Wrapper for LayoutLMv3ForTokenClassification
│   ├── train.py            # Main training loop
│   └── inference.py        # Visualization and Prediction script
└── requirements.txt
```

## ⚡ Usage

### 1. Data Generation & Parsing
Generate mock samples (simulating CubiCasa5k structure) and parse the SVG geometry:
```bash
python src/data/get_data.py
python src/data/svg_parser.py
```

### 2. Training
Fine-tune the LayoutLMv3 model. (Uses CPU fallback if no GPU is detected for testing):
```bash
python -m src.train
```

### 3. Inference & Visualization
Run the model on a sample image to see bounding box predictions:
```bash
python -m src.inference
```
*Output visualization is saved to `models_output/prediction_vis.png`*

## 🔬 Technical Approach

### Geometric Normalization
The raw data comes in SVG format with arbitrary polygon coordinates. We apply a min-max normalization strategy to map these coordinates to the `[0, 1000]` integer space required by the LayoutLM embedding layer:

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} \times 1000$$

### Multi-Modal Fusion
Unlike standard CNNs (YOLO/ResNet) that only look at pixels, this architecture treats the floor plan as a document:
1.  **Visual Embedding:** ViT backbone processes the image patches.
2.  **Text Embedding:** Room labels (e.g., "Living Room") are tokenized.
3.  **Layout Embedding:** 2D positional embeddings represent the bounding boxes.

## 📊 Performance
The model uses standard NER (Named Entity Recognition) metrics for evaluation:
* **Precision**
* **Recall**
* **F1-Score**
* **Accuracy**

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.