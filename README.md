# VANGUARD: ViT-Based Video Attention Network for Generalized Unseen Artifact Recognition in Deepfakes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official PyTorch implementation for the paper: **"VANGUARD: ViT-Based Video Attention Network for Generalized Unseen Artifact Recognition in Deepfakes"**.

VANGUARD addresses the vulnerabilities of spatial-only deepfake detectors under heavy video compression. It explicitly correlates spatial blending boundaries with high-frequency spectral anomalies (Orthogonal DCT) utilizing a synchronized dual-domain cross-attention mechanism (`Asymmetric SFA Block`) and temporal sequence modeling.

## 📊 Performance (Cross-Domain on Celeb-DF)
* **Precision:** 91.03%
* **Recall:** 68.89%
* **AUC:** 91.71%

---

## 1. Repository Structure

* `models/`: Contains the core VANGUARD architecture (`EnhancedMINTIME.py`, `SFA_ViT_module`).
* `scripts/`: Standard pipeline scripts for preprocessing (`extract_faces.py`), dataset building (`build_tensor_dataset.py`), training (`train.py`), and evaluation (`evaluate.py`).
* `weights/`: Contains the YuNet ONNX face detection model used for preprocessing spatial extraction.
* `tsne_feature_space.py`: Replicates the Figure 5 representation learning manifolds.
* `verify_attention.py` / `paper_heatmap_grid.py`: Replicates the Figure 4 spatial cross-attention localizations.

## 2. Environment Setup

It is highly recommended to run this pipeline in an isolated Conda environment.

## Clone the repository
git clone [https://github.com/aashishsaini101/VANGUARD_Deepfake_Detection.git](https://github.com/aashishsaini101/VANGUARD_Deepfake_Detection.git)
cd VANGUARD_Deepfake_Detection

# Create and activate environment
conda create -n vanguard_env python=3.10 -y
conda activate vanguard_env

# Install dependencies
pip install -r requirements.txt

## 📦 Pre-trained Model Weights

Due to GitHub's file size limitations, the pre-trained VANGUARD model weights (`best_mintime_model.pth`) are hosted externally.

### Download

📥 **Pre-trained Weights:**

https://drive.google.com/drive/folders/1YTGa8jSYx28fRkJCD4AiSM-StPZCYQyC?usp=sharing

### Installation

After downloading, place the checkpoint file in:

```text
models/checkpoints/best_mintime_model.pth
```

### Additional Requirements

Ensure the YuNet face detection model is available at:

```text
weights/face_detection_yunet_2023mar.onnx
```

> **Note:** The YuNet weights are already included in this repository.

---

# 🚀 Pipeline Execution

## A. Data Preprocessing

Extract temporally coherent 5-frame face sequences from raw video files (`.mp4`):

### Step 1: Face Extraction

```bash
python scripts/extract_faces.py \
    --input_dir path/to/videos \
    --output_dir dataset_lake/extracted_faces
```

### Step 2: Build Tensor Dataset

```bash
python scripts/build_tensor_dataset.py
```

---

## B. Training

Train the VANGUARD model from scratch using Label-Smoothed Focal Loss:

```bash
python scripts/train.py \
    --batch_size 24 \
    --epochs 100 \
    --lr 1e-4
```

---

## C. Evaluation

Run inference and generate evaluation metrics (AUC, Accuracy):

```bash
python scripts/evaluate.py \
    --checkpoint models/checkpoints/best_mintime_model.pth
```

---

# 📊 Reproducing Publication Figures

The provided scripts allow exact replication of the representation learning manifolds and spatial attention visualizations presented in **Section IV** of the manuscript.

---

## Figure 4 — Spatial Cross-Attention Maps

Generate spatial attention visualizations from the final Asymmetric Cross-Attention block (**Head 4**).

This script extracts pre-dropout Softmax probabilities to visualize spatial blending boundary localization.

```bash
python scripts/attention_visualizer.py \
    --video_path test_fake_1.mp4 \
    --out Figure_4_Attention.png
```

### Output

```text
Figure_4_Attention.png
```

---

## Figure 5 — Latent Feature Manifolds

Extract the 384-dimensional latent embeddings prior to classification, perform L2 normalization, and project the feature space using:

* t-SNE
* UMAP

```bash
python tsne_feature_space.py
```

---

# 📖 Citation

If you find this repository useful in your research, please consider citing our work:

```bibtex
@article{saini2026vanguard,
  title={VANGUARD: ViT-Based Video Attention Network for Generalized Unseen Artifact Recognition in Deepfakes},
  author={Saini, Aashish Kumar and Upadhyay, Govind Murari},
  journal={The Visual Computer},
  year={2026},
  publisher={Springer Nature}
}
```

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for complete details.

