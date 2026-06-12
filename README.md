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

```bash
# Clone the repository
git clone [https://github.com/aashishsaini101/VANGUARD_Deepfake_Detection.git](https://github.com/aashishsaini101/VANGUARD_Deepfake_Detection.git)
cd VANGUARD_Deepfake_Detection

# Create and activate environment
conda create -n vanguard_env python=3.10 -y
conda activate vanguard_env

# Install dependencies
pip install -r requirements.txt
