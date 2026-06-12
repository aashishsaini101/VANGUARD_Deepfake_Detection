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

3. Model Weights
Due to GitHub's file size limits, the pre-trained weights (best_mintime_model.pth) for the VANGUARD architecture are hosted externally.

Download the pre-trained model: [https://drive.google.com/drive/folders/1YTGa8jSYx28fRkJCD4AiSM-StPZCYQyC?usp=sharing]

Place the downloaded .pth file in the following directory:
models/checkpoints/best_mintime_model.pth

Ensure the YuNet face detection weights are present in weights/face_detection_yunet_2023mar.onnx (included in repo).

4. Pipeline Execution
A. Data Preprocessing
To extract temporally coherent 5-frame face sequences from raw .mp4 files:

Bash
python scripts/extract_faces.py --input_dir path/to/videos --output_dir dataset_lake/extracted_faces
python scripts/build_tensor_dataset.py
B. Training
To train the model from scratch using the label-smoothed focal loss:

Bash
python scripts/train.py --batch_size 24 --epochs 100 --lr 1e-4
C. Evaluation
To run inference and generate AUC/ACC metrics on your test splits:

Bash
python scripts/evaluate.py --checkpoint models/checkpoints/best_mintime_model.pth
5. Replicating Publication Figures
The scripts provided allow for exact replication of the representation learning manifolds and spatial attention distributions discussed in Section IV of the manuscript.

Generating Spatial Cross-Attention Maps (Figure 4)
This script performs a white-box extraction of the pre-dropout Softmax probabilities from the final Asymmetric Cross-Attention block (Head 4) to visualize spatial blending boundary localization.

Bash
python scripts/attention_visualizer.py --video_path test_fake_1.mp4 --out Figure_4_Attention.png
Generating Latent Feature Manifolds (Figure 5)
This script explicitly extracts the 384-dimensional latent embeddings preceding the final classification layer, applies L2-normalization, and projects the feature space using optimized t-SNE and UMAP.





***
