import os
import sys
import glob
import random
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

sys.path.append(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models"))
from EnhancedMINTIME import EnhancedMINTIME

# --- DETERMINISTIC FORENSIC STATE ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

CHECKPOINT_PATH = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models/checkpoints/best_mintime_model.pth")
SEQ_LENGTH = 5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_faces(video_path, device, mtcnn):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQ_LENGTH:
        return None
    
    frame_indices = np.linspace(0, total_frames - 1, SEQ_LENGTH, dtype=int).tolist()
    
    raw_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(raw_frames) < SEQ_LENGTH:
        return None

    tensors = []
    for frame_rgb in raw_frames:
        pil_img = Image.fromarray(frame_rgb)
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is not None and len(probs) > 0 and probs[0] > 0.85:
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]
            h_orig, w_orig, _ = frame_rgb.shape
            w, h = x2 - x1, y2 - y1
            
            mx, my = int(0.05 * w), int(0.05 * h)
            x1, y1 = max(0, x1 - mx), max(0, y1 - my)
            x2, y2 = min(w_orig, x2 + mx), min(h_orig, y2 + my)
            
            cropped = frame_rgb[y1:y2, x1:x2].copy()
            tensors.append(transform(Image.fromarray(cropped)))
        else:
            return None 

    if len(tensors) != SEQ_LENGTH:
        return None
        
    return torch.stack(tensors)

def process_dataset_folder(folder_path, label, dataset_name, binary_lbl, model, mtcnn, device, max_videos=80):
    search_path = os.path.join(folder_path, "**", "*.mp4")
    videos = glob.glob(search_path, recursive=True)
    
    print(f"\n📁 Searching: {search_path}")
    print(f"👀 Found {len(videos)} source videos.")
    
    if len(videos) == 0:
        return [], [], []
        
    random.shuffle(videos)
    videos = videos[:max_videos]
    
    features_list = []
    labels_list = []
    binary_list = []
    
    print(f"⚙️ Extracting embeddings for up to {max_videos} balanced samples...")
    for vid in videos:
        try:
            video_tensor = extract_faces(vid, device, mtcnn)
            if video_tensor is None:
                continue
                
            video_tensor = video_tensor.unsqueeze(0).to(device).float()
            with torch.no_grad():
                embedding = model.extract_features(video_tensor, video_tensor)
                
            # Fix: Safe squeeze preserves 1D vector shape [384] without flattening tokens destructively
            features_list.append(embedding.squeeze(0).cpu().numpy())
            labels_list.append(f"{dataset_name} - {label}")
            binary_list.append(binary_lbl)
            
        except Exception as e:
            continue
            
    print(f"🎯 Extracted {len(features_list)} valid feature vectors.")
    return features_list, labels_list, binary_list

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EnhancedMINTIME(dim=384, depth=4, num_heads=6).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    mtcnn = MTCNN(keep_all=False, device=device)

    PATH_FF_REAL = "/mnt/d/Dataset/FaceForensics++_C23/original"
    PATH_FF_FAKE = "/mnt/d/Dataset/FaceForensics++_C23/Deepfakes"
    PATH_CELEB_REAL = "/mnt/d/Dataset/Celeb_DF/Celeb-real"
    PATH_CELEB_FAKE = "/mnt/d/Dataset/Celeb_DF/Celeb-synthesis"

    X, y, y_bin = [], [], []
    
    for path, lbl, ds, bin_val in [(PATH_FF_REAL, "Authentic", "FF++", 0), 
                                   (PATH_FF_FAKE, "Manipulated", "FF++", 1),
                                   (PATH_CELEB_REAL, "Authentic", "Celeb-DF", 0), 
                                   (PATH_CELEB_FAKE, "Manipulated", "Celeb-DF", 1)]:
        feats, lbls, bins = process_dataset_folder(path, lbl, ds, bin_val, model, mtcnn, device, max_videos=80)
        X.extend(feats)
        y.extend(lbls)
        y_bin.extend(bins)

    if len(X) == 0:
        print("\n🚨 CRITICAL ERROR: Feature array is completely empty. Verify paths.")
        sys.exit(1)

    X = np.array(X)
    
    # Fix 1: Mandatory L2 Normalization to align anisotropic scales
    X_norm = normalize(X, norm='l2')
    
    # Fix 2: Calculate explicit structural labels for evaluation
    domain_labels = [0 if "FF++" in lbl else 1 for lbl in y]
    classes = ["Authentic" if "Authentic" in lbl else "Manipulated" for lbl in y]
    datasets = ["FF++" if "FF++" in lbl else "Celeb-DF" for lbl in y]
    
    # Fix 3: Quantitative Clustering Metrics
    class_sil = silhouette_score(X_norm, y_bin, metric='cosine')
    domain_sil = silhouette_score(X_norm, domain_labels, metric='cosine')
    db_index = davies_bouldin_score(X_norm, y_bin)
    
    print(f"\n📊 --- LATENT SPACE QUANTITATIVE METRICS ---")
    print(f"✅ Class Silhouette Score (Real vs Fake - High Desired): {class_sil:.4f}")
    print(f"✅ Domain Silhouette Score (FF++ vs CelebDF - Low Desired): {domain_sil:.4f}")
    print(f"✅ Davies-Bouldin Index (Lower Desired): {db_index:.4f}\n")

    # Fix 4: PCA dimensionality reduction to filter token noise prior to t-SNE
    X_pca = PCA(n_components=min(50, X_norm.shape[0], X_norm.shape[1])).fit_transform(X_norm)
    
    print("Computing optimized t-SNE projection...")
    perp = min(30, max(5, len(X_pca) // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)

    print("Computing optimized UMAP projection...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
    X_umap = reducer.fit_transform(X_norm)

    # --- JOURNAL VISUAL ENCODING CONFIGURATION ---
    df_tsne = pd.DataFrame({'Dim 1': X_tsne[:, 0], 'Dim 2': X_tsne[:, 1], 'Class': classes, 'Dataset': datasets})
    df_umap = pd.DataFrame({'Dim 1': X_umap[:, 0], 'Dim 2': X_umap[:, 1], 'Class': classes, 'Dataset': datasets})
    
    # Color mapped to Class (Real=Blue, Fake=Red)
    palette = {"Authentic": "#1f77b4", "Manipulated": "#d62728"}
    # Marker mapped to Dataset domain (FF++=Circle, Celeb-DF=Square)
    markers = {"FF++": "o", "Celeb-DF": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    sns.set_theme(style="whitegrid", font="serif")
    
    # Plot 1: t-SNE
    sns.scatterplot(
        ax=axes[0], data=df_tsne, x='Dim 1', y='Dim 2', 
        hue='Class', style='Dataset', palette=palette, markers=markers,
        s=90, alpha=0.75, edgecolor='black', linewidth=0.5
    )
    axes[0].set_title(f't-SNE Embedding Manifold', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Dimension 1', fontsize=11)
    axes[0].set_ylabel('Dimension 2', fontsize=11)
    axes[0].legend().remove()

    # Plot 2: UMAP
    sns.scatterplot(
        ax=axes[1], data=df_umap, x='Dim 1', y='Dim 2', 
        hue='Class', style='Dataset', palette=palette, markers=markers,
        s=90, alpha=0.75, edgecolor='black', linewidth=0.5
    )
    axes[1].set_title(f'UMAP Embedding Manifold', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Dimension 1', fontsize=11)
    axes[1].set_ylabel('Dimension 2', fontsize=11)
    
    # Single clean combined legend
    axes[1].legend(loc='best', frameon=True, shadow=False, title_fontsize='11', fontsize='10')

    plt.suptitle(
        f'Latent Feature Space Topology and Domain Invariance\n'
        f'(Class Separation Silhouette Score: {class_sil:.3f} | Domain Alignment Silhouette Score: {domain_sil:.3f})', 
        fontsize=15, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    out_path = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/Figure_5_Manifolds.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ Defensible figure successfully generated and saved to: {out_path}")

if __name__ == "__main__":
    main()