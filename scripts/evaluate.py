import os
import glob
import torch
import sys
import csv
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, 
                             precision_recall_curve, roc_curve)
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models"))
from EnhancedMINTIME import EnhancedMINTIME

class ShardedIterableDataset(IterableDataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        if not self.files: raise ValueError(f"No .pt shards found in {root_dir}.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.files.copy()
        if worker_info is not None: files = files[worker_info.id :: worker_info.num_workers]
        for file_path in files:
            try:
                data = torch.load(file_path, map_location='cpu')
                for i in range(len(data["labels"])): yield data["videos"][i], data["labels"][i]
            except Exception:
                continue

def get_predictions(model, dataloader, device, desc="Processing"):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc=desc, unit="batch", leave=False):
            videos, labels = videos.to(device, non_blocking=True).float(), labels.to(device, non_blocking=True)
            logits, _ = model(videos, videos)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

def plot_histogram(labels, probs, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(probs[labels == 0], bins=50, alpha=0.6, color='blue', density=True, label='Real (Class 0)')
    plt.hist(probs[labels == 1], bins=50, alpha=0.6, color='red', density=True, label='Fake (Class 1)')
    plt.title('Model Probability Distribution (Test Set)')
    plt.xlabel('Predicted Probability of Fake')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_roc(labels, probs, auc_score, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_pr(labels, probs, save_path):
    precisions, recalls, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8, 8))
    plt.plot(recalls, precisions, color='purple', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def main():
    VAL_DIR = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/tensor_dataset/val")
    TEST_DIR = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/tensor_dataset/test")
    CHECKPOINT_PATH = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models/checkpoints/best_mintime_model.pth")
    
    # --- REAL-TIME DIRECTORY SETUP ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = os.path.expanduser(f"~/Deepfake_Project_Root/dataset_lake/eval_results/eval_{timestamp}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    BATCH_SIZE = 8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initiating Production-Grade Evaluation on {device}")
    print(f"📁 Results will be saved to: {RESULTS_DIR}")

    val_loader = DataLoader(ShardedIterableDataset(VAL_DIR), batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    test_loader = DataLoader(ShardedIterableDataset(TEST_DIR), batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    model = EnhancedMINTIME(dim=384, depth=4, num_heads=6).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # --- STAGE 1: PRECISION-CONSTRAINED CALIBRATION ---
    print("\n[STAGE 1] Calibrating threshold for 90% Precision on Validation Set...")
    val_labels, val_probs = get_predictions(model, val_loader, device, desc="Scanning VAL")
    
    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
    target_precision = 0.90
    valid_idxs = np.where(precisions[:-1] >= target_precision)[0]

    if len(valid_idxs) > 0:
        best_idx = valid_idxs[np.argmax(recalls[:-1][valid_idxs])]
    else:
        print("⚠️ Warning: 90% Precision unattainable on VAL. Defaulting to F1 max.")
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])

    frozen_threshold = thresholds[best_idx]
    print(f"✅ Calibration Complete. Frozen Threshold: {frozen_threshold:.4f}")

    # --- STAGE 2: BLIND EVALUATION ---
    print("\n[STAGE 2] Executing Blind Test Set Evaluation...")
    test_labels, test_probs = get_predictions(model, test_loader, device, desc="Scanning TEST")
    test_preds = (test_probs >= frozen_threshold).astype(int)

    # --- STAGE 3: METRICS COMPILATION & EXPORT ---
    acc = accuracy_score(test_labels, test_preds)
    auc = roc_auc_score(test_labels, test_probs)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()

    # 1. Generate and Save Graphs
    plot_histogram(test_labels, test_probs, os.path.join(RESULTS_DIR, "probability_histogram.png"))
    plot_roc(test_labels, test_probs, auc, os.path.join(RESULTS_DIR, "roc_curve.png"))
    plot_pr(test_labels, test_probs, os.path.join(RESULTS_DIR, "pr_curve.png"))

    # 2. Save Metrics to JSON
    metrics_data = {
        "timestamp": timestamp,
        "total_videos_analyzed": len(test_labels),
        "applied_threshold": float(frozen_threshold),
        "roc_auc": float(auc),
        "overall_accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    # 3. Save Raw Predictions to CSV
    with open(os.path.join(RESULTS_DIR, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True_Label", "Predicted_Probability", "Predicted_Class"])
        for t_label, prob, p_class in zip(test_labels, test_probs, test_preds):
            writer.writerow([t_label, prob, p_class])

    print("\n" + "="*50)
    print(" 📊 FINAL BLIND TEST REPORT (PRECISION 90% TARGET)")
    print("="*50)
    print(f"Total Videos Analyzed : {len(test_labels)}")
    print(f"Applied Threshold     : {frozen_threshold:.4f} (Calibrated on VAL)")
    print(f"ROC-AUC Score         : {auc:.4f}")
    print(f"Overall Accuracy      : {acc*100:.2f}%")
    print(f"F1-Score              : {f1:.4f}")
    print(f"Precision             : {precision:.4f} (Target was ~0.90)")
    print(f"Recall                : {recall:.4f}")
    
    print("\n--- Confusion Matrix ---")
    print(f"True Negatives (Real) : {tn}")
    print(f"False Positives (Fake Alarm) : {fp}")
    print(f"False Negatives (Miss)       : {fn}")
    print(f"True Positives (Fake)        : {tp}")
    print("="*50)
    print(f"💾 All evaluation artifacts saved successfully in: {RESULTS_DIR}\n")

if __name__ == "__main__":
    main()