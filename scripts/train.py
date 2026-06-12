import os
import glob
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import GradScaler
from torch import autocast # Updated import for device-specific autocast
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import sys
import numpy as np
import torch.nn.functional as F
import json  

class FocalLossWithSmoothing(nn.Module):
    """
    Publication-Grade Focal Loss + Label Smoothing.
    Mathematically consistent formulation avoiding exp(-CE) approximation.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        num_classes = inputs.size(1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.label_smoothing
            )

        pt = (probs * true_dist).sum(dim=1)
        ce_loss = -(true_dist * log_probs).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        return loss.mean()

# Ensure Python can find your custom modules
sys.path.append(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models"))
from EnhancedMINTIME import EnhancedMINTIME
from augmentations import AttentionConsistentAugmentation

# --- 0. MULTIPROCESS SEEDING ---
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- 1. DOUBLE-SHUFFLED, LOCALLY-SEEDED DATASET ---
class ShardedIterableDataset(IterableDataset):
    def __init__(self, root_dir, shuffle=True, buffer_size=2048):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.rng = random.Random()
        
        if not self.files:
            raise ValueError(f"No .pt shards found in {root_dir}.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            self.rng.seed(worker_info.seed)
        
        if self.shuffle:
            self.rng.shuffle(self.files)

        files = self.files.copy()
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        buffer = []
        for file_path in files:
            try:
                data = torch.load(file_path, map_location='cpu')
                videos, labels = data["videos"], data["labels"]

                # DEFENSIVE CASTING: Prevent mid-run crashes from corrupted shards
                if videos.dtype != torch.float16:
                    videos = videos.half()
                
                assert videos.ndim == 5, f"Dim error in {file_path}"
                assert videos.shape[2:] == (3, 224, 224), f"Shape error in {file_path}"

                indices = list(range(len(labels)))
                if self.shuffle:
                    self.rng.shuffle(indices)

                for idx in indices:
                    if not self.shuffle:
                        yield videos[idx], labels[idx]
                    else:
                        buffer.append((videos[idx], labels[idx]))
                        
                        if len(buffer) >= self.buffer_size:
                            self.rng.shuffle(buffer)
                            for item in buffer:
                                yield item
                            buffer = [] 

            except Exception as e:
                print(f"\n⚠️ I/O WARNING: Skipping {file_path} | {e}")
                continue

        if self.shuffle and buffer:
            self.rng.shuffle(buffer)
            for item in buffer:
                yield item

# --- 2. EVALUATION LOOP (AMP ALIGNED) ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    total_steps = 0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            # Float cast required before AMP for Kornia stability
            videos = videos.to(device, non_blocking=True).float() 
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            # FIX: Wrapping evaluation in AMP to ensure identical mathematical distribution
            with autocast(device_type='cuda'):
                logits, orth_loss = model(videos, videos)
                loss = criterion(logits, labels)
            
            epoch_loss += loss.item()
            total_steps += 1

            probs = torch.softmax(logits, dim=1)[:, 1] 
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0 

    return epoch_loss / max(1, total_steps), acc, auc

# --- 3. MASTER LOOP ---
def main():
    # --- HARDWARE OPTIMIZATION ---
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # FIX: Enable Tensor Cores for massive ViT Matmul speedup
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    TRAIN_DIR = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/tensor_dataset/train")
    VAL_DIR = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/tensor_dataset/val")
    CHECKPOINT_DIR = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models/checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    EPOCHS = 25
    BATCH_SIZE = 16  
    LR = 1e-4
    NUM_WORKERS = 4
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing Publication-Grade Training Pipeline on {device}")

    train_dataset = ShardedIterableDataset(TRAIN_DIR, shuffle=True, buffer_size=2048)
    val_dataset = ShardedIterableDataset(VAL_DIR, shuffle=False)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g
    )

    model = EnhancedMINTIME(dim=384, depth=4, num_heads=6).to(device)
    aug_layer = AttentionConsistentAugmentation().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    
    criterion = FocalLossWithSmoothing(gamma=2.0, label_smoothing=0.1)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = 0.0
    training_history = [] 

    print("Commencing Epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        total_steps = 0 
        lambda_orth = 0.01
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        start_time = time.time()

        for step, (videos, labels) in enumerate(progress_bar):
            
            videos = videos.to(device, non_blocking=True).float()
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            total_steps += 1

            optimizer.zero_grad(set_to_none=True)

            aug_rgb, aug_dct = aug_layer(videos)

            # FIX: Explicit Device context for modern PyTorch
            with autocast(device_type='cuda'):
                logits, orth_loss = model(aug_rgb, aug_dct)
                logits = torch.clamp(logits, min=-10.0, max=10.0)
                cls_loss = criterion(logits, labels)
                loss = cls_loss + (lambda_orth * orth_loss)

            if not torch.isfinite(loss):
                print(f"\n⚠️ NaN detected at step {step}. Skipping batch.")
                scaler.update()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            
            if step % 10 == 0:
                elapsed = time.time() - start_time
                it_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                
                if torch.cuda.is_available():
                    vram_alloc = torch.cuda.memory_allocated(device) / (1024**3)
                    vram_res = torch.cuda.memory_reserved(device) / (1024**3)
                else:
                    vram_alloc, vram_res = 0.0, 0.0
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.3f}", 
                    "orth": f"{orth_loss.item():.3f}",
                    "it/s": f"{it_per_sec:.2f}",
                    "vram": f"{vram_alloc:.1f}G/{vram_res:.1f}G"
                })

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, total_steps)

        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Orth Weight: {lambda_orth:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f}\n")

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_mintime_model.pth"))
            print(f"⭐ New Best Model Saved! (AUC: {best_val_auc:.4f})")

    metrics_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(metrics_path, "w") as f:
        json.dump(training_history, f, indent=4)
    print(f"\n✅ Training Complete. Metrics saved to {metrics_path}. Best Validation AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()