import os
import time
import torch
import torch.nn as nn
import glob
from torch.cuda.amp import autocast, GradScaler
import sys

sys.path.append(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models"))
from EnhancedMINTIME import EnhancedMINTIME

def run_sanity_check():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- INITIATING ARCHITECTURE OVERFIT CHECK ON {device} ---")

    print("\n1. Initializing EnhancedMINTIME (dim=384, depth=4)...")
    model = EnhancedMINTIME().to(device)
    
    shard_dir = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/tensor_dataset/train")
    shard_files = glob.glob(shard_dir + "/*.pt")
    
    if not shard_files:
        print(f"❌ No shards found. Run build_tensor_dataset.py first.")
        return
        
    print(f"2. Loading test shard: {os.path.basename(shard_files[0])}")
    shard = torch.load(shard_files[0], map_location='cpu')
    
    # Grab a fixed batch of 8
    test_batch = shard["videos"][:8].to(device)
    test_labels = shard["labels"][:8].to(device)

    # Initialize Training Components
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    print("\n3. Executing Backward Graph & AMP Overfit Test...")
    model.train()
    
    for iteration in range(1, 151):
        optimizer.zero_grad()
        
        with autocast():
            logits, orth_loss = model(test_batch)
            # Static lambda for overfit test
            loss = criterion(logits, test_labels) + (0.1 * orth_loss)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if iteration == 1 or iteration % 15 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == test_labels).float().mean().item() * 100
            print(f"   Iter {iteration:03d} | Total Loss: {loss.item():.4f} | Orth: {orth_loss.item():.4f} | Acc: {acc:3.0f}%")
            
        # Target Condition
        if loss.item() < 0.05 and acc == 100:
            print(f"\n🚀 SUCCESS: Model successfully overfitted the batch at Iteration {iteration}.")
            print("   Backward graph is stable. AMP is functioning. Ready for Full Training.")
            return

    print("\n⚠️ WARNING: Model failed to memorize the batch within 150 iterations.")

if __name__ == "__main__":
    run_sanity_check()