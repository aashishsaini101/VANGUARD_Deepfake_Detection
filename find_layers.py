import torch
import sys
import os

# Add model directory to Python path
sys.path.append(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models"))

# Import model class
from EnhancedMINTIME import EnhancedMINTIME

print("Loading model structure...")

# Create model architecture
model = EnhancedMINTIME(dim=384, depth=4, num_heads=6)

# OPTIONAL: Load checkpoint weights
checkpoint_path = os.path.expanduser(
    "~/Deepfake_Project_Root/dataset_lake/models/checkpoints/best_mintime_model.pth"
)

if os.path.exists(checkpoint_path):
    print("Loading checkpoint weights...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Depending on how the checkpoint was saved
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

print("\n--- LIST OF ALL ATTENTION LAYERS IN YOUR MODEL ---")

for name, module in model.named_modules():
    if 'attn' in name.lower() or 'attention' in name.lower():
        print(name)