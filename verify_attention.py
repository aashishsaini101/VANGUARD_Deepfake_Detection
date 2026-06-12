import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

sys.path.append(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models"))
from EnhancedMINTIME import EnhancedMINTIME

CHECKPOINT_PATH = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models/checkpoints/best_mintime_model.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def run_verification():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EnhancedMINTIME(dim=384, depth=4, num_heads=6).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Create a dummy video tensor [Batch=1, Seq=5, C=3, H=224, W=224]
    dummy_video = torch.randn(1, 5, 3, 224, 224).to(device)

    with torch.no_grad():
        logits, _ = model(dummy_video, dummy_video)

    # Attempt to extract the true spatial attention from the last block
    try:
        # Access the exact variable you added in hybrid_sfa.py
        raw_attn = model.spatial_blocks[3].last_spatial_attn
        print("\n✅ SUCCESS: Extracted last_spatial_attn")
        print(f"📏 TENSOR TYPE:  {type(raw_attn)}")
        print(f"📐 TENSOR SHAPE: {raw_attn.shape}")
        
        # SCIENTIFIC VALIDATION CHECK
        # Expected shape: [Batch*Seq, Heads, Tokens, Tokens] -> [5, 6, 197, 197]
        if raw_attn.dim() == 4 and raw_attn.shape[2] == 197 and raw_attn.shape[3] == 197:
            print("🔬 VALIDATION PASSED: This is a mathematically valid ViT spatial attention matrix.")
        else:
            print("🚨 VALIDATION FAILED: The tensor dimensions do not match a 197x197 spatial attention matrix. Do not plot this.")
            
    except AttributeError:
        print("\n🚨 ERROR: 'last_spatial_attn' not found. You did not modify hybrid_sfa.py correctly.")

if __name__ == "__main__":
    run_verification()