import os
import sys
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

sys.path.append(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models"))
from EnhancedMINTIME import EnhancedMINTIME

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

CHECKPOINT_PATH = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models/checkpoints/best_mintime_model.pth")
SEQ_LENGTH = 5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_faces(video_path, device):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, SEQ_LENGTH, dtype=int).tolist()
    mtcnn = MTCNN(keep_all=False, device=device)
    
    raw_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    while len(raw_frames) < SEQ_LENGTH:
        raw_frames.append(raw_frames[-1])

    valid_boxes = {}
    for i, frame_rgb in enumerate(raw_frames):
        pil_img = Image.fromarray(frame_rgb)
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is not None and len(probs) > 0 and probs[0] > 0.85:
            valid_boxes[i] = boxes[0]

    tensors, original_crops = [], []
    for i, frame_rgb in enumerate(raw_frames):
        box = valid_boxes.get(i, valid_boxes[min(valid_boxes.keys(), key=lambda k: abs(k - i))])
        x1, y1, x2, y2 = [int(b) for b in box]
        h_orig, w_orig, _ = frame_rgb.shape
        w, h = x2 - x1, y2 - y1
        
        mx, my = int(0.05 * w), int(0.05 * h)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w_orig, x2 + mx), min(h_orig, y2 + my)
        
        cropped = frame_rgb[y1:y2, x1:x2].copy()
        original_crops.append(cropped)
        tensors.append(transform(Image.fromarray(cropped)))

    return torch.stack(tensors), original_crops

def generate_publication_heatmap(original_img, raw_attention, center_idx):
    # raw_attention shape expected: [5, 6, 197, 197]
    attn_center_frame = raw_attention[center_idx].float().cpu()
    
    # ---> THE CRITICAL FIX: ISOLATE HEAD 4 <---
    # We do NOT use .mean(). We explicitly pull index 4 (Head 4), 
    # from the CLS token (0), looking at the spatial tokens (1 to 196)
    cls_to_spatial = attn_center_frame[4, 0, 1:] # Shape: [196]
    
    # Reshape strictly to 14x14
    grid = cls_to_spatial.reshape(14, 14).numpy()
    
    # Smooth the grid for publication-quality rendering
    grid = cv2.GaussianBlur(grid, (3, 3), 0)
    
    # Percentile normalization to prevent a single pixel from blowing out the map
    vmax = np.percentile(grid, 95)
    vmin = grid.min()
    grid = np.clip((grid - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # Resize using INTER_CUBIC for smooth overlays
    heatmap = cv2.resize(grid, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend
    overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 0.5, heatmap_colored, 0.5, 0)
    return overlay

def process_video(model, device, video_path):
    video_tensor, original_crops = extract_faces(video_path, device)
    video_tensor = video_tensor.unsqueeze(0).to(device).float()

    with torch.no_grad():
        logits, _ = model(video_tensor, video_tensor)
        
        # Temperature Scaling
        temperature = 1.5 
        scaled_logits = logits / temperature
        prob = torch.softmax(scaled_logits, dim=1)[0, 1].item()

    # RAW TENSOR EXTRACTION
    raw_attention = model.spatial_blocks[3].last_spatial_attn

    center_frame_idx = SEQ_LENGTH // 2
    original_img = original_crops[center_frame_idx]
    
    heatmap_overlay = generate_publication_heatmap(original_img, raw_attention, center_frame_idx)
    return original_img, heatmap_overlay, prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real1", type=str, required=True)
    parser.add_argument("--real2", type=str, required=True)
    parser.add_argument("--fake1", type=str, required=True)
    parser.add_argument("--fake2", type=str, required=True)
    parser.add_argument("--out", type=str, default="Figure_3_VANGUARD_FINAL.png")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EnhancedMINTIME(dim=384, depth=4, num_heads=6).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    videos = [
        (args.real1, "Real"), (args.real2, "Real"),
        (args.fake1, "Fake"), (args.fake2, "Fake")
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300)
    fig.suptitle("Cross-Attention Localization on Spatial Blending Boundaries (Head 4)", fontsize=18, fontweight='bold')

    for i, (vid_path, label) in enumerate(videos):
        orig, heat, prob = process_video(model, device, vid_path)
        row, col_base = i // 2, (i % 2) * 2
        
        axes[row, col_base].imshow(orig)
        axes[row, col_base].set_title(f"Input ({label})", fontsize=12)
        axes[row, col_base].axis('off')
        
        axes[row, col_base + 1].imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
        axes[row, col_base + 1].set_title(f"Attention (Fake Prob: {prob*100:.1f}%)", fontsize=12)
        axes[row, col_base + 1].axis('off')

    plt.tight_layout()
    out_path = os.path.expanduser(f"~/Deepfake_Project_Root/dataset_lake/{args.out}")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ Final figure saved to {out_path}")

if __name__ == "__main__":
    main()