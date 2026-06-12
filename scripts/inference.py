import os
import sys
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

# Ensure Python can find your custom modules
sys.path.append(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models"))
from EnhancedMINTIME import EnhancedMINTIME

# --- DETERMINISTIC FORENSIC STATE ---
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
np.random.seed(42)

# --- CONFIGURATION ---
CHECKPOINT_PATH = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models/checkpoints/best_mintime_model.pth")
CALIBRATED_THRESHOLD = 0.8825 
SEQ_LENGTH = 5

# STRICT PARITY with build_tensor_dataset.py. DO NOT add ImageNet Normalization.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_face_sequence(video_path, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQ_LENGTH:
        raise ValueError(f"Video is too short. Needs at least {SEQ_LENGTH} frames.")

    # FIX: True temporal coverage using linspace
    frame_indices = np.linspace(0, total_frames - 1, SEQ_LENGTH, dtype=int).tolist()
    
    mtcnn = MTCNN(keep_all=False, device=device)
    
    debug_dir = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/debug_frames")
    os.makedirs(debug_dir, exist_ok=True)
    
    raw_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    while len(raw_frames) < SEQ_LENGTH:
        raw_frames.append(raw_frames[-1])

    # 1. First Pass: Detect faces safely
    valid_boxes = {}
    for i, frame_rgb in enumerate(raw_frames):
        pil_img = Image.fromarray(frame_rgb)
        boxes, probs = mtcnn.detect(pil_img)
        
        # FIX: Strict None-guards to prevent MTCNN crashes
        if (boxes is not None and 
            probs is not None and 
            len(probs) > 0 and 
            probs[0] is not None and 
            probs[0] > 0.85):
            valid_boxes[i] = boxes[0]

    if not valid_boxes:
        raise ValueError("🚨 CRITICAL: MTCNN could not find a face in ANY of the 5 frames.")

    # 2. Second Pass: Interpolate missing boxes and clamp safely
    processed_tensors = []
    for i, frame_rgb in enumerate(raw_frames):
        if i in valid_boxes:
            box = valid_boxes[i]
        else:
            nearest_idx = min(valid_boxes.keys(), key=lambda k: abs(k - i))
            box = valid_boxes[nearest_idx]
            print(f"⚠️ Recovered frame {i} by borrowing bounding box from frame {nearest_idx}")

        x1, y1, x2, y2 = [int(b) for b in box]
        h_orig, w_orig, _ = frame_rgb.shape
        
        # 20% margin
        w, h = x2 - x1, y2 - y1
        margin_x, margin_y = int(0.2 * w), int(0.2 * h)
        
        # FIX: Strict boundary clamping to prevent out-of-bounds array slicing
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w_orig, x2 + margin_x)
        y2 = min(h_orig, y2 + margin_y)
        
        # FIX: Prevent empty crops
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box geometry generated on frame {i}")
            
        cropped = frame_rgb[y1:y2, x1:x2]
        
        if cropped.size == 0:
            raise ValueError(f"Empty crop generated for frame {i}")

        pil_cropped = Image.fromarray(cropped)
        debug_path = os.path.join(debug_dir, f"frame_{i}.jpg")
        pil_cropped.save(debug_path)
        
        processed_tensors.append(transform(pil_cropped))

    return torch.stack(processed_tensors)

def main():
    parser = argparse.ArgumentParser(description="MINTIME Deepfake Inference Engine")
    parser.add_argument("--video", type=str, required=True, help="Path to the .mp4 video file")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing MINTIME Inference on {device}...")

    model = EnhancedMINTIME(dim=384, depth=4, num_heads=6).to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Missing model weights at {CHECKPOINT_PATH}")
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    print(f"🎬 Processing video: {os.path.basename(args.video)}")
    video_tensor = extract_face_sequence(args.video, device)
    
    # [1, T, C, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(device).float()

    print("🧠 Analyzing Temporal and Frequency Domains...")
    with torch.no_grad():
        logits, _ = model(video_tensor, video_tensor)
        probs = torch.softmax(logits, dim=1)
        fake_prob = probs[0, 1].item()

    print("\n" + "="*50)
    print(" 🔎 FORENSIC VERDICT")
    print("="*50)
    
    if fake_prob >= CALIBRATED_THRESHOLD:
        print(f"🚨 RESULT: FAKE (Synthetic Media Detected)")
    else:
        print(f"✅ RESULT: REAL (Pristine Media)")
        
    print(f"Probability of Fake  : {fake_prob * 100:.2f}%")
    print(f"Operating Threshold  : {CALIBRATED_THRESHOLD * 100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()