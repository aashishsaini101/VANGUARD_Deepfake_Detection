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

# Ensure Python can find your custom modules
sys.path.append(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models"))
from EnhancedMINTIME import EnhancedMINTIME

# --- DETERMINISTIC FORENSIC STATE ---
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

# --- CONFIGURATION ---
CHECKPOINT_PATH = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/models/checkpoints/best_mintime_model.pth")
SEQ_LENGTH = 5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attention_map = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Casts a wide net across the entire architecture. Hooks into any 
        attention block or softmax layer to mathematically sniff out the weights.
        """
        for name, module in self.model.named_modules():
            # Hook anything that sounds like attention or acts like a probability distributor
            if 'attn' in name.lower() or 'attention' in name.lower() or isinstance(module, torch.nn.Softmax):
                handle = module.register_forward_hook(self._get_hook_fn(name))
                self.hook_handles.append(handle)

    def _get_hook_fn(self, name):
        def hook_fn(module, input, output):
            weights = None
            
            # Extract tensor whether it's returned alone or in a tuple
            if isinstance(output, tuple):
                # Standard PyTorch nn.MultiheadAttention returns (output, weights)
                weights = output[1]
            elif isinstance(output, torch.Tensor):
                weights = output

            # MATHEMATICAL SNIFFING: 
            # A ViT attention matrix compares every patch to every other patch.
            # Therefore, the last two dimensions MUST be square and relatively large.
            if weights is not None and weights.dim() >= 3:
                if weights.shape[-1] == weights.shape[-2] and weights.shape[-1] > 50:
                    # We overwrite this continuously so we end up with the very last 
                    # attention layer's weights right before the classification head.
                    self.attention_map = weights.detach().cpu()

        return hook_fn

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def extract_faces(video_path, device):
    """
    Extracts the first 5 frames and their spatial bounding boxes 
    using the mathematically flawless MTCNN spatial memory logic.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

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
        if boxes is not None and probs is not None and len(probs) > 0 and probs[0] > 0.85:
            valid_boxes[i] = boxes[0]

    if not valid_boxes:
        raise ValueError("🚨 CRITICAL: MTCNN failed to detect faces.")

    tensors, original_crops = [], []
    for i, frame_rgb in enumerate(raw_frames):
        box = valid_boxes.get(i, valid_boxes[min(valid_boxes.keys(), key=lambda k: abs(k - i))])
        x1, y1, x2, y2 = [int(b) for b in box]
        h_orig, w_orig, _ = frame_rgb.shape
        w, h = x2 - x1, y2 - y1
        mx, my = int(0.2 * w), int(0.2 * h)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w_orig, x2 + mx), min(h_orig, y2 + my)
        
        cropped = frame_rgb[y1:y2, x1:x2]
        pil_cropped = Image.fromarray(cropped)
        
        original_crops.append(cropped)
        tensors.append(transform(pil_cropped))

    return torch.stack(tensors), original_crops

def generate_heatmap(original_img, attention_weights):
    """
    Surgically parses 1D, 2D, or 3D attention matrices, handling 
    the specific 38,612 (196x197) Cross-Attention geometry.
    """
    attn = attention_weights.detach().cpu().float()
    
    # 1. Flatten into a 1D sequence of patch weights
    num_elements = attn.numel()
    
    if num_elements == 38612:
        # FIX: The 196x197 Cross Attention Matrix
        matrix = attn.view(-1, 197)
        if matrix.shape[0] == 196:
            # Average the attention directed at each of the 196 spatial patches
            cls_attention = matrix.mean(dim=1)
        else:
            matrix = attn.view(197, 196)
            cls_attention = matrix.mean(dim=0)
    else:
        # Standard Fallback: Try to find the CLS token row
        if attn.dim() >= 2:
            cls_attention = attn[0] if attn.shape[0] < 20 else attn.mean(dim=0)
        else:
            cls_attention = attn
            
        cls_attention = cls_attention.flatten()
        
        # Grab exactly 196 elements
        if len(cls_attention) > 196:
            cls_attention = cls_attention[-196:]
        elif len(cls_attention) < 196:
            raise ValueError(f"🚨 CRITICAL: Tensor too small ({len(cls_attention)}).")

    # 2. Reshape to 14x14 Grid
    grid_size = 14
    cls_attention = cls_attention.reshape(grid_size, grid_size).numpy()
    
    # 3. Normalize strictly between 0 and 1
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
    
    # 4. FIX: High-Fidelity Interpolation (Smooths the blocky patches)
    heatmap = cv2.resize(cls_attention, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 5. Colormap and Superimpose
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 0.4, heatmap_colored, 0.6, 0)
    return overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4")
    parser.add_argument("--out", type=str, default="attention_figure.png", help="Output filename")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing Attention Surgery on {device}...")

    # Load Model
    model = EnhancedMINTIME(dim=384, depth=4, num_heads=6).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Hook the network
    extractor = AttentionExtractor(model)

    # Process Video
    video_tensor, original_crops = extract_faces(args.video, device)
    video_tensor = video_tensor.unsqueeze(0).to(device).float()

    print("🧠 Forcing Forward Pass to extract Attention Matrix...")
    with torch.no_grad(), torch.cuda.amp.autocast():
        logits, _ = model(video_tensor, video_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    if extractor.attention_map is None:
        raise ValueError("🚨 Hook failed to intercept attention weights.")

    # Generate Publication-Grade Figure
    print(f"📊 Generating High-Res Heatmap... (Model Fake Probability: {prob*100:.2f}%)")
    
    # We will visualize the center frame (Frame 2 of 5)
    center_frame_idx = SEQ_LENGTH // 2
    original_img = original_crops[center_frame_idx]
    
    # Generate the heatmap using the extracted weights
    heatmap_overlay = generate_heatmap(original_img, extractor.attention_map)

    # Plot side-by-side using Matplotlib (300 DPI for IEEE/CVPR)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    axes[0].imshow(original_img)
    axes[0].set_title("Original Crop", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"MINTIME Attention\n(Fake Prob: {prob*100:.2f}%)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    out_path = os.path.expanduser(f"~/Deepfake_Project_Root/dataset_lake/{args.out}")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    extractor.remove_hooks()
    print(f"✅ Paper Figure saved to: {out_path}")

if __name__ == "__main__":
    main()