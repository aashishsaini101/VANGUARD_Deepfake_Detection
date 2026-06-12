import os
import re
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict

# --- Configuration ---
# CHANGE THIS VARIABLE to "train", "val", or "test"
TARGET_SPLIT = "test" 

INPUT_ROOT = f"~/Deepfake_Project_Root/dataset_lake/extracted_faces/split_{TARGET_SPLIT}"
OUTPUT_ROOT = f"~/Deepfake_Project_Root/dataset_lake/tensor_dataset/{TARGET_SPLIT}"
# Note: output for 'val' will be saved in 'tensor_dataset/val'. Make sure your train.py looks for 'val' and not 'validation'.

SEQ_LENGTH = 5
SHARD_SIZE = 512

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Safer Regex from our split script
FRAME_PATTERN = re.compile(r"(.+)_f\d+\.(jpg|png|jpeg)$")

def group_frames(dataset_path):
    video_groups = defaultdict(list)
    # FIX: sorted() guarantees OS-level deterministic temporal ordering
    for img_path in sorted(dataset_path.glob("*.jpg")):
        name = img_path.name
        match = FRAME_PATTERN.match(name)
        base = match.group(1) if match else name.split('.')[0]
        video_groups[base].append(img_path)
    return video_groups

def build():
    input_root = Path(os.path.expanduser(INPUT_ROOT))
    output_root = Path(os.path.expanduser(OUTPUT_ROOT))
    output_root.mkdir(parents=True, exist_ok=True)

    classes = {"real": 0, "fake": 1}
    all_samples = []

    print(f"📦 Building sequence index for [{TARGET_SPLIT.upper()}]... (Video-level isolation enforced)")

    for cls, label in classes.items():
        cls_dir = input_root / cls
        if not cls_dir.exists(): continue

        for dataset_name in sorted(cls_dir.iterdir()):
            if not dataset_name.is_dir(): continue

            groups = group_frames(dataset_name)

            for _, frames in groups.items():
                frames = sorted(frames) # Secondary sort safeguard
                if not frames: continue

                seq = frames[:SEQ_LENGTH]
                if len(seq) < SEQ_LENGTH:
                    seq += [seq[-1]] * (SEQ_LENGTH - len(seq))

                all_samples.append((seq, label))

    print(f"Total 5-frame sequences: {len(all_samples)}")

    shard_id = 0
    
    for i in tqdm(range(0, len(all_samples), SHARD_SIZE), desc="Writing FP16 Tensor Shards"):
        batch = all_samples[i:i+SHARD_SIZE]
        videos, labels = [], []

        for seq, label in batch:
            frames = []
            for path in seq:
                try:
                    img = Image.open(path).convert("RGB")
                    # ToTensor() scales 0-255 to 0.0-1.0. We then cast to float16 to save space.
                    frames.append(transform(img).to(torch.float16))
                except Exception:
                    frames.append(torch.zeros(3, 224, 224, dtype=torch.float16))

            videos.append(torch.stack(frames))
            labels.append(label)

        videos = torch.stack(videos)  
        labels = torch.tensor(labels, dtype=torch.long)

        torch.save({"videos": videos, "labels": labels}, output_root / f"shard_{shard_id:04d}.pt")
        shard_id += 1

    print(f"✅ Tensor Sharding Complete for {TARGET_SPLIT}.")

if __name__ == "__main__":
    build()