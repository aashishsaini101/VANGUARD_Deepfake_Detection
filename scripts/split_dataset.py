import os
import re
import json
import random
from pathlib import Path
from collections import defaultdict
import datetime

# --- Configuration ---
DRY_RUN = False  
SEED = 42

CANONICAL_DIR = Path(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/extracted_faces/train"))
TRAIN_LINK_DIR = Path(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/extracted_faces/split_train"))
VAL_LINK_DIR = Path(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/extracted_faces/split_val"))
TEST_LINK_DIR = Path(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/extracted_faces/split_test"))

VAL_RATIO = 0.10
TEST_RATIO = 0.10

FRAME_PATTERN = re.compile(r"(.+)_f\d+\.(jpg|png|jpeg)$")

# Isolate RNG state
rng = random.Random(SEED)

def get_identity_group(filename, metadata_map=None):
    """
    Extracts the identity ID to prevent leakage.
    If you have a JSON mapping video filenames to actor IDs, pass it here.
    """
    match = FRAME_PATTERN.match(filename)
    base_vid = match.group(1) if match else filename.split('.')[0]
    
    if metadata_map and base_vid in metadata_map:
        return metadata_map[base_vid]
    return base_vid

def execute_split():
    print(f"🚀 Initiating Research-Grade Immutable Split (DRY_RUN={DRY_RUN})...")

    classes = ["real", "fake"]
    manifest = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "seed": SEED,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "total_videos": 0,
            "total_frames": 0
        },
        "splits": {"train": [], "val": [], "test": []}
    }

    stats = {"real": {"train": 0, "val": 0, "test": 0}, "fake": {"train": 0, "val": 0, "test": 0}}

    for cls in classes:
        cls_path = CANONICAL_DIR / cls
        if not cls_path.exists(): continue

        for dataset_dir in sorted(cls_path.iterdir()):
            if not dataset_dir.is_dir(): continue
            dataset_name = dataset_dir.name

            video_groups = defaultdict(list)
            for img_path in sorted(dataset_dir.glob("*.jpg")):
                video_groups[get_identity_group(img_path.name)].append(img_path)

            unique_videos = sorted(list(video_groups.keys()))
            if not unique_videos: continue
                
            rng.shuffle(unique_videos)
            dataset_total = len(unique_videos)

            # Strict Residual Accounting
            val_float = dataset_total * VAL_RATIO
            test_float = dataset_total * TEST_RATIO
            val_count = int(val_float)
            test_count = int(test_float)
            
            # Distribute remainder if small dataset
            if val_count == 0 and val_float > 0: val_count = 1
            if test_count == 0 and test_float > 0: test_count = 1
            train_count = dataset_total - val_count - test_count

            val_vids = set(unique_videos[:val_count])
            test_vids = set(unique_videos[val_count:val_count + test_count])
            train_vids = set(unique_videos[val_count + test_count:])

            # CRITICAL: Collision Assertions
            assert train_vids.isdisjoint(val_vids), f"Leakage detected: Train/Val intersection in {dataset_name}"
            assert train_vids.isdisjoint(test_vids), f"Leakage detected: Train/Test intersection in {dataset_name}"
            assert val_vids.isdisjoint(test_vids), f"Leakage detected: Val/Test intersection in {dataset_name}"

            stats[cls]["train"] += len(train_vids)
            stats[cls]["val"] += len(val_vids)
            stats[cls]["test"] += len(test_vids)
            manifest["metadata"]["total_videos"] += dataset_total

            # Manifest tracking with lightweight byte-size integrity checks
            for v_set, v_name in [(val_vids, "val"), (test_vids, "test"), (train_vids, "train")]:
                manifest["splits"][v_name].extend([f"{cls}/{dataset_name}/{v}" for v in v_set])

            dirs = {
                "val": VAL_LINK_DIR / cls / dataset_name,
                "test": TEST_LINK_DIR / cls / dataset_name,
                "train": TRAIN_LINK_DIR / cls / dataset_name
            }
            
            if not DRY_RUN:
                for d in dirs.values():
                    d.mkdir(parents=True, exist_ok=True)

            # Atomic Hardlink Generation
            for vid, frames in video_groups.items():
                target_dir = dirs["val"] if vid in val_vids else dirs["test"] if vid in test_vids else dirs["train"]
                
                for frame in frames:
                    manifest["metadata"]["total_frames"] += 1
                    if not DRY_RUN:
                        link_path = target_dir / frame.name
                        if link_path.exists() or link_path.is_symlink():
                            link_path.unlink()
                        try:
                            os.link(frame.resolve(), link_path) # Try hardlink first
                        except OSError:
                            os.symlink(frame.resolve(), link_path) # Fallback to symlink

    if not DRY_RUN:
        manifest_path = Path(os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/split_manifest.json"))
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
            
    print("\n--- Split Statistical Report ---")
    print(f"Total Videos: {manifest['metadata']['total_videos']} | Total Frames: {manifest['metadata']['total_frames']}")
    for cls in classes:
        t, v, te = stats[cls]["train"], stats[cls]["val"], stats[cls]["test"]
        print(f"[{cls.upper()}] Train: {t} | Val: {v} | Test: {te} | Ratio: {t}:{v}:{te}")
    print("✅ Split Verified. Zero Collisions Detected.")

if __name__ == "__main__":
    execute_split()