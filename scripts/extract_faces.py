import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image
import multiprocessing as mp

# ---------------- CONFIG ---------------- #
INPUT_ROOT = "/mnt/d/Dataset/"
OUTPUT_ROOT = os.path.expanduser("~/Deepfake_Project_Root/dataset_lake/extracted_faces/train/")

FRAMES_PER_VIDEO = 5
MARGIN = 20
BATCH_SIZE = 8
NUM_WORKERS = 2  # keep low for GPU stability

FAKE_KEYWORDS = ['fake', 'manipulated', 'deepfakes', 'face2face', 'faceshifter', 'faceswap', 'neuraltextures', 'celeb-synthesis']
REAL_KEYWORDS = ['real', 'original', 'youtube-real']

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---------------- GLOBAL MODEL ---------------- #
mtcnn = None

def worker_init():
    global mtcnn
    mtcnn = MTCNN(
        keep_all=False,
        margin=MARGIN,
        post_process=False,
        device=device
    )

# ---------------- LABEL ---------------- #
def determine_label(file_path):
    path_lower = str(file_path).lower()
    if any(k in path_lower for k in FAKE_KEYWORDS):
        return "fake"
    elif any(k in path_lower for k in REAL_KEYWORDS):
        return "real"
    return None

# ---------------- BATCH SAVE ---------------- #
def save_batch(images, paths):
    global mtcnn
    try:
        mtcnn(images, save_path=[str(p) for p in paths])
    except Exception:
        pass

# ---------------- VIDEO PROCESS ---------------- #
def process_video(args):
    file_path, input_root = args

    try:
        label = determine_label(file_path)
        if label is None:
            return

        dataset_name = file_path.relative_to(input_root).parts[0]
        output_dir = Path(OUTPUT_ROOT) / label / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(file_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return

        # ✅ FIX 1: Temporal contiguous sampling (center clip)
        start_frame = max(0, (total_frames // 2) - (FRAMES_PER_VIDEO // 2))
        end_frame = min(total_frames, start_frame + FRAMES_PER_VIDEO)
        frame_indices = set(range(start_frame, end_frame))

        base_name = file_path.stem

        batch_imgs, batch_paths = [], []
        current_idx = 0

        while cap.isOpened() and frame_indices:
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in frame_indices:
                out_path = output_dir / f"{dataset_name}_{base_name}_f{current_idx:04d}.jpg"

                # Resume-safe
                if out_path.exists():
                    frame_indices.remove(current_idx)
                    current_idx += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

                batch_imgs.append(img)
                batch_paths.append(out_path)

                if len(batch_imgs) >= BATCH_SIZE:
                    save_batch(batch_imgs, batch_paths)
                    batch_imgs, batch_paths = [], []

                frame_indices.remove(current_idx)

            current_idx += 1

        if batch_imgs:
            save_batch(batch_imgs, batch_paths)

        cap.release()

    except Exception:
        pass


# ---------------- IMAGE PROCESS ---------------- #
def process_image(args):
    file_path, input_root = args

    try:
        label = determine_label(file_path)
        if label is None:
            return

        dataset_name = file_path.relative_to(input_root).parts[0]
        output_dir = Path(OUTPUT_ROOT) / label / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{dataset_name}_{file_path.stem}.jpg"

        if out_path.exists():
            return

        frame = cv2.imread(str(file_path))
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        global mtcnn
        mtcnn([img], save_path=[str(out_path)])

    except Exception:
        pass


# ---------------- MAIN ---------------- #
def main():
    # ✅ FIX: CUDA-safe multiprocessing
    mp.set_start_method('spawn', force=True)

    input_root = Path(INPUT_ROOT)

    print(f"Device: {device}")
    print(f"Workers: {NUM_WORKERS}")

    all_files = []
    for ext in ['**/*.mp4', '**/*.avi', '**/*.png', '**/*.jpg', '**/*.jpeg']:
        all_files.extend(list(input_root.glob(ext)))

    video_files = [f for f in all_files if f.suffix.lower() in ['.mp4', '.avi']]
    image_files = [f for f in all_files if f.suffix.lower() not in ['.mp4', '.avi']]

    print(f"Videos: {len(video_files)} | Images: {len(image_files)}")

    # ---------------- VIDEOS ---------------- #
    with mp.Pool(NUM_WORKERS, initializer=worker_init) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_video, [(f, input_root) for f in video_files]),
            total=len(video_files),
            desc="Processing Videos"
        ):
            pass

    # ---------------- IMAGES ---------------- #
    with mp.Pool(NUM_WORKERS, initializer=worker_init) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_image, [(f, input_root) for f in image_files]),
            total=len(image_files),
            desc="Processing Images"
        ):
            pass

    print("✅ Face extraction complete.")


if __name__ == "__main__":
    main()