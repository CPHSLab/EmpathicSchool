import os
import cv2
import dlib
import pandas as pd
from pandarallel import pandarallel

# Initialize pandarallel for faster apply
pandarallel.initialize(progress_bar=True)

# Paths are now relative to the repository root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")

# Load dlib’s face detector and landmark predictor
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

def process_video(video_path):
    """
    Process a single video: detect face landmarks frame-by-frame,
    accumulate into a DataFrame, and write out a CSV alongside the video.
    """
    video_name = os.path.basename(video_path)
    folder     = os.path.dirname(video_path)
    print(f"[+] Processing {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Cannot open {video_name}, skipping.")
        return

    records = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Timestamp in seconds
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            coords = [(p.x, p.y) for p in shape.parts()]
            flat   = [coord for point in coords for coord in point]
            records.append([ts] + flat)

    cap.release()

    if not records:
        print(f"[!] No landmarks found in {video_name}")
        return

    # Build DataFrame
    cols = ["timestamp"] + [f"landmark_{i}_{ax}" 
                            for i in range(68) 
                            for ax in ("x", "y")]
    df = pd.DataFrame(records, columns=cols)

    # Save CSV next to the video file
    out_csv = os.path.splitext(video_path)[0] + ".csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved landmarks to {out_csv}")

def get_all_video_files(base_dir, subfolders):
    """
    Walk each subfolder under `base_dir` and collect .mp4/.avi files.
    """
    files = []
    for sf in subfolders:
        folder_path = os.path.join(base_dir, sf)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".mp4", ".avi")):
                files.append(os.path.join(folder_path, fname))
    return files

if __name__ == "__main__":
    # automatically generate s1..s30
    folders = [f"s{i}" for i in range(1, 31)]

    video_paths = get_all_video_files(DATASET_DIR, folders)
    print(f"[+] Found {len(video_paths)} videos across {len(folders)} folders")

    # parallel processing
    pd.Series(video_paths).parallel_apply(process_video)
