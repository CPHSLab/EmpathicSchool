import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ————————————————
# Configuration (relative paths)
# ————————————————
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
INPUT_DIR    = os.path.join(BASE_DIR, "30hz_labeled_scaled")
OUTPUT_DIR   = os.path.join(BASE_DIR, "30hz_full")
FOLDERS      = [f"s{i}" for i in range(1, 31)]

# ————————————————
# AU calculation functions
# ————————————————
def calculate_inner_brow_raiser(data):
    baseline = data[['y_21', 'y_22', 'y_23', 'y_24']].mean().mean()
    return ((data[['y_21', 'y_22', 'y_23', 'y_24']].mean(axis=1) - baseline)
            .tolist())

def calculate_outer_brow_raiser(data):
    baseline = data[['y_25', 'y_26', 'y_19', 'y_20']].mean().mean()
    return ((data[['y_25', 'y_26', 'y_19', 'y_20']].mean(axis=1) - baseline)
            .tolist())

def calculate_brow_lowerer(row):
    left_dist  = np.hypot(row.x_20 - row.x_22, row.y_20 - row.y_22)
    right_dist = np.hypot(row.x_25 - row.x_23, row.y_25 - row.y_23)
    return (left_dist + right_dist) / 2

def calculate_upper_lid_raiser(row):
    left  = np.hypot(row.x_38 - row.x_42, row.y_38 - row.y_42)
    left2 = np.hypot(row.x_39 - row.x_41, row.y_39 - row.y_41)
    right = np.hypot(row.x_44 - row.x_48, row.y_44 - row.y_48)
    right2= np.hypot(row.x_45 - row.x_47, row.y_45 - row.y_47)
    return (left + left2 + right + right2) / 4

def calculate_cheek_raiser(row):
    return np.hypot(row.y_49 - row.y_37, row.y_55 - row.y_46) / 2

def calculate_lid_tightener(row):
    up   = np.hypot(row.x_38 - row.x_42, row.y_38 - row.y_42)
    up2  = np.hypot(row.x_44 - row.x_48, row.y_44 - row.y_48)
    return (up + up2) / 2

def calculate_nose_wrinkler(row):
    return np.hypot(row.x_28 - row.x_34, row.y_28 - row.y_34)

def calculate_upper_lip_raiser(row):
    return np.hypot(row.x_52 - row.x_34, row.y_52 - row.y_34)

def calculate_nasolabial_furrow_deepener(row):
    return np.hypot(row.x_33 - row.x_35, row.y_33 - row.y_35)

# …and so on for AU13–AU41, e.g.:
def au13_cheek_puffer(row):
    return np.hypot(row.x_13 - row.x_5, row.y_13 - row.y_5)

# (Define the rest of your AUxx_* functions similarly, taking a single row.)

# ————————————————
# Feature-wrapping
# ————————————————
def compute_facial_features(df):
    feats = {
        'inner_brow_raiser': calculate_inner_brow_raiser(df),
        'outer_brow_raiser': calculate_outer_brow_raiser(df),
        # per-frame AUs that need row-wise apply:
        'brow_lowerer': df.apply(calculate_brow_lowerer, axis=1),
        'upper_lid_raiser': df.apply(calculate_upper_lid_raiser, axis=1),
        'cheek_raiser': df.apply(calculate_cheek_raiser, axis=1),
        'lid_tightener': df.apply(calculate_lid_tightener, axis=1),
        'nose_wrinkler': df.apply(calculate_nose_wrinkler, axis=1),
        'upper_lip_raiser': df.apply(calculate_upper_lip_raiser, axis=1),
        'nasolabial_furrow_deepener': df.apply(calculate_nasolabial_furrow_deepener, axis=1),
        'cheek_puffer': df.apply(au13_cheek_puffer, axis=1),
        # … add remaining AUs here …
    }
    # Convert any list-valued entries to a Series
    for k, v in feats.items():
        feats[k] = pd.Series(v)
    return pd.DataFrame(feats)

# ————————————————
# Main script
# ————————————————
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sf in FOLDERS:
        inp_sub  = os.path.join(INPUT_DIR, sf)
        out_sub  = os.path.join(OUTPUT_DIR, sf)
        os.makedirs(out_sub, exist_ok=True)

        if not os.path.isdir(inp_sub):
            print(f"[!] Skipping missing folder: {sf}")
            continue

        for fname in os.listdir(inp_sub):
            if not fname.lower().endswith(".csv"):
                continue

            in_path  = os.path.join(inp_sub, fname)
            out_path = os.path.join(out_sub, fname)

            df = pd.read_csv(in_path)
            feats_df = compute_facial_features(df)
            result = pd.concat([df, feats_df], axis=1)

            if len(result) != len(df):
                print(f"[!] Row mismatch in {sf}/{fname}")

            result.to_csv(out_path, index=False)
            print(f"[✓] Wrote features for {sf}/{fname}")
