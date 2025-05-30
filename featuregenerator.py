#!/usr/bin/env python3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

WINDOW_SIZE = 40
STEP       = 20
INPUT_CSV  = Path("Emotions.csv")
OUTPUT_CSV = Path("majid.csv")
FEATURE_COLS = [
    "Angry","Disgust","Scared","Happy","Sad","Surprised","Neutral",
    "hr","eda","temp","stress","id"
]

def window_generator(df, size: int, step: int):
    """Yield (start_index, window_df) for sliding windows."""
    for start in range(0, len(df) - size + 1, step):
        yield start, df.iloc[start:start+size]

def compute_features(win: pd.DataFrame) -> dict:
    """Compute all AU and bio-signal stats on one window."""
    eda  = win["eda"].values
    hr   = win["hr"].values
    temp = win["temp"].values

    stats = {}

    # 1) Emotion means
    for emo in ["Angry","Disgust","Scared","Happy","Sad","Surprised","Neutral"]:
        stats[emo.lower()] = win[emo].mean()

    # 2) EDA basic stats + shape
    stats.update({
        "eda_min":    eda.min(),
        "eda_max":    eda.max(),
        "eda_mean":   eda.mean(),
        "eda_std":    eda.std(),
        "eda_skew":   skew(eda),
        "eda_kurtosis": kurtosis(eda),
    })
    peaks, props = find_peaks(eda, width=5, prominence=0)
    stats["eda_num_peaks"]  = len(peaks)
    stats["eda_amplitude"]  = props.get("prominences", []).sum()
    stats["eda_duration"]   = props.get("widths", []).sum()

    # 3) HR stats + shape
    stats.update({
        "hr_min":  hr.min(),
        "hr_max":  hr.max(),
        "hr_mean": hr.mean(),
        "hr_std":  hr.std(),
        "hr_rms":  np.sqrt(np.mean(np.diff(hr)**2)),
    })
    hr_peaks, hr_props = find_peaks(hr, width=5, prominence=0)
    stats["hr_num_peaks"]  = len(hr_peaks)
    stats["hr_amplitude"]  = hr_props.get("prominences", []).sum()
    stats["hr_duration"]   = hr_props.get("widths", []).sum()

    # 4) Temp stats
    stats.update({
        "temp_min":  temp.min(),
        "temp_max":  temp.max(),
        "temp_mean": temp.mean(),
        "temp_std":  temp.std(),
    })

    # 5) Stress label buckets
    stress_mean = win["stress"].mean()
    if stress_mean <= 6.7:
        stats["stress_label"] = 0
    elif stress_mean <= 13.4:
        stats["stress_label"] = 1
    else:
        stats["stress_label"] = 2

    # 6) Copy user ID
    stats["user"] = win["id"].iloc[0][:2]
    return stats

def process_subject(df_sub: pd.DataFrame) -> list[dict]:
    """Normalize & window a single subject’s data, return list of feature dicts."""
    # 1) MinMax normalize all but id, subject, stress
    scaler = MinMaxScaler()
    to_norm = df_sub[["Angry","Disgust","Scared","Happy","Sad","Surprised","Neutral","hr","eda","temp"]]
    normed = scaler.fit_transform(to_norm)
    norm_df = pd.DataFrame(normed, columns=to_norm.columns, index=df_sub.index)

    # 2) Reassemble with stress & id
    full = pd.concat([norm_df, df_sub[["stress","id"]]], axis=1)

    # 3) Slide windows and compute
    feats = []
    for _, window in window_generator(full, WINDOW_SIZE, STEP):
        feats.append(compute_features(window))
    return feats

def main():
    # Load and prep
    df = pd.read_csv(INPUT_CSV, usecols=FEATURE_COLS)
    df["subject"] = df["id"].str[:2]

    # Process each subject
    all_feats = []
    for subj, group in df.groupby("subject"):
        print(f"Processing subject {subj} ({len(group)} samples)...")
        all_feats.extend(process_subject(group))

    # Save
    out_df = pd.DataFrame(all_feats)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} feature windows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
