#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ————————————————
# Configuration
# ————————————————
BASE_DIR        = Path(__file__).parent.resolve()
EMOTION_DIR     = BASE_DIR / "Cropped"
BIOSIG_DIR      = EMOTION_DIR / "bioSignals_sep"
LABELS_DIR      = BASE_DIR / "Labels"
OUTPUT_DIR      = BASE_DIR / "Significance"
TASK_COUNT      = 8

# ————————————————
# I/O helpers
# ————————————————
def load_csv(df_dir: Path, subject: str, task: int, suffix: str, col_names):
    """
    Load T{task}{suffix}.csv from df_dir/subject, rename cols.
    """
    path = df_dir / subject / f"T{task}{suffix}.csv"
    df = pd.read_csv(path)
    df.columns = col_names
    return df

def load_emotion(subject: str, task: int):
    """
    Load the emotion Excel file: {subject}V{task}XCEPTION.xlsx
    """
    path = EMOTION_DIR / subject / f"{subject}V{task}XCEPTION.xlsx"
    return pd.read_excel(path)

# ————————————————
# Synchronization logic
# ————————————————
def sync_signals(em_df, hr_df, eda_df, temp_df, labels, task):
    """
    Align emotion (4 bins/sec), HR/EDA/TEMP (1 Hz → repeated 4×),
    then concatenate and assign stress labels.
    """
    # Round times, floor to Int, and split to 4-frame bins
    em_df = em_df.copy()
    em_df["Time (seconds)"] = em_df["Time (seconds)"].round(3)
    em_df["TimeInt"]  = np.floor(em_df["Time (seconds)"])
    em_df["FrameSpec"]= ((em_df["Time (seconds)"] - em_df["TimeInt"])
                         .apply(lambda x: 1 if x < 0.24 else
                                          2 if x < 0.48 else
                                          3 if x < 0.72 else 4))
    # aggregate per (TimeInt, FrameSpec)
    emotions = em_df.groupby(
        ["TimeInt", "FrameSpec"]
    ).agg({c:"mean" for c in ['Angry','Disgust','Scared','Happy','Sad','Surprised','Neutral']}
    ).reset_index()

    # fill missing bins up to 540 seconds
    for t in range(540):
        present = emotions.loc[emotions.TimeInt==t, "FrameSpec"].tolist()
        for spec in set(range(1,5)) - set(present):
            # copy last known values
            if spec == 1 and t > 0:
                src = emotions[(emotions.TimeInt==t-1)&(emotions.FrameSpec==4)]
            else:
                src = emotions[(emotions.TimeInt==t)&(emotions.FrameSpec==spec-1)]
            if not src.empty:
                row = dict(TimeInt=t, FrameSpec=spec, **src.iloc[0][2:].to_dict())
                emotions = pd.concat([emotions, pd.DataFrame([row])], ignore_index=True)

    emotions = emotions[emotions.TimeInt < 540].sort_values(["TimeInt","FrameSpec"])

    # repeat HR/EDA/TEMP 4× to align
    hr_rep   = hr_df.loc[hr_df.index.repeat(4)].reset_index(drop=True)
    eda_rep  = eda_df.loc[eda_df.index.repeat(4)].reset_index(drop=True)
    temp_rep = temp_df.loc[temp_df.index.repeat(4)].reset_index(drop=True)

    # concat
    total = pd.concat(
        [emotions.reset_index(drop=True),
         hr_rep["hr"],
         eda_rep["eda"],
         temp_rep["temp"]],
        axis=1
    )

    # assign stress labels
    stress = np.ones(len(total), dtype=int)
    if task in {2,3,5,7}:
        idx0 = 240
        blocks = [labels.iloc[task-1-(task>4)*2, i] for i in range(1,6)]
        # first 240 → blocks[0]; then 4 blocks of 480; then rest
        stress[:240] = blocks[0]
        for i in range(4):
            start = idx0 + i*480
            stress[start:start+480] = blocks[i+1]
        stress[1920:] = blocks[-1]
    total["stress"] = stress
    return total

# ————————————————
# Main
# ————————————————
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    # discover subjects
    subjects = [d.name for d in EMOTION_DIR.iterdir() if d.is_dir()]
    for subj in tqdm(subjects, desc="Subjects"):
        out_sub = OUTPUT_DIR / subj
        out_sub.mkdir(exist_ok=True)

        for task in range(1, TASK_COUNT+1):
            try:
                emo = load_emotion(subj, task)
                hr  = load_csv(BIOSIG_DIR, subj, task, "HR",   ["hr","epoch"])
                eda = load_csv(BIOSIG_DIR, subj, task, "EDA",  ["eda","epoch"])
                tmp = load_csv(BIOSIG_DIR, subj, task, "TEMP", ["temp","epoch"])
                lbl = pd.read_excel(LABELS_DIR / f"{subj}.xlsx")

                merged = sync_signals(emo, hr, eda, tmp, lbl, task)
                merged.to_csv(out_sub / f"{subj}T{task}.csv", index=False)
            except Exception as e:
                print(f"[!] {subj} T{task} failed: {e}")

if __name__ == "__main__":
    main()
