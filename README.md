# Facial & Biometric Signal Processing Toolkit

A collection of Python scripts to:

1. **Extract 68 facial landmarks** from videos
2. **Compute Facial Action Units (AUs)** from landmark CSVs
3. **Synchronize “emotion” (XCEPTION) bins with HR/EDA/TEMP**, generating labeled time-series
4. **Window & featurize** the combined data for machine learning

---

## 📂 Repository Layout

```
.
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── dataset/                   # raw videos
│   ├── s1/ … s30/
│   │   ├── T1.mp4
│   └── …
├── 30hz_labeled_scaled/       # landmark CSVs input
│   ├── s1/ … s30/
│   │   └── T1.csv
├── 30hz_full/                 # AU CSVs output
├── Cropped/                   # emotion data
│   ├── bioSignals_sep/
│   │   ├── <subject>/T1HR.csv, T1EDA.csv, T1TEMP.csv, …
│   ├── <subject1>/subject1V1XCEPTION.xlsx
│   └── …
├── Labels/                    # Excel stress labels per subject
│   └── <subject>.xlsx
├── Synch/                     # synchronized emotion+bio outputs
├── Emotions.csv               # merged emotion+bio signals for all users
├── 68landmarksgenerator.py
├── faugenerator.py
├── combine.py
├── featuregenerator.py
└── README.md


## ⚙️ Installation

1. **Clone** and enter repo:

   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create a virtualenv** (optional but recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   > If you don’t have a `requirements.txt`, you can install directly:
   >
   > ```bash
   > pip install numpy pandas scipy scikit-learn opencv-python dlib pandarallel tqdm openpyxl
   > ```

4. **Download the DLIB model**
   Place `shape_predictor_68_face_landmarks.dat` under `models/`.
   You can get it from: [http://dlib.net/files/shape\_predictor\_68\_face\_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

---

## 🚀 Usage

### 1. Extract 68-landmarks from videos

```bash
python 68landmarksgenerator.py
```

* **Input**: `dataset/s1…s30/*.mp4|.avi`
* **Output**: `dataset/sX/<video>.csv` (columns: `timestamp, landmark_0_x, landmark_0_y, …, landmark_67_y`)

### 2. Generate Facial Action Units (AUs)

```bash
python faugenerator.py
```

* **Input**: `30hz_labeled_scaled/s1…s30/*.csv` (landmark CSVs)
* **Output**: `30hz_full/sX/*.csv` (original cols + AU features)

### 3. Synchronize Emotion & Biometric Signals

```bash
python combine.py
```

* **Input**:

  * `Cropped/<subject>/<subject>V<task>XCEPTION.xlsx`
  * `Cropped/bioSignals_sep/<subject>/T<task>{HR,EDA,TEMP}.csv`
  * `Labels/<subject>.xlsx`
* **Output**: `Significance/<subject>/<subject>T<task>.csv`
  (columns: time-bins, averaged emotions, hr, eda, temp, `stress` label)

### 4. Window & Featurize for ML

```bash
python featuregenerator.py
```

* **Input**: `Emotions.csv`
* **Output**: `majid.csv`

  * Sliding windows (40 samples, step 20)
  * Stats: emotion means, EDA/HR/TEMP min/max/mean/std/skew/kurtosis/peaks, RMS, durations, binned `stress_label`, `user`

---

## 📝 Requirements

* **Python** ≥ 3.7
* **Key packages**:

  * `numpy`, `pandas`, `scipy`, `scikit-learn`
  * `opencv-python`, `dlib`, `pandarallel`, `tqdm`, `openpyxl`

Make sure your directory structure matches the layout above before running.

---

## 📖 Tips

* Scripts are wrapped in `if __name__=="__main__":` so you can also import their functions into other workflows.
* Adjust constants (e.g. window size, folder names) directly at the top of each script.
* Monitor console output for skipped files or errors.
