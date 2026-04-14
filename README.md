# dist-attn-surgical-classification

Code for distance-aware attention for surgical video classification using VideoMAE features.

---

## Setup

Install required packages:

pip install torch torchvision transformers opencv-python numpy scikit-learn

---

## Configure Paths

In `dist_classifier.py`, update the following:

- `VIDEOMAE_CHECKPOINT_PATH` → path to your pretrained VideoMAE checkpoint  
- `CLASS_FOLDERS` → paths to your dataset folders  

Example:

CLASS_FOLDERS = {
    "DDSIT": "<path to DDSIT>",
    "SIIT": "<path to SIIT>",
    "OHSK": "<path to OHSK>",
    "THSK": "<path to THSK>",
}

---

## Dataset Format

Each class folder should contain video folders with frames:

CLASS_NAME/
    video_1/
        frame_000.jpg
        frame_001.jpg
    video_2/
        ...

---

## Run

python dist_classifier.py

---

## What the Code Does

- Loads video frames and splits them into fixed-length chunks  
- Uses a pretrained VideoMAE model to extract features for each chunk  
- Applies a distance-based temporal attention layer to model relationships between chunks  
- Aggregates chunk-level predictions into a video-level classification  
- Trains and evaluates using 5-fold cross-validation  
- Runs three tasks:
  - 4-class classification
  - Binary suturing classification
  - Binary knot-tying classification  
- Outputs metrics, confusion matrices, and best model checkpoints
