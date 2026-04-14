import os
import glob
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from torch.utils.data import Dataset, Subset, DataLoader
from transformers import VideoMAEForVideoClassification, AutoImageProcessor

from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# =========================================================
# CONFIGURATION
# Replace these placeholder strings with your own paths.
# =========================================================
KEYFRAME_BASE_DIR = "LOAD_KEYFRAME_DIRECTORY"
RESULTS_ROOT_DIR = "LOAD_RESULTS_DIRECTORY"
VIDEOMAE_CHECKPOINT_PATH = "LOAD_VIDEOMAE_CHECKPOINT"

CLASS_FOLDERS = {
    "DDSIT": "LOAD_DDSIT_FOLDER",
    "SIIT": "LOAD_SIIT_FOLDER",
    "OHSK": "LOAD_OHSK_FOLDER",
    "THSK": "LOAD_THSK_FOLDER",
}

SEED = 30
EPOCHS = 8
LR = 1e-4
BETA = 0.0
CHUNK_LEN = 16
LAMBDA_INIT = 2.0
N_SPLITS = 5


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=30):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Debug / validation helpers
# -----------------------------
def check_split_disjoint(train_idx, val_idx, dataset, fold):
    train_set = set(map(int, train_idx))
    val_set = set(map(int, val_idx))
    overlap = train_set.intersection(val_set)
    print(f"[CHECK A] Fold {fold}: index overlap count = {len(overlap)}")
    assert len(overlap) == 0, f"[LEAK] Fold {fold}: train/val index overlap detected!"

    train_paths = set()
    val_paths = set()

    for i in train_idx:
        item = dataset.samples[int(i)]
        train_paths.add(os.path.abspath(item["video_dir"]))

    for i in val_idx:
        item = dataset.samples[int(i)]
        val_paths.add(os.path.abspath(item["video_dir"]))

    overlap_paths = train_paths.intersection(val_paths)
    print(f"[CHECK A] Fold {fold}: video_dir overlap count = {len(overlap_paths)}")
    if len(overlap_paths) > 0:
        print("[LEAK] Example overlapping video_dir:", next(iter(overlap_paths)))
        raise AssertionError(f"[LEAK] Fold {fold}: train/val share the same underlying video(s)!")


def dist_debug_stats(l_cls, l_dist, beta):
    cls = float(l_cls.item())
    dist = float(l_dist.item())
    weighted = float(beta) * dist
    ratio = weighted / (cls + 1e-12)
    return cls, dist, weighted, ratio


# -----------------------------
# Distance-based classifier
# -----------------------------
class DistanceBiasedTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=1, lambda_init=2.0):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, F_chunks):
        p, d = F_chunks.shape
        Q = self.q_proj(F_chunks)
        K = self.k_proj(F_chunks)
        V = self.v_proj(F_chunks)

        attn_logits = (Q @ K.T) / (d ** 0.5)
        dist = torch.cdist(F_chunks, F_chunks, p=2)

        attn_logits = attn_logits - self.lambda_param * dist
        attn = F.softmax(attn_logits, dim=-1)
        F_out = attn @ V
        return F_out, dist


class VideoClassifierWithDistanceLoss(nn.Module):
    def __init__(self, feature_dim, num_classes, lambda_init=2.0):
        super().__init__()
        self.temporal_attn = DistanceBiasedTemporalAttention(
            feature_dim,
            lambda_init=lambda_init
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, F_chunks):
        F_att, dist = self.temporal_attn(F_chunks)
        logits_chunks = self.classifier(F_att)
        logits_video = logits_chunks.mean(dim=0, keepdim=True)
        return logits_video, logits_chunks, F_att, dist


def distance_consistency_loss(F_att, dist):
    sim = torch.exp(-dist)
    diff = F_att.unsqueeze(1) - F_att.unsqueeze(0)
    diff_sq = (diff ** 2).sum(dim=-1)
    return (sim * diff_sq).mean()


def training_step(model, F_chunks, label, beta=0.0):
    logits_video, logits_chunks, F_att, dist = model(F_chunks)
    loss_cls = F.cross_entropy(logits_video, label)
    loss_dist = distance_consistency_loss(F_att, dist)
    loss = loss_cls + beta * loss_dist
    return loss, loss_cls, loss_dist, logits_video


# -----------------------------
# Metrics
# -----------------------------
def make_compute_metrics(num_labels):
    def compute_metrics_from_preds(preds, labels):
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average=None,
            labels=list(range(num_labels)),
            zero_division=0,
        )
        macro_f1 = float(np.mean(f1)) if len(f1) else 0.0

        out = {"accuracy": float(acc), "macro_f1": float(macro_f1)}
        for i in range(num_labels):
            out[f"precision_{i}"] = float(prec[i]) if i < len(prec) else 0.0
            out[f"recall_{i}"] = float(rec[i]) if i < len(rec) else 0.0
            out[f"f1_{i}"] = float(f1[i]) if i < len(f1) else 0.0
        return out
    return compute_metrics_from_preds


# -----------------------------
# Frame / chunk helpers
# -----------------------------
def chunkify_video(frame_paths, chunk_len=16):
    chunks = []
    n = len(frame_paths)
    for start in range(0, n, chunk_len):
        chunk = frame_paths[start:start + chunk_len]
        if len(chunk) == 0:
            continue
        while len(chunk) < chunk_len:
            chunk.append(chunk[-1])
        chunks.append(chunk)
    return chunks


def load_chunk_rgb(chunk_paths, chunk_len=16):
    frames = []
    for p in chunk_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if len(frames) == 0:
        frames = [np.zeros((224, 224, 3), dtype=np.uint8)]

    while len(frames) < chunk_len:
        frames.append(frames[-1])

    return frames


# -----------------------------
# Dataset: stores paths only
# -----------------------------
class KeyframeFolderIndexDataset(Dataset):
    def __init__(self, class_roots_with_labels):
        self.samples = []

        for root, lbl in class_roots_with_labels:
            if not os.path.isdir(root):
                raise FileNotFoundError(f"Missing class root: {root}")

            video_dirs = sorted(
                [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
            )

            for vd in video_dirs:
                frame_paths = sorted(glob.glob(os.path.join(vd, "frame_*.jpg")))
                if len(frame_paths) == 0:
                    continue

                self.samples.append({
                    "video_dir": vd,
                    "frame_paths": frame_paths,
                    "label": lbl,
                })

        print(f"[INFO] Loaded {len(self.samples)} videos from {len(class_roots_with_labels)} class roots.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_bs1(features):
    return features[0]


# -----------------------------
# VideoMAE feature extractor (no cache)
# -----------------------------
class OnlineVideoMAEFeatureExtractor:
    def __init__(self, videomae_ckpt_path, chunk_len=16, device=None):
        self.chunk_len = chunk_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )

        self.videomae = VideoMAEForVideoClassification.from_pretrained(
            videomae_ckpt_path
        ).to(self.device)

        self.videomae.eval()
        for p in self.videomae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_F_chunks(self, frame_paths):
        chunks = chunkify_video(frame_paths, self.chunk_len)
        feats = []

        for chunk_paths in chunks:
            frames = load_chunk_rgb(chunk_paths, chunk_len=self.chunk_len)
            proc = self.processor(frames, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(self.device)

            out = self.videomae(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )

            last_h = out.hidden_states[-1].squeeze(0)
            feat = last_h.mean(dim=0)
            feats.append(feat)

        if len(feats) == 0:
            raise RuntimeError("No chunks were extracted for this video.")

        return torch.stack(feats, dim=0)


# -----------------------------
# Sanity check
# -----------------------------
def sanity_check_one_sample(dataset, extractor, num_labels, beta, device, lambda_init):
    if len(dataset) == 0:
        raise RuntimeError("Dataset empty; cannot sanity-check.")

    sample = dataset[0]
    F_chunks = extractor.extract_F_chunks(sample["frame_paths"])
    label = torch.tensor([sample["label"]], dtype=torch.long, device=device)

    print("\n===== SANITY CHECK =====")
    print("[SANITY] video_dir:", sample["video_dir"])
    print("[SANITY] F_chunks shape:", tuple(F_chunks.shape), "(expected: (p, 768))")
    print("[SANITY] label:", int(label.item()))

    if F_chunks.ndim != 2 or F_chunks.shape[1] != 768:
        raise RuntimeError(f"Unexpected F_chunks shape: {F_chunks.shape} (expected (?, 768))")

    p = int(F_chunks.shape[0])
    if p < 2:
        print("[SANITY][WARN] p < 2 (only one chunk). Distance loss is not meaningful when p=1.")

    model = VideoClassifierWithDistanceLoss(768, num_labels, lambda_init=lambda_init).to(device)

    logits_video, logits_chunks, F_att, dist = model(F_chunks)

    print("[SANITY] logits_video:", tuple(logits_video.shape), "(expected: (1, C))")
    print("[SANITY] logits_chunks:", tuple(logits_chunks.shape), "(expected: (p, C))")
    print("[SANITY] F_att:", tuple(F_att.shape), "(expected: (p, 768))")
    print("[SANITY] dist:", tuple(dist.shape), "(expected: (p, p))")

    loss, l_cls, l_dist, _ = training_step(model, F_chunks, label, beta=beta)

    cls, distv, weighted, ratio = dist_debug_stats(l_cls, l_dist, beta)
    print(f"[SANITY] beta={beta}")
    print(f"[SANITY] loss_cls={cls:.6f} loss_dist(raw)={distv:.6e} beta*dist={weighted:.6e} (beta*dist / cls = {ratio:.6e})")

    loss.backward()
    grad_ok = any(p.grad is not None for p in model.parameters())
    print("[SANITY] grads present on classifier params:", grad_ok)
    print("===== SANITY CHECK PASSED =====\n")


# -----------------------------
# 5-fold experiment runner
# -----------------------------
def run_experiment(
    exp_name,
    dataset_index,
    extractor,
    num_labels,
    class_names,
    results_root,
    seed=30,
    epochs=8,
    lr=1e-4,
    beta=0.0,
    lambda_init=2.0,
):
    set_seed(seed)

    exp_dir = os.path.join(results_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    device = extractor.device

    sanity_check_one_sample(
        dataset=dataset_index,
        extractor=extractor,
        num_labels=num_labels,
        beta=beta,
        device=device,
        lambda_init=lambda_init,
    )

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=43)
    all_idx = np.arange(len(dataset_index))

    fold_rows = []
    compute_metrics = make_compute_metrics(num_labels)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_idx), start=1):
        print(f"\n[INFO] ===== {exp_name}: Fold {fold}/{N_SPLITS} ===== Train={len(train_idx)} Val={len(val_idx)}")

        check_split_disjoint(train_idx, val_idx, dataset_index, fold)

        train_ds = Subset(dataset_index, train_idx)
        val_ds = Subset(dataset_index, val_idx)

        fold_out = os.path.join(exp_dir, f"fold{fold}")
        os.makedirs(fold_out, exist_ok=True)

        model = VideoClassifierWithDistanceLoss(
            feature_dim=768,
            num_classes=num_labels,
            lambda_init=lambda_init
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_bs1)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_bs1)

        best_macro_f1 = -1.0
        best_path = None

        for epoch in range(epochs):
            model.train()
            tr_loss = tr_cls = tr_dist = 0.0
            n_tr = 0

            for batch in train_dl:
                F_chunks = extractor.extract_F_chunks(batch["frame_paths"])
                labels = torch.tensor([batch["label"]], dtype=torch.long, device=device)

                opt.zero_grad(set_to_none=True)
                loss, l_cls, l_dist, _ = training_step(model, F_chunks, labels, beta=beta)

                if n_tr == 0:
                    cls, distv, weighted, ratio = dist_debug_stats(l_cls, l_dist, beta)
                    p = int(F_chunks.shape[0])
                    print(f"[CHECK D][train] Fold {fold} Epoch {epoch+1}: p={p} "
                          f"loss_cls={cls:.6f} loss_dist={distv:.6e} beta*dist={weighted:.6e} ratio={ratio:.6e}")

                loss.backward()
                opt.step()

                tr_loss += float(loss.item())
                tr_cls += float(l_cls.item())
                tr_dist += float(l_dist.item())
                n_tr += 1

            model.eval()
            print(f"[CHECK C] Fold {fold} Epoch {epoch+1}: model.training? {model.training}")

            all_preds = []
            all_labels = []
            ev_loss = ev_cls = ev_dist = 0.0
            n_ev = 0

            with torch.no_grad():
                for batch in val_dl:
                    F_chunks = extractor.extract_F_chunks(batch["frame_paths"])
                    labels = torch.tensor([batch["label"]], dtype=torch.long, device=device)

                    loss, l_cls, l_dist, logits_video = training_step(model, F_chunks, labels, beta=beta)
                    pred = int(torch.argmax(logits_video, dim=-1).item())

                    all_preds.append(pred)
                    all_labels.append(int(labels.item()))

                    ev_loss += float(loss.item())
                    ev_cls += float(l_cls.item())
                    ev_dist += float(l_dist.item())
                    n_ev += 1

            metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
            metrics["loss"] = ev_loss / max(1, n_ev)
            metrics["loss_cls"] = ev_cls / max(1, n_ev)
            metrics["loss_dist"] = ev_dist / max(1, n_ev)

            print(
                f"[INFO] Fold {fold} Epoch {epoch+1}/{epochs} "
                f"train_loss={tr_loss/max(1,n_tr):.4f} "
                f"eval_acc={metrics['accuracy']:.4f} eval_macro_f1={metrics['macro_f1']:.4f}"
            )

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_path = os.path.join(fold_out, "best_classifier.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "num_labels": num_labels,
                        "class_names": class_names,
                        "lr": lr,
                        "beta": beta,
                    },
                    best_path
                )

        if best_path and os.path.exists(best_path):
            ck = torch.load(best_path, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            model.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_dl:
                F_chunks = extractor.extract_F_chunks(batch["frame_paths"])
                labels = torch.tensor([batch["label"]], dtype=torch.long, device=device)
                logits_video, _, _, _ = model(F_chunks)
                pred = int(torch.argmax(logits_video, dim=-1).item())
                all_preds.append(pred)
                all_labels.append(int(labels.item()))

        labels_np = np.array(all_labels)
        preds_np = np.array(all_preds)

        eval_results = compute_metrics(preds_np, labels_np)

        report_txt = classification_report(labels_np, preds_np, target_names=class_names, zero_division=0)
        with open(os.path.join(exp_dir, f"classification_report_fold{fold}.txt"), "w") as f:
            f.write(report_txt)

        cm = confusion_matrix(labels_np, preds_np, labels=list(range(num_labels)))
        np.save(os.path.join(exp_dir, f"confusion_fold{fold}.npy"), cm)

        row = {"experiment": exp_name, "fold": fold}
        for k, v in eval_results.items():
            row[k] = v
        fold_rows.append(row)

    csv_path = os.path.join(exp_dir, "metrics_5fold.csv")
    fieldnames = sorted(set().union(*[r.keys() for r in fold_rows]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(RESULTS_ROOT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = OnlineVideoMAEFeatureExtractor(
        videomae_ckpt_path=VIDEOMAE_CHECKPOINT_PATH,
        chunk_len=CHUNK_LEN,
        device=device,
    )

    dataset_4 = KeyframeFolderIndexDataset([
        (CLASS_FOLDERS["DDSIT"], 0),
        (CLASS_FOLDERS["SIIT"], 1),
        (CLASS_FOLDERS["OHSK"], 2),
        (CLASS_FOLDERS["THSK"], 3),
    ])

    run_experiment(
        exp_name="4class_experiment",
        dataset_index=dataset_4,
        extractor=extractor,
        num_labels=4,
        class_names=["DDSIT", "SIIT", "OHSK", "THSK"],
        results_root=RESULTS_ROOT_DIR,
        seed=SEED,
        epochs=EPOCHS,
        lr=LR,
        beta=BETA,
        lambda_init=LAMBDA_INIT,
    )

    dataset_sut = KeyframeFolderIndexDataset([
        (CLASS_FOLDERS["DDSIT"], 0),
        (CLASS_FOLDERS["SIIT"], 1),
    ])

    run_experiment(
        exp_name="suturing_binary_experiment",
        dataset_index=dataset_sut,
        extractor=extractor,
        num_labels=2,
        class_names=["DDSIT", "SIIT"],
        results_root=RESULTS_ROOT_DIR,
        seed=SEED,
        epochs=EPOCHS,
        lr=LR,
        beta=BETA,
        lambda_init=LAMBDA_INIT,
    )

    dataset_knot = KeyframeFolderIndexDataset([
        (CLASS_FOLDERS["OHSK"], 0),
        (CLASS_FOLDERS["THSK"], 1),
    ])

    run_experiment(
        exp_name="knotting_binary_experiment",
        dataset_index=dataset_knot,
        extractor=extractor,
        num_labels=2,
        class_names=["OHSK", "THSK"],
        results_root=RESULTS_ROOT_DIR,
        seed=SEED,
        epochs=EPOCHS,
        lr=LR,
        beta=BETA,
        lambda_init=LAMBDA_INIT,
    )


if __name__ == "__main__":
    main()
