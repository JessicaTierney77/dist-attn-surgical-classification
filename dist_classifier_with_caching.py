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
# Debug / validation helpers (A-C-D)
# -----------------------------
def check_split_disjoint(train_idx, val_idx, cached_dataset, fold):
    # A1: index-level overlap
    train_set = set(map(int, train_idx))
    val_set = set(map(int, val_idx))
    overlap = train_set.intersection(val_set)
    print(f"[CHECK A] Fold {fold}: index overlap count = {len(overlap)}")
    assert len(overlap) == 0, f"[LEAK] Fold {fold}: train/val index overlap detected!"

    # A2: path/video-level overlap (catches duplicates under different indices)
    # NOTE: this forces loading cache for those samples; can be expensive but is the best check.
    train_paths = set()
    val_paths = set()

    for i in train_idx:
        item = cached_dataset[int(i)]  # triggers cache read/compute
        train_paths.add(item["cache_path"])

    for i in val_idx:
        item = cached_dataset[int(i)]
        val_paths.add(item["cache_path"])

    overlap_paths = train_paths.intersection(val_paths)
    print(f"[CHECK A] Fold {fold}: cache_path overlap count = {len(overlap_paths)}")
    if len(overlap_paths) > 0:
        print("[LEAK] Example overlapping cache_path:", next(iter(overlap_paths)))
        raise AssertionError(f"[LEAK] Fold {fold}: train/val share the same underlying video(s)!")

def dist_debug_stats(l_cls, l_dist, beta):
    # D: show whether dist loss is meaningful and how much it contributes
    cls = float(l_cls.item())
    dist = float(l_dist.item())
    weighted = float(beta) * dist
    ratio = weighted / (cls + 1e-12)
    return cls, dist, weighted, ratio

# -----------------------------
# Your distance-based classifier
# -----------------------------
class DistanceBiasedTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=1, lambda_init=2.0):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, F_chunks):
        """
        F_chunks: (p, d)
        """
        p, d = F_chunks.shape
        Q = self.q_proj(F_chunks)
        K = self.k_proj(F_chunks)
        V = self.v_proj(F_chunks)

        attn_logits = (Q @ K.T) / (d ** 0.5)         # (p, p)
        dist = torch.cdist(F_chunks, F_chunks, p=2)  # (p, p)

        attn_logits = attn_logits - self.lambda_param * dist
        attn = F.softmax(attn_logits, dim=-1)        # (p, p)
        F_out = attn @ V                             # (p, d)
        return F_out, dist

class VideoClassifierWithDistanceLoss(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.temporal_attn = DistanceBiasedTemporalAttention(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, F_chunks):
        """
        F_chunks: (p, d)
        """
        F_att, dist = self.temporal_attn(F_chunks)
        logits_chunks = self.classifier(F_att)                   # (p, C)
        logits_video  = logits_chunks.mean(dim=0, keepdim=True)  # (1, C)
        return logits_video, logits_chunks, F_att, dist

def distance_consistency_loss(F_att, dist):
    sim = torch.exp(-dist)  # (p, p)
    diff = F_att.unsqueeze(1) - F_att.unsqueeze(0)  # (p, p, d)
    diff_sq = (diff ** 2).sum(dim=-1)               # (p, p)
    return (sim * diff_sq).mean()

def training_step(model, F_chunks, label, beta=0.0):
    logits_video, logits_chunks, F_att, dist = model(F_chunks)
    loss_cls = F.cross_entropy(logits_video, label)
    loss_dist = distance_consistency_loss(F_att, dist)
    loss = loss_cls + beta * loss_dist
    return loss, loss_cls, loss_dist, logits_video

# -----------------------------
# Metrics (same style as yours)
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
# Dataset index: one folder = one video
# (keeps all frames so we can chunkify)
# -----------------------------
class KeyframeFolderIndexDataset(Dataset):
    """
    Expects directory layout like:
      class_root/
        VIDEO_A/
          frame_000.jpg ...
        VIDEO_B/
          frame_000.jpg ...
    Each VIDEO_* folder is one sample (one video).
    """
    def __init__(self, class_roots_with_labels):
        self.samples = []  # (video_dir, frame_paths, label)
        for root, lbl in class_roots_with_labels:
            if not os.path.isdir(root):
                raise FileNotFoundError(f"Missing class root: {root}")
            video_dirs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
            for vd in video_dirs:
                frame_paths = sorted(glob.glob(os.path.join(vd, "frame_*.jpg")))
                if len(frame_paths) == 0:
                    continue
                self.samples.append((vd, frame_paths, lbl))
        print(f"[INFO] Loaded {len(self.samples)} videos from {len(class_roots_with_labels)} class roots.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, frame_paths, label = self.samples[idx]
        return {"video_dir": video_dir, "frame_paths": frame_paths, "label": label}

# -----------------------------
# VideoMAE chunk feature extraction + caching
# -----------------------------
def chunkify_video(frame_paths, chunk_len=16):
    chunks = []
    n = len(frame_paths)
    for start in range(0, n, chunk_len):
        chunk = frame_paths[start:start + chunk_len]
        if len(chunk) == 0:
            continue
        # within-chunk padding to 16 frames
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

def safe_video_id(video_dir):
    parts = [p for p in video_dir.split(os.sep) if p]
    return "__".join(parts[-2:])  # e.g. CLASSROOT__VIDFOLDER

class FeatureCache:
    def __init__(self, cache_dir, videomae_ckpt_path, chunk_len=16, device=None):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.chunk_len = chunk_len

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        self.videomae = VideoMAEForVideoClassification.from_pretrained(videomae_ckpt_path).to(self.device)
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

            out = self.videomae(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
            last_h = out.hidden_states[-1].squeeze(0)  # (seq_len, 768)
            feat = last_h.mean(dim=0)                  # (768,)
            feats.append(feat.cpu())
        return torch.stack(feats, dim=0)  # (p, 768) CPU

    def get_or_compute(self, video_dir, frame_paths, label):
        vid_id = safe_video_id(video_dir)
        out_path = os.path.join(self.cache_dir, f"{vid_id}.pt")

        if os.path.exists(out_path):
            data = torch.load(out_path, map_location="cpu")

            cached_label = int(data.get("label", -999))
            expected_label = int(label)

            # B: cache label mismatch check
            if cached_label != expected_label:
                print(f"[CHECK B][WARN] Cache label mismatch for {out_path}")
                print(f"  cached_label={cached_label} expected_label={expected_label} video_dir={video_dir}")
                raise RuntimeError("Cache label mismatch detected — possible ID collision or dataset mapping error.")

            return data["F_chunks"], cached_label, out_path

        F_chunks = self.extract_F_chunks(frame_paths)
        torch.save({"F_chunks": F_chunks, "label": int(label), "video_dir": video_dir}, out_path)
        return F_chunks, int(label), out_path

# -----------------------------
# Cached feature dataset (for training)
# -----------------------------
class CachedFeatureDataset(Dataset):
    def __init__(self, base_index_dataset: Dataset, feature_cache: FeatureCache):
        self.base = base_index_dataset
        self.cache = feature_cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        F_chunks, label, cache_path = self.cache.get_or_compute(
            item["video_dir"], item["frame_paths"], item["label"]
        )
        return {
            "F_chunks": F_chunks,  # (p,768) CPU
            "labels": torch.tensor([label], dtype=torch.long),  # (1,)
            "video_dir": item["video_dir"],
            "cache_path": cache_path,
        }

def collate_bs1(features):
    return features[0]

# -----------------------------
# Sanity checks
# -----------------------------
def sanity_check_one_sample(cached_ds, num_labels, beta, device):
    if len(cached_ds) == 0:
        raise RuntimeError("Dataset empty; cannot sanity-check.")

    sample = cached_ds[0]
    F_chunks = sample["F_chunks"]
    label = sample["labels"]

    print("\n===== SANITY CHECK =====")
    print("[SANITY] video_dir:", sample["video_dir"])
    print("[SANITY] cache_path:", sample["cache_path"])
    print("[SANITY] F_chunks shape:", tuple(F_chunks.shape), "(expected: (p, 768))")
    print("[SANITY] label:", int(label.item()))

    if F_chunks.ndim != 2 or F_chunks.shape[1] != 768:
        raise RuntimeError(f"Unexpected F_chunks shape: {F_chunks.shape} (expected (?, 768))")

    p = int(F_chunks.shape[0])
    if p < 2:
        print("[SANITY][WARN] p < 2 (only one chunk). Distance loss is not meaningful when p=1.")

    model = VideoClassifierWithDistanceLoss(768, num_labels).to(device)

    F_chunks_dev = F_chunks.to(device)
    label_dev = label.to(device)

    logits_video, logits_chunks, F_att, dist = model(F_chunks_dev)

    print("[SANITY] logits_video:", tuple(logits_video.shape), "(expected: (1, C))")
    print("[SANITY] logits_chunks:", tuple(logits_chunks.shape), "(expected: (p, C))")
    print("[SANITY] F_att:", tuple(F_att.shape), "(expected: (p, 768))")
    print("[SANITY] dist:", tuple(dist.shape), "(expected: (p, p))")

    loss, l_cls, l_dist, _ = training_step(model, F_chunks_dev, label_dev, beta=beta)

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
    exp_name: str,
    dataset_index: Dataset,
    videomae_ckpt_path: str,
    num_labels: int,
    class_names: list,
    results_root: str,
    cache_dir: str,
    seed: int = 30,
    epochs: int = 8,
    lr: float = 1e-4,
    beta: float = 0.1,
):
    set_seed(seed)

    exp_dir = os.path.join(results_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_cache = FeatureCache(
        cache_dir=cache_dir,
        videomae_ckpt_path=videomae_ckpt_path,
        chunk_len=16,
        device=device,
    )

    cached_dataset = CachedFeatureDataset(dataset_index, feature_cache)

    sanity_check_one_sample(cached_dataset, num_labels=num_labels, beta=beta, device=device)

    kf = KFold(n_splits=5, shuffle=True, random_state=43)
    all_idx = np.arange(len(cached_dataset))

    fold_rows = []
    compute_metrics = make_compute_metrics(num_labels)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_idx), start=1):
        print(f"\n[INFO] ===== {exp_name}: Fold {fold}/5 ===== Train={len(train_idx)} Val={len(val_idx)}")

        # A: confirm disjointness by indices + cache_path
        check_split_disjoint(train_idx, val_idx, cached_dataset, fold)

        train_ds = Subset(cached_dataset, train_idx)
        val_ds   = Subset(cached_dataset, val_idx)

        fold_out = os.path.join(exp_dir, f"fold{fold}")
        os.makedirs(fold_out, exist_ok=True)

        model = VideoClassifierWithDistanceLoss(feature_dim=768, num_classes=num_labels).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_bs1)
        val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_bs1)

        best_macro_f1 = -1.0
        best_path = None

        for epoch in range(epochs):
            # ---- train ----
            model.train()
            tr_loss = tr_cls = tr_dist = 0.0
            n_tr = 0

            for batch in train_dl:
                F_chunks = batch["F_chunks"].to(device)
                labels   = batch["labels"].to(device)

                opt.zero_grad(set_to_none=True)
                loss, l_cls, l_dist, _ = training_step(model, F_chunks, labels, beta=beta)

                # D: print dist contribution on first train batch
                if n_tr == 0:
                    cls, distv, weighted, ratio = dist_debug_stats(l_cls, l_dist, beta)
                    p = int(F_chunks.shape[0])
                    print(f"[CHECK D][train] Fold {fold} Epoch {epoch+1}: p={p} "
                          f"loss_cls={cls:.6f} loss_dist={distv:.6e} beta*dist={weighted:.6e} ratio={ratio:.6e}")
                    if p < 2:
                        print("[CHECK D][train][WARN] p < 2 so dist loss is not meaningful.")

                loss.backward()
                opt.step()

                tr_loss += float(loss.item())
                tr_cls  += float(l_cls.item())
                tr_dist += float(l_dist.item())
                n_tr += 1

            # ---- eval ----
            model.eval()
            print(f"[CHECK C] Fold {fold} Epoch {epoch+1}: model.training? {model.training} (should be False in eval)")

            all_preds = []
            all_labels = []
            ev_loss = ev_cls = ev_dist = 0.0
            n_ev = 0

            with torch.no_grad():
                # C: ensure gradients are off during evaluation
                assert torch.is_grad_enabled() == False, "[CHECK C] Gradients are enabled during eval — should be disabled!"

                for batch in val_dl:
                    F_chunks = batch["F_chunks"].to(device)
                    labels   = batch["labels"].to(device)

                    loss, l_cls, l_dist, logits_video = training_step(model, F_chunks, labels, beta=beta)
                    pred = int(torch.argmax(logits_video, dim=-1).item())

                    # D: print dist contribution on first eval batch
                    if n_ev == 0:
                        cls, distv, weighted, ratio = dist_debug_stats(l_cls, l_dist, beta)
                        p = int(F_chunks.shape[0])
                        print(f"[CHECK D][eval]  Fold {fold} Epoch {epoch+1}: p={p} "
                              f"loss_cls={cls:.6f} loss_dist={distv:.6e} beta*dist={weighted:.6e} ratio={ratio:.6e}")
                        if p < 2:
                            print("[CHECK D][eval][WARN] p < 2 so dist loss is not meaningful.")

                    all_preds.append(pred)
                    all_labels.append(int(labels.item()))

                    ev_loss += float(loss.item())
                    ev_cls  += float(l_cls.item())
                    ev_dist += float(l_dist.item())
                    n_ev += 1

            metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
            metrics["loss"] = ev_loss / max(1, n_ev)
            metrics["loss_cls"] = ev_cls / max(1, n_ev)
            metrics["loss_dist"] = ev_dist / max(1, n_ev)

            print(
                f"[INFO] Fold {fold} Epoch {epoch+1}/{epochs} "
                f"train_loss={tr_loss/max(1,n_tr):.4f} "
                f"eval_acc={metrics['accuracy']:.4f} eval_macro_f1={metrics['macro_f1']:.4f} "
                f"(eval_loss={metrics['loss']:.4f}, cls={metrics['loss_cls']:.4f}, dist={metrics['loss_dist']:.4f})"
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
                F_chunks = batch["F_chunks"].to(device)
                labels   = batch["labels"].to(device)
                logits_video, _, _, _ = model(F_chunks)
                pred = int(torch.argmax(logits_video, dim=-1).item())
                all_preds.append(pred)
                all_labels.append(int(labels.item()))

        labels_np = np.array(all_labels)
        preds_np = np.array(all_preds)

        eval_results = compute_metrics(preds_np, labels_np)
        print("[INFO] Fold eval:", eval_results)

        report_txt = classification_report(labels_np, preds_np, target_names=class_names, zero_division=0)
        with open(os.path.join(exp_dir, f"classification_report_fold{fold}.txt"), "w") as f:
            f.write(report_txt)

        cm = confusion_matrix(labels_np, preds_np, labels=list(range(num_labels)))
        np.save(os.path.join(exp_dir, f"confusion_fold{fold}.npy"), cm)

        print(report_txt)

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

    accs = [r.get("accuracy", np.nan) for r in fold_rows]
    f1s  = [r.get("macro_f1", np.nan) for r in fold_rows]

    mean_acc = float(np.nanmean(accs))
    std_acc  = float(np.nanstd(accs, ddof=1))
    mean_f1  = float(np.nanmean(f1s))
    std_f1   = float(np.nanstd(f1s, ddof=1))

    summary = (
        f"\n===== {exp_name} COMPLETE =====\n"
        f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n"
        f"Macro-F1:  {mean_f1:.4f} ± {std_f1:.4f}\n"
        f"Saved to:  {exp_dir}\n"
    )
    print(summary)
    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        f.write(summary)

# -----------------------------
# Main
# -----------------------------
def main():
    base = "/scratch/vsf6tk/suturing/keyframes_stride8"
    results_root = "/scratch/vsf6tk/suturing/results/videomae_distance_classifier"
    os.makedirs(results_root, exist_ok=True)

    videomae_ckpt_path = "/scratch/vsf6tk/suturing/results/videomae_stride8/4class_allvideos_stride8_chunk/fold1/checkpoint-85640"

    cache_dir = "/scratch/vsf6tk/suturing/results/videomae_feature_cache_allchunks"
    os.makedirs(cache_dir, exist_ok=True)

    # 1) 4-class
    dataset_4 = KeyframeFolderIndexDataset(
        [
            (os.path.join(base, "DDSIT_test"), 0),
            (os.path.join(base, "SIIT_test"), 1),
            (os.path.join(base, "OHSK_test"), 2),
            (os.path.join(base, "THSK_test"), 3),
        ]
    )
    run_experiment(
        exp_name="4class_DDSIT_SIIT_OHSK_THSK_distance_lambda2.0",
        dataset_index=dataset_4,
        videomae_ckpt_path=videomae_ckpt_path,
        num_labels=4,
        class_names=["DDSIT", "SIIT", "OHSK", "THSK"],
        results_root=results_root,
        cache_dir=cache_dir,
        seed=30,
        epochs=8,
        lr=1e-4,
        beta=0.0,
    )

    # 2) Suturing binary
    dataset_sut = KeyframeFolderIndexDataset(
        [
            (os.path.join(base, "DDSIT_test"), 0),
            (os.path.join(base, "SIIT_test"), 1),
        ]
    )
    run_experiment(
        exp_name="binary_suturing_DDSIT_vs_SIIT_distance_lambda2.0",
        dataset_index=dataset_sut,
        videomae_ckpt_path=videomae_ckpt_path,
        num_labels=2,
        class_names=["DDSIT", "SIIT"],
        results_root=results_root,
        cache_dir=cache_dir,
        seed=30,
        epochs=8,
        lr=1e-4,
        beta=0.0,
    )

    # 3) Knot-tying binary
    dataset_knot = KeyframeFolderIndexDataset(
        [
            (os.path.join(base, "OHSK_test"), 0),
            (os.path.join(base, "THSK_test"), 1),
        ]
    )
    run_experiment(
        exp_name="binary_knot_OHSK_vs_THSK_distance_lambda2.0",
        dataset_index=dataset_knot,
        videomae_ckpt_path=videomae_ckpt_path,
        num_labels=2,
        class_names=["OHSK", "THSK"],
        results_root=results_root,
        cache_dir=cache_dir,
        seed=30,
        epochs=8,
        lr=1e-4,
        beta=0.0,
    )

if __name__ == "__main__":
    main()
