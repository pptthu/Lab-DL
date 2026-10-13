"""Module `evaluate` (Evaluation) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `evaluate` (Đánh giá). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Compute average cross-entropy loss and top-1 accuracy without gradients.

    Input : model  – nn.Module in any mode (switched to eval internally).
            loader – DataLoader for the evaluation set.
            device – torch.device.
    Output: (avg_loss, accuracy)  where accuracy ∈ [0, 1].

    VI: Tính loss trung bình và độ chính xác top-1, không tính gradient.
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> np.ndarray:
    """Compute the confusion matrix over an evaluation set.

    Rows = true labels, columns = predicted labels.

    Input : model       – nn.Module.
            loader      – DataLoader.
            device      – torch.device.
            num_classes – number of classes (default 10 for CIFAR-10).
    Output: np.ndarray of shape [num_classes, num_classes] (integer counts).

    VI: Tính ma trận nhầm lẫn.
    """
    model.to(device)
    model.eval()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[int(true), int(pred)] += 1

    return cm


def top_confusions(cm: np.ndarray, class_names: List[str], k: int = 5) -> list:
    """Return the k most frequent off-diagonal confusion pairs.

    Input : cm          – confusion matrix [C, C] as returned by confusion_matrix().
            class_names – list of class name strings (length C).
            k           – how many pairs to return.
    Output: list of (true_class_name, predicted_class_name, count) tuples,
            sorted by count descending.

    VI: Các cặp nhầm lẫn nhiều nhất (không tính đường chéo).
    """
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    out_dir: Optional[str] = None,
    run_label: str = "model",
) -> Dict:
    """Extended evaluation: loss, accuracy, per-class metrics, confusion matrix.

    Also measures inference throughput and optionally saves artefacts.

    Input : model       – nn.Module.
            loader      – DataLoader (test set).
            device      – torch.device.
            class_names – list of class name strings.
            out_dir     – if given, save metrics.json and confusion_matrix.png here.
            run_label   – short label for filenames.
    Output: dict with keys: avg_loss, accuracy, per_class_acc, confusion_matrix,
            top_confusions, precision_macro, recall_macro, f1_macro,
            n_samples, inference_seconds.
    """
    from sklearn.metrics import precision_recall_fscore_support

    model.to(device)
    model.eval()

    criterion  = nn.CrossEntropyLoss()
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    t0 = time.time()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, labels).item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    inference_seconds = time.time() - t0
    n_samples = len(all_labels)
    avg_loss  = total_loss / max(len(loader), 1)
    accuracy  = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n_samples, 1)

    preds_arr  = np.array(all_preds,  dtype=np.int64)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # Per-class accuracy
    n_classes = len(class_names)
    per_class_acc = {}
    for c in range(n_classes):
        mask = labels_arr == c
        per_class_acc[class_names[c]] = (
            float((preds_arr[mask] == c).sum()) / max(int(mask.sum()), 1)
        )

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels_arr, preds_arr):
        cm[t, p] += 1

    # Macro metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_arr, preds_arr, average="macro", zero_division=0
    )

    top_conf = top_confusions(cm, class_names, k=5)

    result = {
        "avg_loss":          float(avg_loss),
        "accuracy":          float(accuracy),
        "precision_macro":   float(precision),
        "recall_macro":      float(recall),
        "f1_macro":          float(f1),
        "per_class_acc":     per_class_acc,
        "confusion_matrix":  cm.tolist(),
        "top_confusions":    [list(t) for t in top_conf],
        "n_samples":         n_samples,
        "inference_seconds": float(inference_seconds),
        "throughput_img_per_sec": float(n_samples / max(inference_seconds, 1e-6)),
    }

    if out_dir is not None:
        _save_evaluation_artefacts(result, cm, class_names, out_dir, run_label)

    return result


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a heatmap confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save_evaluation_artefacts(result, cm, class_names, out_dir, run_label):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON (exclude raw confusion matrix from JSON for readability;
    # it is already included in full).
    metrics_path = out_path / f"{run_label}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Save confusion matrix figure.
    cm_path = out_path / f"{run_label}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, str(cm_path),
                          title=f"Confusion Matrix — {run_label}")
    print(f"[Evaluate] Saved metrics → {metrics_path}")
    print(f"[Evaluate] Saved confusion matrix → {cm_path}")
