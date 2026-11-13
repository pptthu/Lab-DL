"""Module `visualize` (Visualization) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `visualize` (Trực quan hóa). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn
from matplotlib.figure import Figure


def plot_loss_curves(history: dict, out_path: str) -> None:
    """Plot training vs validation loss.

    Input : history from train_model; out_path (.png)
    Output: PNG file

    VI: Vẽ loss train và validation.
    Vào : history từ train_model; out_path (.png)
    Ra  : file PNG
    """
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()

    # One point per epoch: train loss (blue) and validation loss (orange).
    # VI: Mỗi epoch một điểm: train loss (xanh dương) và validation loss (cam).
    ax.plot(epochs, train_loss, label="Train Loss", marker="o", color="#1f77b4")
    if val_loss:
        ax.plot(epochs, val_loss, label="Validation Loss", marker="s", color="#ff7f0e")

    ax.set_title("Training and Validation Loss", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=100)


def show_predictions(
    model: nn.Module,
    loader,
    class_names: Sequence[str],
    device: str | torch.device,
    out_path: str,
    n: int = 16,
) -> None:
    """Grid of images with predicted vs actual labels (wrong ones highlighted).

    Input : model, loader, class_names (10 names), device, out_path, n
    Output: PNG file

    VI: Lưới ảnh có nhãn dự đoán và nhãn thật (ảnh sai được tô nổi bật).
    Vào : model, loader, class_names (10 tên), device, out_path, n
    Ra  : file PNG
    """
    if len(class_names) != 10:
        raise ValueError(f"Expected 10 class_names, got {len(class_names)}")

    model.to(device)
    model.eval()

    collected_images = []
    collected_preds = []
    collected_trues = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)

            # Take the first n images of the loader (validation is not shuffled, so always the same images).
            # VI: Lấy n ảnh đầu của loader (validation không xáo nên lần nào cũng là cùng các ảnh).
            for img, t, p in zip(x_batch, y_batch, preds):
                collected_images.append(img.cpu())
                collected_trues.append(int(t.item()))
                collected_preds.append(int(p.item()))
                if len(collected_images) >= n:
                    break
            if len(collected_images) >= n:
                break

    actual_n = len(collected_images)
    if actual_n == 0:
        return

    ncols = int(math.ceil(math.sqrt(actual_n)))
    nrows = int(math.ceil(actual_n / ncols))

    fig = Figure(figsize=(3 * ncols, 3.2 * nrows))
    axes = fig.subplots(nrows, ncols)
    axes_flat = [axes] if actual_n == 1 else (axes.flat if hasattr(axes, "flat") else [axes])

    for i, ax in enumerate(axes_flat):
        if i < actual_n:
            img = collected_images[i].squeeze().float().numpy()
            # Rescale the normalized image back to 0-1 so it can be shown.
            # VI: Đưa ảnh đã chuẩn hóa về khoảng 0-1 để hiển thị.
            img_min = float(img.min())
            img_max = float(img.max())
            if img_max > img_min:
                img_rescaled = (img - img_min) / (img_max - img_min)
            else:
                img_rescaled = np.zeros_like(img)

            t_idx = collected_trues[i]
            p_idx = collected_preds[i]
            pred_name = class_names[p_idx]
            true_name = class_names[t_idx]

            ax.imshow(img_rescaled, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            is_correct = (p_idx == t_idx)
            # Green title = correct prediction, red = wrong.
            # VI: Tiêu đề xanh lá = đoán đúng, đỏ = đoán sai.
            color = "#2ca02c" if is_correct else "#d62728"
            ax.set_title(f"pred: {pred_name}\ntrue: {true_name}", color=color, fontsize=9)
        else:
            ax.axis("off")

    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=100)


def plot_confusion_matrix(cm, class_names: Sequence[str], out_path: str) -> None:
    """Heatmap of the confusion matrix.

    Input : cm, class_names, out_path
    Output: PNG file

    VI: Heatmap của ma trận nhầm lẫn.
    Vào : cm, class_names, out_path
    Ra  : file PNG
    """
    cm_arr = np.asarray(cm)
    if len(class_names) != cm_arr.shape[0]:
        raise ValueError(
            f"Expected {cm_arr.shape[0]} class_names matching cm rows, got {len(class_names)}"
        )

    fig = Figure(figsize=(8.5, 7.5))
    ax = fig.subplots()

    # Darker cell = more images; the diagonal holds the correct predictions.
    # VI: Ô càng đậm càng nhiều ảnh; đường chéo là các dự đoán đúng.
    cax = ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    num_classes = len(class_names)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12)

    # White numbers on dark cells, black on light cells, so every count is readable.
    # VI: Số màu trắng trên ô đậm, màu đen trên ô nhạt, để đọc được mọi số.
    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=100)
