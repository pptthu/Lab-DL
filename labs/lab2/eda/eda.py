"""Module `eda` (EDA) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `eda` (EDA (khám phá dữ liệu)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to safely generate figures
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.datasets as datasets


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def run_eda(data_root: str, out_dir: str) -> dict:
    """Explore the image dataset (read-only) and save figures.

    Input : data_root: dataset folder (downloaded by torchvision); out_dir
    Output: dict: n_train, n_test, class_counts, image_size, channel_mean, channel_std; PNG files in out_dir

    VI: Khám phá bộ dữ liệu ảnh (chỉ đọc) và lưu biểu đồ.
    Vào : data_root: thư mục dữ liệu (torchvision tải về); out_dir
    Ra  : dict: n_train, n_test, class_counts, image_size, channel_mean, channel_std; file PNG trong out_dir
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load raw datasets without transformation
    train_raw = datasets.CIFAR10(root=data_root, train=True, download=True)
    test_raw = datasets.CIFAR10(root=data_root, train=False, download=True)

    n_train = len(train_raw)
    n_test = len(test_raw)

    # train_raw.data shape: [50000, 32, 32, 3] (uint8), train_raw.targets: list of 50000 ints
    targets = np.array(train_raw.targets)
    class_counts = {CIFAR10_CLASSES[i]: int((targets == i).sum()) for i in range(10)}

    sample_img = train_raw.data[0]
    image_size = tuple(sample_img.shape)  # (32, 32, 3)

    # 2. Compute channel mean and std on train data normalized to [0, 1]
    # Reshape to [-1, 3] to calculate statistics per channel
    data_float = train_raw.data.astype(np.float32) / 255.0
    channel_mean = data_float.mean(axis=(0, 1, 2)).tolist()
    channel_std = data_float.std(axis=(0, 1, 2)).tolist()

    # 3. Figure 1: Sample grid (10 classes x 3 examples)
    fig, axes = plt.subplots(10, 3, figsize=(6, 15))
    for class_idx in range(10):
        idx_matches = np.where(targets == class_idx)[0][:3]
        for col_idx, img_idx in enumerate(idx_matches):
            ax = axes[class_idx, col_idx]
            ax.imshow(train_raw.data[img_idx])
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(CIFAR10_CLASSES[class_idx], loc="left", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path / "cifar10_sample_grid.png", dpi=150)
    plt.close(fig)

    # 4. Figure 2: Class distribution bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    ax.bar(classes, counts, color="#3470a3")
    ax.set_title("CIFAR-10 Class Distribution (Train Set)")
    ax.set_xlabel("Class")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path / "cifar10_class_distribution.png", dpi=150)
    plt.close(fig)

    return {
        "n_train": n_train,
        "n_test": n_test,
        "class_counts": class_counts,
        "image_size": image_size,
        "channel_mean": [round(float(m), 4) for m in channel_mean],
        "channel_std": [round(float(s), 4) for s in channel_std],
    }