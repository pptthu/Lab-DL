"""Module `eda` (EDA) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `eda` (EDA (khám phá dữ liệu)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


# CIFAR-10 class names (official)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def run_eda(data_root: str, out_dir: str) -> dict:
    """Explore the CIFAR-10 dataset and save EDA figures.

    Loads CIFAR-10 WITHOUT any transform (raw PIL images) to compute
    correct pixel statistics.  Saves figures to out_dir.

    Input : data_root – dataset folder (torchvision downloads here).
            out_dir   – directory for output PNG files and summary JSON.
    Output: dict with keys:
              n_train       – number of training samples (before split).
              n_test        – number of test samples.
              class_counts  – {class_name: count} for training set.
              image_size    – (H, W) tuple.
              channel_mean  – per-channel mean of raw [0,1] pixels.
              channel_std   – per-channel std of raw [0,1] pixels.
            Side effects: PNG files and eda_summary.json in out_dir.

    VI: Khám phá CIFAR-10 và lưu biểu đồ.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import torchvision.datasets as datasets

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw datasets
    # ------------------------------------------------------------------
    try:
        train_set = datasets.CIFAR10(root=data_root, train=True,
                                     transform=None, download=True)
        test_set  = datasets.CIFAR10(root=data_root, train=False,
                                     transform=None, download=True)
    except Exception as e:
        print(f"[EDA] Warning: could not load CIFAR-10: {e}")
        return _eda_unavailable()

    n_train = len(train_set)
    n_test  = len(test_set)
    classes = train_set.classes  # official ordering

    # ------------------------------------------------------------------
    # 2. Class distribution
    # ------------------------------------------------------------------
    targets = np.array(train_set.targets)
    class_counts: Dict[str, int] = {
        cls: int((targets == i).sum()) for i, cls in enumerate(classes)
    }
    _plot_class_distribution(class_counts, out_path / "class_distribution.png")

    # ------------------------------------------------------------------
    # 3. Image grid (one per class)
    # ------------------------------------------------------------------
    _plot_image_grid(train_set, classes, out_path / "sample_grid.png")

    # ------------------------------------------------------------------
    # 4. Pixel statistics (sample 5000 images for speed)
    # ------------------------------------------------------------------
    sample_n = min(5000, n_train)
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(n_train, sample_n, replace=False)

    pixels: list = []
    for idx in sample_idx:
        img_pil, _ = train_set[idx]
        arr = np.array(img_pil, dtype=np.float32) / 255.0  # [H,W,C]
        pixels.append(arr)

    pixel_arr = np.stack(pixels, axis=0)  # [N, H, W, C]
    channel_mean = pixel_arr.mean(axis=(0, 1, 2)).tolist()
    channel_std  = pixel_arr.std(axis=(0, 1, 2)).tolist()
    image_size   = (pixel_arr.shape[1], pixel_arr.shape[2])  # (H, W)

    # ------------------------------------------------------------------
    # 5. Pixel intensity histogram
    # ------------------------------------------------------------------
    _plot_pixel_histogram(pixel_arr, channel_mean, channel_std,
                          out_path / "pixel_histogram.png")

    # ------------------------------------------------------------------
    # 6. Summary JSON
    # ------------------------------------------------------------------
    summary = {
        "n_train": n_train,
        "n_test":  n_test,
        "class_counts": class_counts,
        "image_size": list(image_size),
        "channel_mean": channel_mean,
        "channel_std":  channel_std,
        "classes": classes,
    }
    with open(out_path / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[EDA] n_train={n_train}, n_test={n_test}, "
          f"image_size={image_size}, "
          f"channel_mean={[f'{v:.4f}' for v in channel_mean]}, "
          f"channel_std={[f'{v:.4f}' for v in channel_std]}")
    print(f"[EDA] Figures saved to {out_path}")

    return summary


# ---------------------------------------------------------------------------
# Private plot helpers
# ---------------------------------------------------------------------------

def _plot_class_distribution(class_counts: Dict[str, int], save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(class_counts.keys())
    counts  = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(classes, counts, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of samples", fontsize=12)
    ax.set_title("CIFAR-10 Training Set – Class Distribution", fontsize=14)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(cnt), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_image_grid(dataset, classes: list, save_path: Path, n_per_class: int = 5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = np.array(dataset.targets)
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 1.5, n_classes * 1.5))

    for c in range(n_classes):
        class_idx = np.where(targets == c)[0][:n_per_class]
        for j, idx in enumerate(class_idx):
            img, _ = dataset[idx]
            ax = axes[c, j]
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(classes[c], rotation=0, labelpad=40,
                              va="center", fontsize=9)

    fig.suptitle("CIFAR-10 — Sample Images (one row per class)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_pixel_histogram(pixel_arr: np.ndarray,
                          means: list, stds: list,
                          save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channel_names = ["Red", "Green", "Blue"]
    colors = ["red", "green", "blue"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for c, (name, color) in enumerate(zip(channel_names, colors)):
        flat = pixel_arr[..., c].ravel()
        ax.hist(flat, bins=50, alpha=0.5, color=color, label=name,
                density=True)
    ax.set_xlabel("Pixel value (normalised 0–1)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Pixel intensity distribution  "
        f"(mean R={means[0]:.3f} G={means[1]:.3f} B={means[2]:.3f})",
        fontsize=12,
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def _eda_unavailable() -> dict:
    return {
        "n_train": None,
        "n_test": None,
        "class_counts": {},
        "image_size": None,
        "channel_mean": None,
        "channel_std": None,
    }
