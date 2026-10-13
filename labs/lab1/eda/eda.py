"""Module `eda` (EDA) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `eda` (EDA (khám phá dữ liệu)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

IMAGE_SIDE = 28
N_PIXELS = IMAGE_SIDE * IMAGE_SIDE
LABEL_COL = "label"
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def run_eda(train_csv: str, out_dir: str) -> dict:
    """Explore the training CSV (read-only, never modify data) and save figures.

    Input : train_csv: path to the Kaggle training CSV; out_dir: folder for figures
    Output: dict with n_rows, class_counts, pixel_min, pixel_max, n_missing, n_duplicates; PNG files (class bar chart, sample image grid) in out_dir

    VI: Khám phá file CSV train (chỉ đọc, không sửa dữ liệu) và lưu biểu đồ.
    Vào : train_csv: đường dẫn CSV train của Kaggle; out_dir: thư mục lưu hình
    Ra  : dict gồm n_rows, class_counts, pixel_min, pixel_max, n_missing, n_duplicates; file PNG (biểu đồ số lượng mỗi lớp, lưới ảnh mẫu) trong out_dir
    """
    df = pd.read_csv(train_csv)
    _check_columns(df)

    pixels = df.drop(columns=LABEL_COL)
    class_counts = {
        int(label): int(count)
        for label, count in df[LABEL_COL].value_counts().sort_index().items()
    }
    stats = {
        "n_rows": int(len(df)),
        "class_counts": class_counts,
        # min/max skip NaN, so missing cells do not break the statistics.
        # VI: min/max bỏ qua NaN nên ô trống không làm hỏng thống kê.
        "pixel_min": int(pixels.min().min()),
        "pixel_max": int(pixels.max().max()),
        "n_missing": int(df.isna().sum().sum()),
        # A duplicate = a row identical to an earlier row (label + all pixels).
        # VI: Dòng trùng = giống hệt một dòng phía trước (cả nhãn và pixel).
        "n_duplicates": int(df.duplicated().sum()),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _plot_class_counts(class_counts, out / "class_counts.png")
    _plot_sample_grid(df, out / "sample_grid.png")
    return stats


def _check_columns(df: pd.DataFrame) -> None:
    """Fail early if the CSV is not label + 784 pixels. / Báo lỗi sớm nếu CSV sai định dạng."""
    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV must have a '{LABEL_COL}' column")
    n_pixel_cols = df.shape[1] - 1
    if n_pixel_cols != N_PIXELS:
        raise ValueError(f"expected {N_PIXELS} pixel columns, got {n_pixel_cols}")


def _plot_class_counts(class_counts: dict, out_path: Path) -> None:
    """Bar chart: number of samples per class. / Biểu đồ cột số mẫu mỗi lớp."""
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    labels = list(class_counts)
    names = [f"{i}\n{CLASS_NAMES[i]}" for i in labels]
    ax.bar(names, list(class_counts.values()), color="#4C72B0")
    ax.set_title("Samples per class (train)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)


def _plot_sample_grid(df: pd.DataFrame, out_path: Path) -> None:
    """Grid 2x5: the first image of each class. / Lưới 2x5: ảnh đầu tiên của mỗi lớp."""
    fig = Figure(figsize=(10, 5))
    axes = fig.subplots(2, 5)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        rows = df[df[LABEL_COL] == i]
        if rows.empty:
            continue
        img = rows.iloc[0].drop(LABEL_COL).to_numpy(dtype=float).reshape(IMAGE_SIDE, IMAGE_SIDE)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{i}: {CLASS_NAMES[i]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
