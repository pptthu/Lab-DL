"""Module `preprocessing` (Preprocessing) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `preprocessing` (Tiền xử lý). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_SIDE = 28
N_PIXELS = IMAGE_SIDE * IMAGE_SIDE
LABEL_COL = "label"
TRAIN_CSV_NAME = "fashion-mnist_train.csv"
TEST_CSV_NAME = "fashion-mnist_test.csv"
SPLIT_JSON_NAME = "train_val_split.json"


def load_csv(csv_path: str) -> tuple:
    """Read a Fashion-MNIST CSV (label + 784 pixel columns) into arrays.

    Input : csv_path: path to a CSV file
    Output: (images: uint8 array [N,28,28], labels: int64 array [N])

    VI: Đọc CSV Fashion-MNIST (cột label + 784 cột pixel) thành mảng.
    Vào : csv_path: đường dẫn file CSV
    Ra  : (images: mảng uint8 [N,28,28], labels: mảng int64 [N])
    """
    df = pd.read_csv(csv_path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV must have a '{LABEL_COL}' column")
    pixels = df.drop(columns=LABEL_COL).to_numpy()
    if pixels.shape[1] != N_PIXELS:
        raise ValueError(f"expected {N_PIXELS} pixel columns, got {pixels.shape[1]}")
    # Check the range BEFORE casting: uint8 would silently wrap 256 -> 0.
    # VI: Kiểm tra khoảng giá trị TRƯỚC khi ép kiểu: uint8 sẽ âm thầm quay vòng 256 -> 0.
    if pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("pixel values must be in 0-255")
    images = pixels.astype(np.uint8).reshape(-1, IMAGE_SIDE, IMAGE_SIDE)
    labels = df[LABEL_COL].to_numpy(dtype=np.int64)
    return images, labels


def make_split(labels, val_ratio: float = 0.1, seed: int = 42, val_fraction: float | None = None) -> tuple:
    if val_fraction is not None:
        val_ratio = val_fraction
    """Split training indices into train/validation (stratified, reproducible).

    Input : labels: int64 array [N]; val_ratio; seed
    Output: (train_idx, val_idx): disjoint index arrays covering all N samples

    VI: Chia chỉ số train thành train/validation (phân tầng, tái lập được).
    Vào : labels: mảng int64 [N]; val_ratio; seed
    Ra  : (train_idx, val_idx): hai mảng chỉ số không giao nhau, gộp lại đủ N mẫu
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    val_parts = []
    # Stratified: take the same fraction from every class, so validation keeps
    # the class balance of the training set.
    # VI: Phân tầng: lấy cùng một tỉ lệ ở mỗi lớp, nên validation giữ đúng tỉ lệ lớp của train.
    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        n_val = int(round(len(cls_idx) * val_ratio))
        val_parts.append(rng.permutation(cls_idx)[:n_val])
    val_idx = np.sort(np.concatenate(val_parts)).astype(np.int64)
    train_idx = np.setdiff1d(np.arange(len(labels), dtype=np.int64), val_idx)
    return train_idx, val_idx


def save_split(train_idx, val_idx, path: str) -> None:
    """Save the split as JSON so everyone uses the same split.

    Input : train_idx, val_idx, path (.json)
    Output: JSON file at path

    VI: Lưu split ra JSON để cả nhóm dùng chung một cách chia.
    Vào : train_idx, val_idx, path (.json)
    Ra  : file JSON tại path
    """
    # JSON cannot store numpy integers, so convert to plain Python ints.
    # VI: JSON không lưu được số nguyên numpy nên đổi sang int thường của Python.
    split = {
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(split, f)


def load_split(path: str) -> tuple:
    """Load a saved split.

    Input : path (.json)
    Output: (train_idx, val_idx)

    VI: Đọc lại split đã lưu.
    Vào : path (.json)
    Ra  : (train_idx, val_idx)
    """
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)
    return train_idx, val_idx


if __name__ == "__main__":
    # Create the shared split file: python -m labs.lab1.preprocessing.preprocessing
    # VI: Tạo file split dùng chung cho cả nhóm.
    from shared.paths import artifact_dir, raw_data_dir

    images, labels = load_csv(str(raw_data_dir("lab1") / TRAIN_CSV_NAME))
    print("images", images.shape, images.dtype, images.min(), images.max())
    print("labels", labels.shape, labels.dtype, labels.min(), labels.max())
    train_idx, val_idx = make_split(labels)
    split_path = artifact_dir("lab1", "splits") / SPLIT_JSON_NAME
    save_split(train_idx, val_idx, str(split_path))
    print("train", len(train_idx), "val", len(val_idx), "->", split_path)
