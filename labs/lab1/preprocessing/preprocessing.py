"""Module `preprocessing` (Preprocessing) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `preprocessing` (Tiền xử lý). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
# Input / Output of this module
#   Input : labs/lab1/raw_data/fashion-mnist_train.csv and fashion-mnist_test.csv (from download_official.py; not committed)
#   Output: 1) load_csv -> images uint8 [N,28,28], labels int64 [N] (in memory, passed to features)
#           2) make_split -> train_idx (54,000), val_idx (6,000) (in memory)
#           3) save_split -> labs/lab1/artifacts/splits/train_val_split.json
#              ({"train_idx": [...], "val_idx": [...]}; COMMITTED, read-only; main.py and the notebook only load it)
# VI: Input / Output của module
#   Vào : labs/lab1/raw_data/fashion-mnist_train.csv và fashion-mnist_test.csv (từ download_official.py; không commit)
#   Ra  : 1) load_csv -> ảnh uint8 [N,28,28], nhãn int64 [N] (trong bộ nhớ, đưa sang features)
#         2) make_split -> train_idx (54.000), val_idx (6.000) (trong bộ nhớ)
#         3) save_split -> labs/lab1/artifacts/splits/train_val_split.json
#            ({"train_idx": [...], "val_idx": [...]}; CÓ commit, chỉ đọc; main.py và notebook chỉ load)

# Type hints are not evaluated at import time (works on older Python versions too).
# VI: Không tính type hint lúc import (chạy được cả trên Python cũ).
from __future__ import annotations

# json: the split file; Path: output paths; numpy: arrays; pandas: CSV reading.
# VI: json: file split; Path: đường dẫn; numpy: mảng; pandas: đọc CSV.
import json
from pathlib import Path

import numpy as np
import pandas as pd

# One place for sizes and file names; download_official.py and main.py import them,
# so the names never drift between files.
# VI: Một chỗ duy nhất cho kích thước và tên file; download_official.py và main.py import lại,
# nên tên không bị lệch giữa các file.
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
    # No label column -> wrong file; fail early.
    # VI: Không có cột nhãn -> sai file; báo lỗi ngay.
    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV must have a '{LABEL_COL}' column")
    # Pixel matrix [N, 784] (everything except the label).
    # VI: Ma trận pixel [N, 784] (mọi cột trừ nhãn).
    pixels = df.drop(columns=LABEL_COL).to_numpy()
    if pixels.shape[1] != N_PIXELS:
        raise ValueError(f"expected {N_PIXELS} pixel columns, got {pixels.shape[1]}")
    # Check the range BEFORE casting: uint8 would silently wrap 256 -> 0.
    # VI: Kiểm tra khoảng giá trị TRƯỚC khi ép kiểu: uint8 sẽ âm thầm quay vòng 256 -> 0.
    if pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("pixel values must be in 0-255")
    # uint8 = 1 byte per pixel (60,000 images ~ 47 MB; float32 would be ~ 188 MB).
    # reshape(-1, 28, 28): -1 lets numpy compute N.
    # VI: uint8 = 1 byte mỗi pixel (60.000 ảnh ~ 47 MB; float32 sẽ ~ 188 MB).
    # reshape(-1, 28, 28): -1 để numpy tự tính N.
    images = pixels.astype(np.uint8).reshape(-1, IMAGE_SIDE, IMAGE_SIDE)
    # int64 (= torch long) is the label type CrossEntropyLoss needs.
    # VI: int64 (= long của torch) là kiểu nhãn CrossEntropyLoss cần.
    labels = df[LABEL_COL].to_numpy(dtype=np.int64)
    return images, labels


def make_split(labels, val_ratio: float = 0.1, seed: int = 42) -> tuple:
    """Split training indices into train/validation (stratified, reproducible).

    Input : labels: int64 array [N]; val_ratio; seed
    Output: (train_idx, val_idx): disjoint index arrays covering all N samples

    VI: Chia chỉ số train thành train/validation (phân tầng, tái lập được).
    Vào : labels: mảng int64 [N]; val_ratio; seed
    Ra  : (train_idx, val_idx): hai mảng chỉ số không giao nhau, gộp lại đủ N mẫu
    """
    # 0 or 1 would give an empty split.
    # VI: 0 hoặc 1 sẽ cho ra một tập rỗng.
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    # Also accept a list or a tensor.
    # VI: Nhận cả list hoặc tensor.
    labels = np.asarray(labels)
    # Private seeded generator: the same split every run, and the global RNG is not touched.
    # VI: Bộ sinh số riêng có seed: lần nào cũng ra cùng split, không động tới RNG toàn cục.
    rng = np.random.default_rng(seed)
    # Validation indices collected class by class.
    # VI: Gom chỉ số validation theo từng lớp.
    val_parts = []
    # Stratified: take the same fraction from every class, so validation keeps
    # the class balance of the training set.
    # VI: Phân tầng: lấy cùng một tỉ lệ ở mỗi lớp, nên validation giữ đúng tỉ lệ lớp của train.
    for cls in np.unique(labels):
        # Positions of the images of class `cls`.
        # VI: Vị trí các ảnh thuộc lớp `cls`.
        cls_idx = np.flatnonzero(labels == cls)
        # 6,000 x 0.1 = 600 validation images per class.
        # VI: 6.000 x 0,1 = 600 ảnh validation mỗi lớp.
        n_val = int(round(len(cls_idx) * val_ratio))
        # Shuffle the class indices and keep the first n_val.
        # VI: Xáo chỉ số của lớp rồi lấy n_val cái đầu.
        val_parts.append(rng.permutation(cls_idx)[:n_val])
    # Join the 10 parts and sort (a stable, readable file).
    # VI: Gộp 10 phần và sắp tăng dần (file ổn định, dễ đọc).
    val_idx = np.sort(np.concatenate(val_parts)).astype(np.int64)
    # Train = every index not in validation -> disjoint and complete (54,000 + 6,000 = 60,000).
    # VI: Train = mọi chỉ số không thuộc validation -> không giao nhau và đủ (54.000 + 6.000 = 60.000).
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
    # Create the parent folder if missing, then write UTF-8 JSON.
    # VI: Tạo thư mục cha nếu chưa có, rồi ghi JSON UTF-8.
    # Saved file: `path`; the shared split is labs/lab1/artifacts/splits/train_val_split.json (committed).
    # VI: File được lưu: `path`; split dùng chung là labs/lab1/artifacts/splits/train_val_split.json (có commit).
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
    # A missing file raises FileNotFoundError here.
    # VI: File không tồn tại thì báo FileNotFoundError ở đây.
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    # Back to int64 arrays, the same type make_split returns.
    # VI: Đổi lại thành mảng int64, cùng kiểu với make_split trả về.
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)
    return train_idx, val_idx


if __name__ == "__main__":
    # Create the shared split file: python -m labs.lab1.preprocessing.preprocessing
    # VI: Tạo file split dùng chung cho cả nhóm.
    # Imported here so the module itself does not depend on the repo layout.
    # VI: Import ở đây để bản thân module không phụ thuộc cấu trúc repo.
    from shared.paths import artifact_dir, raw_data_dir

    images, labels = load_csv(str(raw_data_dir("lab1") / TRAIN_CSV_NAME))
    # Print shape, dtype and range so a wrong reshape or dtype is seen at once.
    # VI: In shape, dtype và khoảng giá trị để thấy ngay nếu reshape hoặc dtype sai.
    print("images", images.shape, images.dtype, images.min(), images.max())
    print("labels", labels.shape, labels.dtype, labels.min(), labels.max())
    # Defaults: val_ratio=0.1, seed=42.
    # VI: Mặc định: val_ratio=0.1, seed=42.
    train_idx, val_idx = make_split(labels)
    split_path = artifact_dir("lab1", "splits") / SPLIT_JSON_NAME
    save_split(train_idx, val_idx, str(split_path))
    print("train", len(train_idx), "val", len(val_idx), "->", split_path)
