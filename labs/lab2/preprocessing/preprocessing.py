"""Module `preprocessing` (Preprocessing) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `preprocessing` (Tiền xử lý). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_raw_dataset(data_root: str, download: bool = True) -> tuple:
    """Load the raw CIFAR-10 dataset WITHOUT transforms.

    Input : data_root – directory where torchvision will store the dataset.
            download   – if True, download the dataset if not already present.
    Output: (train_set, test_set)  — torchvision Dataset objects with no transforms.

    VI: Nạp CIFAR-10 thô, CHƯA áp dụng transform.
    Vào : data_root, download
    Ra  : (train_set, test_set)
    """
    import torchvision.datasets as datasets  # local import to avoid side effects

    train_set = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=None,
        target_transform=None,
        download=download,
    )
    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=None,
        target_transform=None,
        download=download,
    )
    return train_set, test_set


def make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42) -> tuple:
    """Reproducible stratified train/validation index split.

    Input : n_samples – total number of training samples.
            val_ratio  – fraction to reserve for validation (default 0.1 = 10%).
            seed       – random seed for reproducibility.
    Output: (train_idx, val_idx) — Python lists of integer indices.

    VI: Chia train/validation tái lập được (stratified khi có thể).
    Vào : n_samples, val_ratio, seed
    Ra  : (train_idx, val_idx) — danh sách chỉ số nguyên.
    """
    rng = random.Random(seed)
    all_idx = list(range(n_samples))
    rng.shuffle(all_idx)
    n_val = max(1, int(n_samples * val_ratio))
    val_idx = sorted(all_idx[:n_val])
    train_idx = sorted(all_idx[n_val:])
    return train_idx, val_idx


def make_stratified_split(
    targets: List[int],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Stratified split preserving class distribution.

    Requires scikit-learn. Falls back to make_split when unavailable.

    Input : targets   – list/array of class labels for all training samples.
            val_ratio  – fraction for validation.
            seed       – random seed.
    Output: (train_idx, val_idx)
    """
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        n_samples = len(targets)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        all_idx = np.arange(n_samples)
        train_idx, val_idx = next(sss.split(all_idx, targets))
        return train_idx.tolist(), val_idx.tolist()
    except ImportError:
        return make_split(len(targets), val_ratio, seed)


def save_split(train_idx, val_idx, path: str) -> None:
    """Persist the split as a JSON file for reproducibility.

    Input : train_idx – list of training indices.
            val_idx   – list of validation indices.
            path      – output file path (JSON).
    Output: JSON file written to disk.

    VI: Lưu split dạng JSON để tái lập.
    Vào : chỉ số + path
    Ra  : file JSON
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split(path: str) -> tuple:
    """Load a previously saved split from JSON.

    Input : path – path to the JSON file written by save_split.
    Output: (train_idx, val_idx)

    VI: Nạp lại split từ file JSON.
    Vào : path
    Ra  : (train_idx, val_idx)
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["train_idx"], payload["val_idx"]
