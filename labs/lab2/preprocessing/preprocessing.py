"""Module `preprocessing` (Preprocessing) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `preprocessing` (Tiền xử lý). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torchvision.datasets as datasets


def load_raw_dataset(data_root: str, download: bool = True) -> tuple:
    """Load the raw dataset WITHOUT transforms.

    Input : data_root, download
    Output: (train_set, test_set)

    VI: Nạp dataset thô, CHƯA áp dụng transform.
    Vào : data_root, download
    Ra  : (train_set, test_set)
    """
    train_set = datasets.CIFAR10(root=data_root, train=True, download=download, transform=None)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=download, transform=None)
    return train_set, test_set


def make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42) -> tuple:
    """Reproducible train/validation split.

    Input : n_samples, val_ratio, seed
    Output: (train_idx, val_idx)

    VI: Chia train/validation tái lập được.
    Vào : n_samples, val_ratio, seed
    Ra  : (train_idx, val_idx)
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    val_size = int(round(n_samples * val_ratio))

    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()
    return train_idx, val_idx


def save_split(train_idx: list[int], val_idx: list[int], path: str) -> None:
    """Persist the split as JSON.

    Input : indices + path
    Output: JSON file

    VI: Lưu split dạng JSON.
    Vào : chỉ số + path
    Ra  : file JSON
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split(path: str) -> tuple[list[int], list[int]]:
    """Reload the split from JSON.

    Input : path
    Output: (train_idx, val_idx)

    VI: Nạp lại split dạng JSON.
    Vào : path
    Ra  : (train_idx, val_idx)
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["train_idx"], data["val_idx"]