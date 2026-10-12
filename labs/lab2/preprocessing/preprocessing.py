"""Module `preprocessing` (Preprocessing) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `preprocessing` (Tiền xử lý). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def load_raw_dataset(data_root: str, download: bool = True) -> tuple:
    """Load the raw dataset WITHOUT transforms.

    Input : data_root, download
    Output: (train_set, test_set)

    VI: Nạp dataset thô, CHƯA áp dụng transform.
    Vào : data_root, download
    Ra  : (train_set, test_set)
    """
    raise NotImplementedError("TODO(Member A): implement load_raw_dataset")


def make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42) -> tuple:
    """Reproducible train/validation split.

    Input : n_samples, val_ratio, seed
    Output: (train_idx, val_idx)

    VI: Chia train/validation tái lập được.
    Vào : n_samples, val_ratio, seed
    Ra  : (train_idx, val_idx)
    """
    raise NotImplementedError("TODO(Member A): implement make_split")


def save_split(train_idx, val_idx, path: str) -> None:
    """Persist and reload the split as JSON.

    Input : indices + path / path
    Output: JSON file / (train_idx, val_idx)

    VI: Lưu và nạp lại split dạng JSON.
    Vào : chỉ số + path / path
    Ra  : file JSON / (train_idx, val_idx)
    """
    raise NotImplementedError("TODO(Member A): implement save_split")


def load_split(path: str) -> tuple:
    """Persist and reload the split as JSON.

    Input : indices + path / path
    Output: JSON file / (train_idx, val_idx)

    VI: Lưu và nạp lại split dạng JSON.
    Vào : chỉ số + path / path
    Ra  : file JSON / (train_idx, val_idx)
    """
    raise NotImplementedError("TODO(Member A): implement load_split")
