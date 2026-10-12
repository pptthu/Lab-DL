"""Module `preprocessing` (Preprocessing) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `preprocessing` (Tiền xử lý). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def load_csv(csv_path: str) -> tuple:
    """Read a Fashion-MNIST CSV (label + 784 pixel columns) into arrays.

    Input : csv_path: path to a CSV file
    Output: (images: uint8 array [N,28,28], labels: int64 array [N])

    VI: Đọc CSV Fashion-MNIST (cột label + 784 cột pixel) thành mảng.
    Vào : csv_path: đường dẫn file CSV
    Ra  : (images: mảng uint8 [N,28,28], labels: mảng int64 [N])
    """
    raise NotImplementedError("TODO(Member A): implement load_csv")


def make_split(labels, val_ratio: float = 0.1, seed: int = 42) -> tuple:
    """Split training indices into train/validation (stratified, reproducible).

    Input : labels: int64 array [N]; val_ratio; seed
    Output: (train_idx, val_idx): disjoint index arrays covering all N samples

    VI: Chia chỉ số train thành train/validation (phân tầng, tái lập được).
    Vào : labels: mảng int64 [N]; val_ratio; seed
    Ra  : (train_idx, val_idx): hai mảng chỉ số không giao nhau, gộp lại đủ N mẫu
    """
    raise NotImplementedError("TODO(Member A): implement make_split")


def save_split(train_idx, val_idx, path: str) -> None:
    """Save the split as JSON so everyone uses the same split.

    Input : train_idx, val_idx, path (.json)
    Output: JSON file at path

    VI: Lưu split ra JSON để cả nhóm dùng chung một cách chia.
    Vào : train_idx, val_idx, path (.json)
    Ra  : file JSON tại path
    """
    raise NotImplementedError("TODO(Member A): implement save_split")


def load_split(path: str) -> tuple:
    """Load a saved split.

    Input : path (.json)
    Output: (train_idx, val_idx)

    VI: Đọc lại split đã lưu.
    Vào : path (.json)
    Ra  : (train_idx, val_idx)
    """
    raise NotImplementedError("TODO(Member A): implement load_split")
