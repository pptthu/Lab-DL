"""Module `features` (Feature engineering (transformation + scaling)) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `features` (Feature engineering (biến đổi + chuẩn hóa)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def compute_mean_std(train_images) -> tuple:
    """Compute normalization statistics from the TRAIN split only.

    Input : train_images: uint8 array [N,28,28] (train indices only)
    Output: (mean, std) as floats on the 0-1 pixel scale

    VI: Tính mean/std để chuẩn hóa, CHỈ dùng tập train.
    Vào : train_images: mảng uint8 [N,28,28] (chỉ chỉ số train)
    Ra  : (mean, std) dạng số thực trên thang pixel 0-1
    """
    raise NotImplementedError("TODO(Member A): implement compute_mean_std")


def build_transform(mean: float, std: float):
    """Build the transform applied to every image.

    Input : mean, std
    Output: callable: uint8 [28,28] -> float32 tensor [1,28,28], normalized

    VI: Tạo phép biến đổi áp dụng cho mọi ảnh.
    Vào : mean, std
    Ra  : hàm: uint8 [28,28] -> tensor float32 [1,28,28] đã chuẩn hóa
    """
    raise NotImplementedError("TODO(Member A): implement build_transform")


class FashionDataset:  # TODO: inherit from the right base class (see docstring)
    """Dataset wrapper (__init__, __len__, __getitem__).

    Input : __init__(images, labels, indices, transform)
    Output: __getitem__(i) -> (tensor [1,28,28], int label); __len__() -> number of samples

    VI: Lớp Dataset (__init__, __len__, __getitem__).
    Vào : __init__(images, labels, indices, transform)
    Ra  : __getitem__(i) -> (tensor [1,28,28], nhãn int); __len__() -> số mẫu
    """
    # TODO(Member A): implement FashionDataset


def make_loaders(train_ds, val_ds, test_ds, batch_size: int = 64) -> tuple:
    """Create DataLoaders (shuffle the training loader only).

    Input : three Dataset objects; batch_size
    Output: (train_loader, val_loader, test_loader)

    VI: Tạo DataLoader (chỉ shuffle tập train).
    Vào : ba đối tượng Dataset; batch_size
    Ra  : (train_loader, val_loader, test_loader)
    """
    raise NotImplementedError("TODO(Member A): implement make_loaders")
