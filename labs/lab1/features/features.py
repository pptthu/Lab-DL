"""Module `features` (Feature engineering (transformation + scaling)) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `features` (Feature engineering (biến đổi + chuẩn hóa)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PIXEL_MAX = 255.0


def compute_mean_std(train_images) -> tuple:
    """Compute normalization statistics from the TRAIN split only.

    Input : train_images: uint8 array [N,28,28] (train indices only)
    Output: (mean, std) as floats on the 0-1 pixel scale

    VI: Tính mean/std để chuẩn hóa, CHỈ dùng tập train.
    Vào : train_images: mảng uint8 [N,28,28] (chỉ chỉ số train)
    Ra  : (mean, std) dạng số thực trên thang pixel 0-1
    """
    images = np.asarray(train_images)
    # One mean/std for the whole split (grayscale = 1 channel). Dividing by 255
    # afterwards gives the same result as scaling every pixel to 0-1 first.
    # VI: Một mean/std cho cả tập (ảnh xám = 1 kênh). Chia 255 sau cùng cho kết quả
    # giống như đưa từng pixel về 0-1 trước rồi mới tính.
    mean = float(images.mean(dtype=np.float64) / PIXEL_MAX)
    std = float(images.std(dtype=np.float64) / PIXEL_MAX)
    return mean, std


def build_transform(mean: float, std: float):
    """Build the transform applied to every image.

    Input : mean, std
    Output: callable: uint8 [28,28] -> float32 tensor [1,28,28], normalized

    VI: Tạo phép biến đổi áp dụng cho mọi ảnh.
    Vào : mean, std
    Ra  : hàm: uint8 [28,28] -> tensor float32 [1,28,28] đã chuẩn hóa
    """
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")

    def transform(image) -> torch.Tensor:
        # uint8 [28,28] in 0-255 -> float32 [28,28] in 0-1 -> add channel dim -> normalize.
        # VI: uint8 [28,28] thang 0-255 -> float32 [28,28] thang 0-1 -> thêm chiều kênh -> chuẩn hóa.
        x = torch.as_tensor(np.asarray(image), dtype=torch.float32) / PIXEL_MAX
        return (x.unsqueeze(0) - mean) / std

    return transform


class FashionDataset(Dataset):
    """Dataset wrapper (__init__, __len__, __getitem__).

    Input : __init__(images, labels, indices, transform)
    Output: __getitem__(i) -> (tensor [1,28,28], int label); __len__() -> number of samples

    VI: Lớp Dataset (__init__, __len__, __getitem__).
    Vào : __init__(images, labels, indices, transform)
    Ra  : __getitem__(i) -> (tensor [1,28,28], nhãn int); __len__() -> số mẫu
    """

    def __init__(self, images, labels, indices=None, transform=None) -> None:
        # `indices` selects the rows of this split (train or val). None = use every
        # row (test set). The image array is shared, not copied.
        # VI: `indices` chọn các dòng thuộc split này (train hoặc val). None = dùng hết
        # (tập test). Mảng ảnh được dùng chung, không sao chép.
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple:
        row = self.indices[i]
        image = self.images[row]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[row])


def make_loaders(train_ds, val_ds, test_ds, batch_size: int = 64) -> tuple:
    """Create DataLoaders (shuffle the training loader only).

    Input : three Dataset objects; batch_size
    Output: (train_loader, val_loader, test_loader)

    VI: Tạo DataLoader (chỉ shuffle tập train).
    Vào : ba đối tượng Dataset; batch_size
    Ra  : (train_loader, val_loader, test_loader)
    """
    # Shuffle train only: a new batch order every epoch helps SGD; val/test order
    # does not change the metrics, so keep it fixed.
    # VI: Chỉ shuffle train: thứ tự batch mới mỗi epoch giúp SGD; thứ tự val/test
    # không đổi kết quả đo nên giữ cố định.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
