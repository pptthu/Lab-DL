"""Module `features` (Feature engineering (transforms + extraction setup)) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `features` (Feature engineering (transform + thiết lập trích xuất đặc trưng)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def build_transforms(image_size: int = 224, augment: bool = False) -> tuple:
    """Transforms for pre-trained models: resize, ToTensor, ImageNet normalization; optional light augmentation for training only. Use a smaller image_size if CPU only.

    Input : image_size, augment
    Output: (train_transform, eval_transform)

    VI: Transform cho model pre-trained: resize, ToTensor, chuẩn hóa ImageNet; augmentation nhẹ (tùy chọn) chỉ cho train. Dùng image_size nhỏ hơn nếu chỉ có CPU.
    Vào : image_size, augment
    Ra  : (train_transform, eval_transform)
    """
    raise NotImplementedError("TODO(Member A): implement build_transforms")


def make_loaders(train_set, test_set, train_idx, val_idx, train_tf, eval_tf, batch_size: int = 64) -> tuple:
    """Build DataLoaders; validation uses the eval transform (no augmentation).

    Input : datasets, split indices, transforms, batch_size
    Output: (train_loader, val_loader, test_loader)

    VI: Tạo DataLoader; validation dùng transform eval (không augmentation).
    Vào : dataset, chỉ số split, transform, batch_size
    Ra  : (train_loader, val_loader, test_loader)
    """
    raise NotImplementedError("TODO(Member A): implement make_loaders")
