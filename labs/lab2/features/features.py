"""Module `features` (Feature engineering (transforms + extraction setup)) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `features` (Feature engineering (transform + thiết lập trích xuất đặc trưng)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformedSubset(Dataset):
    """Dataset subset that applies a specific transform on the fly.
    
    VI: Subset của dataset áp dụng transform chỉ định tại thời điểm nạp mẫu.
    """
    def __init__(self, base_dataset, indices: list[int], transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        image, label = self.base_dataset[real_idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TransformedDataset(Dataset):
    """Full dataset wrapper with a specific transform.
    
    VI: Wrapper bọc toàn bộ dataset với transform chỉ định.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int = 224, augment: bool = False) -> tuple:
    """Transforms for pre-trained models: resize, ToTensor, ImageNet normalization; optional light augmentation for training only. Use a smaller image_size if CPU only.

    Input : image_size, augment
    Output: (train_transform, eval_transform)

    VI: Transform cho model pre-trained: resize, ToTensor, chuẩn hóa ImageNet; augmentation nhẹ (tùy chọn) chỉ cho train. Dùng image_size nhỏ hơn nếu chỉ có CPU.
    Vào : image_size, augment
    Ra  : (train_transform, eval_transform)
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    eval_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]
    eval_transform = transforms.Compose(eval_ops)

    if augment:
        train_ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        train_ops = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    train_transform = transforms.Compose(train_ops)

    return train_transform, eval_transform


def make_loaders(
    train_set,
    test_set,
    train_idx: list[int],
    val_idx: list[int],
    train_tf,
    eval_tf,
    batch_size: int = 64
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders; validation uses the eval transform (no augmentation).

    Input : datasets, split indices, transforms, batch_size
    Output: (train_loader, val_loader, test_loader)

    VI: Tạo DataLoader; validation dùng transform eval (không augmentation).
    Vào : dataset, chỉ số split, transform, batch_size
    Ra  : (train_loader, val_loader, test_loader)
    """
    train_subset = TransformedSubset(train_set, train_idx, transform=train_tf)
    val_subset = TransformedSubset(train_set, val_idx, transform=eval_tf)
    test_dataset = TransformedDataset(test_set, transform=eval_tf)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader