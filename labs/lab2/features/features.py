"""Module `features` (Feature engineering (transforms + extraction setup)) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `features` (Feature engineering (transform + thiết lập trích xuất đặc trưng)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.

Design notes
------------
* build_transforms returns (train_transform, eval_transform).
  - The training transform includes RandomCrop + RandomHorizontalFlip when augment=True.
  - Both transforms resize to image_size (important for ImageNet-pretrained models).
  - Normalisation uses ImageNet statistics (mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]) so pretrained feature maps stay in the
    expected distribution range.
* make_loaders wraps torch.utils.data.Subset + DataLoader.
  - Validation uses eval_transform even though it comes from the train dataset.
  - num_workers=0 is the safe default for CPU-only environments.
"""
from __future__ import annotations

from typing import List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# ImageNet normalisation statistics (used for ALL pretrained models here).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_transforms(image_size: int = 224, augment: bool = False) -> tuple:
    """Build torchvision transform pipelines for pre-trained models.

    Always resizes to image_size × image_size, converts to tensor, and
    applies ImageNet normalisation.  When augment=True the training
    transform adds RandomCrop (with padding) and RandomHorizontalFlip.

    Input : image_size – target spatial resolution (224 for standard ResNet-18).
                         Use 64 or 96 on CPU-only machines to speed things up.
            augment    – True → include light augmentation in the training transform.
    Output: (train_transform, eval_transform)

    VI: Xây dựng pipeline transform cho model pre-trained.
    Vào : image_size, augment
    Ra  : (train_transform, eval_transform)
    """
    pad = max(4, image_size // 16)  # proportional padding for random crop

    common_tail = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=pad),
            transforms.RandomHorizontalFlip(),
            *common_tail,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(image_size),
            *common_tail,
        ])

    eval_tf = transforms.Compose([
        transforms.Resize(image_size),
        *common_tail,
    ])

    return train_tf, eval_tf


def make_loaders(
    train_set,
    test_set,
    train_idx: List[int],
    val_idx: List[int],
    train_tf,
    eval_tf,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple:
    """Build DataLoaders for training, validation, and testing.

    * Training loader: Subset(train_set, train_idx) with train_tf applied.
    * Validation loader: Subset(train_set, val_idx) with eval_tf (NO augmentation).
    * Test loader: full test_set with eval_tf (NO augmentation).

    The validation subset applies eval_tf by wrapping the underlying dataset
    via a lightweight TransformSubset to avoid the augmentation bug that
    occurs when Subset uses the parent dataset's transform.

    Input : train_set  – raw CIFAR-10 training dataset (no transform set).
            test_set   – raw CIFAR-10 test dataset (no transform set).
            train_idx  – indices of training samples within train_set.
            val_idx    – indices of validation samples within train_set.
            train_tf   – transform for training samples.
            eval_tf    – transform for validation/test samples (no augmentation).
            batch_size – mini-batch size (default 64).
            num_workers – DataLoader workers (0 = main process, safe on CPU).
            pin_memory – set True only when using CUDA.
    Output: (train_loader, val_loader, test_loader)

    VI: Tạo DataLoader cho train, validation, test.
    Vào : dataset, chỉ số split, transform, batch_size
    Ra  : (train_loader, val_loader, test_loader)
    """

    train_subset = _TransformSubset(train_set, train_idx, train_tf)
    val_subset   = _TransformSubset(train_set, val_idx,   eval_tf)
    # Apply eval_tf to test_set by wrapping it similarly
    test_wrapped = _TransformDataset(test_set, eval_tf)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(
        train_subset, shuffle=True,  drop_last=False, **loader_kwargs
    )
    val_loader = DataLoader(
        val_subset,   shuffle=False, drop_last=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_wrapped, shuffle=False, drop_last=False, **loader_kwargs
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TransformSubset(torch.utils.data.Dataset):
    """A Subset that applies its own transform, independent of the parent dataset's.

    This avoids the well-known bug where Subset uses the parent's transform
    instead of the one we want for validation.
    """

    def __init__(self, dataset, indices: List[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class _TransformDataset(torch.utils.data.Dataset):
    """Wrap a dataset and apply a fixed transform, replacing any existing one."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
