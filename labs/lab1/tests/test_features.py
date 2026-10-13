"""Tests for lab1 features on tiny synthetic data (no real data needed).

VI: Test module features trên dữ liệu giả nhỏ (không cần dữ liệu thật).
"""
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

from labs.lab1.features.features import (
    FashionDataset,
    build_transform,
    compute_mean_std,
    make_loaders,
)


def _fake_data(n: int = 40, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8)
    labels = (np.arange(n) % 10).astype(np.int64)
    return images, labels


def test_compute_mean_std_on_0_1_scale():
    images, _ = _fake_data()

    mean, std = compute_mean_std(images)

    scaled = images.astype(np.float64) / 255.0
    assert isinstance(mean, float) and isinstance(std, float)
    assert mean == pytest.approx(scaled.mean())
    assert std == pytest.approx(scaled.std())
    assert 0.0 <= mean <= 1.0


def test_build_transform_shape_dtype_and_values():
    image = np.zeros((28, 28), dtype=np.uint8)
    image[0, 0] = 255

    x = build_transform(mean=0.5, std=0.25)(image)

    assert x.shape == (1, 28, 28) and x.dtype == torch.float32
    assert x[0, 0, 0].item() == pytest.approx((1.0 - 0.5) / 0.25)
    assert x[0, 1, 1].item() == pytest.approx((0.0 - 0.5) / 0.25)


def test_build_transform_rejects_zero_std():
    with pytest.raises(ValueError):
        build_transform(0.5, 0.0)


def test_transform_with_train_stats_gives_mean_0_std_1():
    images, _ = _fake_data()
    transform = build_transform(*compute_mean_std(images))

    stacked = torch.stack([transform(img) for img in images])

    assert stacked.mean().item() == pytest.approx(0.0, abs=1e-4)
    assert stacked.std().item() == pytest.approx(1.0, abs=1e-3)


def test_fashion_dataset_uses_indices():
    images, labels = _fake_data()
    indices = np.array([5, 17, 23])
    ds = FashionDataset(images, labels, indices, build_transform(0.5, 0.5))

    x, y = ds[1]

    assert isinstance(ds, Dataset)
    assert len(ds) == 3
    assert x.shape == (1, 28, 28) and x.dtype == torch.float32
    assert isinstance(y, int) and y == int(labels[17])
    assert len(FashionDataset(images, labels, None, None)) == len(labels)


def test_make_loaders_batches_and_shuffle_only_train():
    images, labels = _fake_data()
    transform = build_transform(*compute_mean_std(images[:30]))
    train_ds = FashionDataset(images, labels, np.arange(30), transform)
    val_ds = FashionDataset(images, labels, np.arange(30, 40), transform)
    test_ds = FashionDataset(images, labels, None, transform)

    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=8)

    x, y = next(iter(train_loader))
    assert x.shape == (8, 1, 28, 28) and x.dtype == torch.float32
    assert y.shape == (8,) and y.dtype == torch.int64
    assert y.min() >= 0 and y.max() <= 9
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
