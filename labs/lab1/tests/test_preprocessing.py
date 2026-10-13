"""Tests for lab1 preprocessing on tiny synthetic data (no real data needed).

VI: Test module preprocessing trên dữ liệu giả nhỏ (không cần dữ liệu thật).
"""
import numpy as np
import pandas as pd
import pytest

from labs.lab1.preprocessing.preprocessing import (
    N_PIXELS,
    load_csv,
    load_split,
    make_split,
    save_split,
)


def _write_csv(path, n_rows: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = ["label"] + [f"pixel{i}" for i in range(1, N_PIXELS + 1)]
    data = rng.integers(0, 256, size=(n_rows, N_PIXELS + 1))
    data[:, 0] = np.arange(n_rows) % 10
    pd.DataFrame(data, columns=cols).to_csv(path, index=False)
    return data


def test_load_csv_shapes_dtypes_values(tmp_path):
    csv = tmp_path / "train.csv"
    data = _write_csv(csv)

    images, labels = load_csv(str(csv))

    assert images.shape == (20, 28, 28) and images.dtype == np.uint8
    assert labels.shape == (20,) and labels.dtype == np.int64
    assert np.array_equal(labels, data[:, 0])
    # Row-major reshape: pixel1..pixel28 is the first image row.
    assert np.array_equal(images[3].reshape(-1), data[3, 1:])


def test_load_csv_rejects_bad_files(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"label": [0], "pixel1": [0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_csv(str(csv))

    data = _write_csv(csv)
    cols = ["label"] + [f"pixel{i}" for i in range(1, N_PIXELS + 1)]
    data[0, 1] = 256  # out of the uint8 range
    pd.DataFrame(data, columns=cols).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_csv(str(csv))


def test_make_split_disjoint_complete_stratified():
    labels = np.repeat(np.arange(10), 50)  # 500 samples, 50 per class

    train_idx, val_idx = make_split(labels, val_ratio=0.1, seed=42)

    assert len(val_idx) == 50 and len(train_idx) == 450
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert np.array_equal(np.sort(np.concatenate([train_idx, val_idx])), np.arange(500))
    assert np.array_equal(np.bincount(labels[val_idx]), np.full(10, 5))


def test_make_split_is_reproducible():
    labels = np.repeat(np.arange(10), 50)

    a = make_split(labels, seed=42)
    b = make_split(labels, seed=42)
    c = make_split(labels, seed=7)

    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_make_split_rejects_bad_ratio():
    with pytest.raises(ValueError):
        make_split(np.arange(10), val_ratio=1.0)


def test_save_and_load_split_roundtrip(tmp_path):
    labels = np.repeat(np.arange(10), 50)
    train_idx, val_idx = make_split(labels)
    path = tmp_path / "splits" / "split.json"

    save_split(train_idx, val_idx, str(path))
    loaded_train, loaded_val = load_split(str(path))

    assert loaded_train.dtype == np.int64 and loaded_val.dtype == np.int64
    assert np.array_equal(loaded_train, train_idx)
    assert np.array_equal(loaded_val, val_idx)
