"""Unit tests for experiment module (fake data only, fast).

VI: Unit test cho module thí nghiệm (chỉ dùng dữ liệu giả lập, chạy nhanh).
"""
from pathlib import Path
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from labs.lab1.experiment.experiment import run_experiments, save_results


def _make_dummy_loaders():
    x_train = torch.randn(16, 1, 28, 28)
    y_train = torch.randint(0, 10, (16,))
    x_val = torch.randn(8, 1, 28, 28)
    y_val = torch.randint(0, 10, (8,))
    x_test = torch.randn(8, 1, 28, 28)
    y_test = torch.randint(0, 10, (8,))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=4, shuffle=False)
    return train_loader, val_loader, test_loader


def test_run_experiments_and_save_results(tmp_path):
    loaders = _make_dummy_loaders()
    configs = [
        {"hidden_sizes": (16,), "lr": 0.01, "epochs": 1, "seed": 42},
        {"hidden_sizes": (32,), "lr": 0.01, "epochs": 1, "seed": 42},
    ]

    results = run_experiments(configs, loaders, device="cpu")

    assert isinstance(results, list)
    assert len(results) == 2

    for r in results:
        assert "config" in r
        assert "best_val_acc" in r
        assert "n_params" in r
        assert "train_seconds" in r

        assert isinstance(r["best_val_acc"], float)
        assert 0.0 <= r["best_val_acc"] <= 1.0
        assert isinstance(r["n_params"], int)
        assert r["n_params"] > 0
        assert isinstance(r["train_seconds"], float)
        assert r["train_seconds"] >= 0.0

    csv_path = str(tmp_path / "nested" / "results.csv")
    save_results(results, csv_path)

    assert Path(csv_path).is_file()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert "best_val_acc" in df.columns
    assert "n_params" in df.columns
    assert "train_seconds" in df.columns
    assert "hidden_sizes" in df.columns
