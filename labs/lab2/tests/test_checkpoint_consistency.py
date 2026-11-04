"""Tests that experiment reporting matches the saved checkpoint.

The training loop saves the checkpoint at the epoch with the LOWEST validation
loss. Reporting ``max(history["val_acc"])`` can therefore describe a different
epoch than the checkpoint. These tests pin down the consistent behaviour.

VI: Kiểm tra báo cáo của experiment khớp với checkpoint đã lưu (chọn theo val loss nhỏ nhất).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running this file directly: python labs/lab2/tests/test_checkpoint_consistency.py
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.lab2.evaluate.evaluate import evaluate  # noqa: E402
from labs.lab2.experiment.experiment import run_comparison, save_results  # noqa: E402
from labs.lab2.train.train import load_checkpoint, train_model  # noqa: E402


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


class MockNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.zeros(x.size(0), 4))


@pytest.fixture
def tiny_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    torch.manual_seed(42)
    x_train = torch.randn(16, 3, 16, 16)
    y_train = torch.randint(0, 10, (16,))
    x_val = torch.randn(8, 3, 16, 16)
    y_val = torch.randint(0, 10, (8,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4)
    test_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4)
    return train_loader, val_loader, test_loader


def _mock_build_model(name, num_classes=10, freeze_backbone=True, pretrained=True):
    return MockNet(num_classes=num_classes)


def _make_mock_train(history):
    def _train(model, train_loader, val_loader, epochs, lr, device, ckpt_path,
               writer=None, optimizer_name="adam", weight_decay=0.0):
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        return history

    return _train


def test_history_records_the_checkpoint_epoch(tiny_loaders, tmp_path) -> None:
    """best_epoch / best_val_* must describe the epoch with minimum val loss."""
    train_loader, val_loader, _ = tiny_loaders
    ckpt_path = tmp_path / "best.pt"

    history = train_model(
        model=TinyCNN(),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        lr=0.01,
        device="cpu",
        ckpt_path=str(ckpt_path),
    )

    assert os.path.exists(ckpt_path)
    assert len(history["train_loss"]) == len(history["val_loss"]) == len(history["val_acc"]) == 3

    best_index = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
    assert history["best_epoch"] == best_index + 1
    assert history["best_val_loss"] == pytest.approx(history["val_loss"][best_index])
    assert history["best_val_acc"] == pytest.approx(history["val_acc"][best_index])


def test_reloaded_checkpoint_matches_reported_accuracy(tiny_loaders, tmp_path) -> None:
    """Evaluating the saved checkpoint must reproduce the reported best accuracy."""
    train_loader, val_loader, _ = tiny_loaders
    ckpt_path = tmp_path / "best.pt"

    history = train_model(
        model=TinyCNN(),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        lr=0.01,
        device="cpu",
        ckpt_path=str(ckpt_path),
    )

    reloaded = load_checkpoint(TinyCNN(), str(ckpt_path), device="cpu")
    _, accuracy = evaluate(reloaded, val_loader, device="cpu")
    assert accuracy == pytest.approx(history["best_val_acc"], abs=1e-6)


def test_experiment_reports_checkpoint_epoch_not_max_accuracy(tmp_path) -> None:
    """Lowest val loss at epoch 2, highest val accuracy at epoch 3 -> report epoch 2."""
    history = {
        "train_loss": [0.90, 0.70, 0.60],
        "val_loss": [0.60, 0.45, 0.50],
        "val_acc": [0.80, 0.78, 0.85],
        "best_epoch": 2,
        "best_val_loss": 0.45,
        "best_val_acc": 0.78,
    }

    results = run_comparison(
        model_names=["mock"],
        hparam_grid=[{"lr": 1e-3, "epochs": 3}],
        loaders=(None, None, None),
        device="cpu",
        log_root=str(tmp_path),
        build_model_fn=_mock_build_model,
        train_model_fn=_make_mock_train(history),
    )

    assert len(results) == 1
    result = results[0]
    assert result["best_epoch"] == 2
    assert result["best_val_loss"] == pytest.approx(0.45)
    assert result["best_val_acc"] == pytest.approx(0.78)  # NOT max(val_acc) == 0.85
    assert os.path.exists(result["checkpoint"])


def test_experiment_derives_epoch_when_history_has_no_metadata(tmp_path) -> None:
    """Old-style history (lists only) still resolves the minimum-loss epoch."""
    history = {
        "train_loss": [0.90, 0.70, 0.60],
        "val_loss": [0.60, 0.45, 0.50],
        "val_acc": [0.80, 0.78, 0.85],
    }

    results = run_comparison(
        model_names=["mock"],
        hparam_grid=[{"lr": 1e-3, "epochs": 3}],
        loaders=(None, None, None),
        device="cpu",
        log_root=str(tmp_path),
        build_model_fn=_mock_build_model,
        train_model_fn=_make_mock_train(history),
    )

    assert results[0]["best_epoch"] == 2
    assert results[0]["best_val_acc"] == pytest.approx(0.78)


@pytest.mark.parametrize(
    "history, expected",
    [
        ({}, (0.0, None, None)),
        ({"val_acc": [], "val_loss": []}, (0.0, None, None)),
        ({"val_acc": [0.7, 0.9]}, (0.9, None, 2)),
    ],
)
def test_experiment_handles_missing_validation_history(tmp_path, history, expected) -> None:
    results = run_comparison(
        model_names=["mock"],
        hparam_grid=[{"lr": 1e-3, "epochs": 1}],
        loaders=(None, None, None),
        device="cpu",
        log_root=str(tmp_path),
        build_model_fn=_mock_build_model,
        train_model_fn=_make_mock_train(history),
    )

    best_acc, best_loss, best_epoch = expected
    assert results[0]["best_val_acc"] == pytest.approx(best_acc)
    assert results[0]["best_val_loss"] == best_loss
    assert results[0]["best_epoch"] == best_epoch


def test_save_results_includes_checkpoint_columns(tmp_path) -> None:
    results = [
        {
            "model": "resnet18",
            "hparams": {"lr": 1e-3},
            "best_val_acc": 0.78,
            "best_val_loss": 0.45,
            "best_epoch": 2,
            "train_seconds": 3.5,
            "checkpoint": "/tmp/best.pt",
        },
        {
            "model": "vgg16",
            "hparams": {"lr": 1e-3},
            "best_val_acc": 0.5,
            "best_val_loss": None,
            "best_epoch": None,
            "train_seconds": 4.5,
            "checkpoint": "/tmp/best2.pt",
        },
    ]
    csv_path = tmp_path / "results.csv"
    save_results(results, str(csv_path))

    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    header = rows[0]
    for column in ("best_val_acc", "best_val_loss", "best_epoch", "train_seconds", "checkpoint"):
        assert column in header

    first = dict(zip(header, rows[1]))
    second = dict(zip(header, rows[2]))
    assert first["best_epoch"] == "2"
    assert first["best_val_loss"] == "0.4500"
    assert second["best_epoch"] == ""
    assert second["best_val_loss"] == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
