"""Unit tests for Lab 2 Member B: experiment module."""
import csv
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from labs.lab2.experiment.experiment import run_comparison, save_results


class DummyNet(nn.Module):
    def __init__(self, name="dummy", num_classes=10, freeze_backbone=True, pretrained=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=[2, 3]))


def mock_build_model(name: str, num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True):
    return DummyNet(name=name, num_classes=num_classes)


def mock_train_model(model, train_loader, val_loader, epochs, lr, device, ckpt_path, writer=None, optimizer_name="adam", weight_decay=0.0):
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    return {"train_loss": [0.5], "val_loss": [0.4], "val_acc": [0.85]}


def test_run_comparison_records_results_and_logs():
    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 10, (8,))
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=4)
    val_loader = DataLoader(ds, batch_size=4)
    test_loader = DataLoader(ds, batch_size=4)
    loaders = (train_loader, val_loader, test_loader)

    model_names = ["dummy1"]
    hparam_grid = [
        {"lr": 0.01, "epochs": 1, "optimizer_name": "adam"},
        {"lr": 0.05, "epochs": 1, "optimizer_name": "sgd"},
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_comparison(
            model_names=model_names,
            hparam_grid=hparam_grid,
            loaders=loaders,
            device="cpu",
            log_root=tmp_dir,
            build_model_fn=mock_build_model,
            train_model_fn=mock_train_model,
        )

        assert len(results) == 2
        for r in results:
            assert r["model"] == "dummy1"
            assert "best_val_acc" in r
            assert "train_seconds" in r
            assert os.path.exists(r["checkpoint"])


def test_save_results_creates_valid_csv():
    results = [
        {
            "model": "resnet18",
            "hparams": {"lr": 0.001, "epochs": 5},
            "best_val_acc": 0.8523,
            "train_seconds": 12.34,
            "checkpoint": "/tmp/resnet18_best.pt",
        },
        {
            "model": "vgg16",
            "hparams": {"lr": 0.0005, "epochs": 5},
            "best_val_acc": 0.8211,
            "train_seconds": 25.67,
            "checkpoint": "/tmp/vgg16_best.pt",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "test_res.csv")
        save_results(results, csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3  # header + 2 rows
        headers = rows[0]
        assert "model" in headers
        assert "hp_lr" in headers
        assert "hp_epochs" in headers
        assert "best_val_acc" in headers
        assert "train_seconds" in headers
