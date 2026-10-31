"""Unit tests for training module (fake data only, fast).

VI: Unit test cho module huấn luyện (chỉ dùng dữ liệu giả lập, chạy nhanh).
"""
import math
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from labs.lab1.train.train import load_checkpoint, save_checkpoint, train_model


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)


def _make_dummy_loaders():
    x_train = torch.randn(16, 1, 28, 28)
    y_train = torch.randint(0, 10, (16,))
    x_val = torch.randn(8, 1, 28, 28)
    y_val = torch.randint(0, 10, (8,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4, shuffle=False)
    return train_loader, val_loader


def test_bad_optimizer_raises_value_error(tmp_path):
    model = TinyNet()
    train_loader, val_loader = _make_dummy_loaders()
    with pytest.raises(ValueError, match="optimizer"):
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            lr=1e-3,
            device="cpu",
            ckpt_path=str(tmp_path / "best.pt"),
            optimizer_name="invalid_opt",
        )


def test_history_lengths_and_finite_metrics(tmp_path):
    model = TinyNet()
    train_loader, val_loader = _make_dummy_loaders()
    ckpt_file = str(tmp_path / "models" / "best.pt")
    epochs = 3

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device="cpu",
        ckpt_path=ckpt_file,
        optimizer_name="adam",
    )

    assert set(history.keys()) == {"train_loss", "val_loss", "val_acc"}
    assert len(history["train_loss"]) == epochs
    assert len(history["val_loss"]) == epochs
    assert len(history["val_acc"]) == epochs

    for key in ("train_loss", "val_loss", "val_acc"):
        for val in history[key]:
            assert isinstance(val, float)
            assert math.isfinite(val)
    for acc in history["val_acc"]:
        assert 0.0 <= acc <= 1.0

    assert Path(ckpt_file).is_file()


def test_save_and_load_checkpoint(tmp_path):
    model = TinyNet()
    ckpt_file = str(tmp_path / "nested" / "dir" / "model.pt")
    save_checkpoint(model, ckpt_file)
    assert Path(ckpt_file).is_file()

    new_model = TinyNet()
    loaded_model = load_checkpoint(new_model, ckpt_file, device="cpu")

    assert loaded_model is new_model
    assert not loaded_model.training

    dummy_x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out_orig = model(dummy_x)
        out_loaded = loaded_model(dummy_x)
    assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_train_overfits_tiny_set(tmp_path):
    torch.manual_seed(42)
    model = TinyNet()
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
    ckpt_file = str(tmp_path / "overfit.pt")

    history = train_model(
        model=model,
        train_loader=loader,
        val_loader=loader,
        epochs=35,
        lr=0.02,
        device="cpu",
        ckpt_path=ckpt_file,
        optimizer_name="adam",
    )

    assert history["train_loss"][-1] < history["train_loss"][0] * 0.1
    assert history["val_acc"][-1] == 1.0
