"""Unit tests for Lab 2 Member B: train module."""
import os
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from labs.lab2.train.train import train_model, save_checkpoint, load_checkpoint


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


@pytest.fixture
def tiny_loaders():
    # 16 samples for training, 8 for validation, batch_size=4
    torch.manual_seed(42)
    x_train = torch.randn(16, 3, 16, 16)
    y_train = torch.randint(0, 10, (16,))
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    x_val = torch.randn(8, 3, 16, 16)
    y_val = torch.randint(0, 10, (8,))
    val_ds = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    return train_loader, val_loader


def test_train_model_runs_and_records_history(tiny_loaders):
    train_loader, val_loader = tiny_loaders
    model = TinyCNN(num_classes=10)
    device = "cpu"

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "best.pt")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            lr=0.01,
            device=device,
            ckpt_path=ckpt_path,
            optimizer_name="adam",
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert len(history["val_acc"]) == 2
        assert all(isinstance(v, float) for v in history["train_loss"])
        assert all(0.0 <= acc <= 1.0 for acc in history["val_acc"])
        assert os.path.exists(ckpt_path)


def test_train_model_handles_sgd_and_uneven_batches():
    # 7 samples with batch_size=4 (uneven last batch)
    torch.manual_seed(42)
    x = torch.randn(7, 3, 16, 16)
    y = torch.randint(0, 10, (7,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = TinyCNN()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_sgd.pt")
        history = train_model(
            model=model,
            train_loader=loader,
            val_loader=loader,
            epochs=1,
            lr=0.01,
            device="cpu",
            ckpt_path=ckpt_path,
            optimizer_name="sgd",
        )
        assert len(history["train_loss"]) == 1


def test_save_and_load_checkpoint_roundtrip():
    model1 = TinyCNN()
    model2 = TinyCNN()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "sub", "ckpt.pt")
        save_checkpoint(model1, ckpt_path)
        assert os.path.exists(ckpt_path)

        loaded_model = load_checkpoint(model2, ckpt_path, device="cpu")
        assert loaded_model.training is False

        # Verify weights match
        for p1, p2 in zip(model1.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)


def test_train_model_rejects_invalid_inputs(tiny_loaders):
    train_loader, val_loader = tiny_loaders
    model = TinyCNN()

    with pytest.raises(ValueError, match="epochs must be positive"):
        train_model(model, train_loader, val_loader, epochs=0, lr=1e-3, device="cpu", ckpt_path="dummy.pt")

    with pytest.raises(ValueError, match="Unsupported optimizer_name"):
        train_model(model, train_loader, val_loader, epochs=1, lr=1e-3, device="cpu", ckpt_path="dummy.pt", optimizer_name="unknown")
