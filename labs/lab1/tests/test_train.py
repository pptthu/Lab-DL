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

    # history giữ 3 khoá cũ và có thêm các cột của _history.csv cùng tóm tắt lần chạy.
    assert {"train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"} <= set(history.keys())
    assert history["stop_reason"] == "max_epochs"
    assert history["stopped_epoch"] == epochs
    assert 1 <= history["best_epoch"] <= epochs
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


class ConstantNet(nn.Module):
    """Model không học được gì (lr = 0) để val_loss đứng yên, kiểm quy tắc dừng."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_early_stopping_after_patience_epochs(tmp_path):
    # lr = 0 nên val_loss không đổi: epoch 1 là best, dừng sau 5 epoch không giảm, tức ở epoch 6.
    train_loader, val_loader = _make_dummy_loaders()
    history = train_model(
        model=ConstantNet(), train_loader=train_loader, val_loader=val_loader,
        epochs=30, lr=0.0, device="cpu", ckpt_path=str(tmp_path / "r_best.pt"),
        patience=5, min_delta=1e-4, last_ckpt_path=str(tmp_path / "r_last.pt"),
    )
    assert history["stop_reason"] == "early_stopping"
    assert history["best_epoch"] == 1
    assert history["stopped_epoch"] == 6
    assert len(history["val_loss"]) == 6
    assert (tmp_path / "r_best.pt").is_file()
    assert (tmp_path / "r_last.pt").is_file()


def test_writes_three_log_files(tmp_path):
    train_loader, val_loader = _make_dummy_loaders()
    run_info = {"seed": 42, "config": {"hidden_sizes": [16], "lr": 1e-3, "optimizer_name": "adam",
                                       "dropout": 0.0, "weight_decay": 0.0},
                "batch_size": 4, "architecture": "TinyNet", "params": 12730, "data": "fake", "split": "-"}
    history = train_model(
        model=TinyNet(), train_loader=train_loader, val_loader=val_loader,
        epochs=2, lr=1e-3, device="cpu", ckpt_path=str(tmp_path / "m" / "x_best.pt"),
        patience=5, min_delta=1e-4, last_ckpt_path=str(tmp_path / "m" / "x_last.pt"),
        log_dir=str(tmp_path / "logs"), run_name="x", run_info=run_info,
    )
    log = (tmp_path / "logs" / "x.log").read_text(encoding="utf-8")
    assert log.startswith("=== RUN START")
    assert "=== RUN END" in log and "stop_reason: max_epochs" in log
    lines = (tmp_path / "logs" / "x_history.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "epoch,train_loss,train_acc,val_loss,val_acc,lr,time_s"
    assert len(lines) == 1 + history["stopped_epoch"]
    import json
    cfg = json.loads((tmp_path / "logs" / "x_config.json").read_text(encoding="utf-8"))
    assert cfg["seed"] == 42 and cfg["early_stopping"]["patience"] == 5


def test_max_batches_limits_each_epoch(tmp_path):
    # Smoke run: 2 batch mỗi epoch, batch_size 4 nên chỉ đi qua 8 ảnh.
    train_loader, val_loader = _make_dummy_loaders()
    model = TinyNet()
    calls = {"n": 0}
    orig = model.forward

    def counting_forward(x):
        calls["n"] += 1
        return orig(x)

    model.forward = counting_forward
    train_model(model=model, train_loader=train_loader, val_loader=val_loader, epochs=1, lr=1e-3,
                device="cpu", ckpt_path=str(tmp_path / "s.pt"), max_batches=2)
    assert calls["n"] == 4  # 2 batch train + 2 batch val
