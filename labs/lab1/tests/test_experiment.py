"""Unit tests for experiment module (fake data only, fast).

VI: Unit test cho module thí nghiệm (chỉ dùng dữ liệu giả lập, chạy nhanh).
"""
from pathlib import Path
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import labs.lab1.experiment.experiment as experiment_module
from labs.lab1.experiment.experiment import (
    VARIANTS, build_model, make_configs, run_experiments, save_history, save_results, select_config,
)
from labs.lab1.model.model import MLP


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


def test_run_experiments_and_save_results(tmp_path, monkeypatch):
    # Keep fake checkpoints out of the real outputs/models folder.
    # VI: Không để checkpoint giả ghi vào thư mục outputs/models thật.
    monkeypatch.setattr(experiment_module, "output_dir", lambda *_: tmp_path)
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


def test_best_val_acc_comes_from_lowest_val_loss_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_module, "output_dir", lambda *_: tmp_path)
    history = {"train_loss": [1.0, 0.9, 0.8], "val_loss": [0.5, 0.3, 0.4], "val_acc": [0.9, 0.7, 0.8]}
    monkeypatch.setattr(experiment_module, "train_model", lambda **_: history)

    results = run_experiments([{"hidden_sizes": (4,), "seed": 42}], _make_dummy_loaders(), device="cpu")

    # Highest val_acc is 0.9, but the kept checkpoint is epoch 2.
    assert results[0]["best_val_acc"] == 0.7
    assert results[0]["best_val_loss"] == 0.3
    assert results[0]["best_epoch"] == 2


def test_run_experiments_keeps_epoch_history(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_module, "output_dir", lambda *_: tmp_path)
    history = {"train_loss": [1.0, 0.9, 0.8], "val_loss": [0.5, 0.3, 0.4], "val_acc": [0.9, 0.7, 0.8]}
    monkeypatch.setattr(experiment_module, "train_model", lambda **_: history)

    results = run_experiments([{"hidden_sizes": (4,), "seed": 42}], _make_dummy_loaders(), device="cpu")

    assert results[0]["history"] == history


def test_save_history_writes_one_row_per_epoch(tmp_path):
    history = {"train_loss": [1.0, 0.9], "val_loss": [0.5, 0.3], "val_acc": [0.6, 0.7]}
    results = [
        {"config": {"name": "baseline", "seed": 42}, "history": history},
        {"config": {"name": "baseline", "seed": 7}, "history": history},
    ]
    csv_path = tmp_path / "nested" / "history.csv"

    save_history(results, str(csv_path))

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["name", "seed", "epoch", "train_loss", "val_loss", "val_acc"]
    assert len(df) == 4
    assert df["epoch"].tolist() == [1, 2, 1, 2]
    assert df["seed"].tolist() == [42, 42, 7, 7]
    assert df["val_loss"].tolist() == [0.5, 0.3, 0.5, 0.3]


def test_save_results_marks_network_without_hidden_layers(tmp_path):
    csv_path = tmp_path / "results.csv"
    save_results([{"config": {"name": "softmax", "hidden_sizes": ()}, "best_val_acc": 0.8}], str(csv_path))

    assert pd.read_csv(csv_path)["hidden_sizes"].tolist() == ["none"]


def test_build_model_follows_config():
    model = build_model({"hidden_sizes": (4,), "dropout": 0.5})
    assert isinstance(model, MLP)
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def test_run_experiments_uses_ckpt_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_module, "output_dir", lambda *_: tmp_path)
    configs = [{"hidden_sizes": (4,), "epochs": 1, "seed": 1}]

    results = run_experiments(configs, _make_dummy_loaders(), device="cpu", ckpt_prefix="ext_run")

    assert Path(results[0]["ckpt_path"]).name == "ext_run_0.pt"


def test_make_configs_has_five_lab_configs():
    configs = make_configs(epochs=30, seed=42)
    assert [c["name"] for c in configs] == ["baseline", "softmax_regression", "larger_network", "sgd_lr_0.01", "dropout_0.2"]
    assert all(c["seed"] == 42 and c["epochs"] == 30 for c in configs)
    # Mỗi cấu hình khác baseline đúng một yếu tố (sgd đổi optimizer kèm lr).
    assert len(VARIANTS) == 5
    assert configs[1]["hidden_sizes"] == () and configs[2]["hidden_sizes"] == (512, 256, 128)
    assert configs[3]["optimizer_name"] == "sgd" and configs[3]["lr"] == 0.01
    assert configs[4]["dropout"] == 0.2


def test_run_experiments_with_logs_writes_files_and_new_columns(tmp_path):
    configs = make_configs(epochs=2)[:2]
    results = run_experiments(configs, _make_dummy_loaders(), device="cpu", log_dir=str(tmp_path / "logs"),
                              models_dir=str(tmp_path / "models"), patience=5, min_delta=1e-4,
                              run_info={"batch_size": 4, "data": "fake", "split": "-"})
    for r in results:
        assert Path(r["log_path"]).is_file() and Path(r["history_path"]).is_file() and Path(r["config_path"]).is_file()
        assert r["ckpt_path"].endswith("_best.pt") and Path(r["ckpt_path"]).is_file()
        assert Path(r["last_ckpt_path"]).is_file()
        assert r["stop_reason"] in ("early_stopping", "max_epochs")
    csv_path = tmp_path / "cmp.csv"
    save_results(results, str(csv_path))
    df = pd.read_csv(csv_path)
    for col in ("name", "n_params", "best_epoch", "stopped_epoch", "best_val_loss", "best_val_acc",
                "best_train_acc", "train_minus_val_acc", "train_seconds"):
        assert col in df.columns
    assert df["train_minus_val_acc"].iloc[0] == pytest.approx(df["best_train_acc"].iloc[0] - df["best_val_acc"].iloc[0])


def test_select_config_uses_val_acc_then_val_loss():
    results = [
        {"config": {"name": "a"}, "best_val_acc": 0.88, "best_val_loss": 0.33},
        {"config": {"name": "b"}, "best_val_acc": 0.89, "best_val_loss": 0.35},
        {"config": {"name": "c"}, "best_val_acc": 0.89, "best_val_loss": 0.30},
    ]
    chosen = select_config(results)
    assert chosen["name"] == "c"
    assert chosen["index"] == 2
    assert "val_acc" in chosen["reason"]
