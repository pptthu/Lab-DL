"""Unit tests for experiment module (fake data only, fast).

VI: Unit test cho module thí nghiệm (chỉ dùng dữ liệu giả lập, chạy nhanh).
"""
from pathlib import Path
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import labs.lab1.experiment.experiment as experiment_module
from labs.lab1.experiment.experiment import analyze_seed_runs, build_model, compare_pairs, run_experiments, save_results
from labs.lab1.train.train import save_checkpoint
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
    # Keep fake checkpoints out of the real artifacts/models folder.
    # VI: Không để checkpoint giả ghi vào thư mục artifacts/models thật.
    monkeypatch.setattr(experiment_module, "artifact_dir", lambda *_: tmp_path)
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
    monkeypatch.setattr(experiment_module, "artifact_dir", lambda *_: tmp_path)
    history = {"train_loss": [1.0, 0.9, 0.8], "val_loss": [0.5, 0.3, 0.4], "val_acc": [0.9, 0.7, 0.8]}
    monkeypatch.setattr(experiment_module, "train_model", lambda **_: history)

    results = run_experiments([{"hidden_sizes": (4,), "seed": 42}], _make_dummy_loaders(), device="cpu")

    # Highest val_acc is 0.9, but the kept checkpoint is epoch 2.
    assert results[0]["best_val_acc"] == 0.7
    assert results[0]["best_val_loss"] == 0.3
    assert results[0]["best_epoch"] == 2


def test_save_results_marks_network_without_hidden_layers(tmp_path):
    csv_path = tmp_path / "results.csv"
    save_results([{"config": {"name": "softmax", "hidden_sizes": ()}, "best_val_acc": 0.8}], str(csv_path))

    assert pd.read_csv(csv_path)["hidden_sizes"].tolist() == ["none"]


def test_analyze_seed_runs_writes_summaries(tmp_path):
    loaders = _make_dummy_loaders()
    class_names = [f"c{i}" for i in range(10)]
    runs = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        ckpt = tmp_path / f"model_{seed}.pt"
        save_checkpoint(MLP(hidden_sizes=(8,)), str(ckpt))
        runs.append({"config": {"hidden_sizes": (8,), "dropout": 0.0, "seed": seed}, "best_epoch": 1, "ckpt_path": str(ckpt)})

    summary = analyze_seed_runs(runs, loaders, class_names, device="cpu", out_dir=str(tmp_path), top_k=5)

    seeds = pd.read_csv(tmp_path / "seed_results.csv")
    recall = pd.read_csv(tmp_path / "recall_by_class.csv")
    confusions = pd.read_csv(tmp_path / "top_confusions.csv")
    assert seeds["seed"].tolist() == [1, 2]
    assert len(recall) == 10
    assert recall["n_seeds_lowest"].sum() == 2
    assert len(confusions) == 5
    assert summary["test_acc_mean"] == pytest.approx(seeds["test_acc"].mean())


def test_analyze_seed_runs_can_skip_test_set(tmp_path):
    ckpt = tmp_path / "model.pt"
    save_checkpoint(MLP(hidden_sizes=(8,)), str(ckpt))
    run = {"config": {"hidden_sizes": (8,), "dropout": 0.0, "seed": 1}, "best_epoch": 1, "ckpt_path": str(ckpt)}

    summary = analyze_seed_runs([run], _make_dummy_loaders(), [f"c{i}" for i in range(10)], device="cpu",
                                out_dir=str(tmp_path), prefix="x_", use_test=False)

    columns = pd.read_csv(tmp_path / "x_seed_results.csv").columns
    assert "train_acc" in columns
    assert "test_acc" not in columns
    assert "test_acc_mean" not in summary


def test_build_model_follows_config():
    model = build_model({"hidden_sizes": (4,), "dropout": 0.5})
    assert isinstance(model, MLP)
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def test_run_experiments_uses_ckpt_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_module, "artifact_dir", lambda *_: tmp_path)
    configs = [{"hidden_sizes": (4,), "epochs": 1, "seed": 1}]

    results = run_experiments(configs, _make_dummy_loaders(), device="cpu", ckpt_prefix="ext_run")

    assert Path(results[0]["ckpt_path"]).name == "ext_run_0.pt"


def test_compare_pairs_statistics():
    values = pd.DataFrame({"a": [90.0, 91.0, 92.0, 93.0, 94.0], "b": [88.0, 88.5, 89.0, 89.5, 90.5],
                           "c": [90.1, 90.9, 92.2, 92.8, 94.1]})

    df = compare_pairs(values, [("a", "b", "x"), ("c", "a", "y")]).set_index("factor")

    assert df.loc["x", "mean_diff"] == pytest.approx(2.9)
    assert df.loc["x", "seeds_higher"] == 5
    assert bool(df.loc["x", "rule_clear"]) is True
    assert bool(df.loc["y", "rule_clear"]) is False
    assert df.loc["x", "ci95_low"] < 2.9 < df.loc["x", "ci95_high"]
    assert df.loc["x", "p_holm"] == pytest.approx(min(df.loc["x", "p_value"] * 2, 1.0))
