"""Integration tests for labs.lab2.main — stage order and data flow.

Every stage is replaced by a lightweight fake, so these tests never download
CIFAR-10 or train a real network.

VI: Test tích hợp cho labs.lab2.main — thứ tự các bước và luồng dữ liệu.
Mọi bước được thay bằng hàm giả nhẹ nên không cần tải CIFAR-10 hay huấn luyện thật.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Allow running this file directly: python labs/lab2/tests/test_main_pipeline.py
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import labs.lab2.main as lab2_main  # noqa: E402


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.zeros(x.size(0), 4))


class Recorder:
    """Collects the names of the stages that were called, in order."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def add(self, name: str) -> None:
        self.calls.append(name)


@pytest.fixture
def artifact_root(monkeypatch, tmp_path):
    artifacts = tmp_path / "artifacts"
    raw = tmp_path / "raw"
    raw.mkdir()
    for kind in ("figures", "splits", "models", "logs", "runs"):
        (artifacts / kind).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(lab2_main, "artifact_dir", lambda lab, kind: artifacts / kind)
    monkeypatch.setattr(lab2_main, "raw_data_dir", lambda lab: raw)
    return artifacts


@pytest.fixture
def history() -> dict:
    return {
        "train_loss": [0.90, 0.80],
        "val_loss": [0.70, 0.60],
        "val_acc": [0.60, 0.70],
        "best_epoch": 2,
        "best_val_loss": 0.60,
        "best_val_acc": 0.70,
    }


def _install_fakes(monkeypatch, recorder: Recorder, *, history, experiment_results,
                   fail_stage: str | None = None, split_exists: bool = False):
    train_loader = object()
    val_loader = object()
    test_loader = object()
    model = FakeModel()

    def fake_run_eda(data_root, out_dir):
        if fail_stage == "eda":
            raise RuntimeError("EDA failed")
        recorder.add("eda")
        return {
            "n_train": 10,
            "n_test": 4,
            "class_counts": {str(i): 1 for i in range(10)},
            "image_size": (32, 32, 3),
            "channel_mean": [0.5, 0.5, 0.5],
            "channel_std": [0.2, 0.2, 0.2],
        }

    def fake_load_raw_dataset(data_root, download=True):
        recorder.add("preprocessing")
        return [0] * 10, [0] * 4

    def fake_make_split(n_samples, val_ratio=0.1, seed=42):
        recorder.add("make_split")
        assert n_samples == 10
        return list(range(9)), [9]

    def fake_save_split(train_idx, val_idx, path):
        recorder.add("save_split")
        Path(path).write_text("{}", encoding="utf-8")

    def fake_load_split(path):
        recorder.add("load_split")
        return list(range(9)), [9]

    def fake_build_transforms(image_size=224, augment=False):
        recorder.add("features")
        return ("train_tf", "eval_tf")

    def fake_make_loaders(**kwargs):
        recorder.add("make_loaders")
        assert kwargs["batch_size"] == 4
        return train_loader, val_loader, test_loader

    def fake_build_model(name, num_classes=10, freeze_backbone=True, pretrained=True):
        recorder.add("model")
        return model

    def fake_describe_model(module):
        recorder.add("describe_model")
        return {"total_params": 100, "trainable_params": 10, "layer_names": ["fc"]}

    def fake_train_model(**kwargs):
        recorder.add("train")
        assert kwargs["train_loader"] is train_loader
        assert kwargs["val_loader"] is val_loader
        assert kwargs["ckpt_path"].endswith(".pt")
        Path(kwargs["ckpt_path"]).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), kwargs["ckpt_path"])
        return history

    def fake_load_checkpoint(module, path, device):
        recorder.add("load_checkpoint")
        assert Path(path).is_file(), "evaluation must load the saved checkpoint"
        return module

    def fake_evaluate(module, loader, device):
        recorder.add("evaluate")
        if loader is test_loader:
            return 0.6, 0.75
        assert loader is val_loader
        return 0.5, 0.8

    def fake_confusion_matrix(module, loader, device, num_classes=10):
        recorder.add("confusion_matrix")
        assert loader is test_loader
        return np.zeros((10, 10), dtype=int)

    def fake_top_confusions(matrix, class_names, k=5):
        recorder.add("top_confusions")
        return [("cat", "dog", 3)]

    def fake_run_comparison(**kwargs):
        recorder.add("run_comparison")
        assert kwargs["loaders"] == (train_loader, val_loader, test_loader)
        return experiment_results

    def fake_save_results(results, csv_path):
        recorder.add("save_results")
        assert results == experiment_results
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        Path(csv_path).write_text("model,best_val_acc\nresnet18,0.8\n", encoding="utf-8")

    def fake_make_summary_writer(log_dir):
        return None

    monkeypatch.setattr(lab2_main, "run_eda", fake_run_eda)
    monkeypatch.setattr(lab2_main, "load_raw_dataset", fake_load_raw_dataset)
    monkeypatch.setattr(lab2_main, "make_split", fake_make_split)
    monkeypatch.setattr(lab2_main, "save_split", fake_save_split)
    monkeypatch.setattr(lab2_main, "load_split", fake_load_split)
    monkeypatch.setattr(lab2_main, "build_transforms", fake_build_transforms)
    monkeypatch.setattr(lab2_main, "make_loaders", fake_make_loaders)
    monkeypatch.setattr(lab2_main, "build_model", fake_build_model)
    monkeypatch.setattr(lab2_main, "describe_model", fake_describe_model)
    monkeypatch.setattr(lab2_main, "train_model", fake_train_model)
    monkeypatch.setattr(lab2_main, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(lab2_main, "evaluate", fake_evaluate)
    monkeypatch.setattr(lab2_main, "confusion_matrix", fake_confusion_matrix)
    monkeypatch.setattr(lab2_main, "top_confusions", fake_top_confusions)
    monkeypatch.setattr(lab2_main, "run_comparison", fake_run_comparison)
    monkeypatch.setattr(lab2_main, "save_results", fake_save_results)
    monkeypatch.setattr(lab2_main, "make_summary_writer", fake_make_summary_writer)

    if split_exists:
        (lab2_main.artifact_dir("lab2", "splits") / "train_val_split.json").write_text(
            "{}", encoding="utf-8"
        )


def test_pipeline_runs_stages_in_order(monkeypatch, artifact_root, history):
    recorder = Recorder()
    _install_fakes(
        monkeypatch, recorder, history=history,
        experiment_results=[{"model": "resnet18", "best_val_acc": 0.8}],
    )

    lab2_main.main(seed=42, epochs=2, image_size=32, batch_size=4, pretrained=False)

    assert recorder.calls == [
        "eda",
        "preprocessing",
        "make_split",
        "save_split",
        "features",
        "make_loaders",
        "model",
        "describe_model",
        "train",
        "model",
        "load_checkpoint",
        "evaluate",
        "evaluate",
        "confusion_matrix",
        "top_confusions",
        "run_comparison",
        "save_results",
    ]


def test_pipeline_returns_and_persists_summary(monkeypatch, artifact_root, history):
    recorder = Recorder()
    experiment_results = [{"model": "resnet18", "best_val_acc": 0.8}]
    _install_fakes(monkeypatch, recorder, history=history, experiment_results=experiment_results)

    summary = lab2_main.main(seed=42, epochs=2, image_size=32, batch_size=4, pretrained=False)

    assert summary["evaluation"]["test_acc"] == pytest.approx(0.75)
    assert summary["evaluation"]["val_acc"] == pytest.approx(0.80)
    assert summary["history"]["best_epoch"] == 2
    assert summary["experiments"] == experiment_results

    summary_path = Path(summary["artifacts"]["summary_json"])
    history_path = Path(summary["artifacts"]["history_csv"])
    checkpoint_path = Path(summary["artifacts"]["checkpoint"])
    assert summary_path.is_file()
    assert history_path.is_file()
    assert checkpoint_path.is_file()
    assert summary_path.parent == artifact_root / "runs"

    on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk["evaluation"]["test_acc"] == pytest.approx(0.75)


def test_pipeline_skips_experiments_when_disabled(monkeypatch, artifact_root, history):
    recorder = Recorder()
    _install_fakes(monkeypatch, recorder, history=history, experiment_results=[])

    summary = lab2_main.main(
        seed=42, epochs=2, image_size=32, batch_size=4, pretrained=False,
        run_experiments=False,
    )

    assert "run_comparison" not in recorder.calls
    assert "save_results" not in recorder.calls
    assert summary["experiments"] == []


def test_pipeline_reuses_existing_split(monkeypatch, artifact_root, history):
    recorder = Recorder()
    _install_fakes(
        monkeypatch, recorder, history=history, experiment_results=[], split_exists=True
    )

    lab2_main.main(seed=42, epochs=2, image_size=32, batch_size=4, pretrained=False)

    assert "load_split" in recorder.calls
    assert "make_split" not in recorder.calls
    assert "save_split" not in recorder.calls


def test_pipeline_propagates_stage_errors(monkeypatch, artifact_root, history):
    recorder = Recorder()
    _install_fakes(
        monkeypatch, recorder, history=history, experiment_results=[], fail_stage="eda"
    )

    with pytest.raises(RuntimeError, match="EDA failed"):
        lab2_main.main(seed=42, epochs=2, image_size=32, batch_size=4, pretrained=False)


def test_pipeline_validates_configuration_before_any_stage(monkeypatch, artifact_root):
    recorder = Recorder()
    _install_fakes(monkeypatch, recorder, history={}, experiment_results=[])

    with pytest.raises(ValueError, match="epochs must be positive"):
        lab2_main.main(epochs=0)

    assert recorder.calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
