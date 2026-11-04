"""Tests for the optional TensorBoard dependency of lab2.experiment.

TensorBoard must be optional: importing the module and running the experiment
tests must work on a machine where it is not installed, while real logging must
still work when it is installed.

VI: Kiểm tra TensorBoard là phụ thuộc tùy chọn của labs.lab2.experiment.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running this file directly: python labs/lab2/tests/test_tensorboard_optional.py
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import labs.lab2.experiment.experiment as experiment  # noqa: E402


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def _loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)
    return loader, loader, loader


def _build_model(name, num_classes=10, freeze_backbone=True, pretrained=True):
    return TinyNet()


def _train_model(
    model, train_loader, val_loader, epochs, lr, device, ckpt_path,
    writer=None, optimizer_name="adam", weight_decay=0.0,
):
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    if writer is not None:
        writer.add_scalar("Loss/val", 0.5, 1)
    return {
        "train_loss": [0.9],
        "val_loss": [0.5],
        "val_acc": [0.8],
        "best_epoch": 1,
        "best_val_loss": 0.5,
        "best_val_acc": 0.8,
    }


def test_module_imports_and_runs_without_tensorboard_in_subprocess():
    """A fresh interpreter that blocks TensorBoard must still import the module.

    VI: Một tiến trình Python mới chặn TensorBoard vẫn phải import được module.
    """
    code = (
        "import sys\n"
        "sys.modules['torch.utils.tensorboard'] = None\n"
        "import labs.lab2.experiment.experiment as e\n"
        "assert e.make_summary_writer('unused') is None\n"
        "assert callable(e.run_comparison)\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_run_comparison_works_without_tensorboard(monkeypatch, tmp_path):
    """Experiment tests must run when TensorBoard is unavailable (writer is None).

    VI: Test experiment phải chạy được khi không có TensorBoard (writer = None).
    """
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)

    results = experiment.run_comparison(
        model_names=["tiny"],
        hparam_grid=[{"lr": 1e-3, "epochs": 1}],
        loaders=_loaders(),
        device="cpu",
        log_root=str(tmp_path),
        build_model_fn=_build_model,
        train_model_fn=_train_model,
    )

    assert len(results) == 1
    assert results[0]["tensorboard_logged"] is False
    assert os.path.exists(results[0]["checkpoint"])


@pytest.mark.parametrize("enabled", [True, False])
def test_writer_is_closed_after_successful_run(monkeypatch, tmp_path, enabled):
    """The writer must be closed exactly once after training completes.

    VI: Writer phải được đóng đúng một lần sau khi huấn luyện xong.
    """
    writer = MagicMock()
    monkeypatch.setattr(
        experiment, "make_summary_writer", lambda log_dir: writer if enabled else None
    )

    results = experiment.run_comparison(
        model_names=["tiny"],
        hparam_grid=[{"lr": 1e-3, "epochs": 1}],
        loaders=_loaders(),
        device="cpu",
        log_root=str(tmp_path),
        build_model_fn=_build_model,
        train_model_fn=_train_model,
    )

    assert results[0]["tensorboard_logged"] is enabled
    if enabled:
        writer.close.assert_called_once()
    else:
        writer.close.assert_not_called()


def test_writer_is_closed_when_training_raises(monkeypatch, tmp_path):
    """A training failure must not leak the TensorBoard writer.

    VI: Khi huấn luyện lỗi, writer vẫn phải được đóng (try/finally).
    """
    writer = MagicMock()
    monkeypatch.setattr(experiment, "make_summary_writer", lambda log_dir: writer)

    def failing_train(**kwargs):
        raise RuntimeError("training exploded")

    with pytest.raises(RuntimeError, match="training exploded"):
        experiment.run_comparison(
            model_names=["tiny"],
            hparam_grid=[{"lr": 1e-3, "epochs": 1}],
            loaders=_loaders(),
            device="cpu",
            log_root=str(tmp_path),
            build_model_fn=_build_model,
            train_model_fn=failing_train,
        )

    writer.close.assert_called_once()


def test_logging_writes_event_file_when_tensorboard_installed(tmp_path):
    """When TensorBoard is installed, the writer really logs to disk.

    VI: Khi đã cài TensorBoard, writer thực sự ghi log ra đĩa.
    """
    pytest.importorskip("torch.utils.tensorboard")

    writer = experiment.make_summary_writer(str(tmp_path))
    assert writer is not None
    writer.add_scalar("Loss/val", 0.5, 1)
    writer.close()

    assert any(p.name.startswith("events.out.tfevents") for p in tmp_path.iterdir())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
