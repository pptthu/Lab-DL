"""Unit tests for visualize module (fake data only, fast).

VI: Unit test cho module trực quan hóa (chỉ dùng dữ liệu giả lập, chạy nhanh).
"""
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from labs.lab1.visualize.visualize import (
    plot_confusion_matrix,
    plot_loss_curves,
    show_predictions,
)

DUMMY_CLASS_NAMES = [f"c{i}" for i in range(10)]


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.flatten(start_dim=1))


def _make_dummy_loader(n_samples=20):
    x = torch.randn(n_samples, 1, 28, 28)
    y = torch.randint(0, 10, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_plot_loss_curves_creates_non_empty_file(tmp_path):
    out_file = str(tmp_path / "nested" / "loss_curves.png")
    history = {
        "train_loss": [0.6, 0.4, 0.2],
        "val_loss": [0.65, 0.45, 0.25],
        "val_acc": [0.7, 0.8, 0.9],
    }

    plot_loss_curves(history, out_file)

    p = Path(out_file)
    assert p.is_file()
    assert p.stat().st_size > 0


def test_show_predictions_creates_file_and_validates_class_names(tmp_path):
    loader = _make_dummy_loader(n_samples=20)
    model = TinyNet()
    out_file = str(tmp_path / "predictions.png")

    # Wrong class_names length raises ValueError
    with pytest.raises(ValueError, match="class_names"):
        show_predictions(
            model=model,
            loader=loader,
            class_names=["only", "two"],
            device="cpu",
            out_path=out_file,
            n=16,
        )

    # Valid call creates file
    show_predictions(
        model=model,
        loader=loader,
        class_names=DUMMY_CLASS_NAMES,
        device="cpu",
        out_path=out_file,
        n=16,
    )

    p = Path(out_file)
    assert p.is_file()
    assert p.stat().st_size > 0


def test_plot_confusion_matrix_creates_file_and_validates_class_names(tmp_path):
    cm = torch.randint(0, 50, (10, 10)).numpy()
    out_file = str(tmp_path / "confusion_matrix.png")

    # Wrong class_names length raises ValueError
    with pytest.raises(ValueError, match="class_names"):
        plot_confusion_matrix(cm, ["too", "short"], out_file)

    # Valid call creates file
    plot_confusion_matrix(cm, DUMMY_CLASS_NAMES, out_file)

    p = Path(out_file)
    assert p.is_file()
    assert p.stat().st_size > 0
