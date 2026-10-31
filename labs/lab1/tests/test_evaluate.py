"""Unit tests for evaluate module (fake data only, fast).

VI: Unit test cho module đánh giá (chỉ dùng dữ liệu giả lập, chạy nhanh).
"""
import math
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from labs.lab1.evaluate.evaluate import confusion_matrix, evaluate, per_class_accuracy, save_classification_report


class DummyClassifier(nn.Module):
    """Deterministic classifier that outputs large logit for label y = argmax(x) or fixed label."""
    def __init__(self, mode="perfect"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        # x is [B, 1, 28, 28]
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, 10)
        if self.mode == "perfect":
            # first pixel encodes true class
            classes = x[:, 0, 0, 0].long().clamp(0, 9)
            logits.scatter_(1, classes.unsqueeze(1), 10.0)
        elif self.mode == "constant_zero":
            logits[:, 0] = 10.0
        return logits


def _make_dummy_loader(n_samples=20):
    # Set pixel (0,0,0) to target class so DummyClassifier can be perfect
    x = torch.zeros(n_samples, 1, 28, 28)
    y = torch.tensor([i % 10 for i in range(n_samples)], dtype=torch.long)
    for i in range(n_samples):
        x[i, 0, 0, 0] = float(y[i].item())
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def test_evaluate_returns_valid_metrics_and_handles_perfect_model():
    loader = _make_dummy_loader(n_samples=20)
    model = DummyClassifier(mode="perfect")
    avg_loss, accuracy = evaluate(model, loader, device="cpu")

    assert isinstance(avg_loss, float)
    assert isinstance(accuracy, float)
    assert math.isfinite(avg_loss)
    assert avg_loss >= 0.0
    assert accuracy == 1.0
    assert not model.training


def test_confusion_matrix_shape_sum_and_diagonal():
    n_samples = 30
    loader = _make_dummy_loader(n_samples=n_samples)
    model = DummyClassifier(mode="perfect")

    cm = confusion_matrix(model, loader, device="cpu", num_classes=10)

    assert isinstance(cm, np.ndarray)
    assert cm.shape == (10, 10)
    assert cm.sum() == n_samples

    # For perfect model, off-diagonal elements are 0 and diagonal elements match class counts
    diag = np.diag(cm)
    assert np.all(diag == 3)  # 30 samples distributed evenly across 10 classes
    assert cm.sum() == diag.sum()


def test_per_class_accuracy_with_known_matrix():
    cm = np.array([
        [10, 0, 0],
        [2, 8, 0],
        [1, 2, 7],
    ])
    accs = per_class_accuracy(cm)

    assert isinstance(accs, dict)
    assert len(accs) == 3
    assert accs[0] == pytest.approx(10 / 10)
    assert accs[1] == pytest.approx(8 / 10)
    assert accs[2] == pytest.approx(7 / 10)


def test_per_class_accuracy_empty_row_handles_zero_division():
    cm = np.array([
        [5, 0],
        [0, 0],  # empty row, 0 samples
    ])
    accs = per_class_accuracy(cm)

    assert isinstance(accs, dict)
    assert accs[0] == 1.0
    assert accs[1] == 0.0


def test_save_classification_report_values_and_file(tmp_path):
    cm = np.array([[8, 2], [1, 9]])
    csv_path = tmp_path / "nested" / "report.csv"

    df = save_classification_report(cm, ["a", "b"], str(csv_path))

    assert csv_path.is_file()
    assert df.loc[0, "support"] == 10
    assert df.loc[0, "recall"] == pytest.approx(0.8)
    assert df.loc[0, "precision"] == pytest.approx(8 / 9)
    assert df.loc[0, "specificity"] == pytest.approx(0.9)
    assert df.loc[1, "f1"] == pytest.approx(2 * (9 / 11) * 0.9 / ((9 / 11) + 0.9))
