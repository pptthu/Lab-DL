"""Unit tests for Lab 2 Member B: evaluate module."""
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from labs.lab2.evaluate.evaluate import evaluate, confusion_matrix, top_confusions


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def test_evaluate_accuracy_and_loss_valid():
    torch.manual_seed(42)
    x = torch.randn(12, 4)
    y = torch.randint(0, 3, (12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = SimpleLinear()

    loss, acc = evaluate(model, loader, device="cpu")
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert loss >= 0.0
    assert model.training is False


def test_evaluate_empty_loader_raises():
    loader = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))
    model = SimpleLinear()
    with pytest.raises(ValueError, match="Loader yielded 0 samples"):
        evaluate(model, loader, device="cpu")


def test_confusion_matrix_shape_and_sum():
    # 5 samples with known labels
    targets = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    # mock predictions based on input index
    class MockModel(nn.Module):
        def forward(self, x):
            # x is [B, 1], holding the index
            batch_logits = []
            for item in x.squeeze(1).long():
                idx = item.item()
                if idx in (0, 4):
                    batch_logits.append([10.0, 0.0, 0.0])  # pred 0
                elif idx == 1:
                    batch_logits.append([0.0, 10.0, 0.0])  # pred 1
                elif idx in (2, 3):
                    batch_logits.append([0.0, 0.0, 10.0])  # pred 2
            return torch.tensor(batch_logits)

    indices = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    loader = DataLoader(TensorDataset(indices, targets), batch_size=2)
    cm = confusion_matrix(MockModel(), loader, device="cpu", num_classes=3)

    assert cm.shape == (3, 3)
    assert cm.sum() == 5
    assert cm[0, 0] == 2  # true 0 -> pred 0 twice (indices 0 and 4)
    assert cm[1, 1] == 1  # true 1 -> pred 1 once (index 1)
    assert cm[1, 2] == 1  # true 1 -> pred 2 once (index 3)
    assert cm[2, 2] == 1  # true 2 -> pred 2 once (index 2)


def test_top_confusions_orders_descending_and_skips_diagonal():
    # 3 classes: A, B, C
    # A->B: 5 errors, B->C: 10 errors, C->A: 2 errors, A->A: 50 correct
    cm = np.array([
        [50, 5, 0],
        [0, 40, 10],
        [2, 0, 30],
    ])
    class_names = ["A", "B", "C"]

    top = top_confusions(cm, class_names, k=2)
    assert len(top) == 2
    assert top[0] == ("B", "C", 10)
    assert top[1] == ("A", "B", 5)
