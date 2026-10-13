"""Tests for lab1 model on dummy batches (no real data needed).

VI: Test module model trên batch giả (không cần dữ liệu thật).
"""
import torch
from torch import nn

from labs.lab1.model.model import MLP, count_parameters


def test_forward_returns_logits_shape():
    model = MLP()

    logits = model(torch.randn(8, 1, 28, 28))

    assert isinstance(model, nn.Module)
    assert logits.shape == (8, 10) and logits.dtype == torch.float32


def test_output_is_logits_not_probabilities():
    model = MLP()

    logits = model(torch.randn(16, 1, 28, 28))

    assert not any(isinstance(m, (nn.Softmax, nn.LogSoftmax)) for m in model.modules())
    # Softmax rows would sum to 1; raw logits do not.
    assert not torch.allclose(logits.sum(dim=1), torch.ones(16))


def test_count_parameters_default_and_custom():
    # 784*256+256 + 256*128+128 + 128*10+10
    assert count_parameters(MLP()) == 235146
    assert count_parameters(MLP(hidden_sizes=(64,), num_classes=5)) == 784 * 64 + 64 + 64 * 5 + 5


def test_count_parameters_skips_frozen():
    model = MLP()
    for p in model.net[1].parameters():  # first Linear layer
        p.requires_grad = False

    assert count_parameters(model) == 235146 - (784 * 256 + 256)


def test_dropout_layers_only_when_requested():
    def n_dropout(model):
        return sum(isinstance(m, nn.Dropout) for m in model.modules())

    assert n_dropout(MLP(dropout=0.0)) == 0
    assert n_dropout(MLP(dropout=0.3)) == 2


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = MLP()
    x = torch.randn(16, 1, 28, 28)
    y = torch.arange(16) % 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    first = loss_fn(model(x), y).item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1 < first
