"""Test suite cho Module Model (CIFAR-10 thật)."""
from pathlib import Path
import pytest
import torch
import sys
import torch.nn as nn
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shared.seed import set_seed
from labs.lab2.model.model import (
    build_model,
    describe_model,
    unfreeze_last_blocks,
)


def test_resnet18_model():
    set_seed(42)

    # 1. Test build_model với ResNet-18 (đóng băng backbone)
    model = build_model(name="resnet18", num_classes=10, freeze_backbone=True, pretrained=True)
    assert isinstance(model.fc, nn.Linear), "Lớp cuối phải là nn.Linear"
    assert model.fc.out_features == 10, f"Lớp cuối phải có 10 outputs, nhận được: {model.fc.out_features}"

    # 2. Test describe_model khi backbone bị đóng băng
    stats = describe_model(model)
    expected_keys = {"total_params", "trainable_params", "layer_names"}
    assert set(stats.keys()) == expected_keys
    assert stats["total_params"] == 11181642
    # Chỉ có lớp fc là trainable: 512 * 10 + 10 = 5130
    assert stats["trainable_params"] == 5130, f"Trainable params sai: {stats['trainable_params']} (kỳ vọng 5130)"

    # 3. Test forward pass với 1 batch thật [64, 3, 32, 32] -> logits [64, 10]
    dummy_batch = torch.randn(64, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        logits = model(dummy_batch)
    assert logits.shape == (64, 10), f"Logits shape sai: {logits.shape} (kỳ vọng [64, 10])"
    assert not torch.isnan(logits).any(), "Logits chứa giá trị NaN"

    # 4. Test unfreeze_last_blocks (Fine-tuning mở block layer4)
    unfreeze_last_blocks(model, n_blocks=1)
    stats_ft = describe_model(model)
    # Số lượng trainable params sau khi unfreeze layer4 phải tăng lên 8,403,466
    expected_ft_params = sum(p.numel() for p in model.layer4.parameters()) + sum(p.numel() for p in model.fc.parameters())
    assert stats_ft["trainable_params"] == expected_ft_params
    assert stats_ft["trainable_params"] > stats["trainable_params"]


def test_densenet121_model():
    set_seed(42)

    # Test tương tự cho DenseNet-121
    model = build_model(name="densenet121", num_classes=10, freeze_backbone=True, pretrained=True)
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.out_features == 10

    dummy_batch = torch.randn(4, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        logits = model(dummy_batch)
    assert logits.shape == (4, 10)


if __name__ == "__main__":
    test_resnet18_model()
    test_densenet121_model()
    print(">>> [PASSED] Test Model thành công!")