"""Tests for block detection / unfreezing in labs.lab2.model.model.

VGG-16 keeps every layer in one flat ``nn.Sequential``. Taking "the last two
modules" therefore selects ReLU and MaxPool, which have no parameters at all.
These tests pin down the intended behaviour for VGG-16 and guard ResNet-18 and
DenseNet-121 against regressions.

VI: Kiểm tra việc phát hiện khối và mở khóa của labs.lab2.model.model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch.nn as nn

# Allow running this file directly: python labs/lab2/tests/test_model_unfreezing.py
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.lab2.model.model import build_model, describe_model, unfreeze_last_blocks  # noqa: E402


def _refreeze_vgg16(model: nn.Module) -> None:
    """Return a VGG-16 to the state produced by build_model(freeze_backbone=True)."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[6].parameters():
        param.requires_grad = True


def _refreeze_densenet121(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def _vgg_block_index_ranges(features: nn.Sequential) -> list[tuple[int, int]]:
    """Group feature indices into blocks separated by MaxPool2d (independent of the implementation)."""
    ranges: list[tuple[int, int]] = []
    start = 0
    for index, layer in enumerate(features):
        if isinstance(layer, nn.MaxPool2d):
            ranges.append((start, index))
            start = index + 1
    if start < len(features):
        ranges.append((start, len(features) - 1))
    return ranges


@pytest.fixture(scope="module")
def vgg16_frozen() -> nn.Module:
    return build_model("vgg16", num_classes=10, freeze_backbone=True, pretrained=False)


@pytest.fixture
def vgg16(vgg16_frozen: nn.Module) -> nn.Module:
    _refreeze_vgg16(vgg16_frozen)
    return vgg16_frozen


def test_vgg16_is_split_into_five_conv_blocks(vgg16: nn.Module) -> None:
    ranges = _vgg_block_index_ranges(vgg16.features)
    assert len(ranges) == 5, f"VGG-16 should expose 5 conv blocks, got {ranges}"

    # Before unfreezing, only the replacement classifier head is trainable.
    conv_indices = [i for i, layer in enumerate(vgg16.features) if isinstance(layer, nn.Conv2d)]
    assert conv_indices == [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    assert not any(
        vgg16.features[i].weight.requires_grad for i in conv_indices
    ), "Feature extractor must be frozen before unfreeze_last_blocks is called"


def test_vgg16_one_block_unfreezes_only_the_last_conv_block(vgg16: nn.Module) -> None:
    unfreeze_last_blocks(vgg16, 1)

    ranges = _vgg_block_index_ranges(vgg16.features)
    last_start, last_end = ranges[-1]
    assert (last_start, last_end) == (24, 30)

    for name, param in vgg16.named_parameters():
        if name.startswith("classifier."):
            expected = True
        elif name.startswith("features."):
            index = int(name.split(".")[1])
            expected = last_start <= index <= last_end
        else:  # pragma: no cover - VGG has no other parameters
            expected = False
        assert param.requires_grad is expected, f"{name}: expected requires_grad={expected}"

    assert all(p.requires_grad for p in vgg16.classifier.parameters())
    stats = describe_model(vgg16)
    assert stats["trainable_params"] > 0


def test_vgg16_multiple_blocks_and_clamping(vgg16: nn.Module) -> None:
    ranges = _vgg_block_index_ranges(vgg16.features)

    # Two blocks: indices 17..30 (blocks 4 and 5) plus the classifier.
    unfreeze_last_blocks(vgg16, 2)
    start_two = ranges[-2][0]
    for name, param in vgg16.named_parameters():
        if name.startswith("classifier."):
            expected = True
        elif name.startswith("features."):
            expected = int(name.split(".")[1]) >= start_two
        else:  # pragma: no cover
            expected = False
        assert param.requires_grad is expected, name

    # All five blocks, then an over-large value (must clamp, not crash).
    for n_blocks in (5, 99):
        _refreeze_vgg16(vgg16)
        unfreeze_last_blocks(vgg16, n_blocks)
        assert all(p.requires_grad for p in vgg16.parameters()), n_blocks


@pytest.mark.parametrize("bad_value", [0, -1, -5])
def test_vgg16_invalid_n_blocks_raises(vgg16: nn.Module, bad_value: int) -> None:
    with pytest.raises(ValueError, match="n_blocks must be >= 1"):
        unfreeze_last_blocks(vgg16, bad_value)


def test_invalid_n_blocks_type_raises(vgg16: nn.Module) -> None:
    with pytest.raises(TypeError, match="n_blocks must be an int"):
        unfreeze_last_blocks(vgg16, 1.5)


def test_resnet18_unfreezing_is_unchanged() -> None:
    model = build_model("resnet18", num_classes=10, freeze_backbone=True, pretrained=False)
    unfreeze_last_blocks(model, 1)

    assert all(p.requires_grad for p in model.layer4.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())
    for frozen_stage in (model.layer1, model.layer2, model.layer3):
        assert not any(p.requires_grad for p in frozen_stage.parameters())


def test_densenet121_unfreezing_is_unchanged() -> None:
    model = build_model("densenet121", num_classes=10, freeze_backbone=True, pretrained=False)
    unfreeze_last_blocks(model, 1)

    assert all(p.requires_grad for p in model.classifier.parameters())
    assert all(p.requires_grad for p in model.features.denseblock4.parameters())
    assert not any(p.requires_grad for p in model.features.conv0.parameters())

    _refreeze_densenet121(model)
    unfreeze_last_blocks(model, 2)
    assert all(p.requires_grad for p in model.features.denseblock3.parameters())
    assert not any(p.requires_grad for p in model.features.conv0.parameters())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
