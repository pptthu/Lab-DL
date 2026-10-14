"""Module `model` (Model (backbone + header)) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `model` (Mô hình (backbone + header)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.

Design notes
------------
Supported architectures (name argument):
  'scratch'      – small custom CNN trained from scratch (no pretrained weights).
  'resnet18'     – torchvision ResNet-18 with ImageNet1K_V1 weights (default).
  'vgg16'        – torchvision VGG-16 with ImageNet1K weights.
  'densenet121'  – torchvision DenseNet-121 with ImageNet1K weights.
  'mobilenet_v2' – torchvision MobileNetV2 with ImageNet1K weights.

Uses the modern torchvision weights= API (not the deprecated pretrained=True).

BatchNorm behaviour
-------------------
When freeze_backbone=True the pretrained backbone is kept in eval mode
(bn_eval_on_freeze=True by default) so running statistics are not
accidentally updated during feature extraction.  Fine-tuning disables
this behaviour: BN statistics are updated normally.

This is controlled by the _FrozenBNModel wrapper returned for
freeze_backbone=True runs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Scratch CNN
# ---------------------------------------------------------------------------

class _ScratchCNN(nn.Module):
    """Small CNN designed for CIFAR-10 (32×32 or resized inputs).

    Architecture (input: [B, 3, H, W]):
        Conv(3→32, 3×3) → BN → ReLU → MaxPool(2)
        Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)
        Conv(64→128, 3×3) → BN → ReLU → AdaptiveAvgPool(4×4)
        Flatten → Dropout(0.4) → FC(128*16, 256) → ReLU → FC(256, num_classes)

    ~500K parameters – small enough for CPU training.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # H/2 × W/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # H/4 × W/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # → 128 × 4 × 4 = 2048
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Frozen-BN wrapper
# ---------------------------------------------------------------------------

class _FrozenBNModel(nn.Module):
    """Wraps a model so that modules in ``frozen_modules`` stay in eval mode.

    Calling .train() on this wrapper switches the classifier head to
    training mode but keeps the listed backbone modules in eval mode,
    preventing accidental BatchNorm running-stat updates during feature
    extraction.
    """

    def __init__(self, inner: nn.Module, frozen_modules: List[nn.Module]):
        super().__init__()
        self.inner = inner
        self._frozen_modules = frozen_modules

    def train(self, mode: bool = True):
        # Put everything in the requested mode first.
        super().train(mode)
        # Then force frozen modules back to eval.
        if mode:
            for m in self._frozen_modules:
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


# ---------------------------------------------------------------------------
# Weight-enum helper
# ---------------------------------------------------------------------------

def _get_weights_enum(name: str):
    """Return the recommended ImageNet weights enum for a given model name.

    Uses the new torchvision API (weights=...).  Returns None when pretrained
    weights are not requested (scratch CNN).
    """
    import torchvision.models as models

    mapping = {
        "resnet18":     models.ResNet18_Weights.IMAGENET1K_V1,
        "vgg16":        models.VGG16_Weights.IMAGENET1K_V1,
        "densenet121":  models.DenseNet121_Weights.IMAGENET1K_V1,
        "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1,
    }
    if name not in mapping:
        raise ValueError(
            f"Unsupported model name '{name}'.  "
            f"Choose from: scratch, {', '.join(mapping)}."
        )
    return mapping[name]


# ---------------------------------------------------------------------------
# Head-replacement helpers
# ---------------------------------------------------------------------------

def _replace_head(base_model: nn.Module, name: str, num_classes: int) -> nn.Module:
    """Replace the final classification layer with a new num_classes head."""
    if name == "resnet18":
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)
    elif name == "vgg16":
        # vgg16.classifier[-1] is the final Linear(4096, 1000)
        in_features = base_model.classifier[-1].in_features
        base_model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "densenet121":
        in_features = base_model.classifier.in_features
        base_model.classifier = nn.Linear(in_features, num_classes)
    elif name == "mobilenet_v2":
        in_features = base_model.classifier[-1].in_features
        base_model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"_replace_head: unknown architecture '{name}'")
    return base_model


def _get_backbone_modules(model: nn.Module, name: str) -> List[nn.Module]:
    """Return the backbone (feature extractor) sub-modules for freezing."""
    if name == "resnet18":
        # Everything except the final fc
        return [
            model.inner.conv1, model.inner.bn1,
            model.inner.layer1, model.inner.layer2,
            model.inner.layer3, model.inner.layer4,
        ]
    elif name == "vgg16":
        return [model.inner.features, model.inner.avgpool]
    elif name == "densenet121":
        return [model.inner.features]
    elif name == "mobilenet_v2":
        return [model.inner.features]
    return []


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_model(
    name: str,
    num_classes: int = 10,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Load a torchvision pretrained model and replace the final classification layer.

    Parameters
    ----------
    name : str
        Architecture name: 'scratch', 'resnet18', 'vgg16', 'densenet121',
        'mobilenet_v2'.
    num_classes : int
        Output size of the new classification head (default 10 for CIFAR-10).
    freeze_backbone : bool
        If True, the backbone parameters have requires_grad=False and the
        backbone is kept in eval mode (BN stats frozen).  The replacement
        head has requires_grad=True.
    pretrained : bool
        If True, load ImageNet pretrained weights.
        If False (scratch-mode), initialise randomly.
        Ignored when name='scratch'.

    Returns
    -------
    nn.Module
        Model outputting logits of shape [B, num_classes].
        When freeze_backbone=True, the model is wrapped in _FrozenBNModel
        so that model.train() does not accidentally unfreeze BN.

    VI: Nạp model pre-trained của torchvision và thay lớp phân loại cuối.
    """
    import torchvision.models as models

    if name == "scratch":
        model = _ScratchCNN(num_classes=num_classes)
        return model

    # --- pretrained model ---
    weights = _get_weights_enum(name) if pretrained else None

    model_fn = {
        "resnet18":     models.resnet18,
        "vgg16":        models.vgg16,
        "densenet121":  models.densenet121,
        "mobilenet_v2": models.mobilenet_v2,
    }[name]

    base = model_fn(weights=weights)
    base = _replace_head(base, name, num_classes)

    if freeze_backbone:
        # Freeze all parameters, then unfreeze the new head.
        for param in base.parameters():
            param.requires_grad = False

        # Unfreeze the replacement head only.
        head: nn.Module
        if name == "resnet18":
            head = base.fc
        elif name in ("vgg16", "mobilenet_v2"):
            head = base.classifier[-1]
        elif name == "densenet121":
            head = base.classifier
        for param in head.parameters():
            param.requires_grad = True

        # Wrap to keep backbone BN in eval mode.
        backbone_modules = [
            m for n, m in base.named_children()
            if not _is_head_module(name, n)
        ]
        wrapped = _FrozenBNModel(base, backbone_modules)
        wrapped.eval()
        # Only head should be in train mode
        head.train()
        return wrapped

    # Not frozen: return the plain model.
    return base


def _is_head_module(name: str, child_name: str) -> bool:
    """Return True if ``child_name`` is the classification head of model ``name``."""
    head_names = {
        "resnet18": {"fc"},
        "vgg16": {"classifier"},
        "densenet121": {"classifier"},
        "mobilenet_v2": {"classifier"},
    }
    return child_name in head_names.get(name, set())


def describe_model(model: nn.Module) -> dict:
    """Report parameter counts and trainable layer names.

    Input : model – any nn.Module.
    Output: dict with keys:
              total_params      – total number of scalar parameters.
              trainable_params  – number of trainable (requires_grad=True) parameters.
              frozen_params     – number of frozen parameters.
              trainable_layers  – list of layer names with trainable parameters.
              frozen_layers     – list of layer names with frozen parameters.

    VI: Đếm tham số và liệt kê tên lớp.
    Vào : model
    Ra  : dict: total_params, trainable_params, tên các lớp
    """
    total = 0
    trainable = 0
    trainable_names: List[str] = []
    frozen_names: List[str] = []

    # Use the inner model if wrapped.
    inner = model.inner if isinstance(model, _FrozenBNModel) else model

    for name, param in inner.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            trainable_names.append(name)
        else:
            frozen_names.append(name)

    result = {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_layers": trainable_names,
        "frozen_layers": frozen_names,
    }

    print(f"  Total parameters  : {total:,}")
    print(f"  Trainable params  : {trainable:,}")
    print(f"  Frozen params     : {total - trainable:,}")
    print(f"  Trainable layers  : {trainable_names[:5]}{'...' if len(trainable_names) > 5 else ''}")

    return result


def unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None:
    """Unfreeze the last n_blocks residual blocks (or equivalent) for fine-tuning.

    Operates in-place.  Works with ResNet-18 (layer1–layer4), VGG-16
    (feature groups), DenseNet-121 (dense blocks), and MobileNetV2 (feature
    inverted-residual blocks).

    For ResNet-18:
        n_blocks=1 → unfreeze layer4
        n_blocks=2 → unfreeze layer3, layer4
        n_blocks=3 → unfreeze layer2, layer3, layer4
        n_blocks=4 → unfreeze layer1, layer2, layer3, layer4

    Input : model    – nn.Module returned by build_model.
            n_blocks – how many tail blocks to unfreeze (1-4 for ResNet-18).
    Output: model updated in-place (no return value).

    VI: Mở khóa vài block cuối để fine-tune (tại chỗ).
    """
    inner: nn.Module = model.inner if isinstance(model, _FrozenBNModel) else model

    # Detect architecture by inspecting attributes.
    if hasattr(inner, "layer4"):  # ResNet family
        # Backbone blocks only — exclude the classification head.
        blocks = [inner.layer1, inner.layer2, inner.layer3, inner.layer4]
    elif hasattr(inner, "features") and hasattr(inner, "classifier") and hasattr(inner.features, "__len__"):
        # VGG or MobileNetV2
        feature_children = list(inner.features.children())
        # Split into n_blocks equal groups
        group_size = max(1, len(feature_children) // max(n_blocks, 1))
        blocks = [
            nn.Sequential(*feature_children[i:i + group_size])
            for i in range(0, len(feature_children), group_size)
        ]
        blocks.append(inner.classifier)
    elif hasattr(inner, "features") and hasattr(inner, "classifier"):
        # DenseNet
        feature_children = list(inner.features.children())
        blocks = feature_children + [inner.classifier]
    else:
        # Fallback: unfreeze all parameters
        for param in inner.parameters():
            param.requires_grad = True
        return

    # Unfreeze the tail n_blocks.
    tail = blocks[-min(n_blocks, len(blocks)):]
    for block in tail:
        for param in block.parameters():
            param.requires_grad = True

    # Also unfreeze the replacement head unconditionally.
    if hasattr(inner, "fc"):
        for p in inner.fc.parameters():
            p.requires_grad = True
    if hasattr(inner, "classifier"):
        for p in inner.classifier.parameters():
            p.requires_grad = True
