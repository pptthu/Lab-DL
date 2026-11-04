"""Module `model` (Model (backbone + header)) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `model` (Mô hình (backbone + header)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def build_model(
    name: str,
    num_classes: int = 10,
    freeze_backbone: bool = True,
    pretrained: bool = True
) -> nn.Module:
    """Load a torchvision pre-trained model (e.g. resnet18, vgg16, densenet121) and replace the final classification layer. Note: modern torchvision uses weights=..., pretrained=True is deprecated.

    Input : name, num_classes, freeze_backbone, pretrained
    Output: nn.Module that outputs logits [B,num_classes]; backbone frozen if requested

    VI: Nạp model pre-trained của torchvision (resnet18, vgg16, densenet121...) và thay lớp phân loại cuối. Lưu ý: torchvision mới dùng weights=..., pretrained=True đã cũ.
    Vào : name, num_classes, freeze_backbone, pretrained
    Ra  : nn.Module xuất logits [B,num_classes]; backbone bị đóng băng nếu được yêu cầu
    """
    name_lower = name.lower()

    # 1. Map to modern weights argument instead of deprecated pretrained=True
    if name_lower == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name_lower == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
    elif name_lower == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
    else:
        raise ValueError(f"Unsupported model name: {name}. Choose from 'resnet18', 'densenet121', 'vgg16'.")

    # 2. Freeze backbone if requested (Slide Chapter 3, page 20)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 3. Replace final classification layer (the new layer automatically has requires_grad=True)
    if name_lower == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name_lower == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif name_lower == "vgg16":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    return model


def describe_model(model: nn.Module) -> dict:
    """Print the structure and count parameters.

    Input : model
    Output: dict: total_params, trainable_params, layer names

    VI: In cấu trúc và đếm tham số.
    Vào : model
    Ra  : dict: total_params, trainable_params, tên các lớp
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_names = [name for name, _ in model.named_children()]

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "layer_names": layer_names,
    }


def _unfreeze(module: nn.Module) -> None:
    """Set requires_grad=True on every parameter of a module (in place).

    VI: Đặt requires_grad=True cho mọi tham số của một module (tại chỗ).
    """
    for param in module.parameters():
        param.requires_grad = True


def _split_vgg_conv_blocks(features: nn.Sequential) -> list[nn.Sequential]:
    """Group a VGG feature extractor into its convolutional blocks.

    VGG keeps every layer in one flat ``nn.Sequential``, so "the last two
    modules" are usually ReLU and MaxPool (which have no parameters at all).
    In VGG each convolutional block ends with a ``MaxPool2d``, so the pooling
    layers are a reliable separator. VGG-16 yields exactly 5 blocks.

    VI: Gom phần trích xuất đặc trưng của VGG thành các khối tích chập.
    Mỗi khối trong VGG kết thúc bằng MaxPool2d nên lớp pooling là mốc tách đáng tin cậy.
    """
    blocks: list[nn.Sequential] = []
    current: list[nn.Module] = []
    for layer in features.children():
        current.append(layer)
        if isinstance(layer, nn.MaxPool2d):
            blocks.append(nn.Sequential(*current))
            current = []
    if current:  # trailing layers without a final MaxPool (not the case in VGG-16)
        blocks.append(nn.Sequential(*current))
    return blocks


def unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None:
    """Unfreeze the last n_blocks semantic blocks, keeping the head trainable.

    Supported layouts:
      * ResNet-18: layer1 .. layer4 (n_blocks=1 unfreezes layer4).
      * VGG-16: 5 convolutional blocks, each ending with MaxPool2d.
        n_blocks=1 unfreezes the last conv block (features[24:31]):
        Conv, ReLU, Conv, ReLU, Conv, ReLU, MaxPool.
      * DenseNet-121: dense blocks inside ``model.features`` (existing heuristic).
      * Other models: their last n_blocks top-level children.

    The replacement classification head is always left trainable. Values of
    n_blocks larger than the number of available blocks unfreeze everything.

    Input : model, n_blocks
    Output: model updated in place

    Raises
    ------
    TypeError
        If n_blocks is not an integer.
    ValueError
        If n_blocks < 1.

    VI: Mở khóa n_blocks khối ngữ nghĩa cuối để fine-tune, luôn giữ lớp phân loại được huấn luyện.
    Với VGG-16, mỗi khối tích chập kết thúc bằng MaxPool2d; n_blocks=1 mở khối tích chập cuối (features[24:31]).
    Vào : model, n_blocks (phải >= 1)
    Ra  : model được cập nhật tại chỗ
    """
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, int):
        raise TypeError(f"n_blocks must be an int, got {type(n_blocks).__name__}.")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}.")

    # Case ResNet: blocks are layer4, layer3, layer2, layer1 (Slide Chapter 3, page 21)
    if hasattr(model, "layer4"):
        stage_blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
        for block in stage_blocks[-min(n_blocks, len(stage_blocks)):]:
            _unfreeze(block)
        _unfreeze(model.fc)
        return

    # Case VGG: model.features is a flat Sequential, not a list of semantic blocks.
    if isinstance(model, models.VGG):
        conv_blocks = _split_vgg_conv_blocks(model.features)
        for block in conv_blocks[-min(n_blocks, len(conv_blocks)):]:
            _unfreeze(block)
        _unfreeze(model.classifier)
        return

    # Case DenseNet: features has denseblock4, denseblock3, etc.
    if hasattr(model, "features"):
        # Unfreeze classifier always
        if hasattr(model, "classifier"):
            _unfreeze(model.classifier)

        # DenseNet specific blocks in features: each dense block is paired with
        # the transition block that follows it, hence n_blocks * 2 modules.
        feature_blocks = [m for _name, m in model.features.named_children()]
        for block in reversed(feature_blocks[-(n_blocks * 2):]):
            _unfreeze(block)
        return

    # General fallback for Sequential or other models
    children = list(model.children())
    for child in children[-min(n_blocks, len(children)):]:
        _unfreeze(child)