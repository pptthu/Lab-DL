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


def unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None:
    """Unfreeze the last blocks to fine-tune.

    Input : model, n_blocks
    Output: model updated in place

    VI: Mở khóa vài block cuối để fine-tune.
    Vào : model, n_blocks
    Ra  : model được cập nhật tại chỗ
    """
    if n_blocks <= 0:
        return

    # Case ResNet: blocks are layer4, layer3, layer2, layer1 (Slide Chapter 3, page 21)
    if hasattr(model, "layer4"):
        resnet_blocks = [model.layer4, model.layer3, model.layer2, model.layer1]
        for block in resnet_blocks[:n_blocks]:
            for param in block.parameters():
                param.requires_grad = True
        return

    # Case DenseNet: features has denseblock4, denseblock3, etc.
    if hasattr(model, "features"):
        # Unfreeze classifier always
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        # DenseNet specific blocks in features
        feature_blocks = [m for name, m in model.features.named_children()]
        for block in reversed(feature_blocks[- (n_blocks * 2):]):
            for param in block.parameters():
                param.requires_grad = True
        return

    # General fallback for Sequential or other models
    children = list(model.children())
    for child in children[-n_blocks:]:
        for param in child.parameters():
            param.requires_grad = True