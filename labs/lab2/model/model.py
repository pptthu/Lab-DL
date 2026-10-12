"""Module `model` (Model (backbone + header)) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `model` (Mô hình (backbone + header)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def build_model(name: str, num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True):
    """Load a torchvision pre-trained model (e.g. resnet18, vgg16, densenet121) and replace the final classification layer. Note: modern torchvision uses weights=..., pretrained=True is deprecated.

    Input : name, num_classes, freeze_backbone, pretrained
    Output: nn.Module that outputs logits [B,num_classes]; backbone frozen if requested

    VI: Nạp model pre-trained của torchvision (resnet18, vgg16, densenet121...) và thay lớp phân loại cuối. Lưu ý: torchvision mới dùng weights=..., pretrained=True đã cũ.
    Vào : name, num_classes, freeze_backbone, pretrained
    Ra  : nn.Module xuất logits [B,num_classes]; backbone bị đóng băng nếu được yêu cầu
    """
    raise NotImplementedError("TODO(Member A): implement build_model")


def describe_model(model) -> dict:
    """Print the structure and count parameters.

    Input : model
    Output: dict: total_params, trainable_params, layer names

    VI: In cấu trúc và đếm tham số.
    Vào : model
    Ra  : dict: total_params, trainable_params, tên các lớp
    """
    raise NotImplementedError("TODO(Member A): implement describe_model")


def unfreeze_last_blocks(model, n_blocks: int) -> None:
    """Unfreeze the last blocks to fine-tune.

    Input : model, n_blocks
    Output: model updated in place

    VI: Mở khóa vài block cuối để fine-tune.
    Vào : model, n_blocks
    Ra  : model được cập nhật tại chỗ
    """
    raise NotImplementedError("TODO(Member A): implement unfreeze_last_blocks")
