"""Module `model` (Model) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `model` (Mô hình). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import torch
from torch import nn

INPUT_SIZE = 28 * 28


class MLP(nn.Module):
    """Fully connected network (subclass of torch.nn.Module).

    Input : __init__(hidden_sizes=(256,128), num_classes=10, dropout=0.0); forward(x: tensor [B,1,28,28])
    Output: forward returns logits [B,10] (NO softmax: CrossEntropyLoss expects logits)

    VI: Mạng kết nối đầy đủ (kế thừa torch.nn.Module).
    Vào : __init__(hidden_sizes=(256,128), num_classes=10, dropout=0.0); forward(x: tensor [B,1,28,28])
    Ra  : forward trả logits [B,10] (KHÔNG softmax: CrossEntropyLoss cần logits)
    """

    def __init__(self, hidden_sizes=(256, 128), num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [nn.Flatten()]  # [B,1,28,28] -> [B,784]
        in_features = INPUT_SIZE
        # One block per hidden layer: Linear -> ReLU (-> Dropout when dropout > 0).
        # VI: Mỗi tầng ẩn là một khối: Linear -> ReLU (-> Dropout nếu dropout > 0).
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden
        # Last layer has no activation: it returns raw logits.
        # VI: Tầng cuối không có hàm kích hoạt: trả về logits thô.
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model) -> int:
    """Count trainable parameters.

    Input : model
    Output: int

    VI: Đếm số tham số huấn luyện được.
    Vào : model
    Ra  : int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
