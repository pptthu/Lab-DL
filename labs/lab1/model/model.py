"""Module `model` (Model) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `model` (Mô hình). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

class MLP:  # TODO: inherit from the right base class (see docstring)
    """Fully connected network (subclass of torch.nn.Module).

    Input : __init__(hidden_sizes=(256,128), num_classes=10, dropout=0.0); forward(x: tensor [B,1,28,28])
    Output: forward returns logits [B,10] (NO softmax: CrossEntropyLoss expects logits)

    VI: Mạng kết nối đầy đủ (kế thừa torch.nn.Module).
    Vào : __init__(hidden_sizes=(256,128), num_classes=10, dropout=0.0); forward(x: tensor [B,1,28,28])
    Ra  : forward trả logits [B,10] (KHÔNG softmax: CrossEntropyLoss cần logits)
    """
    # TODO(Member A): implement MLP


def count_parameters(model) -> int:
    """Count trainable parameters.

    Input : model
    Output: int

    VI: Đếm số tham số huấn luyện được.
    Vào : model
    Ra  : int
    """
    raise NotImplementedError("TODO(Member A): implement count_parameters")
