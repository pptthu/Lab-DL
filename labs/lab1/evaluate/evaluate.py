"""Module `evaluate` (Evaluation) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `evaluate` (Đánh giá). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def evaluate(model, loader, device) -> tuple:
    """Evaluate without gradients (model.eval + torch.no_grad).

    Input : model, loader, device
    Output: (avg_loss: float, accuracy: float in 0-1)

    VI: Đánh giá không tính gradient (model.eval + torch.no_grad).
    Vào : model, loader, device
    Ra  : (avg_loss: float, accuracy: float trong 0-1)
    """
    raise NotImplementedError("TODO(Member B): implement evaluate")


def confusion_matrix(model, loader, device, num_classes: int = 10):
    """Build the confusion matrix.

    Input : model, loader, device, num_classes
    Output: array [10,10]; rows = true class, columns = predicted class

    VI: Tạo ma trận nhầm lẫn.
    Vào : model, loader, device, num_classes
    Ra  : mảng [10,10]; hàng = lớp thật, cột = lớp dự đoán
    """
    raise NotImplementedError("TODO(Member B): implement confusion_matrix")


def per_class_accuracy(cm) -> dict:
    """Accuracy of each class.

    Input : cm: confusion matrix
    Output: dict: class index -> accuracy

    VI: Độ chính xác từng lớp.
    Vào : cm: ma trận nhầm lẫn
    Ra  : dict: chỉ số lớp -> độ chính xác
    """
    raise NotImplementedError("TODO(Member B): implement per_class_accuracy")
