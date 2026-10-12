"""Module `evaluate` (Evaluation) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `evaluate` (Đánh giá). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def evaluate(model, loader, device) -> tuple:
    """Evaluate without gradients.

    Input : model, loader, device
    Output: (avg_loss, accuracy)

    VI: Đánh giá không tính gradient.
    Vào : model, loader, device
    Ra  : (avg_loss, accuracy)
    """
    raise NotImplementedError("TODO(Member B): implement evaluate")


def confusion_matrix(model, loader, device, num_classes: int = 10):
    """Confusion matrix.

    Input : model, loader, device
    Output: array [C,C] (rows = true, columns = predicted)

    VI: Ma trận nhầm lẫn.
    Vào : model, loader, device
    Ra  : mảng [C,C] (hàng = thật, cột = dự đoán)
    """
    raise NotImplementedError("TODO(Member B): implement confusion_matrix")


def top_confusions(cm, class_names, k: int = 5) -> list:
    """Most frequent confusions.

    Input : cm, class_names, k
    Output: list of (true_class, predicted_class, count)

    VI: Các cặp nhầm lẫn nhiều nhất.
    Vào : cm, class_names, k
    Ra  : danh sách (lớp thật, lớp dự đoán, số lần)
    """
    raise NotImplementedError("TODO(Member B): implement top_confusions")
