"""Module `visualize` (Visualization) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `visualize` (Trực quan hóa). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def plot_loss_curves(history: dict, out_path: str) -> None:
    """Plot training vs validation loss.

    Input : history from train_model; out_path (.png)
    Output: PNG file

    VI: Vẽ loss train và validation.
    Vào : history từ train_model; out_path (.png)
    Ra  : file PNG
    """
    raise NotImplementedError("TODO(Member B): implement plot_loss_curves")


def show_predictions(model, loader, class_names, device, out_path: str, n: int = 16) -> None:
    """Grid of images with predicted vs actual labels (wrong ones highlighted).

    Input : model, loader, class_names (10 names), device, out_path, n
    Output: PNG file

    VI: Lưới ảnh có nhãn dự đoán và nhãn thật (ảnh sai được tô nổi bật).
    Vào : model, loader, class_names (10 tên), device, out_path, n
    Ra  : file PNG
    """
    raise NotImplementedError("TODO(Member B): implement show_predictions")


def plot_confusion_matrix(cm, class_names, out_path: str) -> None:
    """Heatmap of the confusion matrix.

    Input : cm, class_names, out_path
    Output: PNG file

    VI: Heatmap của ma trận nhầm lẫn.
    Vào : cm, class_names, out_path
    Ra  : file PNG
    """
    raise NotImplementedError("TODO(Member B): implement plot_confusion_matrix")
