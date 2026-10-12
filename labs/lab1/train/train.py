"""Module `train` (Training) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `train` (Huấn luyện). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def train_model(model, train_loader, val_loader, epochs: int, lr: float, device, ckpt_path: str, optimizer_name: str = 'adam', weight_decay: float = 0.0) -> dict:
    """Training loop: zero_grad, forward, loss, backward, step; validate after each epoch; keep the best checkpoint by validation loss.

    Input : model, loaders, epochs, lr, device, ckpt_path, optimizer_name ('adam' or 'sgd'), weight_decay
    Output: history dict: train_loss, val_loss, val_acc (one value per epoch); best model saved at ckpt_path

    VI: Vòng lặp huấn luyện; validate sau mỗi epoch; lưu checkpoint tốt nhất theo validation loss.
    Vào : model, loaders, epochs, lr, device, ckpt_path, optimizer_name ('adam' hoặc 'sgd'), weight_decay
    Ra  : dict history: train_loss, val_loss, val_acc (mỗi epoch một giá trị); model tốt nhất lưu tại ckpt_path
    """
    raise NotImplementedError("TODO(Member B): implement train_model")


def save_checkpoint(model, path: str) -> None:
    """Save model weights (state_dict).

    Input : model, path
    Output: file at path

    VI: Lưu trọng số model (state_dict).
    Vào : model, path
    Ra  : file tại path
    """
    raise NotImplementedError("TODO(Member B): implement save_checkpoint")


def load_checkpoint(model, path: str, device) -> object:
    """Load weights into a freshly created model and switch to eval mode.

    Input : model (same architecture), path, device
    Output: the model with loaded weights, in eval mode

    VI: Nạp trọng số vào model mới tạo và chuyển sang chế độ eval.
    Vào : model (cùng kiến trúc), path, device
    Ra  : model đã nạp trọng số, ở chế độ eval
    """
    raise NotImplementedError("TODO(Member B): implement load_checkpoint")
