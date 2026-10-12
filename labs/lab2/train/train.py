"""Module `train` (Training (+ TensorBoard)) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `train` (Huấn luyện (+ TensorBoard)). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def train_model(model, train_loader, val_loader, epochs: int, lr: float, device, ckpt_path: str, writer=None, optimizer_name: str = 'adam', weight_decay: float = 0.0) -> dict:
    """Training loop with validation each epoch; log scalars to TensorBoard when writer is given; keep the best checkpoint.

    Input : model, loaders, epochs, lr, device, ckpt_path, writer (SummaryWriter or None), optimizer_name, weight_decay
    Output: history dict: train_loss, val_loss, val_acc; checkpoint at ckpt_path; TensorBoard logs

    VI: Vòng lặp huấn luyện có validation mỗi epoch; ghi scalar lên TensorBoard nếu có writer; lưu checkpoint tốt nhất.
    Vào : model, loaders, epochs, lr, device, ckpt_path, writer (SummaryWriter hoặc None), optimizer_name, weight_decay
    Ra  : dict history: train_loss, val_loss, val_acc; checkpoint tại ckpt_path; log TensorBoard
    """
    raise NotImplementedError("TODO(Member B): implement train_model")


def save_checkpoint(model, path: str) -> None:
    """Save and load weights.

    Input : model, path (, device)
    Output: file / model in eval mode

    VI: Lưu và nạp trọng số.
    Vào : model, path (, device)
    Ra  : file / model ở chế độ eval
    """
    raise NotImplementedError("TODO(Member B): implement save_checkpoint")


def load_checkpoint(model, path: str, device):
    """Save and load weights.

    Input : model, path (, device)
    Output: file / model in eval mode

    VI: Lưu và nạp trọng số.
    Vào : model, path (, device)
    Ra  : file / model ở chế độ eval
    """
    raise NotImplementedError("TODO(Member B): implement load_checkpoint")
