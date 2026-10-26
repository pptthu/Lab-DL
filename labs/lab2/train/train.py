"""Module `train` (Training (+ TensorBoard)) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `train` (Huấn luyện (+ TensorBoard)). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: Any,
    ckpt_path: str,
    writer: Optional[Any] = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
) -> Dict[str, list]:
    """Training loop with validation each epoch; log scalars to TensorBoard when writer is given; keep the best checkpoint.

    Input : model, loaders, epochs, lr, device, ckpt_path, writer (SummaryWriter or None), optimizer_name, weight_decay
    Output: history dict: train_loss, val_loss, val_acc; checkpoint at ckpt_path; TensorBoard logs

    VI: Vòng lặp huấn luyện có validation mỗi epoch; ghi scalar lên TensorBoard nếu có writer; lưu checkpoint tốt nhất.
    Vào : model, loaders, epochs, lr, device, ckpt_path, writer (SummaryWriter hoặc None), optimizer_name, weight_decay
    Ra  : dict history: train_loss, val_loss, val_acc; checkpoint tại ckpt_path; log TensorBoard
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")

    device = torch.device(device)
    model.to(device)

    # 1. Setup loss and trainable parameters
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has no trainable parameters. Check if all layers are frozen.")

    opt_name_lower = optimizer_name.lower()
    if opt_name_lower == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif opt_name_lower == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}. Expected 'adam' or 'sgd'.")

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")

    # 2. Training and validation loops
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_train_loss += loss.item() * batch_size
            train_samples += batch_size

        if train_samples == 0:
            raise ValueError("train_loader yielded 0 samples.")

        train_epoch_loss = running_train_loss / train_samples
        history["train_loss"].append(float(train_epoch_loss))

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        val_samples = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                batch_size = images.size(0)
                running_val_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                correct_val += (preds == targets).sum().item()
                val_samples += batch_size

        if val_samples == 0:
            raise ValueError("val_loader yielded 0 samples.")

        val_epoch_loss = running_val_loss / val_samples
        val_epoch_acc = correct_val / val_samples

        history["val_loss"].append(float(val_epoch_loss))
        history["val_acc"].append(float(val_epoch_acc))

        # Log scalars to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_epoch_loss, epoch)
            writer.add_scalar("Loss/val", val_epoch_loss, epoch)
            writer.add_scalar("Accuracy/val", val_epoch_acc, epoch)

        # Checkpoint selection: based on best validation loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            save_checkpoint(model, ckpt_path)

    return history


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save weights to path, creating parent directory if necessary.

    Input : model, path
    Output: file saved

    VI: Lưu trọng số mô hình vào đường dẫn, tự động tạo thư mục cha nếu chưa có.
    Vào : model, path
    Ra  : file đã lưu
    """
    ckpt_dir = os.path.dirname(os.path.abspath(path))
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str, device: Any) -> nn.Module:
    """Load weights from path into model on specified device.

    Input : model, path, device
    Output: model in eval mode with weights loaded

    VI: Nạp trọng số từ đường dẫn vào mô hình trên thiết bị chỉ định, đặt ở chế độ eval.
    Vào : model, path, device
    Ra  : model ở chế độ eval với trọng số đã nạp
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    device = torch.device(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
