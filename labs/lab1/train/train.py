"""Module `train` (Training) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `train` (Huấn luyện). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import torch
from torch import nn


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save model weights (state_dict).

    Input : model, path
    Output: file at path

    VI: Lưu trọng số model (state_dict).
    Vào : model, path
    Ra  : file tại path
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)


def load_checkpoint(model: nn.Module, path: str, device: str | torch.device) -> nn.Module:
    """Load weights into a freshly created model and switch to eval mode.

    Input : model (same architecture), path, device
    Output: the model with loaded weights, in eval mode

    VI: Nạp trọng số vào model mới tạo và chuyển sang chế độ eval.
    Vào : model (cùng kiến trúc), path, device
    Ra  : model đã nạp trọng số, ở chế độ eval
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str | torch.device,
    ckpt_path: str,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
) -> Dict[str, List[float]]:
    """Training loop: zero_grad, forward, loss, backward, step; validate after each epoch; keep the best checkpoint by validation loss.

    Input : model, loaders, epochs, lr, device, ckpt_path, optimizer_name ('adam' or 'sgd'), weight_decay
    Output: history dict: train_loss, val_loss, val_acc (one value per epoch); best model saved at ckpt_path

    VI: Vòng lặp huấn luyện; validate sau mỗi epoch; lưu checkpoint tốt nhất theo validation loss.
    Vào : model, loaders, epochs, lr, device, ckpt_path, optimizer_name ('adam' hoặc 'sgd'), weight_decay
    Ra  : dict history: train_loss, val_loss, val_acc (mỗi epoch một giá trị); model tốt nhất lưu tại ckpt_path
    """
    opt_lower = optimizer_name.strip().lower()
    if opt_lower == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_lower == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unsupported optimizer_name: '{optimizer_name}'. Expected 'adam' or 'sgd'."
        )

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")

    for _ in range(epochs):
        # --- Training phase ---
        model.train()
        running_train_loss = 0.0
        n_train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.size(0)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_size
            n_train_samples += batch_size

        epoch_train_loss = running_train_loss / n_train_samples if n_train_samples > 0 else 0.0

        # --- Validation phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        n_val_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                batch_size = x_batch.size(0)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                running_val_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                running_val_correct += (preds == y_batch).sum().item()
                n_val_samples += batch_size

        epoch_val_loss = running_val_loss / n_val_samples if n_val_samples > 0 else 0.0
        epoch_val_acc = running_val_correct / n_val_samples if n_val_samples > 0 else 0.0

        history["train_loss"].append(float(epoch_train_loss))
        history["val_loss"].append(float(epoch_val_loss))
        history["val_acc"].append(float(epoch_val_acc))

        # Checkpoint: save model when validation loss improves
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(model, ckpt_path)

    return history
