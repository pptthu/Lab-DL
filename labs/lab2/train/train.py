"""Module `train` (Training (+ TensorBoard)) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `train` (Huấn luyện (+ TensorBoard)). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.

Design notes
------------
* train_model drives the epoch loop, calls _train_epoch and _val_epoch.
* Best checkpoint is selected on val_acc (higher is better).
* SummaryWriter is optional; pass writer=None to disable TensorBoard logging.
* Optimizer: 'adam' (default) or 'sgd'.
* No mixed precision – keeps compatibility with CPU-only environments.
* Early stopping: triggered when val_acc does not improve for `patience` epochs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    ckpt_path: str,
    writer=None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    patience: int = 10,
    lr_step_size: int = 7,
    lr_gamma: float = 0.1,
) -> Dict:
    """Training loop with per-epoch validation and best-checkpoint saving.

    Parameters
    ----------
    model          : nn.Module – model to train (modified in-place).
    train_loader   : DataLoader for training batches.
    val_loader     : DataLoader for validation batches.
    epochs         : number of training epochs.
    lr             : initial learning rate for trainable parameters.
    device         : torch.device for computation.
    ckpt_path      : path where the best checkpoint is saved.
    writer         : torch.utils.tensorboard.SummaryWriter or None.
    optimizer_name : 'adam' or 'sgd'.
    weight_decay   : L2 regularisation coefficient.
    patience       : early-stopping patience (epochs without improvement).
    lr_step_size   : StepLR step size (epochs).
    lr_gamma       : StepLR decay factor.

    Returns
    -------
    dict with keys:
        train_loss : List[float] – per-epoch mean training loss.
        val_loss   : List[float] – per-epoch mean validation loss.
        val_acc    : List[float] – per-epoch validation accuracy (0–1).
        best_val_acc : float – best validation accuracy achieved.
        epochs_run : int – number of epochs actually completed.
        train_seconds : float – total wall-clock training time.

    VI: Vòng lặp huấn luyện có validation, lưu checkpoint tốt nhất.
    """
    model.to(device)

    # Only pass parameters that require gradients to the optimizer.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(
            "train_model: no trainable parameters found.  "
            "Check freeze_backbone configuration."
        )

    optimizer = _build_optimizer(optimizer_name, trainable_params, lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )
    criterion = nn.CrossEntropyLoss()

    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss":   [],
        "val_acc":    [],
        "lr":         [],
    }
    best_val_acc = 0.0
    no_improve   = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        # --- training phase ---
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)

        # --- validation phase ---
        val_loss, val_acc = _val_epoch(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t_epoch
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # --- TensorBoard ---
        if writer is not None:
            writer.add_scalar("Loss/train",    train_loss, epoch)
            writer.add_scalar("Loss/val",      val_loss,   epoch)
            writer.add_scalar("Accuracy/val",  val_acc,    epoch)
            writer.add_scalar("LR",            current_lr, epoch)

        # --- best checkpoint ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            _save_full_checkpoint(
                model, optimizer, scheduler, epoch, best_val_acc, ckpt_path
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {epoch} epochs (no improvement for {patience} epochs).")
                break

    total_seconds = time.time() - t0
    history["best_val_acc"]   = best_val_acc
    history["epochs_run"]     = epoch
    history["train_seconds"]  = total_seconds

    print(f"  Training complete. Best val_acc={best_val_acc:.4f} | "
          f"total time={total_seconds:.1f}s")

    return history


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save the model weights (state_dict) to disk and set model to eval mode.

    Input : model – nn.Module.
            path  – output file path (.pt).
    Output: .pt file written to disk.

    VI: Lưu trọng số model ra file và chuyển sang eval mode.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    inner = model.inner if hasattr(model, "inner") else model
    torch.save(inner.state_dict(), path)
    model.eval()


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load weights from a checkpoint file into model.

    Input : model  – nn.Module (must have compatible architecture).
            path   – path to the .pt checkpoint written by save_checkpoint.
            device – target device.
    Output: model in eval mode with loaded weights.

    VI: Nạp trọng số từ file checkpoint vào model, chuyển sang eval mode.
    """
    inner = model.inner if hasattr(model, "inner") else model
    state = torch.load(path, map_location=device, weights_only=True)
    # Support both raw state_dict and full checkpoint dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    inner.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose 'adam' or 'sgd'.")


def _train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


def _val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def _save_full_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, path):
    """Save full training state for potential resume."""
    inner = model.inner if hasattr(model, "inner") else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": inner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, path)
