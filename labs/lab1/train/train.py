"""Module `train` (Training) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `train` (Huấn luyện). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch import nn

from shared.paths import REPO_ROOT


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
    # Save only the weights (state_dict), not the whole model object.
    # VI: Chỉ lưu trọng số (state_dict), không lưu cả đối tượng model.
    torch.save(model.state_dict(), ckpt_path)


def load_checkpoint(model: nn.Module, path: str, device: str | torch.device) -> nn.Module:
    """Load weights into a freshly created model and switch to eval mode.

    Input : model (same architecture), path, device
    Output: the model with loaded weights, in eval mode

    VI: Nạp trọng số vào model mới tạo và chuyển sang chế độ eval.
    Vào : model (cùng kiến trúc), path, device
    Ra  : model đã nạp trọng số, ở chế độ eval
    """
    # weights_only=True loads tensors only, which is safer than loading arbitrary objects.
    # VI: weights_only=True chỉ nạp tensor, an toàn hơn nạp đối tượng bất kỳ.
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


HISTORY_COLUMNS = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s")


def _now_iso() -> str:
    # Giờ địa phương kèm múi giờ, ví dụ 2026-11-14T02:10:00+07:00.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rel(path) -> str:
    # Ghi đường dẫn tương đối từ gốc repo để log đọc được trên máy khác.
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _header_lines(run_name: str, device, run_info: Dict[str, Any], epochs: int,
                  patience: Optional[int], min_delta: float, max_batches: Optional[int]) -> List[str]:
    # Phần đầu log theo docs/templates/LOG_TEMPLATE_v1.md §2.
    cfg = run_info.get("config", {})
    hidden = list(cfg.get("hidden_sizes", []))
    stop = (f"monitor=val_loss patience={patience} min_delta={min_delta}"
            if patience is not None else "none (chạy đủ max_epochs)")
    lines = [
        f"=== RUN START {_now_iso()} ===",
        f"run: {run_name} | seed: {run_info.get('seed')} | device: {device}",
        (f"config: hidden_sizes={hidden} lr={cfg.get('lr')} optimizer={cfg.get('optimizer_name')} "
         f"dropout={cfg.get('dropout')} weight_decay={cfg.get('weight_decay')} "
         f"batch_size={run_info.get('batch_size')} max_epochs={epochs}"),
        f"early_stopping: {stop}",
        f"architecture: {run_info.get('architecture')} | params: {run_info.get('params', 0):,}",
        f"data: {run_info.get('data')} | split: {run_info.get('split')}",
    ]
    if max_batches is not None:
        lines.append(f"smoke: max_batches={max_batches} (số liệu không dùng để báo cáo)")
    lines.append(",".join(HISTORY_COLUMNS))
    return lines


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
    patience: Optional[int] = None,
    min_delta: float = 0.0,
    last_ckpt_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    run_name: str = "run",
    run_info: Optional[Dict[str, Any]] = None,
    max_batches: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Training loop: zero_grad, forward, loss, backward, step; validate after each epoch; keep the best checkpoint by validation loss.

    Input : model, loaders, epochs, lr, device, ckpt_path, optimizer_name ('adam' or 'sgd'), weight_decay
    Output: history dict: train_loss, val_loss, val_acc (one value per epoch); best model saved at ckpt_path

    VI: Vòng lặp huấn luyện; validate sau mỗi epoch; lưu checkpoint tốt nhất theo validation loss.
    Vào : model, loaders, epochs (= số epoch tối đa), lr, device, ckpt_path (= `_best.pt`), optimizer_name ('adam' hoặc 'sgd'), weight_decay
          patience, min_delta: quy tắc dừng theo val_loss (patience=None thì chạy đủ epochs)
          last_ckpt_path: nơi lưu `_last.pt` (None thì không lưu)
          log_dir, run_name, run_info: có log_dir thì ghi 3 file `<run_name>.log`, `_history.csv`, `_config.json`
          max_batches: chỉ dùng cho smoke run, giới hạn số batch mỗi epoch
    Ra  : dict history gồm các list theo epoch (train_loss, train_acc, val_loss, val_acc, lr, time_s)
          và best_epoch, stopped_epoch, stop_reason, best_val_loss, các đường dẫn file
    """
    # Choose the optimizer; SGD always uses momentum 0.9.
    # VI: Chọn optimizer; SGD luôn dùng momentum 0,9.
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
    # CrossEntropyLoss takes raw logits: it applies log-softmax itself.
    # VI: CrossEntropyLoss nhận logits thô: nó tự làm log-softmax bên trong.
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, Any] = {k: [] for k in HISTORY_COLUMNS if k != "epoch"}

    # Chuẩn bị 3 file log nếu có log_dir.
    log_path = history_path = config_path = None
    log_lines: List[str] = []
    if log_dir is not None:
        out = Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / f"{run_name}.log"
        history_path = out / f"{run_name}_history.csv"
        config_path = out / f"{run_name}_config.json"
        info = dict(run_info or {})
        log_lines = _header_lines(run_name, device, info, epochs, patience, min_delta, max_batches)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        history_path.write_text(",".join(HISTORY_COLUMNS) + "\n", encoding="utf-8")
        config = {
            "run": run_name,
            "started_at": _now_iso(),
            "device": str(device),
            **info,
            "max_epochs": epochs,
            "early_stopping": {"monitor": "val_loss", "patience": patience, "min_delta": min_delta},
            "max_batches": max_batches,
            "best_ckpt": _rel(ckpt_path),
            "last_ckpt": _rel(last_ckpt_path) if last_ckpt_path else None,
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if verbose:
            print("\n".join(log_lines[:-1]))

    # Lowest validation loss so far; starts at infinity so the first epoch is always saved.
    # VI: Val loss thấp nhất tới giờ; bắt đầu từ vô cực nên epoch đầu luôn được lưu.
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    stop_reason = "max_epochs"
    epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        # --- Training phase: train() turns dropout on ---
        # VI: --- Pha huấn luyện: train() bật dropout ---
        model.train()
        running_train_loss = 0.0
        running_train_correct = 0
        n_train_samples = 0

        for b, (x_batch, y_batch) in enumerate(train_loader):
            if max_batches is not None and b >= max_batches:
                break
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.size(0)

            # The 5 steps: clear old gradients (PyTorch adds them up), forward, loss, backward (gradients), update weights.
            # VI: 5 bước: xóa gradient cũ (PyTorch cộng dồn), forward, tính loss, backward (tính gradient), cập nhật trọng số.
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            # Weight by batch size so a smaller last batch counts exactly.
            # VI: Nhân với số ảnh của batch để batch cuối nhỏ hơn vẫn được tính đúng.
            running_train_loss += loss.item() * batch_size
            # train_acc đo ngay trong lúc train (dropout đang bật), chỉ để so với val_acc.
            running_train_correct += (torch.argmax(logits, dim=1) == y_batch).sum().item()
            n_train_samples += batch_size

        epoch_train_loss = running_train_loss / n_train_samples if n_train_samples > 0 else 0.0
        epoch_train_acc = running_train_correct / n_train_samples if n_train_samples > 0 else 0.0

        # --- Validation phase: eval() turns dropout off ---
        # VI: --- Pha đánh giá: eval() tắt dropout ---
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        n_val_samples = 0

        # No gradients during validation: faster and uses less memory.
        # VI: Không tính gradient khi đánh giá: nhanh hơn, ít bộ nhớ hơn.
        with torch.no_grad():
            for b, (x_batch, y_batch) in enumerate(val_loader):
                if max_batches is not None and b >= max_batches:
                    break
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                batch_size = x_batch.size(0)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                running_val_loss += loss.item() * batch_size
                # Predicted class = index of the largest logit.
                # VI: Lớp dự đoán = vị trí có logit lớn nhất.
                preds = torch.argmax(logits, dim=1)
                running_val_correct += (preds == y_batch).sum().item()
                n_val_samples += batch_size

        epoch_val_loss = running_val_loss / n_val_samples if n_val_samples > 0 else 0.0
        epoch_val_acc = running_val_correct / n_val_samples if n_val_samples > 0 else 0.0
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.perf_counter() - t0

        history["train_loss"].append(float(epoch_train_loss))
        history["train_acc"].append(float(epoch_train_acc))
        history["val_loss"].append(float(epoch_val_loss))
        history["val_acc"].append(float(epoch_val_acc))
        history["lr"].append(epoch_lr)
        history["time_s"].append(float(epoch_time))

        # Checkpoint: save model when validation loss improves (the test set is never used here)
        # VI: Lưu checkpoint khi validation loss giảm ít nhất min_delta (ở đây không bao giờ dùng tập test)
        if best_val_loss - epoch_val_loss > min_delta:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(model, ckpt_path)
        else:
            bad_epochs += 1

        # Ghi ngay dòng epoch vào log và _history.csv, dừng giữa chừng vẫn còn bằng chứng.
        row = (f"{epoch},{epoch_train_loss:.6f},{epoch_train_acc:.6f},{epoch_val_loss:.6f},"
               f"{epoch_val_acc:.6f},{epoch_lr:g},{epoch_time:.2f}")
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(row + "\n")
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(row + "\n")
        if verbose:
            print(row)

        # Quy tắc dừng: val_loss không giảm đủ min_delta trong `patience` epoch liên tiếp.
        if patience is not None and bad_epochs >= patience:
            stop_reason = "early_stopping"
            break

    stopped_epoch = epoch
    if last_ckpt_path is not None:
        save_checkpoint(model, last_ckpt_path)

    history.update({
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "best_val_loss": float(best_val_loss),
        "best_ckpt": str(ckpt_path),
        "last_ckpt": str(last_ckpt_path) if last_ckpt_path else None,
        "log_path": str(log_path) if log_path else None,
        "history_path": str(history_path) if history_path else None,
        "config_path": str(config_path) if config_path else None,
    })

    if log_path is not None:
        end_lines = [
            f"=== RUN END {_now_iso()} ===",
            (f"best_epoch: {best_epoch} | stopped_epoch: {stopped_epoch} | stop_reason: {stop_reason} | "
             f"best_val_loss: {best_val_loss:.6f} | best_val_acc: {history['val_acc'][best_epoch - 1]:.6f}"),
            f"best_ckpt: {_rel(ckpt_path)} | last_ckpt: {_rel(last_ckpt_path) if last_ckpt_path else '-'}",
        ]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(end_lines) + "\n")
        if verbose:
            print("\n".join(end_lines))

    return history
