"""Module `experiment` (Experiments) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `experiment` (Thí nghiệm). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from labs.lab2.train.train import train_model


def run_comparison(
    model_names: List[str],
    hparam_grid: List[Dict[str, Any]],
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    device: Any,
    log_root: str,
    build_model_fn: Optional[Any] = None,
    unfreeze_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Train/fine-tune several pre-trained models and hyperparameter settings; one TensorBoard run per setting.

    Input : model_names (e.g. ['resnet18','vgg16']), hparam_grid (list of dicts: lr, batch_size, freeze...), loaders, device, log_root
    Output: list of result dicts: model, hparams, best_val_acc, train_seconds; TensorBoard runs under log_root

    VI: Huấn luyện/fine-tune nhiều model pre-trained và nhiều bộ siêu tham số; mỗi bộ một run TensorBoard.
    Vào : model_names (vd ['resnet18','vgg16']), hparam_grid (danh sách dict: lr, batch_size, freeze...), loaders, device, log_root
    Ra  : danh sách dict kết quả: model, hparams, best_val_acc, train_seconds; run TensorBoard trong log_root
    """
    # Dynamic import of Member A model builder if not provided
    if build_model_fn is None:
        try:
            from labs.lab2.model.model import build_model as imported_build_model
            build_model_fn = imported_build_model
        except Exception as exc:
            raise ImportError("Could not import build_model from labs.lab2.model.model. Pass build_model_fn explicitly.") from exc

    if unfreeze_fn is None:
        try:
            from labs.lab2.model.model import unfreeze_last_blocks as imported_unfreeze
            unfreeze_fn = imported_unfreeze
        except Exception:
            unfreeze_fn = None

    train_loader, val_loader, _ = loaders

    results: List[Dict[str, Any]] = []
    os.makedirs(log_root, exist_ok=True)

    for model_name in model_names:
        for idx, hparams in enumerate(hparam_grid):
            # Extract hyperparameters with sensible defaults
            lr = float(hparams.get("lr", 1e-3))
            epochs = int(hparams.get("epochs", 1))
            optimizer_name = str(hparams.get("optimizer_name", "adam"))
            weight_decay = float(hparams.get("weight_decay", 0.0))
            freeze_backbone = bool(hparams.get("freeze_backbone", True))
            pretrained = bool(hparams.get("pretrained", True))
            unfreeze_blocks = int(hparams.get("unfreeze_blocks", 0))

            run_name = f"{model_name}_run{idx}_lr{lr}_freeze{freeze_backbone}"
            run_log_dir = os.path.join(log_root, run_name)
            writer = SummaryWriter(log_dir=run_log_dir)

            ckpt_path = os.path.join(log_root, f"{run_name}_best.pt")

            # 1. Build model according to configuration
            model = build_model_fn(
                name=model_name,
                num_classes=10,
                freeze_backbone=freeze_backbone,
                pretrained=pretrained,
            )

            # If fine-tuning unfreezes blocks:
            if unfreeze_blocks > 0 and unfreeze_fn is not None:
                unfreeze_fn(model, unfreeze_blocks)

            start_time = time.perf_counter()
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                device=device,
                ckpt_path=ckpt_path,
                writer=writer,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
            )
            elapsed_seconds = time.perf_counter() - start_time
            writer.close()

            val_acc_list = history.get("val_acc", [0.0])
            best_val_acc = max(val_acc_list) if val_acc_list else 0.0

            run_result: Dict[str, Any] = {
                "model": model_name,
                "hparams": dict(hparams),
                "best_val_acc": float(best_val_acc),
                "train_seconds": float(elapsed_seconds),
                "checkpoint": str(ckpt_path),
            }
            results.append(run_result)

    return results


def save_results(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Save the comparison table.

    Input : results, csv_path
    Output: CSV file

    VI: Lưu bảng so sánh.
    Vào : results, csv_path
    Ra  : file CSV
    """
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    if not results:
        # Write empty CSV with standard headers
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "best_val_acc", "train_seconds", "checkpoint"])
        return

    # Extract all unique hyperparameter keys to flatten them
    hparam_keys = set()
    for r in results:
        hp = r.get("hparams", {})
        if isinstance(hp, dict):
            hparam_keys.update(hp.keys())
    sorted_hp_keys = sorted(list(hparam_keys))

    headers = ["model"] + [f"hp_{k}" for k in sorted_hp_keys] + ["best_val_acc", "train_seconds", "checkpoint"]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            hp = r.get("hparams", {})
            row = [r.get("model", "")]
            for k in sorted_hp_keys:
                row.append(hp.get(k, ""))
            row.append(f"{r.get('best_val_acc', 0.0):.4f}")
            row.append(f"{r.get('train_seconds', 0.0):.2f}")
            row.append(r.get("checkpoint", ""))
            writer.writerow(row)
