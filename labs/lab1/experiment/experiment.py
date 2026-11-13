"""Module `experiment` (Experiments) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `experiment` (Thí nghiệm). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from torch import nn

from shared.paths import output_dir
from shared.seed import set_seed
from labs.lab1.model.model import MLP, count_parameters
from labs.lab1.train.train import train_model

# 5 cấu hình của desc lab (docs/lab1_description_v1.md), mỗi cấu hình khác baseline đúng một yếu tố.
# SGD luôn có momentum 0.9 (train/train.py).
BASELINE_CONFIG: Dict[str, Any] = {
    "hidden_sizes": (256, 128), "lr": 1e-3, "dropout": 0.0,
    "optimizer_name": "adam", "weight_decay": 0.0, "epochs": 30, "seed": 42,
}
VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "softmax_regression": {"hidden_sizes": ()},
    "larger_network": {"hidden_sizes": (512, 256, 128)},
    "sgd_lr_0.01": {"optimizer_name": "sgd", "lr": 0.01},
    "dropout_0.2": {"dropout": 0.2},
}


def make_configs(epochs: int = 30, seed: int = 42) -> List[Dict[str, Any]]:
    """Tạo danh sách 5 config theo thứ tự của desc lab.

    Vào : epochs (số epoch tối đa), seed
    Ra  : list dict, mỗi dict có thêm khoá "name"
    """
    return [{**BASELINE_CONFIG, **change, "epochs": epochs, "seed": seed, "name": name}
            for name, change in VARIANTS.items()]


def build_model(cfg: Dict[str, Any], num_classes: int = 10) -> nn.Module:
    """Create the MLP of a config.

    Input : cfg; num_classes
    Output: new, untrained model

    VI: Tạo MLP của một config.
    Vào : cfg; num_classes
    Ra  : model mới, chưa huấn luyện
    """
    return MLP(
        hidden_sizes=tuple(cfg.get("hidden_sizes", (256, 128))),
        num_classes=num_classes,
        dropout=float(cfg.get("dropout", 0.0)),
    )


def describe_architecture(cfg: Dict[str, Any], num_classes: int = 10) -> str:
    """Chuỗi mô tả kiến trúc cho log, ví dụ 'MLP 784-256-128-10 (ReLU)'."""
    sizes = [784, *cfg.get("hidden_sizes", (256, 128)), num_classes]
    text = "MLP " + "-".join(str(s) for s in sizes)
    if cfg.get("hidden_sizes", (256, 128)):
        text += " (ReLU)"
    if float(cfg.get("dropout", 0.0)) > 0:
        text += f" + dropout {cfg['dropout']}"
    return text


def run_experiments(
    configs: List[Dict[str, Any]],
    loaders: Tuple[Any, Any, Any],
    device: str,
    ckpt_prefix: str = "exp_run",
    log_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    patience: Optional[int] = None,
    min_delta: float = 0.0,
    max_batches: Optional[int] = None,
    run_info: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Train one model per config (e.g. hidden sizes, lr, dropout, optimizer) and compare on VALIDATION data.

    Input : configs: list of dicts; loaders: (train, val, test); device; ckpt_prefix: checkpoint file name prefix
    Output: list of result dicts: config, best_val_acc, best_val_loss, best_epoch, final_val_loss, n_params, train_seconds, ckpt_path, history

    VI: Huấn luyện mỗi config một model và so sánh trên VALIDATION (không dùng tập test).
    Vào : configs; loaders: (train, val, test); device; ckpt_prefix (dùng khi không có log_dir)
          log_dir: có thì mỗi run ghi 3 file log, tên `<ts>_<name>`; checkpoint `<ts>_<name>_best.pt`, `_last.pt`
          models_dir: thư mục checkpoint (mặc định outputs/models); patience, min_delta, max_batches: chuyển cho train_model
          run_info: thông tin chung cho phần đầu log (batch_size, data, split)
    Ra  : list dict kết quả: config, run_name, best_val_acc, best_val_loss, best_train_acc, best_epoch, stopped_epoch,
          stop_reason, final_val_loss, n_params, train_seconds, ckpt_path, last_ckpt_path, log/history/config path, history
    """
    train_loader, val_loader, _test_loader = loaders
    results: List[Dict[str, Any]] = []

    models_path = Path(models_dir) if models_dir is not None else output_dir("lab1", "models")

    for idx, cfg in enumerate(configs):
        # Same seed -> same initial weights and same shuffling, so the runs can be repeated.
        # VI: Cùng seed -> cùng trọng số ban đầu và cùng thứ tự xáo, nên chạy lại ra đúng kết quả.
        seed = cfg.get("seed", 42)
        set_seed(seed)

        lr = float(cfg.get("lr", 1e-3))
        optimizer_name = str(cfg.get("optimizer_name", "adam"))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        epochs = int(cfg.get("epochs", 5))

        model = build_model(cfg)
        n_params = count_parameters(model)

        # Có log_dir: đặt tên theo giờ bắt đầu + tên cấu hình để không ghi đè lần chạy cũ.
        if log_dir is not None:
            run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.get('name', f'run{idx}')}"
            ckpt_path = str(models_path / f"{run_name}_best.pt")
            last_ckpt_path: Optional[str] = str(models_path / f"{run_name}_last.pt")
        else:
            run_name = f"{ckpt_prefix}_{idx}"
            ckpt_path = str(models_path / f"{ckpt_prefix}_{idx}.pt")
            last_ckpt_path = None

        info = {
            **(run_info or {}),
            "seed": seed,
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "architecture": describe_architecture(cfg),
            "params": n_params,
        }

        t0 = time.perf_counter()
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            ckpt_path=ckpt_path,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            last_ckpt_path=last_ckpt_path,
            log_dir=log_dir,
            run_name=run_name,
            run_info=info,
            max_batches=max_batches,
            verbose=verbose,
        )
        train_seconds = float(time.perf_counter() - t0)

        val_accs = history.get("val_acc", [])
        val_losses = history.get("val_loss", [])
        train_accs = history.get("train_acc", [])
        # Report the epoch whose checkpoint was kept (lowest val loss).
        # VI: Báo cáo đúng epoch được lưu checkpoint (val loss thấp nhất).
        best_idx = val_losses.index(min(val_losses)) if val_losses else 0
        if history.get("best_epoch"):
            best_idx = int(history["best_epoch"]) - 1
        best_val_acc = float(val_accs[best_idx]) if val_accs else 0.0
        best_val_loss = float(val_losses[best_idx]) if val_losses else 0.0
        best_train_acc = float(train_accs[best_idx]) if train_accs else None

        results.append({
            "config": cfg,
            "run_name": run_name,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_train_acc": best_train_acc,
            "best_epoch": best_idx + 1,
            "stopped_epoch": int(history.get("stopped_epoch", len(val_losses))),
            "stop_reason": history.get("stop_reason", "max_epochs"),
            "final_val_loss": float(val_losses[-1]) if val_losses else 0.0,
            "n_params": n_params,
            "train_seconds": train_seconds,
            "ckpt_path": ckpt_path,
            "last_ckpt_path": last_ckpt_path,
            "log_path": history.get("log_path"),
            "history_path": history.get("history_path"),
            "config_path": history.get("config_path"),
            # Per-epoch curves, kept as evidence for the plots.
            # VI: Đường cong theo từng epoch, giữ làm bằng chứng cho các hình.
            "history": history,
        })

    return results


def save_results(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Save the comparison table.

    Input : results, csv_path
    Output: CSV file

    VI: Lưu bảng so sánh (mỗi run một dòng).
    Vào : results, csv_path
    Ra  : file CSV; có cột train_minus_val_acc = train_acc − val_acc ở best_epoch (chỉ để báo cáo)
    """
    rows = []
    for r in results:
        row: Dict[str, Any] = {}
        cfg = r.get("config", {})
        for k, v in cfg.items():
            if isinstance(v, (tuple, list)):
                row[k] = "-".join(str(x) for x in v) or "none"
            else:
                row[k] = v
        row["run_name"] = r.get("run_name")
        row["n_params"] = r.get("n_params")
        row["best_epoch"] = r.get("best_epoch")
        row["stopped_epoch"] = r.get("stopped_epoch")
        row["stop_reason"] = r.get("stop_reason")
        row["best_val_loss"] = r.get("best_val_loss")
        row["best_val_acc"] = r.get("best_val_acc")
        row["best_train_acc"] = r.get("best_train_acc")
        tr, va = r.get("best_train_acc"), r.get("best_val_acc")
        row["train_minus_val_acc"] = (tr - va) if tr is not None and va is not None else None
        row["final_val_loss"] = r.get("final_val_loss")
        row["train_seconds"] = r.get("train_seconds")
        row["ckpt_path"] = r.get("ckpt_path")
        row["log_path"] = r.get("log_path")
        rows.append(row)

    df = pd.DataFrame(rows)
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def save_history(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Save the per-epoch history of every run, one row per run and epoch.

    Input : results from run_experiments (each with config name, seed and history), csv_path
    Output: CSV file with columns name, seed, epoch, train_loss, val_loss, val_acc (+ train_acc if present)

    VI: Gộp lịch sử theo epoch của mọi lần chạy vào một file, mỗi dòng là một lần chạy và một epoch.
    Vào : results từ run_experiments, csv_path
    Ra  : file CSV gồm các cột name, seed, epoch, train_loss, val_loss, val_acc (+ train_acc nếu có)
    """
    rows = []
    has_train_acc = all(r.get("history", {}).get("train_acc") for r in results) and bool(results)
    for r in results:
        cfg = r.get("config", {})
        history = r.get("history", {})
        for i, val_loss in enumerate(history.get("val_loss", [])):
            row = {
                "name": cfg.get("name"),
                "seed": cfg.get("seed"),
                "epoch": i + 1,
                "train_loss": history["train_loss"][i],
                "val_loss": val_loss,
                "val_acc": history["val_acc"][i],
            }
            if has_train_acc:
                row["train_acc"] = history["train_acc"][i]
            rows.append(row)

    columns = ["name", "seed", "epoch", "train_loss", "val_loss", "val_acc"] + (["train_acc"] if has_train_acc else [])
    df = pd.DataFrame(rows, columns=columns)
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def select_config(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Chọn cấu hình theo val_acc tại best_epoch (bằng nhau thì lấy val_loss thấp hơn). Không dùng tập test.

    Vào : results từ run_experiments
    Ra  : dict: name, run_name, index, best_val_acc, best_val_loss, reason (câu giải thích)
    """
    if not results:
        raise ValueError("results is empty")
    order = sorted(range(len(results)),
                   key=lambda i: (-results[i]["best_val_acc"], results[i]["best_val_loss"]))
    best = results[order[0]]
    name = best["config"].get("name", best.get("run_name"))
    reason = f"{name} có val_acc cao nhất tại best_epoch ({best['best_val_acc'] * 100:.2f}%)"
    if len(order) > 1:
        second = results[order[1]]
        gap = (best["best_val_acc"] - second["best_val_acc"]) * 100
        reason += (f", hơn cấu hình thứ hai {second['config'].get('name')} {gap:.2f} điểm"
                   f" ({second['best_val_acc'] * 100:.2f}%)")
    return {
        "name": name,
        "run_name": best.get("run_name"),
        "index": order[0],
        "best_val_acc": best["best_val_acc"],
        "best_val_loss": best["best_val_loss"],
        "reason": reason,
    }
