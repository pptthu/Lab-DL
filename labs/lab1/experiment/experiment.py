"""Module `experiment` (Experiments) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `experiment` (Thí nghiệm). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import pandas as pd

from shared.paths import artifact_dir
from shared.seed import set_seed
from labs.lab1.model.model import MLP, count_parameters
from labs.lab1.train.train import train_model


def run_experiments(configs: List[Dict[str, Any]], loaders: Tuple[Any, Any, Any], device: str) -> List[Dict[str, Any]]:
    """Train one model per config (e.g. hidden sizes, lr, dropout, optimizer) and compare on VALIDATION data.

    Input : configs: list of dicts; loaders: (train, val, test); device
    Output: list of result dicts: config, best_val_acc, n_params, train_seconds

    VI: Huấn luyện mỗi config một model (hidden sizes, lr, dropout, optimizer...) và so sánh trên VALIDATION.
    Vào : configs: danh sách dict; loaders: (train, val, test); device
    Ra  : danh sách dict kết quả: config, best_val_acc, n_params, train_seconds
    """
    train_loader, val_loader, _test_loader = loaders
    results: List[Dict[str, Any]] = []

    models_dir = artifact_dir("lab1", "models")

    for idx, cfg in enumerate(configs):
        seed = cfg.get("seed", 42)
        set_seed(seed)

        hidden_sizes = tuple(cfg.get("hidden_sizes", (256, 128)))
        dropout = float(cfg.get("dropout", 0.0))
        lr = float(cfg.get("lr", 1e-3))
        optimizer_name = str(cfg.get("optimizer_name", "adam"))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        epochs = int(cfg.get("epochs", 5))

        model = MLP(hidden_sizes=hidden_sizes, num_classes=10, dropout=dropout)
        n_params = count_parameters(model)

        ckpt_path = str(models_dir / f"exp_run_{idx}.pt")

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
        )
        train_seconds = float(time.perf_counter() - t0)

        val_accs = history.get("val_acc", [])
        best_val_acc = float(max(val_accs)) if val_accs else 0.0

        results.append({
            "config": cfg,
            "best_val_acc": best_val_acc,
            "n_params": n_params,
            "train_seconds": train_seconds,
        })

    return results


def save_results(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Save the comparison table.

    Input : results, csv_path
    Output: CSV file

    VI: Lưu bảng so sánh.
    Vào : results, csv_path
    Ra  : file CSV
    """
    rows = []
    for r in results:
        row: Dict[str, Any] = {}
        cfg = r.get("config", {})
        for k, v in cfg.items():
            if isinstance(v, (tuple, list)):
                row[k] = "-".join(str(x) for x in v)
            else:
                row[k] = v
        row["best_val_acc"] = r.get("best_val_acc")
        row["n_params"] = r.get("n_params")
        row["train_seconds"] = r.get("train_seconds")
        rows.append(row)

    df = pd.DataFrame(rows)
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
