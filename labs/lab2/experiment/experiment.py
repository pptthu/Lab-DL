"""Module `experiment` (Experiments) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `experiment` (Thí nghiệm). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.

Design
------
run_comparison trains each (model_name, hparam_dict) combination.
hparam_dict keys:
    lr            : float  – learning rate (default 1e-3).
    epochs        : int    – number of epochs (default 5).
    freeze        : bool   – freeze backbone (True = feature extraction).
    pretrained    : bool   – use ImageNet weights (True).
    optimizer     : str    – 'adam' or 'sgd'.
    weight_decay  : float  – L2 regularisation (default 0.0).
    n_unfreeze    : int    – unfreeze last N blocks after head warmup (0 = no fine-tune).
    warmup_epochs : int    – head-only warmup epochs before unfreezing.

Experiments A / B / C convention
----------------------------------
  A (scratch)          : name='scratch', pretrained=False, freeze=False
  B (feature extract.) : name='resnet18', pretrained=True,  freeze=True,  n_unfreeze=0
  C (fine-tuning)      : name='resnet18', pretrained=True,  freeze=True,  n_unfreeze=2,
                         warmup_epochs=<some positive integer>
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from labs.lab2.model.model import build_model, describe_model, unfreeze_last_blocks
from labs.lab2.train.train import train_model, load_checkpoint


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_comparison(
    model_names: List[str],
    hparam_grid: List[Dict],
    loaders: tuple,
    device,
    log_root: str,
) -> List[Dict]:
    """Train / fine-tune each (model_name, hparam_dict) pair.

    For each combination:
    1. Builds the model (build_model).
    2. Optionally runs a head-warmup phase (freeze backbone).
    3. Optionally unfreezes tail blocks and continues training (fine-tuning).
    4. Saves a best checkpoint and logs to TensorBoard when available.

    Input : model_names – list of architecture names, e.g. ['resnet18', 'scratch'].
            hparam_grid – list of hyperparameter dicts (one per run).
            loaders     – (train_loader, val_loader, test_loader).
            device      – torch.device.
            log_root    – root directory for TensorBoard runs and artefacts.
    Output: list of result dicts:
                model, hparams, best_val_acc, train_seconds, ckpt_path, status.

    VI: Huấn luyện/fine-tune nhiều model và nhiều bộ siêu tham số.
    """
    train_loader, val_loader, test_loader = loaders
    results: List[Dict] = []

    for model_name in model_names:
        for hparams in hparam_grid:
            run_id = _make_run_id(model_name, hparams)
            run_dir = Path(log_root) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = str(run_dir / "checkpoint_best.pt")

            print(f"\n{'='*60}")
            print(f"[Experiment] Run: {run_id}")
            print(f"  model={model_name}, hparams={hparams}")

            # Save config.
            config = {"model": model_name, **hparams, "run_id": run_id}
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            try:
                result = _single_run(
                    model_name=model_name,
                    hparams=hparams,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    run_dir=run_dir,
                    ckpt_path=ckpt_path,
                )
                result["run_id"]    = run_id
                result["model"]     = model_name
                result["hparams"]   = hparams
                result["ckpt_path"] = ckpt_path
                result["status"]    = "completed"
            except Exception as exc:
                print(f"[Experiment] ERROR in run {run_id}: {exc}")
                result = {
                    "run_id":      run_id,
                    "model":       model_name,
                    "hparams":     hparams,
                    "status":      "error",
                    "error":       str(exc),
                    "best_val_acc": None,
                    "train_seconds": None,
                }

            # Save summary.
            with open(run_dir / "summary.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

            results.append(result)
            print(f"[Experiment] Run {run_id} → status={result['status']}, "
                  f"best_val_acc={result.get('best_val_acc')}")

    return results


def save_results(results: List[Dict], csv_path: str) -> None:
    """Save the comparison table to a CSV file.

    Input : results  – list of result dicts returned by run_comparison.
            csv_path – output CSV path.
    Output: CSV file written to disk.

    VI: Lưu bảng so sánh vào CSV.
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return

    # Flatten hparams into top-level columns.
    rows = []
    for r in results:
        row = {
            "run_id":        r.get("run_id", ""),
            "model":         r.get("model", ""),
            "status":        r.get("status", ""),
            "best_val_acc":  r.get("best_val_acc", ""),
            "train_seconds": r.get("train_seconds", ""),
            "ckpt_path":     r.get("ckpt_path", ""),
        }
        for k, v in (r.get("hparams") or {}).items():
            row[f"hp_{k}"] = v
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Experiment] Results saved → {csv_path}")


# ---------------------------------------------------------------------------
# Plotting helpers (comparison figures)
# ---------------------------------------------------------------------------

def plot_training_curves(
    histories: Dict[str, Dict],
    out_dir: str,
) -> None:
    """Plot accuracy and loss curves for multiple runs side by side.

    Input : histories – {run_label: history_dict} where history_dict has
                        keys train_loss, val_loss, val_acc (lists).
            out_dir   – directory to save the PNG figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for label, h in histories.items():
        epochs = range(1, len(h.get("val_acc", [])) + 1)
        ax1.plot(epochs, h.get("val_acc", []), marker="o", markersize=3, label=label)
        ax2.plot(epochs, h.get("val_loss", []), marker="o", markersize=3, label=label)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Validation Accuracy")
    ax1.set_title("Validation Accuracy vs Epoch"); ax1.legend(); ax1.grid(True)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss vs Epoch");    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    fig_path = out_path / "training_curves.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"[Experiment] Training curves saved → {fig_path}")


def plot_comparison_bar(results: List[Dict], out_dir: str) -> None:
    """Bar chart comparing best_val_acc across all completed runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [r for r in results if r.get("status") == "completed"]
    if not completed:
        return

    labels = [r["run_id"] for r in completed]
    accs   = [r.get("best_val_acc", 0) or 0 for r in completed]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, accs, color="steelblue", edgecolor="black")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{acc:.4f}", ha="center", fontsize=9)
    ax.set_xlabel("Run")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Experiment Comparison — Best Validation Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    plt.tight_layout()
    fig_path = out_path / "comparison_bar.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"[Experiment] Comparison bar chart saved → {fig_path}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_run_id(model_name: str, hparams: Dict) -> str:
    """Build a short, unique run identifier from model name and key hparams."""
    freeze_tag  = "fe"  if hparams.get("freeze",    True)  else "sc"
    pretrain_tag = "pt" if hparams.get("pretrained", True)  else "rnd"
    lr    = hparams.get("lr", 1e-3)
    ep    = hparams.get("epochs", 5)
    ts    = int(time.time()) % 10000  # short disambiguator
    return f"{model_name}_{freeze_tag}_{pretrain_tag}_lr{lr:.0e}_ep{ep}_{ts}"


def _single_run(
    model_name: str,
    hparams: Dict,
    train_loader,
    val_loader,
    device,
    run_dir: Path,
    ckpt_path: str,
) -> Dict:
    """Execute one training run: warmup → optional unfreeze → continued training."""

    freeze        = hparams.get("freeze", True)
    pretrained    = hparams.get("pretrained", True)
    lr            = float(hparams.get("lr", 1e-3))
    epochs        = int(hparams.get("epochs", 5))
    n_unfreeze    = int(hparams.get("n_unfreeze", 0))
    warmup_epochs = int(hparams.get("warmup_epochs", 0))
    optimizer_nm  = hparams.get("optimizer", "adam")
    weight_decay  = float(hparams.get("weight_decay", 0.0))

    # --- Try to build TensorBoard writer (optional) ---
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))
    except Exception:
        pass  # TensorBoard not available; continue without it.

    # --- Build model ---
    model = build_model(
        name=model_name,
        num_classes=10,
        freeze_backbone=freeze,
        pretrained=pretrained,
    )
    info = describe_model(model)
    with open(run_dir / "model_info.json", "w") as f:
        json.dump({
            "total_params":     info["total_params"],
            "trainable_params": info["trainable_params"],
            "frozen_params":    info["frozen_params"],
        }, f, indent=2)

    histories = []

    # --- Phase 1: Head warmup (always, or full training when no unfreeze) ---
    phase1_epochs = warmup_epochs if (n_unfreeze > 0 and warmup_epochs > 0) else epochs
    print(f"  Phase 1: {phase1_epochs} epoch(s) with backbone "
          f"{'frozen' if freeze else 'trainable'}.")

    h1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=phase1_epochs,
        lr=lr,
        device=device,
        ckpt_path=ckpt_path,
        writer=writer,
        optimizer_name=optimizer_nm,
        weight_decay=weight_decay,
    )
    histories.append(h1)

    # --- Phase 2: Fine-tuning (unfreeze tail blocks) ---
    if n_unfreeze > 0 and warmup_epochs > 0:
        fine_tune_epochs = max(1, epochs - warmup_epochs)
        fine_tune_lr     = lr * 0.1  # smaller LR for pretrained layers
        print(f"  Phase 2: Unfreeze last {n_unfreeze} block(s), "
              f"{fine_tune_epochs} epoch(s) at lr={fine_tune_lr:.2e}.")
        unfreeze_last_blocks(model, n_unfreeze)
        h2 = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=fine_tune_epochs,
            lr=fine_tune_lr,
            device=device,
            ckpt_path=ckpt_path,
            writer=writer,
            optimizer_name=optimizer_nm,
            weight_decay=weight_decay,
        )
        histories.append(h2)

    if writer is not None:
        writer.close()

    # Merge histories.
    merged_train_loss = []
    merged_val_loss   = []
    merged_val_acc    = []
    for h in histories:
        merged_train_loss.extend(h["train_loss"])
        merged_val_loss.extend(h["val_loss"])
        merged_val_acc.extend(h["val_acc"])

    best_val_acc   = max(merged_val_acc) if merged_val_acc else 0.0
    total_seconds  = sum(h["train_seconds"] for h in histories)

    # Save history CSV.
    _save_history_csv(
        merged_train_loss, merged_val_loss, merged_val_acc,
        str(run_dir / "history.csv"),
    )

    return {
        "best_val_acc":   best_val_acc,
        "train_seconds":  total_seconds,
        "trainable_params": info["trainable_params"],
        "total_params":   info["total_params"],
        "history": {
            "train_loss": merged_train_loss,
            "val_loss":   merged_val_loss,
            "val_acc":    merged_val_acc,
        },
    }


def _save_history_csv(train_loss, val_loss, val_acc, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        for i, (tl, vl, va) in enumerate(zip(train_loss, val_loss, val_acc), 1):
            writer.writerow([i, f"{tl:.6f}", f"{vl:.6f}", f"{va:.6f}"])
