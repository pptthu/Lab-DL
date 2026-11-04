"""Entry point of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Run from the repo root:
    python -m labs.lab2.main
    python -m labs.lab2.main --epochs 1 --image-size 64 --no-experiments

Pipeline: eda -> preprocessing -> features -> model -> train -> evaluate ->
experiment -> results.

Owner: team leader (see progress_status.md).

VI: Điểm chạy chính của Lab 2. Chạy từ gốc repo: python -m labs.lab2.main
Pipeline: eda -> preprocessing -> features -> model -> train -> evaluate -> experiment -> results.
Người phụ trách: nhóm trưởng.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

# Allow both `python -m labs.lab2.main` (from the repo root) and
# `python labs/lab2/main.py` (directly). This must run before the package
# imports below, so those imports carry an E402 suppression marker.
if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from shared.paths import artifact_dir, raw_data_dir  # noqa: E402
from shared.seed import set_seed  # noqa: E402

from labs.lab2.eda.eda import CIFAR10_CLASSES, run_eda  # noqa: E402
from labs.lab2.evaluate.evaluate import confusion_matrix, evaluate, top_confusions  # noqa: E402
from labs.lab2.experiment.experiment import (  # noqa: E402
    make_summary_writer,
    run_comparison,
    save_results,
)
from labs.lab2.features.features import build_transforms, make_loaders  # noqa: E402
from labs.lab2.model.model import build_model, describe_model, unfreeze_last_blocks  # noqa: E402
from labs.lab2.preprocessing.preprocessing import (  # noqa: E402
    load_raw_dataset,
    load_split,
    make_split,
    save_split,
)
from labs.lab2.train.train import load_checkpoint, train_model  # noqa: E402

STAGES = [
    "eda",
    "preprocessing",
    "features",
    "model",
    "train",
    "evaluate",
    "experiment",
    "results",
]


def _save_history_csv(history: Dict[str, Any], path: Any) -> None:
    """Write the per-epoch training history to a CSV file.

    VI: Ghi lịch sử huấn luyện từng epoch ra file CSV.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        for epoch, (train_loss, val_loss, val_acc) in enumerate(
            zip(history["train_loss"], history["val_loss"], history["val_acc"]),
            start=1,
        ):
            writer.writerow([epoch, train_loss, val_loss, val_acc])


def default_comparison_grid(
    lr: float = 1e-3,
    epochs: int = 3,
    weight_decay: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Default experiment grid: frozen backbone vs. fine-tuning the last block.

    VI: Lưới thí nghiệm mặc định: đóng băng backbone so với fine-tune khối cuối.
    """
    return [
        {
            "lr": lr,
            "epochs": epochs,
            "optimizer_name": "adam",
            "freeze_backbone": True,
            "unfreeze_blocks": 0,
            "weight_decay": weight_decay,
        },
        {
            "lr": lr,
            "epochs": epochs,
            "optimizer_name": "adam",
            "freeze_backbone": True,
            "unfreeze_blocks": 1,
            "weight_decay": weight_decay,
        },
    ]


def main(
    seed: int = 42,
    model_name: str = "resnet18",
    epochs: int = 3,
    image_size: int = 128,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    augment: bool = True,
    freeze_backbone: bool = True,
    unfreeze_blocks: int = 0,
    pretrained: bool = True,
    download: bool = True,
    run_experiments: bool = True,
    comparison_models: Optional[Sequence[str]] = None,
    comparison_grid: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the full Lab 2 pipeline and return a summary dictionary.

    Stages and data flow:
      EDA          run_eda                       -> figures + eda_summary.json
      Preprocessing load_raw_dataset / make_split -> train_idx, val_idx (+ split JSON)
      Features     build_transforms / make_loaders -> train, val, test DataLoaders
      Model        build_model (+ unfreeze)       -> nn.Module
      Train        train_model                    -> best checkpoint + history
      Evaluate     load_checkpoint / evaluate     -> val and test metrics + confusion matrix
      Experiment   run_comparison / save_results  -> comparison table
      Results      JSON summary under artifacts/runs

    The test set is only used for the final evaluation; model selection uses
    the validation split (checkpoint criterion: lowest validation loss).

    Input : hyperparameters and paths are derived from `shared.paths`
    Output: summary dict (also written to artifacts/runs/pipeline_summary.json)

    VI: Chạy toàn bộ pipeline Lab 2 và trả về dict tổng hợp.
    Tập test chỉ dùng cho đánh giá cuối; chọn model dựa trên tập validation
    (tiêu chí checkpoint: validation loss nhỏ nhất).
    Vào : siêu tham số; đường dẫn lấy từ `shared.paths`
    Ra  : dict tổng hợp (đồng thời ghi ra artifacts/runs/pipeline_summary.json)
    """
    # 0. Validate configuration early, before downloading data or training.
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if unfreeze_blocks < 0:
        raise ValueError(f"unfreeze_blocks must be >= 0, got {unfreeze_blocks}")

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    seed = set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== [Lab 2 Pipeline] Device: {device} | Seed: {seed} ===")

    # 1. Paths (shared/paths.py keeps them repo-relative)
    raw_dir = raw_data_dir("lab2")
    figures_dir = artifact_dir("lab2", "figures")
    splits_dir = artifact_dir("lab2", "splits")
    models_dir = artifact_dir("lab2", "models")
    logs_dir = artifact_dir("lab2", "logs")
    runs_dir = artifact_dir("lab2", "runs")

    # 2. EDA
    print("\n--- 1. Exploratory Data Analysis (EDA) ---")
    eda_stats = run_eda(str(raw_dir), str(figures_dir))
    eda_summary_path = figures_dir / "eda_summary.json"
    eda_summary_path.write_text(json.dumps(eda_stats, indent=2), encoding="utf-8")
    print(
        f"EDA completed: {eda_stats['n_train']} train / "
        f"{eda_stats['n_test']} test images; summary saved to {eda_summary_path}"
    )

    # 3. Preprocessing (load + reproducible split)
    print("\n--- 2. Preprocessing & Splitting ---")
    train_set_raw, test_set_raw = load_raw_dataset(str(raw_dir), download=download)
    split_path = splits_dir / "train_val_split.json"
    if split_path.is_file():
        train_idx, val_idx = load_split(str(split_path))
        print(f"Loaded existing split: {split_path}")
    else:
        train_idx, val_idx = make_split(len(train_set_raw), val_ratio=0.1, seed=seed)
        save_split(train_idx, val_idx, str(split_path))
        print(f"Created reproducible split with seed={seed}: {split_path}")
    print(
        f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, "
        f"test: {len(test_set_raw)}"
    )

    # 4. Features (transforms + DataLoaders)
    print("\n--- 3. Features & DataLoaders ---")
    train_tf, eval_tf = build_transforms(image_size=image_size, augment=augment)
    train_loader, val_loader, test_loader = make_loaders(
        train_set=train_set_raw,
        test_set=test_set_raw,
        train_idx=train_idx,
        val_idx=val_idx,
        train_tf=train_tf,
        eval_tf=eval_tf,
        batch_size=batch_size,
    )
    print(
        f"Image size: {image_size} | batch size: {batch_size} "
        f"| augmentation: {augment}"
    )

    # 5. Model
    print("\n--- 4. Model Instantiation ---")
    set_seed(seed)
    model = build_model(
        name=model_name,
        num_classes=10,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
    )
    if unfreeze_blocks > 0:
        unfreeze_last_blocks(model, unfreeze_blocks)
        print(f"Fine-tuning: unfroze the last {unfreeze_blocks} block(s).")
    model_stats = describe_model(model)
    print(
        f"{model_name}: total={model_stats['total_params']:,} | "
        f"trainable={model_stats['trainable_params']:,} parameters"
    )

    # 6. Train (TensorBoard is optional; writer is None when it is not installed)
    print("\n--- 5. Training ---")
    ckpt_path = str(models_dir / f"{model_name}_baseline_best.pt")
    history_path = runs_dir / f"{model_name}_baseline_history.csv"
    writer = make_summary_writer(str(logs_dir / f"{model_name}_baseline"))
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            ckpt_path=ckpt_path,
            writer=writer,
            optimizer_name="adam",
            weight_decay=weight_decay,
        )
    finally:
        if writer is not None:
            writer.close()
    _save_history_csv(history, history_path)
    print(
        f"Best checkpoint: epoch {history['best_epoch']} "
        f"(val_loss={history['best_val_loss']:.4f}, "
        f"val_acc={history['best_val_acc'] * 100:.2f}%) -> {ckpt_path}"
    )

    # 7. Evaluate the selected checkpoint (validation + final test evaluation)
    print("\n--- 6. Evaluation (best checkpoint) ---")
    set_seed(seed)
    eval_model = build_model(
        name=model_name,
        num_classes=10,
        freeze_backbone=freeze_backbone,
        pretrained=False,
    )
    best_model = load_checkpoint(eval_model, ckpt_path, device=device)

    val_loss, val_acc = evaluate(best_model, val_loader, device=device)
    test_loss, test_acc = evaluate(best_model, test_loader, device=device)
    cm = confusion_matrix(best_model, test_loader, device=device, num_classes=10)
    top_errors = top_confusions(cm, list(CIFAR10_CLASSES), k=5)

    print(f"Validation: loss={val_loss:.4f}, accuracy={val_acc * 100:.2f}%")
    print(f"Test      : loss={test_loss:.4f}, accuracy={test_acc * 100:.2f}%")
    for rank, (true_class, pred_class, count) in enumerate(top_errors, start=1):
        print(f"  top confusion {rank}: {true_class} -> {pred_class} ({count})")

    # 8. Experiment (multi-model / multi-hyperparameter comparison)
    print("\n--- 7. Experiments (model / hyperparameter comparison) ---")
    experiment_results: List[Dict[str, Any]] = []
    if run_experiments:
        model_names = (
            list(comparison_models)
            if comparison_models
            else [model_name, "densenet121"]
        )
        grid = (
            comparison_grid
            if comparison_grid is not None
            else default_comparison_grid(lr, epochs, weight_decay)
        )
        set_seed(seed)
        experiment_results = run_comparison(
            model_names=model_names,
            hparam_grid=grid,
            loaders=(train_loader, val_loader, test_loader),
            device=device,
            log_root=str(runs_dir / "experiment"),
        )
        results_csv = runs_dir / "experiment_results.csv"
        save_results(experiment_results, str(results_csv))
        print(f"Comparison results saved to: {results_csv}")
    else:
        print("Skipped (--no-experiments).")

    # 9. Results summary
    print("\n--- 8. Results ---")
    summary_path = runs_dir / "pipeline_summary.json"
    summary: Dict[str, Any] = {
        "seed": seed,
        "device": device,
        "model": model_name,
        "config": {
            "epochs": epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "augment": augment,
            "freeze_backbone": freeze_backbone,
            "unfreeze_blocks": unfreeze_blocks,
            "pretrained": pretrained,
        },
        "dataset": {
            "n_train": eda_stats.get("n_train"),
            "n_test": eda_stats.get("n_test"),
            "n_train_split": len(train_idx),
            "n_val_split": len(val_idx),
        },
        "model_stats": model_stats,
        "history": {
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "best_val_acc": history["best_val_acc"],
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_acc": history["val_acc"],
        },
        "evaluation": {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "top_confusions": [list(item) for item in top_errors],
        },
        "experiments": experiment_results,
        "artifacts": {
            "figures": str(figures_dir),
            "split": str(split_path),
            "checkpoint": ckpt_path,
            "history_csv": str(history_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Pipeline summary saved to: {summary_path}")
    print("\n=== [Lab 2 Pipeline Completed Successfully] ===")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. / Tạo bộ phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Run the Lab 2 pipeline: EDA -> preprocessing -> features -> "
        "model -> train -> evaluate -> experiment -> results."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="resnet18", help="resnet18 | densenet121 | vgg16")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false",
        help="disable training-time augmentation",
    )
    parser.add_argument(
        "--no-freeze", dest="freeze_backbone", action="store_false",
        help="train the whole backbone",
    )
    parser.add_argument("--unfreeze-blocks", type=int, default=0)
    parser.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false",
        help="random initialisation (debugging only)",
    )
    parser.add_argument(
        "--no-download", dest="download", action="store_false",
        help="do not download CIFAR-10",
    )
    parser.add_argument("--no-experiments", dest="run_experiments", action="store_false")
    parser.add_argument(
        "--comparison-models", nargs="+", default=None,
        help="models used by the comparison stage (default: baseline model + densenet121)",
    )
    return parser


def cli(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Parse command-line arguments and run main(). / Đọc tham số dòng lệnh rồi chạy main()."""
    args = _build_parser().parse_args(argv)
    return main(
        seed=args.seed,
        model_name=args.model,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        augment=args.augment,
        freeze_backbone=args.freeze_backbone,
        unfreeze_blocks=args.unfreeze_blocks,
        pretrained=args.pretrained,
        download=args.download,
        run_experiments=args.run_experiments,
        comparison_models=args.comparison_models,
    )


if __name__ == "__main__":
    cli()
