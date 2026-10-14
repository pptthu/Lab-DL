"""Entry point of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Run from the repo root:
    python -m labs.lab2.main --help
    python -m labs.lab2.main --mode smoke
    python -m labs.lab2.main --mode full --experiments scratch,feature_extraction,fine_tuning
    python -m labs.lab2.main --compare
    python -m labs.lab2.main --eda

VI: Điểm chạy chính. Chạy từ gốc repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the repo root is on sys.path (works when invoked as -m labs.lab2.main
# from the repo root, but also as a script directly).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.seed import set_seed
from shared.paths import raw_data_dir, artifact_dir

LAB = "lab2"

# Experiment configs for each mode
SMOKE_EXPERIMENTS = {
    "scratch": {
        "model_name": "scratch",
        "pretrained": False,
        "freeze": False,
        "lr": 1e-3,
        "epochs": 2,
        "n_unfreeze": 0,
        "warmup_epochs": 0,
        "optimizer": "adam",
    },
    "feature_extraction": {
        "model_name": "resnet18",
        "pretrained": True,
        "freeze": True,
        "lr": 1e-3,
        "epochs": 2,
        "n_unfreeze": 0,
        "warmup_epochs": 0,
        "optimizer": "adam",
    },
    "fine_tuning": {
        "model_name": "resnet18",
        "pretrained": True,
        "freeze": True,
        "lr": 1e-3,
        "epochs": 4,        # 2 warmup + 2 fine-tune
        "n_unfreeze": 1,
        "warmup_epochs": 2,
        "optimizer": "adam",
    },
}

FULL_EXPERIMENTS = {
    "scratch": {
        "model_name": "scratch",
        "pretrained": False,
        "freeze": False,
        "lr": 1e-3,
        "epochs": 20,
        "n_unfreeze": 0,
        "warmup_epochs": 0,
        "optimizer": "adam",
        "weight_decay": 1e-4,
    },
    "feature_extraction": {
        "model_name": "resnet18",
        "pretrained": True,
        "freeze": True,
        "lr": 1e-3,
        "epochs": 10,
        "n_unfreeze": 0,
        "warmup_epochs": 0,
        "optimizer": "adam",
    },
    "fine_tuning": {
        "model_name": "resnet18",
        "pretrained": True,
        "freeze": True,
        "lr": 1e-3,
        "epochs": 15,
        "n_unfreeze": 2,
        "warmup_epochs": 5,
        "optimizer": "adam",
        "weight_decay": 1e-4,
    },
}


def main(seed: int = 42) -> None:
    """Pipeline: parse args → execute requested mode."""
    parser = _build_parser()
    args = parser.parse_args()

    set_seed(seed)

    if args.eda:
        _run_eda(args)
        return

    if args.compare:
        _run_compare(args)
        return

    if args.mode in ("smoke", "full"):
        _run_experiments(args, seed=seed)
        return

    # Default: print help.
    parser.print_help()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m labs.lab2.main",
        description="Lab 2 – Pre-trained CNNs & Fine-tuning on CIFAR-10",
    )
    p.add_argument(
        "--mode", choices=["smoke", "full"], default=None,
        help="Run mode: 'smoke' (fast pipeline check) or 'full' (real training).",
    )
    p.add_argument(
        "--experiments", default="scratch,feature_extraction,fine_tuning",
        help="Comma-separated experiment names (default: all three).",
    )
    p.add_argument(
        "--eda", action="store_true",
        help="Run exploratory data analysis and save figures.",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Load and compare existing experiment results (CSV).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default 42).",
    )
    p.add_argument(
        "--image-size", type=int, default=64,
        help="Input image size (default 64; use 224 for full ImageNet resolution).",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size (default 64).",
    )
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default 0 = main process).",
    )
    p.add_argument(
        "--download", action="store_true", default=True,
        help="Download CIFAR-10 if not present (default True).",
    )
    p.add_argument(
        "--subset-size", type=int, default=None,
        help="Limit training to N samples (smoke test shortcut).",
    )
    return p


def _run_eda(args) -> None:
    from labs.lab2.eda.eda import run_eda
    data_root = str(raw_data_dir(LAB))
    out_dir   = str(artifact_dir(LAB, "figures") / "eda")
    print(f"[EDA] Running EDA.  data_root={data_root}, out_dir={out_dir}")
    summary = run_eda(data_root=data_root, out_dir=out_dir)
    print(f"[EDA] Summary: n_train={summary.get('n_train')}, "
          f"n_test={summary.get('n_test')}, "
          f"classes={list(summary.get('class_counts', {}).keys())[:3]}...")


def _run_compare(args) -> None:
    import glob
    import json

    runs_dir = artifact_dir(LAB, "runs")
    summaries = []
    for p in sorted(runs_dir.glob("*/summary.json")):
        with open(p) as f:
            summaries.append(json.load(f))

    if not summaries:
        print("[Compare] No completed runs found in artifacts/runs/.")
        return

    print(f"\n{'Run ID':<45} {'Status':<12} {'Best Val Acc':>14} {'Time (s)':>10}")
    print("-" * 85)
    for s in summaries:
        acc  = s.get("best_val_acc")
        secs = s.get("train_seconds")
        print(
            f"{s.get('run_id','?'):<45} "
            f"{s.get('status','?'):<12} "
            f"{acc if acc is not None else 'N/A':>14} "
            f"{secs if secs is not None else 'N/A':>10}"
        )


def _run_experiments(args, seed: int) -> None:
    import torch
    from labs.lab2.preprocessing.preprocessing import (
        load_raw_dataset, make_split, save_split,
    )
    from labs.lab2.features.features import build_transforms, make_loaders
    from labs.lab2.experiment.experiment import (
        run_comparison, save_results, plot_training_curves, plot_comparison_bar,
    )

    device     = torch.device("cpu")  # CPU only on this machine
    data_root  = str(raw_data_dir(LAB))
    splits_dir = artifact_dir(LAB, "splits")
    runs_dir   = artifact_dir(LAB, "runs")
    figs_dir   = artifact_dir(LAB, "figures")
    image_size = args.image_size
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Parse requested experiments.
    requested = [e.strip() for e in args.experiments.split(",")]
    cfg_map = SMOKE_EXPERIMENTS if args.mode == "smoke" else FULL_EXPERIMENTS

    print(f"[Main] mode={args.mode}, experiments={requested}, "
          f"image_size={image_size}, batch_size={batch_size}, device={device}")

    # --- 1. Dataset ---
    print("[Main] Loading CIFAR-10 ...")
    train_set, test_set = load_raw_dataset(data_root=data_root, download=args.download)
    n_train = len(train_set)

    # Apply subset if requested (smoke shortcut).
    subset_size = args.subset_size
    if args.mode == "smoke" and subset_size is None:
        subset_size = 1000

    # --- 2. Split ---
    split_path = str(splits_dir / f"split_seed{seed}.json")
    from pathlib import Path
    if Path(split_path).exists():
        from labs.lab2.preprocessing.preprocessing import load_split
        train_idx, val_idx = load_split(split_path)
        print(f"[Main] Loaded existing split from {split_path}")
    else:
        train_idx, val_idx = make_split(n_train, val_ratio=0.1, seed=seed)
        save_split(train_idx, val_idx, split_path)
        print(f"[Main] Created new split → {split_path}")

    # Apply subset to training indices.
    if subset_size is not None:
        train_idx = train_idx[:subset_size]
        val_idx   = val_idx[:max(1, subset_size // 10)]
        print(f"[Main] Subset mode: {len(train_idx)} train, {len(val_idx)} val samples.")

    # --- 3. Transforms ---
    train_tf, eval_tf = build_transforms(image_size=image_size, augment=True)

    # --- 4. DataLoaders ---
    train_loader, val_loader, test_loader = make_loaders(
        train_set=train_set,
        test_set=test_set,
        train_idx=train_idx,
        val_idx=val_idx,
        train_tf=train_tf,
        eval_tf=eval_tf,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    print(f"[Main] DataLoaders ready: "
          f"train={len(train_loader)} batches, "
          f"val={len(val_loader)} batches, "
          f"test={len(test_loader)} batches.")

    # --- 5. Build experiment lists ---
    model_names = []
    hparam_grid = []
    for exp_name in requested:
        if exp_name not in cfg_map:
            print(f"[Main] WARNING: unknown experiment '{exp_name}'. Skipping.")
            continue
        cfg = dict(cfg_map[exp_name])
        mn  = cfg.pop("model_name")
        model_names.append(mn)
        hparam_grid.append(cfg)

    if not model_names:
        print("[Main] No valid experiments selected. Exiting.")
        return

    # --- 6. Run comparison ---
    loaders = (train_loader, val_loader, test_loader)
    results = run_comparison(
        model_names=model_names,
        hparam_grid=hparam_grid,
        loaders=loaders,
        device=device,
        log_root=str(runs_dir),
    )

    # --- 7. Save results CSV ---
    csv_path = str(runs_dir / f"results_{args.mode}.csv")
    save_results(results, csv_path)

    # --- 8. Comparison plots ---
    histories = {
        r["run_id"]: r.get("history", {})
        for r in results
        if r.get("status") == "completed" and r.get("history")
    }
    plot_training_curves(histories, str(figs_dir))
    plot_comparison_bar(results, str(figs_dir))

    # --- 9. Print summary ---
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run ID':<45} {'Best Val Acc':>14} {'Time (s)':>10}")
    print("-" * 70)
    for r in results:
        acc  = r.get("best_val_acc")
        secs = r.get("train_seconds")
        acc_str  = f"{acc:.4f}" if acc is not None else "N/A"
        secs_str = f"{secs:.1f}" if secs is not None else "N/A"
        print(f"{r.get('run_id','?'):<45} {acc_str:>14} {secs_str:>10}")
    print(f"{'='*70}")
    print(f"Results CSV → {csv_path}")


if __name__ == "__main__":
    main()
