"""Entry point of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Run from the repo root:  python -m labs.lab1.main
Owner: team leader. Wire the stages here after the modules are implemented
and each one has passed its smoke test.

VI: Điểm chạy chính. Chạy từ gốc repo:
    python -m labs.lab1.main --smoke   # smoke run: 1 epoch, 2 batch, mọi đầu ra vào outputs/_smoke/
    python -m labs.lab1.main           # normal run: 5 cấu hình, tối đa 30 epoch + early stopping
Thứ tự: EDA -> đọc dữ liệu + split -> loader -> train 5 cấu hình (experiment) -> chọn cấu hình trên val
-> test một lần cho baseline và cấu hình được chọn (evaluate) -> hình từ log (visualize) -> demo.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from shared.paths import REPO_ROOT, data_dir, output_dir
from shared.seed import set_seed
from labs.lab1.eda.eda import run_eda
from labs.lab1.preprocessing.preprocessing import load_csv, load_split, make_split, save_split
from labs.lab1.features.features import compute_mean_std, build_transform, FashionDataset, make_loaders
from labs.lab1.train.train import load_checkpoint
from labs.lab1.evaluate.evaluate import (
    TEST_ACC_THRESHOLD, confusion_matrix, evaluate, save_classification_report, save_test_results,
)
from labs.lab1.visualize.visualize import (
    demo_predictions, plot_accuracy_curves, plot_confusion_matrix, plot_loss_curves, show_predictions,
)
from labs.lab1.experiment.experiment import (
    build_model, make_configs, run_experiments, save_history, save_results, select_config,
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Fixed so that main.py and the notebook run under the same CPU setup.
# VI: Cố định để main.py và notebook chạy cùng cấu hình CPU.
N_THREADS = 4
BATCH_SIZE = 64
# Quy tắc dừng theo docs/templates/LOG_TEMPLATE_v1.md §3.
MAX_EPOCHS = 30
PATIENCE = 5
MIN_DELTA = 1e-4
# Smoke run theo docs/rules/RULES_v1.md mục 7.
SMOKE_EPOCHS = 1
SMOKE_BATCHES = 2


def _rel(path) -> str:
    # Đường dẫn tương đối từ gốc repo, để file tóm tắt dùng được trên máy khác.
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _out_dirs(smoke: bool) -> dict:
    # Normal run: outputs/<kind>/. Smoke run: mọi thứ vào outputs/_smoke/<kind>/.
    kinds = ("models", "logs", "figures", "results", "demo")
    if smoke:
        root = output_dir("lab1", "_smoke")
        dirs = {k: root / k for k in kinds}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs
    return {k: output_dir("lab1", k) for k in kinds}


def main(seed: int = 42, epochs: int = MAX_EPOCHS, smoke: bool = False) -> dict:
    """Chạy toàn bộ pipeline Lab 1 theo desc lab (docs/lab1_description_v1.md).

    Vào : seed; epochs (số epoch tối đa); smoke (True: 1 epoch, 2 batch, ghi vào outputs/_smoke/)
    Ra  : dict tóm tắt (cũng ghi ra results/run_summary.json)
    """
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(N_THREADS)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if smoke:
        epochs = SMOKE_EPOCHS
    max_batches = SMOKE_BATCHES if smoke else None
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    mode = "SMOKE RUN (số liệu không dùng để báo cáo)" if smoke else "NORMAL RUN"
    print(f"=== [Lab 1] {mode} | device: {device} | seed: {seed} | max_epochs: {epochs} ===")

    # 1. Đường dẫn
    raw_dir = data_dir("lab1", "raw")
    train_csv_path = raw_dir / "fashion-mnist_train.csv"
    test_csv_path = raw_dir / "fashion-mnist_test.csv"
    split_file = data_dir("lab1", "splits") / "train_val_split.json"
    out = _out_dirs(smoke)

    # 2. EDA (chỉ trên file train)
    print("\n--- 1. EDA ---")
    eda_stats = run_eda(str(train_csv_path), str(out["figures"]))
    print(f"rows={eda_stats['n_rows']} classes={len(eda_stats['class_counts'])} duplicates={eda_stats['n_duplicates']}")

    # 3. Đọc dữ liệu và split; dùng lại file split đã commit
    print("\n--- 2. Preprocessing ---")
    images, labels = load_csv(str(train_csv_path))
    test_images, test_labels = load_csv(str(test_csv_path))
    if split_file.is_file():
        train_idx, val_idx = load_split(str(split_file))
    else:
        train_idx, val_idx = make_split(labels, val_ratio=0.1, seed=42)
        save_split(train_idx, val_idx, str(split_file))
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_labels)} | split: {_rel(split_file)}")

    # 4. Chuẩn hoá theo mean/std của tập train, tạo loader
    print("\n--- 3. Features ---")
    mean, std = compute_mean_std(images[train_idx])
    transform = build_transform(mean, std)
    train_ds = FashionDataset(images, labels, indices=train_idx, transform=transform)
    val_ds = FashionDataset(images, labels, indices=val_idx, transform=transform)
    test_ds = FashionDataset(test_images, test_labels, indices=None, transform=transform)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=BATCH_SIZE)
    print(f"mean={mean:.4f} std={std:.4f} batch_size={BATCH_SIZE}")

    # 5. Train 5 cấu hình (experiment gọi train cho từng cấu hình), chỉ dùng train + val
    print("\n--- 4. Train 5 cấu hình ---")
    configs = make_configs(epochs=epochs, seed=seed)
    run_info = {"batch_size": BATCH_SIZE, "data": f"train={len(train_idx)} val={len(val_idx)}",
                "split": _rel(split_file), "normalize": {"mean": mean, "std": std}}
    loaders = (train_loader, val_loader, test_loader)
    results = run_experiments(
        configs, loaders, device=device, log_dir=str(out["logs"]), models_dir=str(out["models"]),
        patience=PATIENCE, min_delta=MIN_DELTA, max_batches=max_batches, run_info=run_info, verbose=True,
    )
    comparison_csv = out["results"] / "experiment_results.csv"
    save_results(results, str(comparison_csv))
    save_history(results, str(out["results"] / "epoch_history.csv"))

    # 6. Chọn cấu hình trên val (không nhìn test)
    chosen = select_config(results)
    print(f"\nChọn: {chosen['reason']}")
    by_name = {r["config"]["name"]: r for r in results}
    baseline = by_name["baseline"]
    selected = results[chosen["index"]]

    # 7. Đánh giá: nạp _best.pt; val để đối chiếu với log; test đúng một lần cho mỗi model
    print("\n--- 5. Evaluate (test một lần cho baseline và cấu hình được chọn) ---")
    # Nếu cấu hình được chọn chính là baseline thì chỉ chạy test một lần.
    if selected is baseline:
        to_test = [(baseline, "baseline+selected")]
    else:
        to_test = [(baseline, "baseline"), (selected, "selected")]
    test_rows = []
    loaded = {}
    for r, role in to_test:
        model = load_checkpoint(build_model(r["config"]), r["ckpt_path"], device=device)
        loaded[r["config"]["name"]] = model
        val_loss, val_acc = evaluate(model, val_loader, device=device)
        test_loss, test_acc = evaluate(model, test_loader, device=device)
        test_rows.append({
            "name": r["config"]["name"], "role": role, "run_name": r["run_name"], "ckpt_path": _rel(r["ckpt_path"]),
            "best_epoch": r["best_epoch"], "val_loss": val_loss, "val_acc": val_acc,
            "val_acc_in_log": r["best_val_acc"], "test_loss": test_loss, "test_acc": test_acc,
        })
        print(f"{r['config']['name']:<20} ({role}) val_acc={val_acc * 100:.2f}% (log {r['best_val_acc'] * 100:.2f}%) "
              f"| test_loss={test_loss:.4f} test_acc={test_acc * 100:.2f}%")
    test_csv = out["results"] / "test_results.csv"
    test_df = save_test_results(test_rows, str(test_csv))
    baseline_test_acc = float(test_df.loc[test_df["role"].str.contains("baseline"), "test_acc"].iloc[0])
    print(f"Baseline test {baseline_test_acc * 100:.2f}% | mức desc lab ≥ {TEST_ACC_THRESHOLD * 100:.0f}%: "
          f"{'ĐẠT' if baseline_test_acc >= TEST_ACC_THRESHOLD else 'CHƯA ĐẠT'}")

    # Phân tích lỗi theo lớp trên val của cấu hình được chọn
    sel_name = selected["config"]["name"]
    sel_model = loaded[sel_name]
    cm = confusion_matrix(sel_model, val_loader, device=device, num_classes=10)
    report_csv = out["results"] / f"classification_report_val_{sel_name}.csv"
    save_classification_report(cm, CLASS_NAMES, str(report_csv))

    # 8. Hình: loss và accuracy vẽ từ _history.csv của từng run
    print("\n--- 6. Visualize ---")
    figures = {"eda": [_rel(out["figures"] / "class_counts.png"), _rel(out["figures"] / "sample_grid.png")]}
    for r in results:
        name = r["config"]["name"]
        loss_png = out["figures"] / f"{r['run_name']}_loss.png"
        acc_png = out["figures"] / f"{r['run_name']}_acc.png"
        plot_loss_curves(r["history_path"], str(loss_png), title=f"{name}: loss (train / val)")
        plot_accuracy_curves(r["history_path"], str(acc_png), title=f"{name}: accuracy (train / val)")
        figures[name] = {"loss": _rel(loss_png), "acc": _rel(acc_png)}
    cm_png = out["figures"] / f"{stamp}_confusion_matrix_val_{sel_name}.png"
    plot_confusion_matrix(cm, CLASS_NAMES, str(cm_png))
    pred_png = out["figures"] / f"{stamp}_predictions_test_{sel_name}.png"
    show_predictions(sel_model, test_loader, CLASS_NAMES, device=device, out_path=str(pred_png), n=16)
    figures["confusion_matrix_val"] = _rel(cm_png)
    figures["predictions_test"] = _rel(pred_png)

    # 9. Demo: nạp lại _best.pt của cấu hình được chọn vào model mới, dự đoán 16 ảnh test đầu
    print("\n--- 7. Demo ---")
    demo_model = load_checkpoint(build_model(selected["config"]), selected["ckpt_path"], device=device)
    demo_png = out["demo"] / f"{stamp}_predictions.png"
    demo_csv = out["demo"] / f"{stamp}_predictions.csv"
    demo_df = demo_predictions(demo_model, test_loader, CLASS_NAMES, device=device,
                               out_png=str(demo_png), out_csv=str(demo_csv), n=16)
    print(f"demo: {int(demo_df['correct'].sum())}/{len(demo_df)} đúng -> {_rel(demo_png)}")

    # 10. File tóm tắt cho notebook và tài liệu
    finished = datetime.now().astimezone()
    summary = {
        "mode": "smoke" if smoke else "normal",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_s": round((finished - started).total_seconds(), 1),
        "device": device, "seed": seed, "max_epochs": epochs, "batch_size": BATCH_SIZE,
        "early_stopping": {"monitor": "val_loss", "patience": PATIENCE, "min_delta": MIN_DELTA},
        "max_batches": max_batches,
        "torch": torch.__version__, "python": sys.version.split()[0], "threads": N_THREADS,
        "data": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_labels)),
                 "split": _rel(split_file), "mean": mean, "std": std},
        "runs": [{
            "name": r["config"]["name"], "run_name": r["run_name"], "n_params": r["n_params"],
            "best_epoch": r["best_epoch"], "stopped_epoch": r["stopped_epoch"], "stop_reason": r["stop_reason"],
            "best_val_loss": r["best_val_loss"], "best_val_acc": r["best_val_acc"],
            "best_train_acc": r["best_train_acc"], "train_seconds": r["train_seconds"],
            "log": _rel(r["log_path"]), "history": _rel(r["history_path"]), "config": _rel(r["config_path"]),
            "best_ckpt": _rel(r["ckpt_path"]), "last_ckpt": _rel(r["last_ckpt_path"]),
        } for r in results],
        "selection": {k: chosen[k] for k in ("name", "run_name", "best_val_acc", "best_val_loss", "reason")},
        "test": test_df.to_dict(orient="records"),
        "threshold": TEST_ACC_THRESHOLD,
        "baseline_test_acc": baseline_test_acc,
        "baseline_meets_threshold": bool(baseline_test_acc >= TEST_ACC_THRESHOLD),
        "files": {
            "experiment_results": _rel(comparison_csv), "epoch_history": _rel(out["results"] / "epoch_history.csv"),
            "test_results": _rel(test_csv), "classification_report_val": _rel(report_csv),
            "figures": figures, "demo_png": _rel(demo_png), "demo_csv": _rel(demo_csv),
        },
    }
    summary_path = out["results"] / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nTóm tắt: {_rel(summary_path)} | thời gian {summary['duration_s']:.0f} giây")
    print("=== [Lab 1] Xong ===")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 1: Fashion-MNIST MLP")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 2 batch, ghi vào outputs/_smoke/")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="số epoch tối đa (mặc định 30)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed, epochs=args.epochs, smoke=args.smoke)
