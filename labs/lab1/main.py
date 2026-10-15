"""Entry point of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Run from the repo root:  python -m labs.lab1.main
Owner: team leader. Wire the stages here after the modules are implemented
and each one has passed its smoke test.

VI: Điểm chạy chính. Chạy từ gốc repo: python -m labs.lab1.main
Người phụ trách: nhóm trưởng. Nối các bước sau khi từng module đã qua smoke test.
"""
from __future__ import annotations

import sys
import torch

from shared.paths import artifact_dir, raw_data_dir
from shared.seed import set_seed
from labs.lab1.eda.eda import run_eda
from labs.lab1.preprocessing.preprocessing import load_csv, load_split, make_split, save_split
from labs.lab1.features.features import compute_mean_std, build_transform, FashionDataset, make_loaders
from labs.lab1.model.model import MLP, count_parameters
from labs.lab1.train.train import train_model, load_checkpoint
from labs.lab1.evaluate.evaluate import evaluate, confusion_matrix, per_class_accuracy
from labs.lab1.visualize.visualize import plot_loss_curves, show_predictions, plot_confusion_matrix
from labs.lab1.experiment.experiment import run_experiments, save_results

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main(seed: int = 42, epochs: int = 5) -> dict:
    """Run full Lab 1 pipeline:
    set_seed -> EDA -> preprocessing -> features -> model -> train ->
    load best checkpoint -> evaluate on test once -> visualize -> experiments.

    VI: Chạy toàn bộ pipeline Lab 1: cố định seed, EDA, tiền xử lý, trích xuất đặc trưng,
    tạo model, huấn luyện, nạp checkpoint tốt nhất, đánh giá test đúng 1 lần, trực quan hóa và thí nghiệm.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== [Lab 1 Pipeline] Device: {device} | Seed: {seed} ===")

    # 1. Paths setup
    raw_dir = raw_data_dir("lab1")
    train_csv_path = raw_dir / "fashion-mnist_train.csv"
    test_csv_path = raw_dir / "fashion-mnist_test.csv"
    figures_dir = artifact_dir("lab1", "figures")
    splits_dir = artifact_dir("lab1", "splits")
    models_dir = artifact_dir("lab1", "models")
    runs_dir = artifact_dir("lab1", "runs")

    # 2. EDA
    print("\n--- 1. Exploratory Data Analysis (EDA) ---")
    eda_stats = run_eda(str(train_csv_path), str(figures_dir))
    print(f"EDA completed: {eda_stats['n_rows']} rows, {len(eda_stats['class_counts'])} classes, {eda_stats['n_duplicates']} duplicates.")

    # 3. Preprocessing (Data Loading & Split)
    print("\n--- 2. Preprocessing & Splitting ---")
    images, labels = load_csv(str(train_csv_path))
    test_images, test_labels = load_csv(str(test_csv_path))

    split_file = splits_dir / "train_val_split.json"
    if split_file.is_file():
        train_idx, val_idx = load_split(str(split_file))
    else:
        train_idx, val_idx = make_split(labels, val_fraction=0.1, seed=seed)
        save_split(train_idx, val_idx, str(split_file))
    print(f"Split loaded: train={len(train_idx)}, val={len(val_idx)}, test={len(test_labels)}")

    # 4. Features (Normalization & DataLoaders)
    print("\n--- 3. Features & DataLoaders ---")
    mean, std = compute_mean_std(images[train_idx])
    transform = build_transform(mean, std)
    print(f"Train-only normalization statistics: mean={mean:.4f}, std={std:.4f}")

    train_ds = FashionDataset(images, labels, indices=train_idx, transform=transform)
    val_ds = FashionDataset(images, labels, indices=val_idx, transform=transform)
    test_ds = FashionDataset(test_images, test_labels, indices=None, transform=transform)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=64)

    # 5. Model
    print("\n--- 4. Model Instantiation ---")
    model = MLP(hidden_sizes=(256, 128), num_classes=10, dropout=0.0)
    n_params = count_parameters(model)
    print(f"Baseline MLP instantiated: 784 -> 256 -> 128 -> 10 ({n_params:,} trainable parameters)")

    # 6. Train
    print("\n--- 5. Training Baseline Model ---")
    ckpt_path = str(models_dir / "mlp_baseline.pt")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device=device,
        ckpt_path=ckpt_path,
        optimizer_name="adam",
        weight_decay=0.0,
    )
    print(f"Baseline training finished. Best checkpoint saved to: {ckpt_path}")

    # 7. Evaluate (Load Best Checkpoint)
    print("\n--- 6. Evaluation ---")
    eval_model = MLP(hidden_sizes=(256, 128), num_classes=10, dropout=0.0)
    best_model = load_checkpoint(eval_model, ckpt_path, device=device)

    val_loss, val_acc = evaluate(best_model, val_loader, device=device)
    cm = confusion_matrix(best_model, val_loader, device=device, num_classes=10)
    per_class = per_class_accuracy(cm)
    print(f"Validation Performance: loss={val_loss:.4f}, accuracy={val_acc*100:.2f}%")

    test_loss, test_acc = evaluate(best_model, test_loader, device=device)
    print(f"Test Performance (evaluated ONCE): loss={test_loss:.4f}, accuracy={test_acc*100:.2f}%")

    # 8. Visualize
    print("\n--- 7. Visualization ---")
    loss_fig = str(figures_dir / "loss_curves.png")
    pred_fig = str(figures_dir / "predictions_grid.png")
    cm_fig = str(figures_dir / "confusion_matrix.png")

    plot_loss_curves(history, loss_fig)
    show_predictions(best_model, val_loader, CLASS_NAMES, device=device, out_path=pred_fig, n=16)
    plot_confusion_matrix(cm, CLASS_NAMES, cm_fig)
    print("Saved figures: loss_curves.png, predictions_grid.png, confusion_matrix.png")

    # 9. Experiments
    print("\n--- 8. Ablation Experiments ---")
    configs = [
        {"name": "baseline", "hidden_sizes": (256, 128), "lr": 1e-3, "dropout": 0.0, "optimizer_name": "adam", "weight_decay": 0.0, "epochs": epochs, "seed": seed},
        {"name": "optimizer_sgd", "hidden_sizes": (256, 128), "lr": 0.01, "dropout": 0.0, "optimizer_name": "sgd", "weight_decay": 0.0, "epochs": epochs, "seed": seed},
        {"name": "larger_network", "hidden_sizes": (512, 256, 128), "lr": 1e-3, "dropout": 0.0, "optimizer_name": "adam", "weight_decay": 0.0, "epochs": epochs, "seed": seed},
        {"name": "with_dropout_reg", "hidden_sizes": (256, 128), "lr": 1e-3, "dropout": 0.2, "optimizer_name": "adam", "weight_decay": 1e-4, "epochs": epochs, "seed": seed},
    ]
    exp_results = run_experiments(configs, (train_loader, val_loader, test_loader), device=device)
    exp_csv = str(runs_dir / "experiment_results.csv")
    save_results(exp_results, exp_csv)
    print(f"Experiments completed and saved to: {exp_csv}")

    print("\n=== [Lab 1 Pipeline Completed Successfully] ===")
    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "n_params": n_params,
        "experiment_count": len(exp_results),
    }


if __name__ == "__main__":
    main()
