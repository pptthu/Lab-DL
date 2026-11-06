"""Extended study of Lab 1: longer training.

Run from the repo root, after main.py:  python -m labs.lab1.extended_study
Question: within 20 epochs, does the baseline overfit, and does dropout change the 5-epoch finding?

VI: Nghiên cứu mở rộng của Lab 1: huấn luyện lâu hơn.
Chạy từ gốc repo, sau main.py: python -m labs.lab1.extended_study
Câu hỏi: trong 20 epoch, baseline có overfit không, và dropout có làm đổi kết luận của 5 epoch không?
"""
from __future__ import annotations

import sys
import torch

from shared.paths import artifact_dir, raw_data_dir
from labs.lab1.main import CLASS_NAMES, N_THREADS, SEEDS
from labs.lab1.preprocessing.preprocessing import load_csv, load_split
from labs.lab1.features.features import compute_mean_std, build_transform, FashionDataset, make_loaders
from labs.lab1.experiment.experiment import analyze_seed_runs, run_experiments, save_results

EPOCHS = 20
# The test set is reported only for the baseline, fixed before the run.
# VI: Tập test chỉ báo cho baseline, chốt trước khi chạy.
VARIANTS = {
    "baseline_20": {},
    "dropout_0.2_20": {"dropout": 0.2},
}
TEST_REPORTED = ("baseline_20",)


def main() -> None:
    """Train every variant with every seed, then save the summaries of each one.

    VI: Huấn luyện mọi biến thể với mọi seed, rồi lưu bảng tổng hợp của từng biến thể.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(N_THREADS)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_dir = raw_data_dir("lab1")
    images, labels = load_csv(str(raw_dir / "fashion-mnist_train.csv"))
    test_images, test_labels = load_csv(str(raw_dir / "fashion-mnist_test.csv"))
    train_idx, val_idx = load_split(str(artifact_dir("lab1", "splits") / "train_val_split.json"))
    mean, std = compute_mean_std(images[train_idx])
    transform = build_transform(mean, std)
    loaders = make_loaders(
        FashionDataset(images, labels, indices=train_idx, transform=transform),
        FashionDataset(images, labels, indices=val_idx, transform=transform),
        FashionDataset(test_images, test_labels, indices=None, transform=transform),
        batch_size=64,
    )

    base = {"hidden_sizes": (256, 128), "lr": 1e-3, "dropout": 0.0, "optimizer_name": "adam", "weight_decay": 0.0, "epochs": EPOCHS}
    configs = [
        {**base, **change, "name": name, "seed": s}
        for name, change in VARIANTS.items()
        for s in SEEDS
    ]
    results = run_experiments(configs, loaders, device=device, ckpt_prefix="ext_run")

    runs_dir = artifact_dir("lab1", "runs")
    save_results(results, str(runs_dir / "extended_results.csv"))
    for name in VARIANTS:
        runs = [r for r in results if r["config"]["name"] == name]
        summary = analyze_seed_runs(
            runs, loaders, CLASS_NAMES, device=device, out_dir=str(runs_dir),
            prefix=f"{name}_", use_test=name in TEST_REPORTED,
        )
        print(f"{name}: {summary}")


if __name__ == "__main__":
    main()
