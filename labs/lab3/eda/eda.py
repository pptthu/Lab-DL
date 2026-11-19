"""Module `eda` (EDA (text)) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `eda` (EDA (văn bản)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window needed: figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np


def run_eda(dataset, out_dir: str, text_col: str = 'text', label_col: str = 'label') -> dict:
    """Explore the text dataset (read-only) and save figures.

    Length is counted in words (split on whitespace). Saved files in `out_dir`:
    label_distribution.png (label bar chart), length_distribution.png (length histogram),
    length_by_label.png (extra box plot). The dataset itself is never modified.

    Input : dataset (train split), out_dir, column names
    Output: dict: n_samples, label_counts, length_stats (words); PNG files (label bar chart, length histogram)

    VI: Khám phá dữ liệu văn bản (chỉ đọc) và lưu biểu đồ.
    Độ dài tính bằng số từ (tách theo khoảng trắng). File lưu trong `out_dir`: label_distribution.png
    (biểu đồ cột nhãn), length_distribution.png (histogram độ dài), length_by_label.png (boxplot thêm).
    Dataset không bị sửa.
    Vào : dataset (tập train), out_dir, tên cột
    Ra  : dict: n_samples, label_counts, length_stats (số từ); file PNG (biểu đồ nhãn, histogram độ dài)
    """
    texts = list(dataset[text_col])
    labels = [int(x) for x in dataset[label_col]]
    if not texts:
        raise ValueError("dataset is empty / dataset rỗng")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = dict(sorted(Counter(labels).items()))
    lengths = np.array([len(str(t).split()) for t in texts])
    labels_arr = np.array(labels)
    stats = {
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": int(lengths.min()),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
    }
    keys = [str(k) for k in counts]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))          # 1) label distribution
    bars = ax.bar(keys, list(counts.values()), color="#2563EB")
    ax.bar_label(bars)
    ax.set(xlabel="label (0 = negative, 1 = positive)", ylabel="samples", title="Label distribution")
    fig.tight_layout()
    fig.savefig(out / "label_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))            # 2) length histogram
    ax.hist(lengths, bins=40, color="#2563EB", alpha=0.85)
    ax.axvline(stats["median"], color="#EF4444", linestyle="--", label=f"median = {stats['median']:.0f}")
    ax.axvline(stats["p95"], color="#F59E0B", linestyle="--", label=f"p95 = {stats['p95']:.0f}")
    ax.set(xlabel="words per review", ylabel="samples", title="Length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "length_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))          # 3) length by label
    groups = [lengths[labels_arr == k] for k in counts]
    try:                                                # matplotlib >= 3.9
        ax.boxplot(groups, tick_labels=keys, showfliers=False)
    except TypeError:                                   # older matplotlib
        ax.boxplot(groups, labels=keys, showfliers=False)
    ax.set(xlabel="label", ylabel="words", title="Length by label")
    fig.tight_layout()
    fig.savefig(out / "length_by_label.png", dpi=150)
    plt.close(fig)

    return {"n_samples": len(texts), "label_counts": counts, "length_stats": stats}
