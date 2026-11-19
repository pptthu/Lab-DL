"""EDA of the official IMDB splits: labels, ratings, token lengths, <br /> cost, duplicates, negation.

VI: Khám phá dữ liệu IMDB (train, test chính thức) trước khi chia val.
Số liệu ở đây là căn cứ cho: làm sạch <br /> và gộp review trùng (preprocessing),
chọn max_length (plan §4 S1), bỏ nhóm phủ định trong phân tích lỗi (visualize).
Desc: labs/lab3/docs/phases/eda_v1.md

Run from the repo root / Chạy từ gốc repo:
    python -m labs.lab3.eda.eda --smoke   # vài chục review, đầu ra vào outputs/_smoke/
    python -m labs.lab3.eda.eda           # full 50k review
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from labs.lab3 import config  # noqa: E402
from labs.lab3.preprocessing.preprocessing import (  # noqa: E402
    clean_text, download_imdb, load_split, sample_per_label, text_hash,
)
from shared.paths import output_dir  # noqa: E402

LENGTH_LIMITS = config.MAX_LENGTH_CANDIDATES  # 128, 256, 512
# Dấu hiệu phủ định: "not", "no" chiếm > 55% dấu hiệu phủ định trong SFU Review corpus
# (Konstantinova et al. 2012); thêm các từ phủ định thường gặp khác.
NEGATION_RE = re.compile(r"\b(not|no|never|nothing|none|nor|neither|nobody|without)\b|n't", re.IGNORECASE)


# ---------------------------------------------------------------- tính số liệu
def token_lengths(texts: list[str], tokenizer, batch_size: int = 1000) -> np.ndarray:
    """Number of tokens per text, special tokens included, NO truncation (count only).

    VI: Số token mỗi văn bản (gồm [CLS], [SEP]), không cắt — chỉ đếm, không đưa vào model.
    """
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], add_special_tokens=True, truncation=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return np.asarray(out, dtype=np.int64)


def length_summary(lengths: np.ndarray, limits=LENGTH_LIMITS) -> dict:
    """Mean, median, P90, P95, max and % of texts longer than each limit (= truncated).

    VI: Thống kê độ dài; review có số token > max_length sẽ bị cắt.
    """
    lengths = np.asarray(lengths)
    out = {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
    }
    for limit in limits:
        out[f"pct_over_{limit}"] = float((lengths > limit).mean() * 100)
    return out


def duplicate_counts(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Duplicate reviews (same cleaned text) inside train, inside test and between them.

    VI: Đếm review trùng nội dung. "extra" = số bản thừa (nhóm 3 bản giống nhau → 2 bản thừa).
    """
    h_train = train["text"].map(text_hash)
    h_test = test["text"].map(text_hash)
    return {
        "train_extra_copies": int(h_train.duplicated().sum()),
        "train_reviews_in_dup_groups": int(h_train.duplicated(keep=False).sum()),
        "test_extra_copies": int(h_test.duplicated().sum()),
        "test_reviews_in_dup_groups": int(h_test.duplicated(keep=False).sum()),
        "test_reviews_also_in_train": int(h_test.isin(set(h_train)).sum()),
        "train_dup_groups_label_conflict": int(
            train.assign(h=h_train).groupby("h")["label"].nunique().gt(1).sum()
        ),
    }


def negation_share(texts) -> float:
    """% of texts with at least one negation cue. / % văn bản có ít nhất một từ phủ định."""
    texts = list(texts)
    return float(np.mean([bool(NEGATION_RE.search(t)) for t in texts]) * 100)


def split_summary(df: pd.DataFrame, tokenizer) -> tuple[dict, np.ndarray]:
    """All numbers for one split + the token length of every cleaned review.

    VI: Số liệu của một split; trả thêm độ dài token từng review (đã làm sạch).
    """
    raw = df["text"].tolist()
    clean = [clean_text(t) for t in raw]
    len_raw = token_lengths(raw, tokenizer)
    len_clean = token_lengths(clean, tokenizer)
    row = {
        "split": df["split"].iloc[0],
        "n_reviews": len(df),
        "n_pos": int((df["label"] == 1).sum()),
        "n_neg": int((df["label"] == 0).sum()),
        "n_movies": int(df["movie_id"].nunique()),
        "max_reviews_per_movie": int(df["movie_id"].value_counts().max()),
        "words_median": float(np.median([len(t.split()) for t in clean])),
        "pct_with_br": float(np.mean(["<br" in t.lower() for t in raw]) * 100),
        "br_tokens_mean": float((len_raw - len_clean).mean()),
        "pct_with_negation": negation_share(clean),
    }
    row.update({f"tokens_{k}": v for k, v in length_summary(len_clean).items()})
    return row, len_clean


# ---------------------------------------------------------------- hình
def plot_label_distribution(summary: pd.DataFrame, path: Path) -> None:
    """Bar chart of neg/pos counts per split. / Phân bố nhãn mỗi split."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["n_neg"], 0.4, label="negative (0)")
    ax.bar(x + 0.2, summary["n_pos"], 0.4, label="positive (1)")
    ax.set_xticks(x, summary["split"])
    ax.set_ylabel("số review")
    ax.set_title("Phân bố nhãn")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_length_histogram(lengths: pd.DataFrame, path: Path) -> None:
    """Histogram of token lengths (cleaned text) with lines at 128/256/512.

    VI: Histogram độ dài token; vạch dọc = các giá trị max_length đang xét.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    upper = int(np.percentile(lengths["n_tokens"], 99.5))
    bins = np.linspace(0, upper, 80)
    for split, g in lengths.groupby("split"):
        ax.hist(g["n_tokens"].clip(upper=upper), bins=bins, alpha=0.5, label=split)
    for limit in LENGTH_LIMITS:
        pct = (lengths["n_tokens"] > limit).mean() * 100
        ax.axvline(limit, color="k", ls="--", lw=1)
        ax.text(limit, ax.get_ylim()[1] * 0.95, f" {limit}\n >{limit}: {pct:.1f}%", fontsize=8, va="top")
    ax.set_xlabel("số token (đã làm sạch, gồm [CLS]/[SEP]; cắt hiển thị ở P99.5)")
    ax.set_ylabel("số review")
    ax.set_title("Độ dài review theo token (DistilBERT uncased)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_length_by_label(lengths: pd.DataFrame, path: Path) -> None:
    """Box plot of token length by label (train + test). / Độ dài theo nhãn."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    data = [lengths.loc[lengths["label"] == lab, "n_tokens"] for lab in (0, 1)]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks([1, 2], ["negative (0)", "positive (1)"])
    for limit in LENGTH_LIMITS:
        ax.axhline(limit, color="k", ls="--", lw=0.8)
    ax.set_ylabel("số token")
    ax.set_title("Độ dài theo nhãn (không vẽ điểm ngoại lai)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- chạy
def run_eda(train: pd.DataFrame, test: pd.DataFrame, tokenizer, results_dir: Path, figures_dir: Path) -> dict:
    """Write eda_summary.csv, eda_duplicates.csv, eda_ratings.csv, eda_token_lengths.csv and 3 figures.

    VI: Ghi 4 CSV + 3 hình; trả về dict đường dẫn và bảng tóm tắt.
    """
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows, length_parts = [], []
    for df in (train, test):
        row, lens = split_summary(df, tokenizer)
        rows.append(row)
        length_parts.append(pd.DataFrame({
            "uid": df["uid"].values, "split": df["split"].values,
            "label": df["label"].values, "rating": df["rating"].values, "n_tokens": lens,
        }))
    summary = pd.DataFrame(rows)
    lengths = pd.concat(length_parts, ignore_index=True)
    dups = pd.DataFrame([duplicate_counts(train, test)])
    ratings = (pd.concat([train, test]).groupby(["split", "rating"]).size()
               .unstack("split", fill_value=0).reset_index())

    paths = {
        "summary": results_dir / "eda_summary.csv",
        "duplicates": results_dir / "eda_duplicates.csv",
        "ratings": results_dir / "eda_ratings.csv",
        "token_lengths": results_dir / "eda_token_lengths.csv",
        "fig_labels": figures_dir / "eda_label_distribution.png",
        "fig_lengths": figures_dir / "eda_token_length_hist.png",
        "fig_by_label": figures_dir / "eda_token_length_by_label.png",
    }
    summary.round(3).to_csv(paths["summary"], index=False)
    dups.to_csv(paths["duplicates"], index=False)
    ratings.to_csv(paths["ratings"], index=False)
    lengths.to_csv(paths["token_lengths"], index=False)
    plot_label_distribution(summary, paths["fig_labels"])
    plot_length_histogram(lengths, paths["fig_lengths"])
    plot_length_by_label(lengths, paths["fig_by_label"])
    return {"paths": paths, "summary": summary, "duplicates": dups, "ratings": ratings}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Lab 3 EDA")
    parser.add_argument("--smoke", action="store_true", help="vài chục review, ghi vào outputs/_smoke/")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # bỏ cảnh báo "longer than 512": chỉ đếm, không đưa vào model
    t0 = time.time()
    download_imdb()
    train, test = load_split("train"), load_split("test")
    if args.smoke:
        train = sample_per_label(train, config.SMOKE_N_PER_SPLIT)
        test = sample_per_label(test, config.SMOKE_N_PER_SPLIT)
        base = output_dir(config.LAB, "_smoke")
        results_dir, figures_dir = base / "results", base / "figures"
    else:
        results_dir, figures_dir = output_dir(config.LAB, "results"), output_dir(config.LAB, "figures")
    tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT)
    out = run_eda(train, test, tokenizer, results_dir, figures_dir)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(out["summary"].round(2).T)
        print(out["duplicates"].T)
    for name, p in out["paths"].items():
        print(f"{name}: {p}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
