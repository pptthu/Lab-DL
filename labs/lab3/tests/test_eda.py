"""Tests for labs/lab3/eda (offline: fake aclImdb + tiny tokenizer)."""
import numpy as np
import pandas as pd

from labs.lab3.eda.eda import duplicate_counts, length_summary, negation_share, run_eda, token_lengths
from labs.lab3.preprocessing.preprocessing import load_split


def test_length_summary_percentiles_and_over_limits():
    lengths = np.array([10, 100, 129, 300, 600])
    s = length_summary(lengths, limits=(128, 256, 512))
    assert s["median"] == 129
    assert s["max"] == 600
    assert s["pct_over_128"] == 60.0   # 129, 300, 600
    assert s["pct_over_256"] == 40.0   # 300, 600
    assert s["pct_over_512"] == 20.0   # 600
    assert s["p90"] == np.percentile(lengths, 90)


def test_token_lengths_count_special_tokens(tiny_tokenizer):
    # "great movie" -> [CLS] great movie [SEP] = 4 token; không cắt
    assert token_lengths(["great movie", "the film was bad"], tiny_tokenizer).tolist() == [4, 6]


def test_negation_share():
    assert negation_share(["not good", "I don't know", "great", "no way"]) == 75.0
    assert negation_share(["knot", "nobody came"]) == 50.0  # "knot" không phải phủ định


def test_duplicate_counts(fake_imdb_root):
    train, test = load_split("train", fake_imdb_root), load_split("test", fake_imdb_root)
    d = duplicate_counts(train, test)
    assert d["train_extra_copies"] == 1           # "Very good film." 2 bản
    assert d["train_reviews_in_dup_groups"] == 2
    assert d["test_extra_copies"] == 0
    assert d["test_reviews_also_in_train"] == 1   # "Good acting and good plot."
    assert d["train_dup_groups_label_conflict"] == 1  # cùng nội dung, khác nhãn


def test_run_eda_writes_files(fake_imdb_root, tiny_tokenizer, tmp_path):
    train, test = load_split("train", fake_imdb_root), load_split("test", fake_imdb_root)
    out = run_eda(train, test, tiny_tokenizer, tmp_path / "results", tmp_path / "figures")
    for p in out["paths"].values():
        assert p.exists() and p.stat().st_size > 0
    summary = pd.read_csv(out["paths"]["summary"])
    assert summary["n_reviews"].sum() == len(train) + len(test)
    assert set(summary["split"]) == {"train", "test"}
    # <br /> bị tách thành nhiều token -> bản gốc dài hơn bản đã làm sạch
    assert summary.loc[summary["split"] == "train", "br_tokens_mean"].item() > 0
    lengths = pd.read_csv(out["paths"]["token_lengths"])
    assert len(lengths) == len(train) + len(test)
