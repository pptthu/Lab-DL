"""Tests for `preprocessing` (Lab 3). No network: the Hugging Face `datasets` module is replaced by a tiny fake.

VI: Test cho `preprocessing` (Lab 3). Không cần mạng: module `datasets` của Hugging Face được thay bằng bản giả nhỏ.
"""
import random
import sys
import types

import pytest

from labs.lab3.preprocessing.preprocessing import clean_text, label_counts, load_text_dataset


class FakeDS:
    """Minimal stand-in for datasets.Dataset. / Bản giả tối thiểu của datasets.Dataset."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0]) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return [r[key] for r in self.rows]

    def shuffle(self, seed):
        rows = self.rows[:]
        random.Random(seed).shuffle(rows)
        return FakeDS(rows)

    def select(self, indices):
        return FakeDS([self.rows[i] for i in indices])

    def map(self, fn, batched=False):
        out = fn({"text": self["text"]})
        return FakeDS([dict(r, text=t) for r, t in zip(self.rows, out["text"])])


def _fake_datasets(n=200):
    """Rows ordered by label like the real Hub data: first half 0, second half 1."""
    def make(prefix):
        return FakeDS([{"text": f"{prefix} review {i}<br />ok", "label": 0 if i < n // 2 else 1} for i in range(n)])
    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda name, cache_dir=None: {"train": make("tr"), "test": make("te")}
    return mod


def test_clean_text_strips_html_and_spaces():
    assert clean_text("Great movie!<br /><br />Loved   it &amp; more.\n\n") == "Great movie! Loved it & more."


def test_clean_text_keeps_case_punctuation_and_less_than_sign():
    assert clean_text("WOW!!! I give it <3 stars, 5 > 4?") == "WOW!!! I give it <3 stars, 5 > 4?"


def test_clean_text_none_and_blank():
    assert clean_text(None) == ""
    assert clean_text("   \n ") == ""


def test_load_text_dataset_sizes_columns_and_balance(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets())
    train, val, test = load_text_dataset("imdb", n_train=60, n_val=20, n_test=30, seed=42)
    assert (len(train), len(val), len(test)) == (60, 20, 30)
    for ds in (train, val, test):
        assert {"text", "label"} <= set(ds.column_names)
        assert "<br" not in " ".join(ds["text"])
    # shuffle before select: the subset must contain BOTH labels although the source is ordered by label
    assert set(label_counts(train)) == {0, 1}


def test_load_text_dataset_is_reproducible_and_disjoint(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets())
    a = load_text_dataset("imdb", n_train=60, n_val=20, n_test=30, seed=7)
    b = load_text_dataset("imdb", n_train=60, n_val=20, n_test=30, seed=7)
    assert a[0]["text"] == b[0]["text"]
    assert not set(a[0]["text"]) & set(a[1]["text"])      # train and val do not overlap


def test_load_text_dataset_rejects_too_large_subset(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets(n=50))
    with pytest.raises(ValueError):
        load_text_dataset("imdb", n_train=40, n_val=20, n_test=10)
