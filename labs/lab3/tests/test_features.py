"""Tests for `features` (Lab 3). A fake tokenizer and a fake dataset are used: no network, no `transformers`.

VI: Test cho `features` (Lab 3). Dùng tokenizer và dataset giả: không cần mạng, không cần `transformers`.
"""
import pytest

from labs.lab3.features.features import tokenize_dataset


class FakeTokenizer:
    """Turns each word into an id; honours truncation/max_length like a real tokenizer."""

    def __call__(self, texts, truncation=False, max_length=None, return_token_type_ids=True):
        ids = [[101] + [len(w) for w in t.split()] + [102] for t in texts]
        if truncation and max_length:
            ids = [x[:max_length] for x in ids]
        out = {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}
        if return_token_type_ids:                      # like a real BERT-style tokenizer
            out["token_type_ids"] = [[0] * len(x) for x in ids]
        return out


class FakeDS:
    def __init__(self, columns):
        self.columns = dict(columns)
        self.column_names = list(self.columns)

    def __getitem__(self, key):
        return self.columns[key]

    def map(self, fn, batched=False, remove_columns=None):
        new = fn(self.columns)
        cols = {k: v for k, v in self.columns.items() if k not in (remove_columns or [])}
        cols.update(new)
        return FakeDS(cols)


def _ds():
    return FakeDS({"text": ["I love this movie a lot", "bad"], "label": [1, 0]})


def test_tokenize_dataset_has_three_columns_and_drops_text():
    out = tokenize_dataset(_ds(), FakeTokenizer(), max_length=256)
    assert set(out.column_names) == {"input_ids", "attention_mask", "label"}
    assert out["label"] == [1, 0]


def test_tokenize_dataset_truncates_to_max_length():
    out = tokenize_dataset(_ds(), FakeTokenizer(), max_length=4)
    assert all(len(x) <= 4 for x in out["input_ids"])
    assert all(len(a) == len(i) for a, i in zip(out["attention_mask"], out["input_ids"]))


def test_tokenize_dataset_missing_column_raises():
    with pytest.raises(KeyError):
        tokenize_dataset(_ds(), FakeTokenizer(), text_col="review")
