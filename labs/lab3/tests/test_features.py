"""Tests for labs/lab3/features (offline: tiny tokenizer, small in-memory DatasetDict)."""
import pytest
from datasets import Dataset, DatasetDict

from labs.lab3.features.features import make_collator, tokenize_datasets


def small_ds():
    rows = {"uid": ["a", "b"], "text": ["great movie", "the film was very very very boring and bad"],
            "label": [1, 0], "rating": [9, 1], "movie_id": ["tt1", "tt2"]}
    return DatasetDict({"train": Dataset.from_dict(rows), "test": Dataset.from_dict(rows)})


def test_tokenize_keeps_uid_label_and_truncates(tiny_tokenizer):
    out = tokenize_datasets(small_ds(), tiny_tokenizer, max_length=6)
    for split in out.values():
        assert set(split.column_names) == {"uid", "label", "input_ids", "attention_mask"}
        assert all(len(ids) <= 6 for ids in split["input_ids"])
        assert all(ids[0] == tiny_tokenizer.cls_token_id for ids in split["input_ids"])
        assert all(ids[-1] == tiny_tokenizer.sep_token_id for ids in split["input_ids"])
    assert len(out["train"]["input_ids"][0]) == 4  # câu ngắn không bị pad cố định


def test_tokenize_requires_max_length(tiny_tokenizer):
    with pytest.raises(ValueError, match="max_length"):
        tokenize_datasets(small_ds(), tiny_tokenizer, max_length=None)


def test_collator_pads_to_longest_in_batch(tiny_tokenizer):
    tok = tokenize_datasets(small_ds(), tiny_tokenizer, max_length=8)["train"]
    batch = make_collator(tiny_tokenizer)([{k: tok[i][k] for k in ("input_ids", "attention_mask", "label")}
                                          for i in range(2)])
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert batch["labels"].tolist() == [1, 0]
