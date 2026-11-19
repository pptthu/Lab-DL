"""Tests for `model` (Lab 3). A fake `transformers` module is injected: no network, no download.

VI: Test cho `model` (Lab 3). Chèn module `transformers` giả: không cần mạng, không tải gì.
"""
import sys
import types

import pytest

from labs.lab3.model.model import load_model_and_tokenizer


@pytest.fixture
def fake_hf(monkeypatch):
    calls = {}

    class Tok:
        @staticmethod
        def from_pretrained(name):
            calls["tok"] = name
            return ("tokenizer", name)

    class Mod:
        @staticmethod
        def from_pretrained(name, **kw):
            calls["model"] = (name, kw)
            return ("model", name)

    mod = types.ModuleType("transformers")
    mod.AutoTokenizer, mod.AutoModelForSequenceClassification = Tok, Mod
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return calls


def test_returns_model_and_tokenizer_from_same_checkpoint(fake_hf):
    model, tok = load_model_and_tokenizer()
    assert model == ("model", "distilbert-base-uncased") and tok == ("tokenizer", "distilbert-base-uncased")
    assert fake_hf["tok"] == fake_hf["model"][0]


def test_binary_gets_label_names(fake_hf):
    load_model_and_tokenizer(num_labels=2)
    kw = fake_hf["model"][1]
    assert kw["num_labels"] == 2 and kw["id2label"] == {0: "NEGATIVE", 1: "POSITIVE"}


def test_three_labels_has_no_binary_names(fake_hf):
    load_model_and_tokenizer(num_labels=3)
    assert "id2label" not in fake_hf["model"][1]


@pytest.mark.parametrize("bad", [1, 0, -2, "2", None])
def test_invalid_num_labels_raises(bad):
    with pytest.raises(ValueError):
        load_model_and_tokenizer(num_labels=bad)
