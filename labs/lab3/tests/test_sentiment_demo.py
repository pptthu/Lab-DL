"""Tests for `sentiment_demo` (Lab 3). Fake tokenizer and pipeline: no network, no model download.

VI: Test cho `sentiment_demo` (Lab 3). Tokenizer và pipeline giả: không cần mạng, không tải model.
"""
import pytest

from labs.lab3.sentiment_demo import sentiment_demo as sd


class FakeTokenizer:
    vocab = {101: "[CLS]", 102: "[SEP]"}

    def __call__(self, sentence):
        ids = [101] + [1000 + i for i, _ in enumerate(sentence.split())] + [102]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        return [self.vocab.get(i, f"tok{i}") for i in ids]


def fake_pipeline(sentences, truncation=True, batch_size=1):
    return [{"label": "NEGATIVE" if "boring" in s else "POSITIVE", "score": 0.9} for s in sentences]


def test_show_tokens_contract(monkeypatch):
    monkeypatch.setattr(sd, "_get_tokenizer", lambda name: FakeTokenizer())
    out = sd.show_tokens("I love it")
    assert set(out) == {"tokens", "input_ids"}
    assert out["tokens"][0] == "[CLS]" and out["tokens"][-1] == "[SEP]"
    assert len(out["tokens"]) == len(out["input_ids"]) == 5


def test_show_tokens_rejects_non_string():
    with pytest.raises(TypeError):
        sd.show_tokens(123)


def test_analyze_sentences_contract(monkeypatch):
    monkeypatch.setattr(sd, "_get_pipeline", lambda name: fake_pipeline)
    out = sd.analyze_sentences(["I love this movie", "The plot was boring", "Not bad at all"])
    assert [set(r) for r in out] == [{"sentence", "label", "score"}] * 3
    assert out[0]["label"] == "POSITIVE" and out[1]["label"] == "NEGATIVE"
    assert all(isinstance(r["score"], float) for r in out)


def test_analyze_sentences_accepts_single_string(monkeypatch):
    monkeypatch.setattr(sd, "_get_pipeline", lambda name: fake_pipeline)
    assert len(sd.analyze_sentences("so boring")) == 1
