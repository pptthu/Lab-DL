"""Tests for labs/lab3/sentiment_demo (offline: fake pipeline + tiny tokenizer)."""
import pandas as pd

from labs.lab3.sentiment_demo.sentiment_demo import (
    CONTROL_SENTENCES, predict_sentiment, run_demo, tokenize_sentence,
)


class FakePipe:
    """Mimics transformers' sentiment pipeline: 'not'/'waste'/'dull' -> NEGATIVE."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model = type("M", (), {"name_or_path": "fake-sst2",
                                    "config": type("C", (), {"_commit_hash": "abc123"})()})()
        self.calls = []

    def __call__(self, texts, truncation=True, batch_size=1):
        self.calls.append({"truncation": truncation, "n": len(texts)})
        neg = ("not", "waste", "dull")
        return [{"label": "NEGATIVE" if any(w in t.lower() for w in neg) else "POSITIVE", "score": 0.9}
                for t in texts]


def test_control_sentences_cover_checklist_capabilities():
    assert len(CONTROL_SENTENCES) == 6
    assert {s["expected"] for s in CONTROL_SENTENCES} == {"POSITIVE", "NEGATIVE", None}
    assert any("not" in s["text"].lower() for s in CONTROL_SENTENCES)


def test_tokenize_sentence_starts_with_cls_ends_with_sep(tiny_tokenizer):
    tok = tokenize_sentence("I love this movie", tiny_tokenizer)
    assert tok["tokens"][0] == "[CLS]" and tok["tokens"][-1] == "[SEP]"
    assert len(tok["tokens"]) == len(tok["input_ids"]) == len(tok["attention_mask"])
    assert set(tok["attention_mask"]) == {1}


def test_predict_sentiment_uses_truncation(tiny_tokenizer):
    pipe = FakePipe(tiny_tokenizer)
    out = predict_sentiment(["great", "not good"], pipe)
    assert [o["label"] for o in out] == ["POSITIVE", "NEGATIVE"]
    assert all(0.0 <= o["score"] <= 1.0 for o in out)
    assert pipe.calls[0]["truncation"] is True


def test_run_demo_writes_both_csv(tiny_tokenizer, tmp_path):
    out = run_demo(tmp_path, pipe=FakePipe(tiny_tokenizer))
    preds = pd.read_csv(out["paths"]["predictions"])
    toks = pd.read_csv(out["paths"]["tokens"])
    assert len(preds) == 6
    assert set(preds.columns) >= {"capability", "text", "expected", "predicted", "score", "correct", "revision"}
    assert set(preds["predicted"]) <= {"POSITIVE", "NEGATIVE"}
    assert preds["revision"].iloc[0] == "abc123"
    assert preds.loc[preds["expected"].isna(), "correct"].isna().all()  # câu trung tính: không chấm
    assert toks["token"].iloc[0] == "[CLS]" and toks["token"].iloc[-1] == "[SEP]"
