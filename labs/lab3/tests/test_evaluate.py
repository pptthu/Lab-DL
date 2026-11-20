"""Tests for labs/lab3/evaluate (offline: hand-computed metrics, fake pipeline, tiny model)."""
import json

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from labs.lab3.evaluate.evaluate import (
    accuracy_f1, compute_metrics, evaluate_control_sentences, evaluate_on_test, evaluate_zero_shot,
    softmax_pos, summarize, duplicate_test_uids,
)
from labs.lab3.features.features import make_collator, tokenize_datasets


class FakePipe:
    """Sentiment pipeline stand-in: 'bad'/'not'/'waste'/'dull' -> NEGATIVE."""

    def __call__(self, texts, truncation=True, batch_size=1):
        neg = ("bad", "not", "waste", "dull", "boring")
        return [{"label": "NEGATIVE" if any(w in t.lower() for w in neg) else "POSITIVE", "score": 0.8}
                for t in texts]


def test_metrics_match_hand_computation():
    # labels 1,1,0,0,1 ; preds 1,0,0,1,1 → acc 3/5; TP=2 FP=1 FN=1 → F1 = 2/3
    m = accuracy_f1([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert m["accuracy"] == pytest.approx(0.6) and m["f1"] == pytest.approx(2 / 3)
    logits = np.array([[0.1, 2.0], [3.0, 0.0], [0.0, 1.0]])
    assert compute_metrics((logits, np.array([1, 0, 0])))["accuracy"] == pytest.approx(2 / 3)


def test_softmax_pos():
    p = softmax_pos(np.array([[0.0, 0.0], [0.0, 10.0]]))
    assert p[0] == pytest.approx(0.5) and p[1] > 0.999


def test_zero_shot_maps_labels_and_runs_once(tmp_path):
    ds = Dataset.from_dict({"uid": ["a", "b"], "text": ["great film", "bad film"], "label": [1, 0]})
    out = evaluate_zero_shot(ds, pipe=FakePipe(), results_dir=tmp_path)
    df = pd.read_csv(out["predictions"])
    assert df["pred"].tolist() == [1, 0] and df["prob_pos"].round(2).tolist() == [0.8, 0.2]
    assert out["accuracy"] == 1.0
    with pytest.raises(RuntimeError, match="once"):
        evaluate_zero_shot(ds, pipe=FakePipe(), results_dir=tmp_path)


def test_evaluate_on_test_writes_predictions_and_refuses_second_call(tmp_path, tiny_tokenizer):
    from transformers import DistilBertConfig

    from labs.lab3.model.model import build_model

    model = build_model(seed=0, model_config=DistilBertConfig(vocab_size=len(tiny_tokenizer), dim=16, n_layers=1,
                                                               n_heads=2, hidden_dim=32, max_position_embeddings=64))
    ds = Dataset.from_dict({"uid": ["a", "b", "c"], "text": ["great movie", "bad film", "i love it"],
                            "label": [1, 0, 1]})
    tok = tokenize_datasets(ds, tiny_tokenizer, max_length=16)
    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(json.dumps({"run": "r", "test_evaluated": False}))
    out = evaluate_on_test(None, tok, "r", collator=make_collator(tiny_tokenizer), run_config_path=cfg_path,
                           results_dir=tmp_path, model=model)
    df = pd.read_csv(out["predictions"])
    assert list(df.columns) == ["uid", "label", "pred", "prob_pos"] and df["uid"].tolist() == ["a", "b", "c"]
    assert df["prob_pos"].between(0, 1).all()
    assert json.loads(cfg_path.read_text())["test_evaluated"] is True
    out["predictions"].unlink()            # cờ trong _config.json vẫn chặn
    with pytest.raises(RuntimeError, match="already evaluated"):
        evaluate_on_test(None, tok, "r", collator=make_collator(tiny_tokenizer), run_config_path=cfg_path,
                         results_dir=tmp_path, model=model)


def test_summarize_mean_std_pass_and_no_dup(tmp_path):
    def write(name, preds):
        p = tmp_path / f"{name}.csv"
        pd.DataFrame({"uid": list("abcd"), "label": [1, 1, 0, 0], "pred": preds, "prob_pos": 0.5}).to_csv(p, index=False)
        return p

    files = {"s1": write("s1", [1, 1, 0, 0]), "s2": write("s2", [1, 1, 0, 1]), "s3": write("s3", [1, 0, 0, 1])}
    t = summarize(files, write("zs", [1, 0, 0, 0]), seeds={"s1": 1, "s2": 2, "s3": 3}, duplicate_uids={"d"},
                  out_path=tmp_path / "test_summary.csv", pass_accuracy=0.7)
    mean = t[t["run"] == "mean"].iloc[0]
    std = t[t["run"] == "std"].iloc[0]
    assert mean["accuracy"] == pytest.approx(0.75) and std["accuracy"] == pytest.approx(0.25)  # 1, .75, .5
    assert bool(mean["pass"]) is True
    assert t[t["run"] == "s2"].iloc[0]["accuracy_no_dup"] == pytest.approx(1.0)  # bỏ review d (sai)
    assert (tmp_path / "test_summary.csv").exists() and "zero_shot" in t["run"].tolist()


def test_duplicate_test_uids_use_cleaned_text():
    train = pd.DataFrame({"uid": ["t1"], "text": ["Good  film.<br />Really"]})
    test = pd.DataFrame({"uid": ["x1", "x2"], "text": ["Good film. Really", "Other"]})
    assert duplicate_test_uids(train, test) == {"x1"}


def test_control_sentences_two_models(tmp_path):
    t = evaluate_control_sentences("unused", "seed42", pipe_zero_shot=FakePipe(), pipe_finetuned=FakePipe(),
                                   out_path=tmp_path / "control.csv")
    assert len(t) == 12 and set(t["model"]) == {"zero-shot (SST-2)", "fine-tuned (seed42)"}
    assert t["correct"].isna().sum() == 2   # câu trung tính không có nhãn đúng
