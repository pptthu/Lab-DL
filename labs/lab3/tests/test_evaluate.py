"""Tests for `evaluate` (Lab 3). Fake data and a fake `torch`: no network, no model, no download.

VI: Test cho `evaluate` (Lab 3). Dữ liệu giả và `torch` giả: không cần mạng, không cần model, không tải gì.
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from labs.lab3.evaluate.evaluate import (
    classification_metrics,
    compute_metrics,
    evaluate_on_test,
    predict_texts,
    zero_shot_labels,
)


# ---------- compute_metrics / classification_metrics ----------
def _logits(preds):
    """Logits whose argmax is `preds` (0/1). / Logits có argmax bằng `preds`."""
    return np.array([[2.0, -1.0] if p == 0 else [-1.0, 2.0] for p in preds])


def test_compute_metrics_hand_computed_values():
    # labels 1,1,0,0,1 ; preds 1,0,0,1,1 -> tp=2 fp=1 fn=1 -> accuracy 3/5, f1 = 4/6
    m = compute_metrics((_logits([1, 0, 0, 1, 1]), np.array([1, 1, 0, 0, 1])))
    assert set(m) == {"accuracy", "f1"}
    assert m["accuracy"] == pytest.approx(0.6) and m["f1"] == pytest.approx(4 / 6)
    assert all(type(v) is float for v in m.values())


def test_compute_metrics_perfect_and_all_wrong():
    y = np.array([0, 1, 1, 0])
    assert compute_metrics((_logits(y), y)) == {"accuracy": 1.0, "f1": 1.0}
    assert compute_metrics((_logits(1 - y), y)) == {"accuracy": 0.0, "f1": 0.0}


def test_f1_is_zero_without_division_error_when_class_1_never_hit():
    m = compute_metrics((_logits([0, 0, 0]), np.array([0, 0, 0])))  # no positive at all
    assert m == {"accuracy": 1.0, "f1": 0.0}


def test_compute_metrics_accepts_eval_prediction_like_object_and_tuple_logits():
    class EvalPred:  # like transformers.EvalPrediction: indexable, 0 = predictions, 1 = label_ids
        def __getitem__(self, i):
            return (_logits([1, 0]), np.array([1, 1]))[i]

    assert compute_metrics(EvalPred())["accuracy"] == pytest.approx(0.5)
    assert compute_metrics(((_logits([1, 0]), "extra"), np.array([1, 0])))["accuracy"] == 1.0


def test_compute_metrics_rejects_bad_shapes():
    with pytest.raises(ValueError):
        compute_metrics((np.array([0.1, 0.9]), np.array([1, 0])))  # logits not 2-D
    with pytest.raises(ValueError):
        compute_metrics((_logits([1, 0, 1]), np.array([1, 0])))  # different lengths


def test_classification_metrics_empty_raises():
    with pytest.raises(ValueError):
        classification_metrics([], [])


# ---------- evaluate_on_test ----------
class FakeTrainer:
    def __init__(self, raw):
        self.raw, self.calls = raw, []

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval"):
        self.calls.append((eval_dataset, metric_key_prefix))
        return self.raw


def test_evaluate_on_test_uses_test_prefix_and_strips_it():
    tr = FakeTrainer({"test_loss": 0.31, "test_accuracy": 0.9, "test_f1": 0.89, "test_runtime": 3, "epoch": 2.0})
    out = evaluate_on_test(tr, [1, 2, 3])
    assert tr.calls == [([1, 2, 3], "test")]
    assert out["accuracy"] == 0.9 and out["f1"] == 0.89 and out["loss"] == 0.31 and out["epoch"] == 2.0
    assert not any(k.startswith("test_") for k in out)


def test_evaluate_on_test_empty_raises():
    with pytest.raises(ValueError):
        evaluate_on_test(FakeTrainer({}), [])


# ---------- predict_texts (fake torch: no real torch needed) ----------
class T:  # tiny tensor made of a numpy array / tensor giả bằng numpy
    def __init__(self, a):
        self.a = np.asarray(a)

    def to(self, device):
        return self

    def max(self, dim=-1):
        return T(self.a.max(axis=dim)), T(self.a.argmax(axis=dim))

    def tolist(self):
        return self.a.tolist()


@pytest.fixture
def fake_env(monkeypatch):
    seen = {"batches": [], "model_kwargs": []}

    class NoGrad:
        def __enter__(self):
            seen["no_grad"] = True

        def __exit__(self, *a):
            return False

    def softmax(t, dim=-1):
        e = np.exp(t.a - t.a.max(axis=dim, keepdims=True))
        return T(e / e.sum(axis=dim, keepdims=True))

    torch = types.ModuleType("torch")
    torch.no_grad, torch.softmax = NoGrad, softmax
    monkeypatch.setitem(sys.modules, "torch", torch)

    def tokenizer(chunk, truncation, max_length, padding, return_tensors):
        seen["batches"].append((list(chunk), truncation, max_length, padding, return_tensors))
        return {"input_ids": T([[1]] * len(chunk)), "attention_mask": T([[1]] * len(chunk)), "token_type_ids": T([[0]] * len(chunk))}

    class Model:
        training = True

        def __init__(self):
            self.mode_calls = []

        def parameters(self):
            return iter([SimpleNamespace(device="cpu")])

        def eval(self):
            self.mode_calls.append("eval")
            self.training = False

        def train(self, mode=True):
            self.mode_calls.append(("train", mode))
            self.training = mode

        def __call__(self, **kw):
            seen["model_kwargs"].append(sorted(kw))
            # "good" text -> class 1 ; any other text -> class 0 (decided from the text length parity)
            n = len(kw["input_ids"].a)
            return SimpleNamespace(logits=T([[0.0, 3.0] if (len(seen["batches"][-1][0][i]) % 2 == 0) else [3.0, 0.0] for i in range(n)]))

    return Model(), tokenizer, seen


def test_predict_texts_output_format_order_and_batching(fake_env):
    model, tok, seen = fake_env
    texts = ["aa", "b", "cccc", "d", "ee"]  # even length -> class 1, odd -> class 0
    out = predict_texts(model, tok, texts, batch_size=2, max_length=64)
    assert [o["text"] for o in out] == texts
    assert [o["label"] for o in out] == [1, 0, 1, 0, 1]
    assert all(type(o["label"]) is int and type(o["score"]) is float and 0.5 < o["score"] <= 1.0 for o in out)
    assert out[0]["score"] == pytest.approx(np.exp(3) / (np.exp(3) + 1))
    assert [b[0] for b in seen["batches"]] == [["aa", "b"], ["cccc", "d"], ["ee"]]
    assert all(b[1:] == (True, 64, True, "pt") for b in seen["batches"])  # truncation, max_length, padding, tensors
    assert seen["no_grad"]
    assert all("token_type_ids" not in k for k in seen["model_kwargs"])


def test_predict_texts_restores_training_mode(fake_env):
    model, tok, _ = fake_env
    predict_texts(model, tok, ["x"])
    assert model.mode_calls == ["eval", ("train", True)] and model.training is True


def test_predict_texts_empty_list_returns_empty_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # importing torch would raise ImportError
    assert predict_texts(object(), object(), []) == []


@pytest.mark.parametrize("bad", ["one review", ["ok", 3], None])
def test_predict_texts_rejects_non_list_of_str(bad):
    with pytest.raises(TypeError):
        predict_texts(object(), object(), bad)


# ---------- zero_shot_labels ----------
def test_zero_shot_labels_maps_positive_to_1_negative_to_0():
    res = [{"sentence": "a", "label": "POSITIVE", "score": 0.9}, {"sentence": "b", "label": "NEGATIVE", "score": 0.8}]
    assert zero_shot_labels(res) == [1, 0]


def test_zero_shot_labels_rejects_unknown_label():
    with pytest.raises(ValueError):
        zero_shot_labels([{"label": "NEUTRAL"}])
