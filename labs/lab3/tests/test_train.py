"""Tests for `train` (Lab 3). A fake `transformers` module is injected: no network, no model.

VI: Test cho `train` (Lab 3). Chèn module `transformers` giả: không cần mạng, không cần model.
"""
import dataclasses
import sys
import types

import pytest

from labs.lab3.train.train import build_training_args, train_with_trainer


def _install(monkeypatch, eval_field="eval_strategy", trainer_kw="processing_class"):
    log = {"saved": None, "trained": False}
    fields = [("output_dir", str, "out"), ("num_train_epochs", float, 3.0), ("learning_rate", float, 5e-5),
              ("per_device_train_batch_size", int, 8), ("per_device_eval_batch_size", int, 8),
              ("weight_decay", float, 0.0), (eval_field, str, "no"), ("save_strategy", str, "steps"),
              ("load_best_model_at_end", bool, False), ("metric_for_best_model", str, None),
              ("greater_is_better", bool, None), ("save_total_limit", int, None), ("logging_steps", int, 500),
              ("max_steps", int, -1), ("seed", int, 0), ("report_to", str, "all")]
    TA = dataclasses.make_dataclass("TrainingArguments", [(n, t, dataclasses.field(default=d)) for n, t, d in fields])

    def init(self, model=None, args=None, train_dataset=None, eval_dataset=None, data_collator=None,
             compute_metrics=None, **kw):
        log["init"] = dict(model=model, train=train_dataset, eval=eval_dataset, collator=data_collator, extra=kw)

    ns = {"__init__": init, "train": lambda self: log.update(trained=True),
          "save_model": lambda self, d: log.update(saved=d)}
    if trainer_kw == "tokenizer":  # old style: explicit `tokenizer` parameter
        def init_old(self, model=None, args=None, train_dataset=None, eval_dataset=None, tokenizer=None,
                     data_collator=None, compute_metrics=None):
            log["init"] = dict(model=model, train=train_dataset, eval=eval_dataset, collator=data_collator,
                               extra={"tokenizer": tokenizer})
        ns["__init__"] = init_old
    else:  # new style: explicit `processing_class` parameter
        def init_new(self, model=None, args=None, train_dataset=None, eval_dataset=None, processing_class=None,
                     data_collator=None, compute_metrics=None):
            log["init"] = dict(model=model, train=train_dataset, eval=eval_dataset, collator=data_collator,
                               extra={"processing_class": processing_class})
        ns["__init__"] = init_new
    Trainer = type("Trainer", (), ns)

    class Auto:
        @staticmethod
        def from_pretrained(name):
            log["tok_name"] = name
            return f"tok:{name}"

    mod = types.ModuleType("transformers")
    mod.TrainingArguments, mod.Trainer, mod.AutoTokenizer = TA, Trainer, Auto
    mod.DataCollatorWithPadding = lambda tok: ("collator", tok)
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return log


def test_args_values_and_eval_strategy_new_name(monkeypatch):
    _install(monkeypatch, "eval_strategy")
    a = build_training_args("out_dir", epochs=2, lr=2e-5, batch_size=16)
    assert (a.num_train_epochs, a.learning_rate, a.per_device_train_batch_size) == (2, 2e-5, 16)
    assert a.eval_strategy == "epoch" and a.save_strategy == "epoch"
    assert a.load_best_model_at_end and a.metric_for_best_model == "f1" and a.report_to == "none"


def test_args_eval_strategy_old_name(monkeypatch):
    _install(monkeypatch, "evaluation_strategy")
    assert build_training_args("o").evaluation_strategy == "epoch"


def test_args_max_steps_and_validation(monkeypatch):
    _install(monkeypatch)
    assert build_training_args("o", max_steps=5).max_steps == 5
    for bad in ({"epochs": 0}, {"lr": 0}, {"batch_size": -1}):
        with pytest.raises(ValueError):
            build_training_args("o", **bad)


@pytest.mark.parametrize("kw", ["processing_class", "tokenizer"])
def test_train_with_trainer_trains_saves_and_pads_per_batch(monkeypatch, kw):
    log = _install(monkeypatch, trainer_kw=kw)
    args = build_training_args("saved_here")
    tr = train_with_trainer("MODEL", [1, 2], [3], args, lambda p: {}, tokenizer="TOK")
    assert tr is not None and log["trained"] and log["saved"] == "saved_here"
    assert log["init"]["collator"] == ("collator", "TOK")
    assert log["init"]["extra"] == {kw: "TOK"}
    assert (log["init"]["train"], log["init"]["eval"]) == ([1, 2], [3])


def test_tokenizer_falls_back_to_model_name(monkeypatch):
    log = _install(monkeypatch)
    model = types.SimpleNamespace(name_or_path="distilbert-base-uncased")
    train_with_trainer(model, [1], [2], build_training_args("o"), lambda p: {})
    assert log["tok_name"] == "distilbert-base-uncased"


def test_no_tokenizer_and_no_name_raises(monkeypatch):
    _install(monkeypatch)
    with pytest.raises(ValueError):
        train_with_trainer(object(), [1], [2], build_training_args("o"), lambda p: {})


def test_empty_datasets_raise(monkeypatch):
    _install(monkeypatch)
    with pytest.raises(ValueError):
        train_with_trainer("M", [], [1], build_training_args("o"), lambda p: {}, tokenizer="T")
