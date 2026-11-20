"""Tests for labs/lab3/train (offline: tiny DistilBERT from a config, tiny tokenizer, CPU)."""
import json

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from labs.lab3.features.features import make_collator, tokenize_datasets
from labs.lab3.model.model import build_model
from labs.lab3.train.train import (
    HISTORY_COLUMNS, default_cfg, eval_steps_for, make_training_args, summarize_history, train_model,
)


def tiny_model(seed=0, vocab=40):
    from transformers import DistilBertConfig

    cfg = DistilBertConfig(vocab_size=vocab, dim=16, n_layers=1, n_heads=2, hidden_dim=32,
                           max_position_embeddings=64)
    return build_model(seed=seed, model_config=cfg)


def tiny_data(tokenizer, n=24):
    pos, neg = "i love this great movie", "boring bad film i hate it"
    rows = {"uid": [f"u{i}" for i in range(n)], "text": [pos if i % 2 else neg for i in range(n)],
            "label": [i % 2 for i in range(n)]}
    ds = DatasetDict({"train": Dataset.from_dict(rows), "validation": Dataset.from_dict(rows)})
    return tokenize_datasets(ds, tokenizer, max_length=16)


def np_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean()), "f1": 0.0}


def test_default_cfg_reads_config_and_rejects_unknown_keys():
    from labs.lab3 import config

    cfg = default_cfg(run="x")
    assert (cfg["lr"], cfg["batch_size"], cfg["warmup_ratio"], cfg["weight_decay"]) == (
        config.LEARNING_RATE, config.BATCH_SIZE, config.WARMUP_RATIO, config.WEIGHT_DECAY)
    with pytest.raises(KeyError):
        default_cfg(learning_rate=1e-3)


def test_eval_steps_is_a_quarter_epoch():
    assert eval_steps_for(22_499, default_cfg()) == 352          # ceil(22499/16)=1407 → 1407/4
    assert eval_steps_for(22_499, default_cfg(max_steps=8)) == 2  # smoke: 8 step / 4


def test_training_args_match_the_plan(tmp_path):
    args = make_training_args(default_cfg(max_epochs=3), tmp_path, n_train=22_499)
    assert args.learning_rate == 2e-5 and args.warmup_steps == 0.1 and args.weight_decay == 0.01
    assert args.per_device_train_batch_size == 16 and args.lr_scheduler_type.value == "linear"
    assert args.metric_for_best_model == "accuracy" and args.greater_is_better is True
    assert args.load_best_model_at_end and args.eval_steps == args.save_steps == 352
    assert args.num_train_epochs == 3 and args.save_total_limit == 2


def test_training_args_refuse_missing_epoch_ceiling(tmp_path):
    with pytest.raises(ValueError, match="max_epochs"):
        make_training_args(default_cfg(), tmp_path, n_train=100)


def test_summarize_history_first_best_and_stop_reason():
    rows = [{"step": s, "epoch": s / 10, "val_acc": a, "val_f1": a, "val_loss": 1 - a}
            for s, a in [(2, 0.5), (4, 0.8), (6, 0.8), (8, 0.7)]]
    out = summarize_history(rows, max_steps_planned=20, stopped_step=8)
    assert (out["best_step"], out["stop_reason"], out["best_is_last_eval"]) == (4, "early_stopping", False)
    assert summarize_history(rows[:2], 4, 4)["best_is_last_eval"] is True


def test_train_model_writes_three_logs_and_a_reloadable_checkpoint(tmp_path, tiny_tokenizer):
    tok = tiny_data(tiny_tokenizer)
    cfg = default_cfg(run="unit", max_steps=4, max_length=16, batch_size=8, eval_batch_size=8)
    out = train_model(cfg, tiny_model(vocab=len(tiny_tokenizer)), tok, make_collator(tiny_tokenizer),
                      compute_metrics=np_metrics, tokenizer=tiny_tokenizer,
                      logs_dir=tmp_path / "logs", models_dir=tmp_path / "models")
    stem = f"{out['ts']}_unit"
    log = (tmp_path / "logs" / f"{stem}.log").read_text(encoding="utf-8")
    assert "=== RUN START" in log and "=== RUN END" in log and "best_is_last_eval:" in log
    hist = pd.read_csv(tmp_path / "logs" / f"{stem}_history.csv")
    assert tuple(hist.columns) == HISTORY_COLUMNS and len(hist) == out["n_evals"] == 4   # mỗi 1 step
    assert hist["train_loss"].notna().all() and hist["lr"].notna().all()
    info = json.loads((tmp_path / "logs" / f"{stem}_config.json").read_text(encoding="utf-8"))
    assert info["test_evaluated"] is False and info["result"]["best_step"] == out["best_step"]
    assert info["env"]["versions"]["transformers"]
    assert out["stop_reason"] == "max_epochs" and out["stopped_step"] == 4
    best = tmp_path / "models" / f"{stem}_best"
    assert best.is_dir() and not (tmp_path / "models" / f"{stem}_ckpt").exists()
    from transformers import AutoModelForSequenceClassification

    reloaded = AutoModelForSequenceClassification.from_pretrained(str(best))
    assert reloaded.config.id2label == {0: "NEGATIVE", 1: "POSITIVE"}


def test_early_stopping_needs_patience(tmp_path, tiny_tokenizer):
    tok = tiny_data(tiny_tokenizer)
    with pytest.raises(ValueError, match="patience"):
        train_model(default_cfg(run="x", max_steps=2, early_stopping=True), tiny_model(), tok,
                    make_collator(tiny_tokenizer), compute_metrics=np_metrics,
                    logs_dir=tmp_path, models_dir=tmp_path)
