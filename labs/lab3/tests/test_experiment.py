"""Tests for labs/lab3/experiment (offline: fake train_model, tiny tokenizer)."""
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from labs.lab3 import config
from labs.lab3.experiment import experiment as ex


def test_pilot_and_main_need_values_fixed_after_measuring(monkeypatch):
    monkeypatch.setattr(config, "PILOT_MAX_EPOCHS", None)
    monkeypatch.setattr(config, "PATIENCE", None)
    with pytest.raises(ValueError, match="cần chốt"):
        ex.make_configs("pilot")
    with pytest.raises(ValueError, match="cần chốt"):
        ex.make_configs("main")


def test_make_configs_counts_and_values(monkeypatch):
    for k, v in {"PILOT_MAX_EPOCHS": 6, "MAIN_MAX_EPOCHS": 4, "PATIENCE": 3, "THRESHOLD": 0.001,
                 "MAXLEN_SEEDS": (42,)}.items():
        monkeypatch.setattr(config, k, v)
    pilot = ex.make_configs("pilot")
    assert len(pilot) == 1 and pilot[0]["early_stopping"] is False and pilot[0]["max_epochs"] == 6
    main = ex.make_configs("main")
    assert [c["seed"] for c in main] == [42, 43, 44] and all(c["early_stopping"] for c in main)
    assert all((c["patience"], c["threshold"], c["max_epochs"]) == (3, 0.001, 4) for c in main)
    maxlen = ex.make_configs("maxlen")
    assert sorted(c["max_length"] for c in maxlen) == [128, 256]   # 512 đã chạy ở main
    timing = ex.make_configs("timing")
    assert timing[0]["max_steps"] == config.TIMING_MAX_STEPS and timing[0]["save_best"] is False


def test_smoke_configs_do_not_need_fixed_values(monkeypatch):
    monkeypatch.setattr(config, "PATIENCE", None)
    monkeypatch.setattr(config, "MAIN_MAX_EPOCHS", None)
    cfgs = ex.make_configs("main", smoke=True)
    assert len(cfgs) == 3 and all(c["max_steps"] == config.SMOKE_MAX_STEPS for c in cfgs)


def test_run_stage_writes_table_and_filters_runs(tmp_path, tiny_tokenizer):
    rows = {"uid": ["a", "b"], "text": ["great movie", "bad film"], "label": [1, 0]}
    parts = {"datasets": DatasetDict({k: Dataset.from_dict(rows) for k in ("train", "validation", "test")}),
             "tokenizer": tiny_tokenizer, "collator": None}
    calls = []

    def fake_train(cfg, model, tokenized, collator, tokenizer=None, **kw):
        calls.append((cfg["run"], model))
        return {"run": cfg["run"], "ts": "20260101_000000", "best_step": 2, "best_epoch": 1.0,
                "best_val_acc": 0.9, "best_val_f1": 0.9, "best_val_loss": 0.3, "stopped_step": 4,
                "max_steps_planned": 4, "stop_reason": "max_epochs", "best_is_last_eval": False, "n_evals": 2,
                "train_time_s": 8.0, "best_ckpt": None, "log": "x.log"}

    out = ex.run_stage("main", parts, smoke=True, runs=["main_seed43"], train_fn=fake_train,
                       model_fn=lambda seed: f"model{seed}", results_dir=tmp_path)
    assert calls == [("main_seed43", "model43")]
    table = pd.read_csv(tmp_path / "main_runs.csv")
    assert table["run"].tolist() == ["main_seed43"] and table["sec_per_step"].iloc[0] == 2.0
    assert out["est_epoch_min"].iloc[0] == pytest.approx(2.0 * 1 / 60, abs=0.01)   # 2 mẫu / batch 16 = 1 step
    with pytest.raises(ValueError, match="unknown runs"):
        ex.run_stage("main", parts, smoke=True, runs=["nope"], train_fn=fake_train, model_fn=str,
                     results_dir=tmp_path)


def test_seed_spread_advice():
    df = pd.DataFrame({"best_val_acc": [0.920, 0.921, 0.922]})
    s = ex.seed_spread(df)
    assert s["std"] == pytest.approx(0.001) and s["advice"].startswith("1 seed")
    assert ex.seed_spread(pd.DataFrame({"best_val_acc": [0.90, 0.92, 0.94]}))["advice"].startswith("3 seed")
    assert "cần chốt" in ex.seed_spread(pd.DataFrame({"best_val_acc": [0.910, 0.915, 0.920]}))["advice"]


def test_merge_run_tables_skips_rows_already_there(tmp_path):
    src, dst = tmp_path / "src.csv", tmp_path / "dst.csv"
    pd.DataFrame([{"run": "a", "ts": "1"}, {"run": "b", "ts": "2"}]).to_csv(src, index=False)
    ex.append_run_row({"run": "a", "ts": "1"}, dst)
    assert ex.merge_run_tables(src, dst) == 1
    assert pd.read_csv(dst)["run"].tolist() == ["a", "b"]
