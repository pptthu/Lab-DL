"""Tests for labs/lab3/visualize (offline: small fake CSV files)."""
import numpy as np
import pytest
import pandas as pd

from labs.lab3.visualize.visualize import (
    confusion_counts, error_analysis, maxlen_table, plot_confusion, plot_maxlen, plot_training_curves,
)


def fake_history(path):
    pd.DataFrame({"step": [2, 4, 6, 8], "epoch": [0.25, 0.5, 0.75, 1.0], "train_loss": [0.7, 0.5, 0.4, 0.3],
                  "val_loss": [0.6, 0.4, 0.45, 0.5], "val_acc": [0.7, 0.85, 0.86, 0.84], "val_f1": 0.8,
                  "lr": 1e-5, "time_s": [1, 2, 3, 4]}).to_csv(path, index=False)
    return path


def fake_preds(path, preds):
    pd.DataFrame({"uid": ["a", "b", "c", "d", "e"], "label": [1, 1, 0, 0, 1], "pred": preds,
                  "prob_pos": [0.9, 0.2, 0.1, 0.95, 0.6]}).to_csv(path, index=False)
    return path


def test_training_curves_file(tmp_path):
    out = plot_training_curves(fake_history(tmp_path / "x_history.csv"), tmp_path / "x_curves.png")
    assert out.exists() and out.stat().st_size > 1000


def test_confusion_sums_to_n(tmp_path):
    p = fake_preds(tmp_path / "p.csv", [1, 0, 0, 1, 1])
    m = confusion_counts(pd.read_csv(p))
    assert m.sum() == 5 and m.tolist() == [[1, 1], [1, 2]]
    assert plot_confusion({"zs": p, "ft": p}, tmp_path / "cm.png").exists()


def test_maxlen_table_uses_main_runs_for_512(tmp_path):
    maxlen = pd.DataFrame({"max_length": [128, 256], "best_val_acc": [0.88, 0.90]})
    main = pd.DataFrame({"max_length": [512, 512, 512], "best_val_acc": [0.92, 0.93, 0.94]})
    t = maxlen_table(maxlen, main)
    assert t["max_length"].tolist() == [128, 256, 512] and t["n_seeds"].tolist() == [1, 1, 3]
    assert t.iloc[2]["std_val_acc"] == pytest.approx(np.std([0.92, 0.93, 0.94], ddof=1))
    assert plot_maxlen(maxlen, main, tmp_path / "m.png").exists()


def test_error_analysis_groups_and_examples(tmp_path):
    p = fake_preds(tmp_path / "p.csv", [1, 0, 0, 1, 1])     # sai: b (label 1), d (label 0)
    test = pd.DataFrame({"uid": list("abcde"), "text": ["x" * 400] * 5, "rating": [9, 7, 1, 4, 8]})
    lengths = pd.DataFrame({"uid": list("abcde"), "n_tokens": [100, 600, 50, 700, 30]})
    out = error_analysis(p, test, lengths, max_length=512, results_dir=tmp_path, demo_dir=tmp_path)
    g = out["by_group"].set_index(["group_type", "group"])
    assert g.loc[("all", "all"), "n_errors"] == 2
    assert g.loc[("truncation", "bị cắt (> 512 token)"), "error_rate"] == 1.0
    assert g.loc[("rating", "7"), "n_errors"] == 1
    ex = out["examples"]
    assert set(ex["uid"]) == {"b", "d"} and ex["text_preview"].str.len().max() == 300
    assert out["paths"]["by_group"].exists() and out["paths"]["examples"].exists()
