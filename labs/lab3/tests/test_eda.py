"""Tests for `eda` (Lab 3). Fake data, no network.

VI: Test cho `eda` (Lab 3). Dữ liệu giả, không cần mạng.
"""
import pytest

from labs.lab3.eda.eda import run_eda

DATA = {"text": ["good film", "bad", "a very very long boring movie review here", "nice"], "label": [1, 0, 0, 1]}


def test_run_eda_contract(tmp_path):
    res = run_eda(DATA, str(tmp_path))
    assert set(res) == {"n_samples", "label_counts", "length_stats"}
    assert res["n_samples"] == 4
    assert res["label_counts"] == {0: 2, 1: 2}
    assert res["length_stats"]["min"] == 1 and res["length_stats"]["max"] == 8
    for name in ("label_distribution.png", "length_distribution.png", "length_by_label.png"):
        assert (tmp_path / name).exists()


def test_run_eda_custom_columns(tmp_path):
    res = run_eda({"review": ["a b", "c"], "y": [0, 1]}, str(tmp_path), text_col="review", label_col="y")
    assert res["n_samples"] == 2 and res["length_stats"]["max"] == 2


def test_run_eda_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        run_eda({"text": [], "label": []}, str(tmp_path))
