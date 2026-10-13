"""Tests for lab1 eda on a tiny synthetic CSV (no real data needed).

VI: Test module eda trên CSV giả nhỏ (không cần dữ liệu thật).
"""
import numpy as np
import pandas as pd
import pytest

from labs.lab1.eda.eda import N_PIXELS, run_eda


def _write_csv(path, n_rows: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["label"] + [f"pixel{i}" for i in range(1, N_PIXELS + 1)]
    data = rng.integers(0, 256, size=(n_rows, N_PIXELS + 1))
    data[:, 0] = np.arange(n_rows) % 10
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(path, index=False)
    return df


def test_run_eda_stats_and_figures(tmp_path):
    csv = tmp_path / "train.csv"
    df = _write_csv(csv)
    out = tmp_path / "figs"

    stats = run_eda(str(csv), str(out))

    assert stats["n_rows"] == 20
    assert stats["class_counts"] == {i: 2 for i in range(10)}
    assert stats["pixel_min"] == int(df.iloc[:, 1:].to_numpy().min())
    assert stats["pixel_max"] == int(df.iloc[:, 1:].to_numpy().max())
    assert stats["n_missing"] == 0
    assert stats["n_duplicates"] == 0
    assert (out / "class_counts.png").is_file()
    assert (out / "sample_grid.png").is_file()


def test_run_eda_counts_missing_and_duplicates(tmp_path):
    csv = tmp_path / "train.csv"
    df = _write_csv(csv)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # 1 duplicate row
    df.loc[1, "pixel5"] = np.nan                            # 1 missing cell
    df.to_csv(csv, index=False)

    stats = run_eda(str(csv), str(tmp_path / "figs"))

    assert stats["n_duplicates"] == 1
    assert stats["n_missing"] == 1


def test_run_eda_rejects_wrong_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"label": [0], "pixel1": [0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        run_eda(str(csv), str(tmp_path / "figs"))
