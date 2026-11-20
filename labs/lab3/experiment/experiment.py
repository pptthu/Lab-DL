"""Experiments: timing run, pilot, main run (3 seeds), max_length experiment; run tables.

VI: Thí nghiệm (plan §4 S2–S4). Sinh cfg cho từng giai đoạn, gọi train.train_model cho từng cfg,
ghi outputs/results/<stage>_runs.csv ngay sau mỗi run (Kaggle ngắt giữa chừng vẫn còn kết quả).
Desc: labs/lab3/docs/phases/experiment_v1.md
Owner: Phúc (Member B).

Run from the repo root / Chạy từ gốc repo:
    python -m labs.lab3.experiment.experiment --stage timing|pilot|main|maxlen [--runs main_seed43] [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd

from labs.lab3 import config
from labs.lab3.train.train import default_cfg
from shared.paths import output_dir

STAGES = ("timing", "pilot", "main", "maxlen")
RUN_COLUMNS = ("run", "stage", "seed", "max_length", "ts", "best_step", "best_epoch", "best_val_acc", "best_val_f1",
               "best_val_loss", "stopped_step", "max_steps_planned", "stop_reason", "best_is_last_eval", "n_evals",
               "train_time_s", "sec_per_step", "est_epoch_min", "best_ckpt", "log")


# ---------------------------------------------------------------- cấu hình
def _need(value, name: str, when: str):
    if value is None:
        raise ValueError(f"{name} is not set yet: [cần chốt sau {when}] in labs/lab3/config.py")
    return value


def make_configs(stage: str, smoke: bool = False) -> list[dict]:
    """List of run configurations for one stage; every value is read from config.py.

    VI: timing: 1 run ngắn (config.TIMING_MAX_STEPS step) đo thời gian/epoch trên Kaggle GPU.
    pilot: seed 42, chạy hết trần, không dừng sớm. main: 3 seed cố định, dừng sớm theo patience/threshold.
    maxlen: các giá trị MAX_LENGTH_CANDIDATES khác MAX_LENGTH, MAXLEN_SEEDS seed mỗi giá trị.
    smoke: giữ cấu trúc, chỉ chạy config.SMOKE_MAX_STEPS step.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    smoke_kw = {"max_steps": config.SMOKE_MAX_STEPS} if smoke else {}   # smoke: max_epochs = None
    if stage == "timing":
        return [default_cfg(run="timing_seed42", stage=stage, seed=config.PILOT_SEED, save_best=False,
                            max_steps=config.SMOKE_MAX_STEPS if smoke else config.TIMING_MAX_STEPS)]
    if stage == "pilot":
        max_epochs = None if smoke else _need(config.PILOT_MAX_EPOCHS, "PILOT_MAX_EPOCHS", "đo thời gian trên Kaggle")
        return [default_cfg(run=f"pilot_seed{config.PILOT_SEED}", stage=stage, seed=config.PILOT_SEED,
                            max_epochs=max_epochs, early_stopping=False, **smoke_kw)]
    if stage == "main":
        es = {"early_stopping": True,
              "patience": _need(config.PATIENCE, "PATIENCE", "pilot") if not smoke else 2,
              "threshold": _need(config.THRESHOLD, "THRESHOLD", "pilot") if not smoke else 0.0}
        max_epochs = None if smoke else _need(config.MAIN_MAX_EPOCHS, "MAIN_MAX_EPOCHS", "pilot")
        return [default_cfg(run=f"main_seed{s}", stage=stage, seed=s, max_epochs=max_epochs, **es, **smoke_kw)
                for s in config.MAIN_SEEDS]
    # maxlen
    seeds = (config.PILOT_SEED,) if smoke else _need(config.MAXLEN_SEEDS, "MAXLEN_SEEDS", "S3 (std 3 seed)")
    es = {"early_stopping": True,
          "patience": _need(config.PATIENCE, "PATIENCE", "pilot") if not smoke else 2,
          "threshold": _need(config.THRESHOLD, "THRESHOLD", "pilot") if not smoke else 0.0}
    max_epochs = None if smoke else _need(config.MAIN_MAX_EPOCHS, "MAIN_MAX_EPOCHS", "pilot")
    return [default_cfg(run=f"maxlen{L}_seed{s}", stage=stage, seed=s, max_length=L, max_epochs=max_epochs,
                        save_best=False, **es, **smoke_kw)
            for L in config.MAX_LENGTH_CANDIDATES if L != config.MAX_LENGTH for s in seeds]


# ---------------------------------------------------------------- chạy
def append_run_row(row: dict, path: Path) -> None:
    """Append one run to <stage>_runs.csv (header written once). / Thêm một dòng vào bảng run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def run_row(cfg: dict, result: dict, n_train: int) -> dict:
    """One line of <stage>_runs.csv from a train_model result (+ timing estimate per epoch).

    VI: sec_per_step = thời gian train / số step đã chạy (gồm cả thời gian đánh giá val);
    est_epoch_min = sec_per_step × số step mỗi epoch / 60.
    """
    steps_per_epoch = math.ceil(n_train / cfg["batch_size"])
    sps = result["train_time_s"] / max(1, result["stopped_step"])
    return {**{k: cfg[k] for k in ("run", "stage", "seed", "max_length")}, **result,
            "sec_per_step": round(sps, 4), "est_epoch_min": round(sps * steps_per_epoch / 60, 2)}


def run_stage(stage: str, parts: dict | None = None, smoke: bool = False, runs: list[str] | None = None,
              train_fn=None, model_fn=None, results_dir: Path | None = None, **train_kw) -> pd.DataFrame:
    """Run every configuration of a stage (or only `runs`); write <stage>_runs.csv after each run.

    parts: {"datasets": DatasetDict (văn bản đã làm sạch), "tokenizer", "collator"} (từ main.prepare_parts).
    VI: Mỗi max_length tokenize một lần; model tạo mới với seed của run (đặt seed trước khi tạo model).
    train_fn / model_fn chỉ thay trong test.
    """
    from labs.lab3.features.features import tokenize_datasets

    if train_fn is None:
        from labs.lab3.train.train import train_model as train_fn
    if model_fn is None:
        from labs.lab3.model.model import build_model as model_fn
    if results_dir is None:
        results_dir = output_dir(config.LAB, "_smoke") / "results" if smoke else output_dir(config.LAB, "results")
    if smoke:  # smoke run: mọi đầu ra vào outputs/_smoke/ (RULES_v1 bước 7)
        train_kw.setdefault("logs_dir", output_dir(config.LAB, "_smoke") / "logs")
        train_kw.setdefault("models_dir", output_dir(config.LAB, "_smoke") / "models")
    cfgs = make_configs(stage, smoke=smoke)
    if runs:
        unknown = set(runs) - {c["run"] for c in cfgs}
        if unknown:
            raise ValueError(f"unknown runs for stage {stage}: {sorted(unknown)}")
        cfgs = [c for c in cfgs if c["run"] in runs]
    tokenized_by_len: dict[int, object] = {}
    out_path = Path(results_dir) / f"{stage}_runs.csv"
    rows = []
    for cfg in cfgs:
        L = cfg["max_length"]
        if L not in tokenized_by_len:
            tokenized_by_len[L] = tokenize_datasets(parts["datasets"], parts["tokenizer"], L)
        tokenized = tokenized_by_len[L]
        print(f"[experiment] {cfg['run']}: seed={cfg['seed']} max_length={L} max_epochs={cfg['max_epochs']} "
              f"max_steps={cfg['max_steps']} early_stopping={cfg['early_stopping']}", flush=True)
        model = model_fn(cfg["seed"])
        result = train_fn(cfg, model, tokenized, parts["collator"], tokenizer=parts.get("tokenizer"), **train_kw)
        row = run_row(cfg, result, len(tokenized["train"]))
        append_run_row(row, out_path)
        rows.append(row)
        print(f"[experiment] {cfg['run']} done: best_val_acc={row['best_val_acc']} best_step={row['best_step']} "
              f"stop={row['stop_reason']} best_is_last_eval={row['best_is_last_eval']}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- phân tích
def seed_spread(main_runs) -> dict:
    """Mean and sample std (n-1) of the best validation accuracy of the main runs; advice for S4.

    VI: std nhỏ (≤ 0,003 = 0,3 điểm) → 1 seed mỗi giá trị max_length; std lớn (≥ 0,01) → 3 seed;
    ở giữa → [cần chốt]: Thư quyết (experiment_v1.md bước 4).
    """
    df = pd.read_csv(main_runs) if isinstance(main_runs, (str, Path)) else main_runs
    acc = df["best_val_acc"].astype(float)
    std = float(acc.std(ddof=1)) if len(acc) > 1 else float("nan")
    if std <= 0.003:
        advice = "1 seed / max_length"
    elif std >= 0.01:
        advice = "3 seed / max_length"
    else:
        advice = "[cần chốt] std giữa 0,3 và 1 điểm: Thư quyết"
    return {"n": int(len(acc)), "mean": float(acc.mean()), "std": std, "min": float(acc.min()),
            "max": float(acc.max()), "advice": advice}


def merge_run_tables(src: Path, dst: Path) -> int:
    """Append rows of a downloaded <stage>_runs.csv into the repo table, skipping (run, ts) already there.

    VI: Gộp bảng run tải từ Kaggle vào bảng trong repo (không ghi trùng). Trả số dòng thêm.
    """
    new = pd.read_csv(src)
    if Path(dst).exists():
        old = pd.read_csv(dst)
        seen = set(zip(old["run"], old["ts"].astype(str)))
        new = new[[(r, str(t)) not in seen for r, t in zip(new["run"], new["ts"])]]
    for row in new.to_dict("records"):
        append_run_row(row, Path(dst))
    return len(new)


def main(argv=None) -> None:
    from labs.lab3.main import prepare_parts, utf8_stdout

    utf8_stdout()
    parser = argparse.ArgumentParser(description="Lab 3 experiments (Member B)")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--runs", nargs="*", help="chỉ chạy các run này (vd main_seed43)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    parts = prepare_parts(args.smoke)
    print(run_stage(args.stage, parts, smoke=args.smoke, runs=args.runs).to_string(index=False))


if __name__ == "__main__":
    main()
