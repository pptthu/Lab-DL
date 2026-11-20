"""Exercises 2.5-2.6: TrainingArguments, Trainer and fine-tuning, with a run log (LOG_TEMPLATE_v1).

VI: Bài 2.5–2.6 — tạo TrainingArguments, Trainer, fine-tune; mỗi lần chạy ghi 3 file log
(.log, _history.csv, _config.json); chọn checkpoint theo val accuracy.
Desc: labs/lab3/docs/phases/train_v1.md | Khuôn log: labs/lab3/docs/templates/LOG_TEMPLATE_v1.md
Owner: Phúc (Member B).
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from labs.lab3 import config
from shared.paths import REPO_ROOT, output_dir

VN_TZ = timezone(timedelta(hours=7))  # giờ Việt Nam, cố định (Kaggle chạy giờ UTC)
HISTORY_COLUMNS = ("step", "epoch", "train_loss", "val_loss", "val_acc", "val_f1", "lr", "time_s")


# ---------------------------------------------------------------- cấu hình
def default_cfg(**overrides) -> dict:
    """One run configuration; every value comes from labs/lab3/config.py unless overridden.

    VI: Cấu hình một lần chạy. Giá trị mặc định đọc từ config.py (một chỗ duy nhất).
    Khoá: run, stage, seed, max_length, lr, batch_size, eval_batch_size, warmup_ratio, weight_decay,
    lr_scheduler, max_epochs, max_steps (chỉ smoke/đo thời gian), evals_per_epoch, early_stopping,
    patience, threshold, save_best.
    """
    cfg = {
        "run": "run", "stage": "manual", "seed": config.SEED, "max_length": config.MAX_LENGTH,
        "lr": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE,
        "eval_batch_size": config.EVAL_BATCH_SIZE, "warmup_ratio": config.WARMUP_RATIO,
        "weight_decay": config.WEIGHT_DECAY, "lr_scheduler": config.LR_SCHEDULER,
        "max_epochs": None, "max_steps": None, "evals_per_epoch": config.EVALS_PER_EPOCH,
        "early_stopping": False, "patience": None, "threshold": None, "save_best": True,
    }
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown cfg keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg


def eval_steps_for(n_train: int, cfg: dict) -> int:
    """Steps between two validation evaluations: about 1/evals_per_epoch of an epoch.

    VI: Số step giữa hai lần đánh giá val (~1/4 epoch, plan L11). Smoke/đo thời gian (max_steps nhỏ
    hơn một epoch) thì chia max_steps.
    """
    steps_per_epoch = math.ceil(n_train / cfg["batch_size"])
    span = min(steps_per_epoch, cfg["max_steps"]) if cfg.get("max_steps") else steps_per_epoch
    return max(1, round(span / cfg["evals_per_epoch"]))


def make_training_args(cfg: dict, output_dir: Path | str, n_train: int):
    """TrainingArguments for one run (values and sources: docs/phases/train_v1.md).

    VI: TrainingArguments của một lần chạy. fp16 chỉ bật khi có GPU.
    """
    import torch
    from transformers import TrainingArguments

    if cfg.get("max_epochs") is None and not cfg.get("max_steps"):
        raise ValueError("max_epochs is not set: it is fixed after the pilot (config.py, [cần chốt sau pilot])")
    steps = eval_steps_for(n_train, cfg)
    return TrainingArguments(
        output_dir=str(output_dir),
        seed=cfg["seed"],                       # thứ tự dữ liệu (model đã được tạo với cùng seed)
        learning_rate=cfg["lr"],
        lr_scheduler_type=cfg["lr_scheduler"],
        warmup_steps=cfg["warmup_ratio"],       # transformers 5: số thực trong [0, 1) = tỉ lệ tổng step
        weight_decay=cfg["weight_decay"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        num_train_epochs=cfg["max_epochs"] or 1,
        max_steps=cfg["max_steps"] or -1,
        eval_strategy="steps", eval_steps=steps,
        save_strategy="steps", save_steps=steps,
        logging_strategy="steps", logging_steps=steps,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=config.METRIC_FOR_BEST,
        greater_is_better=True,                  # thiếu thì dễ bị hiểu "nhỏ hơn là tốt" (train_v1.md)
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=True,
    )


# ---------------------------------------------------------------- môi trường chạy
def code_commit() -> str:
    """Git commit of the code that runs (env LAB3_CODE_COMMIT on Kaggle, else git, else 'unknown').

    VI: Mã của code đang chạy: <HEAD>+tree:<cây index> (code chưa commit). Trên Kaggle không có .git
    → kernel đặt biến môi trường từ BUILD_INFO.json.
    """
    env = os.environ.get("LAB3_CODE_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=10)
        if out.returncode == 0:
            tree = subprocess.run(["git", "write-tree"], cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()   # cây index: code đã stage, chưa commit
            unstaged = subprocess.run(["git", "diff", "--name-only", "--", "labs/lab3", "shared"], cwd=REPO_ROOT,
                                      capture_output=True, text=True, timeout=10).stdout.strip()
            return f"{out.stdout.strip()[:7]}+tree:{tree[:12]}" + ("+unstaged" if unstaged else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def run_environment() -> dict:
    """Device, GPU name, where the run happens and library versions. / Thiết bị, GPU, phiên bản thư viện."""
    import datasets
    import torch
    import transformers

    try:
        import evaluate as hf_evaluate
        ev = hf_evaluate.__version__
    except ImportError:  # pragma: no cover - evaluate là thư viện bắt buộc của đề
        ev = "missing"
    cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if cuda else "cpu",
        "gpu": torch.cuda.get_device_name(0) if cuda else "none",
        "where": "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "local",
        "code_commit": code_commit(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "datasets": datasets.__version__,
                     "evaluate": ev},
    }


def now_vn() -> datetime:
    return datetime.now(VN_TZ)


# ---------------------------------------------------------------- ghi log
class RunLog:
    """Writes <ts>_<run>.log, _history.csv and _config.json (LOG_TEMPLATE_v1 §1-§2).

    VI: Ghi 3 file log. history.csv ghi từng dòng ngay khi có (Kaggle ngắt giữa chừng vẫn còn log).
    """

    def __init__(self, logs_dir: Path, ts: str, run: str):
        self.dir = Path(logs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stem = f"{ts}_{run}"
        self.log_path = self.dir / f"{stem}.log"
        self.history_path = self.dir / f"{stem}_history.csv"
        self.config_path = self.dir / f"{stem}_config.json"
        self.rows: list[dict] = []

    def _write(self, text: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    def header(self, info: dict) -> None:
        c, env, d = info["cfg"], info["env"], info["data"]
        v = env["versions"]
        es = (f"patience={c['patience']} threshold={c['threshold']}" if c["early_stopping"]
              else "none")
        lines = [
            f"=== RUN START {info['start']} ===",
            f"run: {c['run']} | stage: {c['stage']} | seed: {c['seed']} | device: {env['device']} | "
            f"gpu: {env['gpu']} | where: {env['where']} | code: {env['code_commit']}",
            "versions: " + " ".join(f"{k}={val}" for k, val in v.items()),
            f"model: {info['model']['checkpoint']} + classification head (2 lớp) | params: "
            f"{info['model']['params_total']} (trainable {info['model']['params_trainable']})",
            f"config: max_length={c['max_length']} lr={c['lr']} warmup_ratio={c['warmup_ratio']} "
            f"scheduler={c['lr_scheduler']} batch_size={c['batch_size']} fp16={info['fp16']} "
            f"weight_decay={c['weight_decay']} max_epochs={c['max_epochs']} max_steps={c['max_steps']}",
            f"eval: every {info['eval_steps']} steps (~1/{c['evals_per_epoch']} epoch) | select: val accuracy "
            f"(cao nhất) | monitor thêm: val_loss",
            f"early_stopping: {es}",
            f"data: train={d['train']} val={d['val']} test={d['test']} | split: {d['split_file']}",
            ",".join(HISTORY_COLUMNS),
        ]
        self._write("\n".join(lines))
        with self.history_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HISTORY_COLUMNS)
        self.save_config(info)

    def save_config(self, info: dict) -> None:
        self.config_path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    def row(self, row: dict) -> None:
        self.rows.append(row)
        values = [row.get(k) for k in HISTORY_COLUMNS]
        with self.history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(values)
        self._write(",".join("" if x is None else str(x) for x in values))

    def footer(self, result: dict, end: str) -> None:
        self._write("\n".join([
            f"=== RUN END {end} ===",
            f"best_step: {result['best_step']} (epoch {result['best_epoch']}) | best_val_acc: "
            f"{result['best_val_acc']} | stopped_step: {result['stopped_step']} | stop_reason: "
            f"{result['stop_reason']}",
            f"best_is_last_eval: {'yes' if result['best_is_last_eval'] else 'no'}",
            f"best_ckpt: {result['best_ckpt']} | train_time_s: {result['train_time_s']}",
        ]))


def _round(x, nd=6):
    return None if x is None else round(float(x), nd)


def make_log_callback(log: RunLog, t0_holder: dict):
    """TrainerCallback: one history row per validation evaluation. / Mỗi lần đánh giá val một dòng."""
    from transformers import TrainerCallback

    class _LogCallback(TrainerCallback):
        def __init__(self):
            self.last = {}

        def on_log(self, args, state, control, logs=None, **kwargs):
            # Trainer ghi log train (loss, learning_rate) TRƯỚC khi đánh giá ở cùng step
            if logs and "loss" in logs:
                self.last = {"train_loss": logs["loss"], "lr": logs.get("learning_rate"), "step": state.global_step}

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return
            same_step = self.last.get("step") == state.global_step
            log.row({
                "step": state.global_step,
                "epoch": _round(state.epoch, 4),
                "train_loss": _round(self.last.get("train_loss")) if same_step else None,
                "val_loss": _round(metrics.get("eval_loss")),
                "val_acc": _round(metrics.get("eval_accuracy")),
                "val_f1": _round(metrics.get("eval_f1")),
                "lr": _round(self.last.get("lr"), 10) if same_step else None,
                "time_s": round(time.time() - t0_holder["t0"], 1),
            })

    return _LogCallback()


def summarize_history(rows: list[dict], max_steps_planned: int, stopped_step: int) -> dict:
    """best_step = first evaluation with the highest val accuracy (same rule as Trainer: strictly greater).

    VI: Checkpoint tốt nhất = lần đánh giá đầu tiên có val acc cao nhất. stop_reason: dừng trước số step
    dự kiến → early_stopping, chạy hết → max_epochs (smoke/đo thời gian: max_steps).
    """
    if not rows:
        raise ValueError("no validation evaluation was logged")
    best = max(rows, key=lambda r: (r["val_acc"], -r["step"]))
    return {
        "best_step": best["step"], "best_epoch": best["epoch"], "best_val_acc": best["val_acc"],
        "best_val_f1": best["val_f1"], "best_val_loss": best["val_loss"],
        "stopped_step": stopped_step,
        "stop_reason": "early_stopping" if stopped_step < max_steps_planned else "max_epochs",
        "best_is_last_eval": best["step"] == rows[-1]["step"],
        "n_evals": len(rows),
    }


# ---------------------------------------------------------------- train
def train_model(cfg: dict, model, tokenized, collator, compute_metrics=None, tokenizer=None,
                logs_dir: Path | None = None, models_dir: Path | None = None) -> dict:
    """Fine-tune one model; keep the checkpoint with the highest validation accuracy.

    Returns best_step, best_val_acc, stopped_step, stop_reason, best_is_last_eval, best_ckpt, log paths.
    VI: Fine-tune một lần chạy. Pilot (early_stopping=False) chạy hết trần; lần chạy chính thêm
    EarlyStoppingCallback(patience, threshold). Checkpoint trung gian xoá sau khi lưu bản tốt nhất.
    """
    from transformers import EarlyStoppingCallback, Trainer

    from labs.lab3.model.model import count_parameters

    if compute_metrics is None:
        from labs.lab3.evaluate.evaluate import compute_metrics
    logs_dir = Path(logs_dir) if logs_dir is not None else output_dir(config.LAB, "logs")
    models_dir = Path(models_dir) if models_dir is not None else output_dir(config.LAB, "models")
    if cfg["early_stopping"] and (cfg["patience"] is None or cfg["threshold"] is None):
        raise ValueError("early stopping needs patience and threshold ([cần chốt sau pilot] in config.py)")

    start = now_vn()
    ts = start.strftime("%Y%m%d_%H%M%S")
    log = RunLog(logs_dir, ts, cfg["run"])
    ckpt_dir = models_dir / f"{ts}_{cfg['run']}_ckpt"
    best_dir = models_dir / f"{ts}_{cfg['run']}_best"
    train_ds, val_ds = tokenized["train"], tokenized["validation"]
    args = make_training_args(cfg, ckpt_dir, len(train_ds))
    params = count_parameters(model)
    info = {
        "run": cfg["run"], "ts": ts, "start": start.isoformat(timespec="seconds"), "cfg": cfg,
        "env": run_environment(), "fp16": bool(args.fp16), "eval_steps": args.eval_steps,
        "model": {"checkpoint": config.CHECKPOINT, "params_total": params["total"],
                  "params_trainable": params["trainable"]},
        "data": {"train": len(train_ds), "val": len(val_ds),
                 "test": len(tokenized["test"]) if "test" in tokenized else None,
                 "split_file": "labs/lab3/data/splits/train_val_split.json"},
        "selection": {"metric": "val accuracy", "rule": "LOG_TEMPLATE_v1 §3"},
        "test_evaluated": False,   # evaluate.evaluate_on_test đặt True; gọi lần hai thì báo lỗi
    }
    log.header(info)

    t0 = {"t0": time.time()}
    callbacks = [make_log_callback(log, t0)]
    if cfg["early_stopping"]:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg["patience"],
                                               early_stopping_threshold=cfg["threshold"]))
    trainer = Trainer(model=model, args=args, data_collator=collator, train_dataset=train_ds,
                      eval_dataset=val_ds, processing_class=tokenizer, compute_metrics=compute_metrics,
                      callbacks=callbacks)
    t0["t0"] = time.time()
    trainer.train()
    train_time = round(time.time() - t0["t0"], 1)

    result = summarize_history(log.rows, trainer.state.max_steps, trainer.state.global_step)
    if cfg.get("save_best", True):
        trainer.save_model(str(best_dir))       # load_best_model_at_end=True → đây là checkpoint tốt nhất
        best_ckpt = best_dir
    else:
        best_ckpt = None
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # checkpoint trung gian: tránh đầy ổ Kaggle
    result.update({
        "run": cfg["run"], "ts": ts, "max_steps_planned": trainer.state.max_steps,
        "best_ckpt": _rel(best_ckpt), "train_time_s": train_time,
        "log": _rel(log.log_path), "history": _rel(log.history_path), "config": _rel(log.config_path),
    })
    end = now_vn().isoformat(timespec="seconds")
    log.footer(result, end)
    info["end"], info["result"] = end, result
    log.save_config(info)
    return result


def _rel(p) -> str | None:
    if p is None:
        return None
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()
