"""Exercise 2.7: evaluate the fine-tuned model (accuracy / F1 with the `evaluate` library).

VI: Bài 2.7 — metric bằng thư viện `evaluate`; test 25k chỉ một lần mỗi model; baseline zero-shot
(pipeline SST-2 của Bài 1) trên cùng 25k test; bảng tổng hợp mean ± std 3 seed; 6 câu kiểm soát.
Desc: labs/lab3/docs/phases/evaluate_v1.md
Owner: Phúc (Member B).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from labs.lab3 import config
from shared.paths import output_dir

PRED_COLUMNS = ("uid", "label", "pred", "prob_pos")
_METRICS: dict = {}


# ---------------------------------------------------------------- metric
def _metric(name: str):
    """evaluate.load once per process (lần đầu tải script metric từ Hub → Kaggle bật Internet)."""
    if name not in _METRICS:
        import evaluate as hf_evaluate

        _METRICS[name] = hf_evaluate.load(name)
    return _METRICS[name]


def accuracy_f1(labels, preds) -> dict:
    """Accuracy and binary F1 (positive class = 1) with the `evaluate` library. / Accuracy, F1 nhị phân."""
    labels, preds = np.asarray(labels).astype(int), np.asarray(preds).astype(int)
    return {
        "accuracy": float(_metric("accuracy").compute(predictions=preds, references=labels)["accuracy"]),
        "f1": float(_metric("f1").compute(predictions=preds, references=labels, pos_label=1)["f1"]),
    }


def compute_metrics(eval_pred) -> dict:
    """Trainer hook: logits -> argmax -> accuracy, f1 (keys become eval_accuracy, eval_f1).

    VI: Dùng trong Trainer khi đánh giá val.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return accuracy_f1(labels, np.argmax(logits, axis=-1))


def softmax_pos(logits) -> np.ndarray:
    """Probability of class 1 (POSITIVE) from 2-class logits. / Xác suất lớp dương."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


# ---------------------------------------------------------------- test một lần
def _check_and_mark_test(run_config_path: Path | None, pred_path: Path) -> None:
    """Refuse a second test evaluation of the same model (plan L2: test used once per model).

    VI: Cờ test_evaluated trong _config.json của lần chạy; file dự đoán đã có cũng coi là đã test.
    """
    if pred_path.exists():
        raise RuntimeError(f"test already evaluated for this model: {pred_path} exists (test is used once)")
    if run_config_path is not None:
        info = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
        if info.get("test_evaluated"):
            raise RuntimeError(f"test already evaluated for run {info.get('run')} ({run_config_path})")


def _mark_tested(run_config_path: Path | None, pred_path: Path) -> None:
    if run_config_path is None:
        return
    path = Path(run_config_path)
    info = json.loads(path.read_text(encoding="utf-8"))
    info["test_evaluated"] = True
    info["test_predictions"] = pred_path.name
    path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def predict_logits(model, tokenized_split, collator, batch_size: int = config.EVAL_BATCH_SIZE):
    """Logits of a tokenized split with Trainer.predict (fp16 on GPU). / Logits bằng Trainer.predict."""
    import tempfile

    import torch
    from transformers import Trainer, TrainingArguments

    cols = [c for c in tokenized_split.column_names if c in ("input_ids", "attention_mask", "label")]
    args = TrainingArguments(output_dir=tempfile.mkdtemp(prefix="lab3_pred_"), per_device_eval_batch_size=batch_size,
                             fp16=torch.cuda.is_available(), report_to="none", disable_tqdm=True)
    trainer = Trainer(model=model, args=args, data_collator=collator)
    return trainer.predict(tokenized_split.select_columns(cols)).predictions


def evaluate_on_test(model_dir, tokenized_test, run: str, collator=None, run_config_path=None,
                     results_dir: Path | None = None, model=None) -> dict:
    """Predict the test set once with the best checkpoint; write <run>_test_predictions.csv.

    VI: Đánh giá test đúng một lần cho mỗi model (gọi lần hai → RuntimeError). model: chỉ dùng trong test.
    Cột CSV: uid, label, pred, prob_pos.
    """
    results_dir = Path(results_dir) if results_dir is not None else output_dir(config.LAB, "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / f"{run}_test_predictions.csv"
    _check_and_mark_test(run_config_path, pred_path)
    if model is None:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    if collator is None:
        from labs.lab3.features.features import make_collator
        from labs.lab3.model.model import load_tokenizer

        collator = make_collator(load_tokenizer())
    logits = predict_logits(model, tokenized_test, collator)
    prob = softmax_pos(logits)
    preds = (prob >= 0.5).astype(int)
    df = pd.DataFrame({"uid": tokenized_test["uid"], "label": tokenized_test["label"], "pred": preds,
                       "prob_pos": np.round(prob, 6)})
    df.to_csv(pred_path, index=False)
    _mark_tested(run_config_path, pred_path)
    return {"run": run, "n": len(df), **accuracy_f1(df["label"], df["pred"]), "predictions": pred_path}


def evaluate_zero_shot(test_ds, pipe=None, results_dir: Path | None = None, batch_size: int = 64) -> dict:
    """Exercise 1 pipeline (SST-2) on the same test set: POSITIVE -> 1, NEGATIVE -> 0; truncation at 512.

    VI: Baseline zero-shot (plan L6). Ghi zero_shot_test_predictions.csv cùng định dạng; chỉ chạy một lần.
    """
    from labs.lab3.sentiment_demo.sentiment_demo import load_sentiment_pipeline, predict_sentiment

    results_dir = Path(results_dir) if results_dir is not None else output_dir(config.LAB, "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / "zero_shot_test_predictions.csv"
    _check_and_mark_test(None, pred_path)
    pipe = pipe or load_sentiment_pipeline()
    out = predict_sentiment(list(test_ds["text"]), pipe, batch_size=batch_size)
    labels = {"POSITIVE": 1, "NEGATIVE": 0}
    pred = np.array([labels[o["label"]] for o in out])
    prob = np.array([o["score"] if o["label"] == "POSITIVE" else 1.0 - o["score"] for o in out])
    df = pd.DataFrame({"uid": test_ds["uid"], "label": test_ds["label"], "pred": pred, "prob_pos": np.round(prob, 6)})
    df.to_csv(pred_path, index=False)
    return {"run": "zero_shot", "n": len(df), **accuracy_f1(df["label"], df["pred"]), "predictions": pred_path}


# ---------------------------------------------------------------- tổng hợp
def duplicate_test_uids(train_df: pd.DataFrame, test_df: pd.DataFrame) -> set:
    """uids of test reviews whose cleaned text also appears in train (EDA: 123 reviews).

    VI: Review test trùng nội dung một review train (cùng cách băm với preprocessing/eda).
    """
    from labs.lab3.preprocessing.preprocessing import text_hash

    train_hashes = set(train_df["text"].map(text_hash))
    return set(test_df.loc[test_df["text"].map(text_hash).isin(train_hashes), "uid"])


def summarize(pred_files: dict, zero_shot_file=None, seeds: dict | None = None, duplicate_uids=None,
              out_path: Path | None = None, pass_accuracy: float = config.PASS_ACCURACY) -> pd.DataFrame:
    """test_summary.csv: one row per seed, mean and std (sample, n-1), zero-shot; pass = mean acc >= 0.91.

    VI: pred_files {run: csv}; seeds {run: seed}; duplicate_uids: thêm cột acc/F1 trên test bỏ review trùng
    (chỉ báo thêm, chỉ số chính vẫn là test chính thức — Thư duyệt 2026-11-20).
    """
    dup = set(duplicate_uids or ())
    rows = []

    def _row(model, run, seed, path):
        df = pd.read_csv(path)
        r = {"model": model, "run": run, "seed": seed, "n": len(df), **accuracy_f1(df["label"], df["pred"])}
        if dup:
            keep = ~df["uid"].isin(dup)
            m = accuracy_f1(df.loc[keep, "label"], df.loc[keep, "pred"])
            r.update({"n_no_dup": int(keep.sum()), "accuracy_no_dup": m["accuracy"], "f1_no_dup": m["f1"]})
        return r

    for run, path in pred_files.items():
        rows.append(_row("fine-tuned", run, (seeds or {}).get(run), path))
    ft = pd.DataFrame(rows)
    metric_cols = [c for c in ("accuracy", "f1", "accuracy_no_dup", "f1_no_dup") if c in ft]
    if len(ft) > 0:
        mean = {"model": "fine-tuned", "run": "mean", "n": int(ft["n"].iloc[0])}
        std = {"model": "fine-tuned", "run": "std", "n": int(ft["n"].iloc[0])}
        for c in metric_cols:
            mean[c] = float(ft[c].mean())
            std[c] = float(ft[c].std(ddof=1)) if len(ft) > 1 else float("nan")
        mean["pass"] = bool(mean["accuracy"] >= pass_accuracy)
        rows += [mean, std]
    if zero_shot_file is not None:
        rows.append(_row("zero-shot (SST-2)", "zero_shot", None, zero_shot_file))
    table = pd.DataFrame(rows)
    for c in metric_cols:
        table[c] = table[c].round(4)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
    return table


def evaluate_control_sentences(model_dir, label: str, pipe_zero_shot=None, pipe_finetuned=None,
                               out_path: Path | None = None) -> pd.DataFrame:
    """The 6 CheckList sentences of Exercise 1 through the zero-shot and the fine-tuned model.

    VI: Cùng 6 câu kiểm soát (sentiment_demo.CONTROL_SENTENCES) cho model trước và sau fine-tune
    → outputs/demo/control_sentences.csv (12 dòng). label: tên model fine-tune (vd main_seed42).
    """
    from labs.lab3.sentiment_demo.sentiment_demo import (
        CONTROL_SENTENCES, load_sentiment_pipeline, predict_sentiment,
    )

    pipe_zero_shot = pipe_zero_shot or load_sentiment_pipeline()
    pipe_finetuned = pipe_finetuned or load_sentiment_pipeline(str(model_dir))
    texts = [s["text"] for s in CONTROL_SENTENCES]
    rows = []
    for name, pipe in (("zero-shot (SST-2)", pipe_zero_shot), (f"fine-tuned ({label})", pipe_finetuned)):
        for s, p in zip(CONTROL_SENTENCES, predict_sentiment(texts, pipe)):
            rows.append({"model": name, "capability": s["capability"], "text": s["text"],
                         "expected": s["expected"], "predicted": p["label"], "score": round(p["score"], 4),
                         "correct": None if s["expected"] is None else p["label"] == s["expected"]})
    table = pd.DataFrame(rows)
    out_path = Path(out_path) if out_path is not None else output_dir(config.LAB, "demo") / "control_sentences.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return table
