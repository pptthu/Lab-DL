"""Exercise 1: sentiment analysis with a pre-trained Hugging Face model.

VI: Bài 1 — nạp pipeline cảm xúc pre-trained, tokenize câu mẫu, dự đoán nhãn + điểm.
6 câu kiểm soát theo CheckList (Ribeiro et al. 2020); evaluate dùng lại cho model sau fine-tune.
Desc: labs/lab3/docs/phases/sentiment_demo_v1.md

Run from the repo root / Chạy từ gốc repo:
    python -m labs.lab3.sentiment_demo.sentiment_demo --smoke
    python -m labs.lab3.sentiment_demo.sentiment_demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from labs.lab3 import config
from shared.paths import output_dir

# Mỗi câu kiểm một năng lực (CheckList). expected=None: câu trung tính, model nhị phân không có nhãn đúng.
CONTROL_SENTENCES = [
    {"capability": "rõ tích cực", "text": "This movie was absolutely wonderful, I loved every minute of it.",
     "expected": "POSITIVE"},
    {"capability": "rõ tiêu cực", "text": "The film was a complete waste of time and the acting was terrible.",
     "expected": "NEGATIVE"},
    {"capability": "phủ định câu tiêu cực", "text": "The movie was not bad at all.", "expected": "POSITIVE"},
    {"capability": "phủ định câu tích cực", "text": "I would not say this film is good.", "expected": "NEGATIVE"},
    {"capability": "tương phản (but)",
     "text": "The cinematography was beautiful, but the plot was dull and the ending made no sense.",
     "expected": "NEGATIVE"},
    {"capability": "trung tính", "text": "The movie was released in 2009 and runs for about two hours.",
     "expected": None},
]


def load_sentiment_pipeline(model_name: str = config.SST2_MODEL, device: int | None = None):
    """Hugging Face `sentiment-analysis` pipeline with an explicit model name (GPU 0 if available).

    VI: Ghi rõ tên model thay vì để pipeline tự chọn mặc định (mặc định có thể đổi theo phiên bản).
    """
    from transformers import pipeline

    if device is None:
        import torch

        device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=model_name, device=device)


def tokenize_sentence(text: str, tokenizer) -> dict:
    """Tokens, input_ids and attention_mask of one sentence (with [CLS] ... [SEP]).

    VI: Bài 1.3 — tokenize một câu mẫu.
    """
    enc = tokenizer(text)
    return {
        "tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"]),
        "input_ids": list(enc["input_ids"]),
        "attention_mask": list(enc["attention_mask"]),
    }


def predict_sentiment(texts: list[str], pipe, batch_size: int = 32) -> list[dict]:
    """Label (POSITIVE/NEGATIVE) and score per text; long texts truncated to the model limit (512).

    VI: Bài 1.4 — dự đoán nhãn + điểm. truncation=True để dùng lại cho review IMDB dài (evaluate).
    """
    out = pipe(list(texts), truncation=True, batch_size=batch_size)
    return [{"label": o["label"], "score": float(o["score"])} for o in out]


def model_revision(pipe) -> str:
    """Hub commit hash of the loaded model, to rerun with the same weights. / Commit hash model trên Hub."""
    return str(getattr(pipe.model.config, "_commit_hash", None) or "unknown")


def run_demo(out_dir: Path, pipe=None, sentences: list[dict] | None = None) -> dict:
    """Predict the control sentences and tokenize sentence 1; write 2 CSV files.

    VI: Ghi sentiment_demo.csv (6 câu) và sentiment_demo_tokens.csv (token câu 1).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = pipe or load_sentiment_pipeline()
    sentences = sentences or CONTROL_SENTENCES
    preds = predict_sentiment([s["text"] for s in sentences], pipe)
    rows = []
    for s, p in zip(sentences, preds):
        rows.append({
            **s, "predicted": p["label"], "score": round(p["score"], 4),
            "correct": None if s["expected"] is None else p["label"] == s["expected"],
            "model": getattr(pipe.model, "name_or_path", config.SST2_MODEL),
            "revision": model_revision(pipe),
        })
    table = pd.DataFrame(rows)
    tok = tokenize_sentence(sentences[0]["text"], pipe.tokenizer)
    tokens = pd.DataFrame({"position": range(len(tok["tokens"])), "token": tok["tokens"],
                           "input_id": tok["input_ids"], "attention_mask": tok["attention_mask"]})
    paths = {"predictions": out_dir / "sentiment_demo.csv", "tokens": out_dir / "sentiment_demo_tokens.csv"}
    table.to_csv(paths["predictions"], index=False)
    tokens.to_csv(paths["tokens"], index=False)
    return {"paths": paths, "predictions": table, "tokens": tokens}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Lab 3 Exercise 1")
    parser.add_argument("--smoke", action="store_true", help="ghi vào outputs/_smoke/demo/")
    args = parser.parse_args(argv)
    out_dir = output_dir(config.LAB, "_smoke") / "demo" if args.smoke else output_dir(config.LAB, "demo")
    out = run_demo(out_dir)
    print("Câu 1 sau khi tokenize:")
    print(out["tokens"].to_string(index=False))
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(out["predictions"][["capability", "text", "expected", "predicted", "score", "correct"]])
    for p in out["paths"].values():
        print(p)


if __name__ == "__main__":
    main()
