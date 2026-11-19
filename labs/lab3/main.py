"""Entry point of Lab 3: sentiment analysis and fine-tuning with Hugging Face (IMDB, DistilBERT).

Run from the repo root:  python -m labs.lab3.main [--smoke]
Owner: Thư (Member A, integration). Member B adds the training stages (train, evaluate,
experiment, visualize) at the marked place after the handoff.

VI: Điểm chạy chính. Chạy từ gốc repo:
    python -m labs.lab3.main --smoke   # smoke run: vài chục review, mọi đầu ra vào outputs/_smoke/
    python -m labs.lab3.main           # normal run
Thứ tự (plan §4): Bài 1 → tải dữ liệu → EDA → split train/val theo phim → tokenize → model
→ [Member B: pilot / 3 seed / max_length trên Kaggle → test một lần → hình, phân tích lỗi].
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from labs.lab3 import config
from shared.paths import output_dir


def _utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_member_a(smoke: bool) -> dict:
    """Stages of Member A; returns the objects Member B needs (tokenized data, collator, tokenizer).

    VI: Chạy phần A. Smoke: ghi vào outputs/_smoke/, dùng config.SMOKE_N_PER_SPLIT review mỗi split.
    """
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    from labs.lab3.eda.eda import run_eda
    from labs.lab3.features.features import make_collator, tokenize_datasets
    from labs.lab3.model.model import build_model, load_tokenizer, model_summary
    from labs.lab3.preprocessing.preprocessing import download_imdb, load_datasets, load_split, sample_per_label
    from labs.lab3.sentiment_demo.sentiment_demo import hub_revision, run_demo

    hf_logging.set_verbosity_error()
    smoke_dir = output_dir(config.LAB, "_smoke")
    out = (lambda kind: smoke_dir / kind) if smoke else (lambda kind: output_dir(config.LAB, kind))

    print("[1/6] Bài 1: sentiment_demo")
    demo = run_demo(out("demo"))
    print(demo["predictions"][["capability", "predicted", "score", "correct"]].to_string(index=False))

    print("[2/6] Tải dữ liệu IMDB gốc")
    download_imdb()
    train, test = load_split("train"), load_split("test")
    if smoke:
        train = sample_per_label(train, config.SMOKE_N_PER_SPLIT)
        test = sample_per_label(test, config.SMOKE_N_PER_SPLIT)

    print("[3/6] EDA")
    eda = run_eda(train, test, AutoTokenizer.from_pretrained(config.CHECKPOINT), out("results"), out("figures"))
    print(eda["summary"][["split", "n_reviews", "tokens_median", "tokens_pct_over_256", "tokens_pct_over_512"]]
          .to_string(index=False))

    print("[4/6] Split train/val theo phim (file đã commit) + tokenize")
    ds = load_datasets(smoke_n=config.SMOKE_N_PER_SPLIT if smoke else None)
    tok = load_tokenizer()
    tokenized = tokenize_datasets(ds, tok, config.MAX_LENGTH)
    print({k: len(v) for k, v in tokenized.items()}, "max_length =", config.MAX_LENGTH)

    print("[5/6] Model")
    summary = model_summary(build_model(config.SEED), hub_revision(config.CHECKPOINT))
    (out("results") / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    print(summary["parameters"])
    return {"tokenized": tokenized, "collator": make_collator(tok), "tokenizer": tok, "datasets": ds}


def run_member_b(parts: dict, smoke: bool) -> None:
    """[6/6] Placeholder for Member B (Phúc): pilot, 3 seeds, max_length, test once, figures.

    VI: Chỗ để Member B gắn train / evaluate / experiment / visualize (docs/phases/*_v1.md).
    """
    print("[6/6] Phần Member B (train, evaluate, experiment, visualize): chưa có — xem progress_status.md")


def main(argv=None) -> None:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description="Lab 3 pipeline")
    parser.add_argument("--smoke", action="store_true", help="smoke run, đầu ra vào outputs/_smoke/")
    args = parser.parse_args(argv)
    t0 = time.time()
    parts = run_member_a(args.smoke)
    run_member_b(parts, args.smoke)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
