"""Entry point of Lab 3: sentiment analysis and fine-tuning with Hugging Face (IMDB, DistilBERT).

Run from the repo root:  python -m labs.lab3.main [--smoke] [--stage STAGE]
Owner: Thư (Member A, integration). Member B (Phúc) stages: train, evaluate, experiment, visualize.

VI: Điểm chạy chính. Chạy từ gốc repo:
    python -m labs.lab3.main --smoke   # smoke run cả A và B: vài chục review, mọi đầu ra vào outputs/_smoke/
    python -m labs.lab3.main           # normal run phần A (Bài 1, EDA, split, model)
    python -m labs.lab3.main --stage timing|pilot|main|maxlen [--runs main_seed43]   # một bước GPU (Kaggle)
    python -m labs.lab3.main --stage test --model-dirs <dir_best> ...                 # S5: test một lần
    python -m labs.lab3.main --stage figures                                          # hình, phân tích lỗi
Mỗi bước GPU là một kernel Kaggle riêng (labs/lab3/kaggle/), để lỗi ở bước sau không mất bước trước.
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


def utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


_utf8_stdout = utf8_stdout  # tên cũ, giữ cho code của phần A


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


def prepare_parts(smoke: bool) -> dict:
    """Data, tokenizer and collator only (no Exercise 1, no EDA): what one GPU stage needs.

    VI: Chuẩn bị đầu vào cho một bước của Member B (Kaggle): tải dữ liệu, đọc split đã commit.
    Tokenize theo max_length làm trong experiment (mỗi max_length một lần).
    """
    from transformers.utils import logging as hf_logging

    from labs.lab3.features.features import make_collator
    from labs.lab3.model.model import load_tokenizer
    from labs.lab3.preprocessing.preprocessing import download_imdb, load_datasets

    hf_logging.set_verbosity_error()
    download_imdb()
    ds = load_datasets(smoke_n=config.SMOKE_N_PER_SPLIT if smoke else None)
    tok = load_tokenizer()
    print({k: len(v) for k, v in ds.items()}, flush=True)
    return {"datasets": ds, "tokenizer": tok, "collator": make_collator(tok)}


def run_test(parts: dict, model_dirs: list, smoke: bool = False) -> None:
    """S5: test 25k once for every main-run model + zero-shot, summary table, control sentences.

    VI: Mỗi model đúng một lần (evaluate báo lỗi nếu đã test). model_dirs: thư mục <ts>_<run>_best;
    file _config.json của run (outputs/logs/, đã commit) được đánh dấu test_evaluated.
    """
    from pathlib import Path

    from labs.lab3.evaluate.evaluate import (
        evaluate_control_sentences, evaluate_on_test, evaluate_zero_shot, summarize, duplicate_test_uids,
    )
    from labs.lab3.features.features import tokenize_datasets
    from labs.lab3.preprocessing.preprocessing import load_split

    base = output_dir(config.LAB, "_smoke")
    res = (base / "results") if smoke else output_dir(config.LAB, "results")
    logs = (base / "logs") if smoke else output_dir(config.LAB, "logs")
    demo = (base / "demo") if smoke else output_dir(config.LAB, "demo")
    res.mkdir(parents=True, exist_ok=True)
    test_ds = parts["datasets"]["test"]
    tokenized, preds, seeds, val_acc = {}, {}, {}, {}
    for d in map(Path, model_dirs):
        stem = d.name[: -len("_best")]                       # <ts>_<run>
        ts, run = stem[:15], stem[16:]
        cfg_path = logs / f"{stem}_config.json"
        info = json.loads(cfg_path.read_text(encoding="utf-8"))
        L = info["cfg"]["max_length"]
        if L not in tokenized:
            tokenized[L] = tokenize_datasets(test_ds, parts["tokenizer"], L)
        out = evaluate_on_test(d, tokenized[L], run, collator=parts["collator"], run_config_path=cfg_path,
                               results_dir=res)
        preds[run], seeds[run] = out["predictions"], info["cfg"]["seed"]
        val_acc[run] = (info.get("result") or {}).get("best_val_acc", 0)
        print(f"[test] {run} ({ts}): accuracy={out['accuracy']:.4f} f1={out['f1']:.4f} n={out['n']}", flush=True)
    zs = evaluate_zero_shot(test_ds, results_dir=res)
    print(f"[test] zero-shot: accuracy={zs['accuracy']:.4f} f1={zs['f1']:.4f}", flush=True)
    dup = duplicate_test_uids(load_split("train"), load_split("test"))
    table = summarize(preds, zs["predictions"], seeds=seeds, duplicate_uids=dup, out_path=res / "test_summary.csv")
    print(table.to_string(index=False))
    best_run = max(val_acc, key=val_acc.get)                 # seed có val acc cao nhất (chọn trên val)
    best_dir = next(d for d in map(Path, model_dirs) if d.name.endswith(f"_{best_run}_best"))
    ctrl = evaluate_control_sentences(best_dir, best_run, out_path=demo / "control_sentences.csv")
    print(ctrl[["model", "capability", "predicted", "score", "correct"]].to_string(index=False))


def run_member_b(parts: dict, smoke: bool) -> None:
    """[6/6] Member B in a smoke run: pilot, main (3 seeds), test once, figures — a few steps on CPU.

    VI: Smoke run phần B (RULES_v1 bước 7): mọi đầu ra vào outputs/_smoke/. Normal run của phần B
    chạy từng bước trên Kaggle GPU bằng --stage (labs/lab3/kaggle/README.md).
    """
    import shutil

    if not smoke:
        print("[6/6] Phần Member B chạy từng bước trên Kaggle: python -m labs.lab3.main --stage ... "
              "(xem labs/lab3/kaggle/README.md)")
        return
    from labs.lab3.experiment.experiment import run_stage
    from labs.lab3.visualize.visualize import run_all

    base = output_dir(config.LAB, "_smoke")
    for sub in ("logs", "models", "figures"):
        shutil.rmtree(base / sub, ignore_errors=True)       # smoke chạy lại được từ đầu
    for p in [*(base / "results").glob("*_runs.csv"), *(base / "results").glob("*_test_predictions.csv")]:
        p.unlink()
    print("[6/6] Member B smoke: pilot → main 3 seed → test → hình")
    run_stage("pilot", parts, smoke=True)
    run_stage("main", parts, smoke=True)
    run_test(parts, sorted(str(p) for p in (base / "models").glob("*_main_seed*_best")), smoke=True)
    for p in run_all(results_dir=base / "results", logs_dir=base / "logs", figures_dir=base / "figures",
                     demo_dir=base / "demo"):
        print(p)


def main(argv=None) -> None:
    utf8_stdout()
    parser = argparse.ArgumentParser(description="Lab 3 pipeline")
    parser.add_argument("--smoke", action="store_true", help="smoke run, đầu ra vào outputs/_smoke/")
    parser.add_argument("--stage", choices=("timing", "pilot", "main", "maxlen", "test", "figures"),
                        help="chỉ chạy một bước của Member B (Kaggle)")
    parser.add_argument("--runs", nargs="*", help="chỉ chạy các run này của --stage (vd main_seed43)")
    parser.add_argument("--model-dirs", nargs="*", help="--stage test: thư mục checkpoint tốt nhất")
    args = parser.parse_args(argv)
    t0 = time.time()
    if args.stage == "figures":
        from labs.lab3.visualize.visualize import run_all

        for p in run_all():
            print(p)
    elif args.stage == "test":
        if not args.model_dirs:
            parser.error("--stage test needs --model-dirs")
        run_test(prepare_parts(args.smoke), args.model_dirs, smoke=args.smoke)
    elif args.stage:
        from labs.lab3.experiment.experiment import run_stage

        print(run_stage(args.stage, prepare_parts(args.smoke), smoke=args.smoke, runs=args.runs)
              .to_string(index=False))
    else:
        parts = run_member_a(args.smoke)
        run_member_b(parts, args.smoke)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
