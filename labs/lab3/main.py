
"""Entry point for Lab 3: Hugging Face sentiment analysis and fine-tuning.

Run from the repository root:
    python -m labs.lab3.main --help
    python -m labs.lab3.main --run-demo
    python -m labs.lab3.main --run-training --max-steps 2
"""
from __future__ import annotations

import argparse
import json

from shared.paths import artifact_dir
from shared.seed import set_seed

STAGES = [
    "sentiment_demo",
    "preprocessing",
    "eda",
    "model",
    "features",
    "train",
    "evaluate",
]


def run_sentiment_demo() -> None:
    from labs.lab3.sentiment_demo.sentiment_demo import (
        analyze_sentences,
        show_tokens,
    )

    samples = [
        "I love this movie",
        "The plot was boring",
        "Not bad at all",
    ]

    print("\n=== Exercise 1: Sentiment Demo ===")
    print("\nTokenization:")
    print(json.dumps(show_tokens(samples[0]), indent=2))

    print("\nSentiment predictions:")
    for result in analyze_sentences(samples):
        print(
            f"{result['sentence']}: "
            f"{result['label']} ({result['score']:.4f})"
        )


def run_training(args: argparse.Namespace) -> None:
    from labs.lab3.preprocessing.preprocessing import load_text_dataset
    from labs.lab3.eda.eda import run_eda
    from labs.lab3.model.model import load_model_and_tokenizer
    from labs.lab3.features.features import tokenize_dataset
    from labs.lab3.train.train import (
        build_training_args,
        train_with_trainer,
    )
    from labs.lab3.evaluate.evaluate import (
        compute_metrics,
        evaluate_on_test,
    )

    print("\n=== Exercise 2: Fine-tuning Pipeline ===")

    # 1. Load reproducible train, validation and test subsets.
    print("\n[1/6] Loading dataset...")
    train_ds, val_ds, test_ds = load_text_dataset(
        name="imdb",
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )
    print(
        f"Train: {len(train_ds)}, "
        f"Validation: {len(val_ds)}, "
        f"Test: {len(test_ds)}"
    )

    # 2. EDA uses only the training split.
    print("\n[2/6] Running EDA...")
    figures_dir = artifact_dir("lab3", "figures")
    eda_result = run_eda(train_ds, str(figures_dir))
    print(json.dumps(eda_result, indent=2))

    # 3. Load model and tokenizer from the same checkpoint.
    print("\n[3/6] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        num_labels=2,
    )

    # 4. Tokenize all splits with the same tokenizer.
    print("\n[4/6] Tokenizing datasets...")
    tokenized_train = tokenize_dataset(
        train_ds, tokenizer, max_length=args.max_length
    )
    tokenized_val = tokenize_dataset(
        val_ds, tokenizer, max_length=args.max_length
    )
    tokenized_test = tokenize_dataset(
        test_ds, tokenizer, max_length=args.max_length
    )

    # 5. Train using validation only for model selection.
    print("\n[5/6] Training...")
    models_dir = artifact_dir("lab3", "models")
    output_dir = models_dir / "distilbert_sentiment"

    training_args = build_training_args(
        output_dir=str(output_dir),
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    trainer = train_with_trainer(
        model,
        tokenized_train,
        tokenized_val,
        training_args,
        compute_metrics,
        tokenizer=tokenizer,
    )

    # 6. Evaluate the held-out test split exactly once, at the end.
    print("\n[6/6] Final test evaluation...")
    metrics = evaluate_on_test(trainer, tokenized_test)

    logs_dir = artifact_dir("lab3", "logs")
    metrics_path = logs_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nFinal test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Model saved to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lab 3: Hugging Face sentiment analysis pipeline"
    )

    parser.add_argument("--run-demo", action="store_true",
                        help="Run Exercise 1 sentiment demo")
    parser.add_argument("--run-training", action="store_true",
                        help="Run Exercise 2 training pipeline")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)

    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Set a small positive number for a smoke training run",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.run_demo and not args.run_training:
        parser.print_help()
        return

    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.max_length <= 0:
        parser.error("--max-length must be > 0")
    if min(args.n_train, args.n_val, args.n_test) <= 0:
        parser.error("dataset subset sizes must be > 0")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.batch_size <= 0:
        parser.error("epochs, learning-rate and batch-size must be > 0")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer")

    set_seed(args.seed)

    if args.run_demo:
        run_sentiment_demo()

    if args.run_training:
        run_training(args)


if __name__ == "__main__":
    main()
