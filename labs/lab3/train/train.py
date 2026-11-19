"""Training for Lab 3 (Member B): TrainingArguments + Trainer. `transformers` is imported inside the functions.

VI: Huấn luyện cho Lab 3 (Member B): TrainingArguments + Trainer. `transformers` được import bên trong hàm.
"""


def build_training_args(output_dir, epochs: int = 2, lr: float = 2e-5, batch_size: int = 16,
                        *, max_steps: int = -1, seed: int = 42, logging_steps: int = 10):
    """Build TrainingArguments. Evaluation and saving run once per epoch; the best epoch on VALIDATION f1 is kept.

    The name of the evaluation argument depends on the installed transformers version
    (`eval_strategy` in recent versions, `evaluation_strategy` in older ones): it is detected here.
    `max_steps` > 0 overrides `epochs` (used for the very short smoke fine-tune). Reports to external services are disabled.

    Input : output_dir, epochs, lr, batch_size (+ optional max_steps, seed, logging_steps)
    Output: a TrainingArguments object

    VI: Tạo TrainingArguments. Đánh giá và lưu mỗi epoch; giữ epoch tốt nhất theo f1 trên VALIDATION.
    Tên tham số đánh giá phụ thuộc phiên bản transformers (`eval_strategy` ở bản mới, `evaluation_strategy` ở bản cũ): được tự dò ở đây.
    `max_steps` > 0 sẽ thay `epochs` (dùng cho lần fine-tune rất ngắn để kiểm tra). Tắt báo cáo ra dịch vụ ngoài.
    Vào : output_dir, epochs, lr, batch_size (+ tùy chọn max_steps, seed, logging_steps)
    Ra  : đối tượng TrainingArguments
    """
    if epochs <= 0 or lr <= 0 or batch_size <= 0:
        raise ValueError("epochs, lr and batch_size must be > 0 / epochs, lr và batch_size phải > 0")
    import dataclasses

    from transformers import TrainingArguments

    names = {f.name for f in dataclasses.fields(TrainingArguments)}
    eval_key = "eval_strategy" if "eval_strategy" in names else "evaluation_strategy"
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "weight_decay": 0.01,
        eval_key: "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_steps": logging_steps,
        "max_steps": max_steps,
        "seed": seed,
        "report_to": "none",
    }
    return TrainingArguments(**kwargs)


def train_with_trainer(model, tokenized_train, tokenized_val, args, compute_metrics, tokenizer=None):
    """Fine-tune with the Hugging Face Trainer, save the model to `args.output_dir`, return the trainer.

    Padding is done per batch by DataCollatorWithPadding (Member A's `tokenize_dataset` does not pad).
    That needs the tokenizer, so it is an extra OPTIONAL argument; if omitted it is loaded from `model.name_or_path`.
    The Trainer argument is `processing_class` in recent versions and `tokenizer` in older ones: detected here.
    The test set is NOT passed here: it is only used once, in `evaluate_on_test`.

    Input : model, tokenized_train, tokenized_val, args, compute_metrics (+ optional tokenizer)
    Output: the trained Trainer (model saved in args.output_dir)

    VI: Fine-tune bằng Trainer của Hugging Face, lưu model vào `args.output_dir`, trả về trainer.
    Padding theo từng batch bằng DataCollatorWithPadding (`tokenize_dataset` của A không padding).
    Cần tokenizer nên đây là tham số THÊM, TÙY CHỌN; nếu bỏ qua sẽ tải từ `model.name_or_path`.
    Tham số của Trainer là `processing_class` ở bản mới, `tokenizer` ở bản cũ: được tự dò ở đây.
    Tập test KHÔNG được đưa vào đây: chỉ dùng một lần ở `evaluate_on_test`.
    Vào : model, tokenized_train, tokenized_val, args, compute_metrics (+ tokenizer tùy chọn)
    Ra  : Trainer đã huấn luyện (model lưu trong args.output_dir)
    """
    if len(tokenized_train) == 0 or len(tokenized_val) == 0:
        raise ValueError("train/validation set is empty / tập train hoặc validation rỗng")
    import inspect

    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer

    if tokenizer is None:
        name = getattr(model, "name_or_path", None)
        if not name:
            raise ValueError("pass `tokenizer=`: cannot infer the checkpoint / hãy truyền `tokenizer=`: không suy ra được checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(name)
    tok_key = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        **{tok_key: tokenizer},
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    return trainer
