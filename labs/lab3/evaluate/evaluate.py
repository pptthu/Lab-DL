"""Module `evaluate` (Evaluation) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `evaluate` (Đánh giá). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def compute_metrics(eval_pred) -> dict:
    """Metrics callback for the Trainer.

    Input : eval_pred: (logits, labels)
    Output: dict: accuracy, f1

    VI: Hàm tính metric cho Trainer.
    Vào : eval_pred: (logits, labels)
    Ra  : dict: accuracy, f1
    """
    raise NotImplementedError("TODO(Member B): implement compute_metrics")


def evaluate_on_test(trainer, tokenized_test) -> dict:
    """Final evaluation on the test split (use once, at the end).

    Input : trainer, tokenized_test
    Output: dict of metrics

    VI: Đánh giá cuối cùng trên tập test (chỉ dùng một lần ở cuối).
    Vào : trainer, tokenized_test
    Ra  : dict các metric
    """
    raise NotImplementedError("TODO(Member B): implement evaluate_on_test")


def predict_texts(model, tokenizer, texts: list) -> list:
    """Predict labels for new texts.

    Input : model, tokenizer, texts
    Output: list of dicts: text, label, score

    VI: Dự đoán nhãn cho văn bản mới.
    Vào : model, tokenizer, texts
    Ra  : danh sách dict: text, label, score
    """
    raise NotImplementedError("TODO(Member B): implement predict_texts")
