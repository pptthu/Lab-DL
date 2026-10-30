"""Module `evaluate` (Evaluation) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

Metrics are computed with NumPy only (no `evaluate` library, no network), so the code is easy to test
and the file name `evaluate` cannot clash with the Hugging Face package of the same name.
Binary task: label 1 = POSITIVE is the positive class for F1. `torch` is imported inside `predict_texts`.

VI: Module `evaluate` (Đánh giá). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
Metric chỉ dùng NumPy (không dùng thư viện `evaluate`, không cần mạng) nên dễ test và tên file `evaluate`
không trùng với package Hugging Face cùng tên. Bài toán nhị phân: nhãn 1 = POSITIVE là lớp dương khi tính F1.
`torch` được import bên trong `predict_texts`.
"""
from __future__ import annotations

import numpy as np

# Pipeline label (Member A's `analyze_sentences`) -> integer label of the dataset. / Nhãn pipeline -> nhãn số.
LABEL_TO_ID = {"NEGATIVE": 0, "POSITIVE": 1}


def classification_metrics(labels, predictions) -> dict:
    """Accuracy and binary F1 (positive class = 1) from integer labels and integer predictions.

    F1 is 0.0 when the model never predicts class 1 correctly (no division by zero).

    Input : labels (n,), predictions (n,) with values 0/1
    Output: dict: accuracy, f1 (python floats)

    VI: Accuracy và F1 nhị phân (lớp dương = 1) từ nhãn thật và nhãn dự đoán (số nguyên).
    F1 bằng 0.0 khi model không đoán đúng lớp 1 lần nào (không chia cho 0).
    Vào : labels (n,), predictions (n,) giá trị 0/1
    Ra  : dict: accuracy, f1 (số float)
    """
    y = np.asarray(labels).astype(int).reshape(-1)
    p = np.asarray(predictions).astype(int).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"labels {y.shape} and predictions {p.shape} differ / nhãn và dự đoán khác kích thước")
    if y.size == 0:
        raise ValueError("empty input / đầu vào rỗng")
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    denom = 2 * tp + fp + fn
    return {"accuracy": float(np.mean(p == y)), "f1": float(2 * tp / denom) if denom else 0.0}


def compute_metrics(eval_pred) -> dict:
    """Metrics callback for the Trainer.

    `eval_pred` is a tuple (logits, labels) or a transformers `EvalPrediction` (both can be indexed 0 and 1).
    The predicted class is the argmax of the logits. The key `f1` is the one used to keep the best epoch.

    Input : eval_pred: (logits, labels)
    Output: dict: accuracy, f1

    VI: Hàm tính metric cho Trainer.
    `eval_pred` là tuple (logits, labels) hoặc `EvalPrediction` của transformers (đều lấy được phần tử 0 và 1).
    Lớp dự đoán là argmax của logits. Khóa `f1` được dùng để giữ epoch tốt nhất.
    Vào : eval_pred: (logits, labels)
    Ra  : dict: accuracy, f1
    """
    logits, labels = eval_pred[0], eval_pred[1]
    if isinstance(logits, (tuple, list)):  # some models return (logits, extra...) / một số model trả (logits, ...)
        logits = logits[0]
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (n, num_labels), got shape {logits.shape} / logits phải có dạng (n, num_labels)")
    return classification_metrics(labels, np.argmax(logits, axis=-1))


def evaluate_on_test(trainer, tokenized_test) -> dict:
    """Final evaluation on the test split (use once, at the end, never to choose the model).

    Calls `trainer.evaluate` with the prefix `test` and returns the keys without the prefix.
    The trainer holds the best epoch chosen on VALIDATION (`load_best_model_at_end`).

    Input : trainer, tokenized_test
    Output: dict of metrics (accuracy, f1, loss, ...)

    VI: Đánh giá cuối cùng trên tập test (chỉ dùng một lần ở cuối, không dùng để chọn model).
    Gọi `trainer.evaluate` với tiền tố `test` và trả về khóa đã bỏ tiền tố.
    Trainer đang giữ epoch tốt nhất chọn trên VALIDATION (`load_best_model_at_end`).
    Vào : trainer, tokenized_test
    Ra  : dict các metric (accuracy, f1, loss, ...)
    """
    if len(tokenized_test) == 0:
        raise ValueError("test set is empty / tập test rỗng")
    raw = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    out = {}
    for key, value in raw.items():
        name = key[len("test_"):] if key.startswith("test_") else key
        out[name] = float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value
    return out


def predict_texts(model, tokenizer, texts: list, *, max_length: int = 256, batch_size: int = 16) -> list:
    """Predict labels for new texts (raw strings, no tokenization needed by the caller).

    `label` is the integer class (0 = NEGATIVE, 1 = POSITIVE) and `score` is the softmax probability of that class.
    The model is put in eval mode for the call and its previous mode is restored. Texts are tokenized with the
    SAME tokenizer as the model, truncated to `max_length` (same as training) and padded per batch.

    Input : model, tokenizer, texts (+ optional max_length, batch_size)
    Output: list of dicts: text, label, score

    VI: Dự đoán nhãn cho văn bản mới (chuỗi thô, người gọi không cần tự tokenize).
    `label` là lớp số nguyên (0 = NEGATIVE, 1 = POSITIVE), `score` là xác suất softmax của lớp đó.
    Model được đặt ở chế độ eval khi gọi và trả về chế độ cũ sau đó. Văn bản được tokenize bằng CÙNG tokenizer
    của model, cắt tới `max_length` (giống lúc train) và padding theo batch.
    Vào : model, tokenizer, texts (+ tùy chọn max_length, batch_size)
    Ra  : danh sách dict: text, label, score
    """
    if isinstance(texts, str) or not all(isinstance(t, str) for t in texts):
        raise TypeError("texts must be a list of str / texts phải là danh sách chuỗi")
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be > 0 / batch_size và max_length phải > 0")
    texts = list(texts)
    if not texts:
        return []
    import torch

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    results = []
    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                enc = tokenizer(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}  # DistilBERT takes no token_type_ids
                probs = torch.softmax(model(**enc).logits, dim=-1)
                scores, labels = probs.max(dim=-1)
                for text, label, score in zip(chunk, labels.tolist(), scores.tolist()):
                    results.append({"text": text, "label": int(label), "score": float(score)})
    finally:
        model.train(was_training)
    return results


def zero_shot_labels(results: list) -> list:
    """Map the output of Member A's `analyze_sentences` to integer labels (POSITIVE -> 1, NEGATIVE -> 0).

    Input : results: list of dicts with key `label` ("POSITIVE" / "NEGATIVE")
    Output: list of int (0 or 1), same order

    VI: Đổi kết quả hàm `analyze_sentences` của Member A sang nhãn số (POSITIVE -> 1, NEGATIVE -> 0).
    Vào : results: danh sách dict có khóa `label` ("POSITIVE" / "NEGATIVE")
    Ra  : danh sách int (0 hoặc 1), cùng thứ tự
    """
    out = []
    for r in results:
        name = str(r["label"]).upper()
        if name not in LABEL_TO_ID:
            raise ValueError(f"unexpected label {r['label']!r}, expected POSITIVE or NEGATIVE / nhãn không hợp lệ")
        out.append(LABEL_TO_ID[name])
    return out
