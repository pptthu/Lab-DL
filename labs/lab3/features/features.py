"""Exercise 2.4: tokenize the datasets for the model (truncate to max_length, dynamic padding).

VI: Bài 2.4 — tokenize 3 split, cắt theo max_length, pad theo batch (DataCollatorWithPadding).
Desc: labs/lab3/docs/phases/features_v1.md
"""
from __future__ import annotations

KEEP_COLUMNS = ("uid", "label")  # uid để nối dự đoán với review khi phân tích lỗi (visualize)


def tokenize_datasets(ds, tokenizer, max_length: int | None, text_column: str = "text"):
    """Tokenize every split with truncation=True and no fixed padding.

    Output columns: uid, label, input_ids, attention_mask (text and other columns removed).
    VI: max_length bắt buộc (chốt sau EDA); không pad cố định — collator pad theo batch.
    """
    if not max_length:
        raise ValueError("max_length is required: set it after EDA (labs/lab3/config.py MAX_LENGTH)")

    def _tok(batch):
        # return_token_type_ids=False: DistilBERT không dùng token_type_ids, nhưng tokenizer của
        # transformers 5 vẫn trả cột này (đo trên dữ liệu thật 2026-11-20) → tắt cho đúng đầu vào model
        return tokenizer(batch[text_column], truncation=True, max_length=max_length,
                         return_token_type_ids=False)

    first = next(iter(ds.values())) if hasattr(ds, "values") else ds
    drop = [c for c in first.column_names if c not in KEEP_COLUMNS]
    return ds.map(_tok, batched=True, remove_columns=drop, desc=f"tokenize (max_length={max_length})")


def make_collator(tokenizer):
    """Pad each batch to its longest sequence (HF text classification tutorial).

    VI: Pad theo câu dài nhất trong batch — nhanh hơn pad cố định tới max_length.
    """
    from transformers import DataCollatorWithPadding

    return DataCollatorWithPadding(tokenizer=tokenizer)
