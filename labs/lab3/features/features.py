"""Module `features` (Feature engineering (tokenization)) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `features` (Feature engineering (tokenization)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def tokenize_dataset(dataset, tokenizer, max_length: int = 256, text_col: str = 'text'):
    """Tokenize with truncation (and padding if needed).

    Decision for this lab: texts are truncated to `max_length` tokens but NOT padded here.
    Member B pads per batch (DataCollatorWithPadding, which the Trainer uses by default when it
    receives the tokenizer). token_type_ids are not returned (DistilBERT does not use them). The text column is dropped; the label column is kept (the Trainer
    renames it to `labels`). The tokenizer must come from the same checkpoint as the model.

    Input : dataset, tokenizer, max_length, text_col
    Output: tokenized dataset with input_ids, attention_mask and label

    VI: Tokenize với truncation (và padding nếu cần).
    Quyết định cho lab này: văn bản bị cắt còn `max_length` token nhưng CHƯA padding ở đây.
    Member B padding theo từng batch (DataCollatorWithPadding, Trainer tự dùng khi được truyền tokenizer).
    Cột text bị bỏ; cột label được giữ (Trainer tự đổi tên thành `labels`). Tokenizer phải cùng checkpoint với model.
    Vào : dataset, tokenizer, max_length, text_col
    Ra  : dataset đã tokenize có input_ids, attention_mask và label
    """
    if text_col not in dataset.column_names:
        raise KeyError(f"column {text_col!r} not found; columns: {dataset.column_names} / không có cột {text_col!r}")

    def _tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_length, return_token_type_ids=False)

    return dataset.map(_tokenize, batched=True, remove_columns=[text_col])
