"""Module `features` (Feature engineering (tokenization)) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `features` (Feature engineering (tokenization)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def tokenize_dataset(dataset, tokenizer, max_length: int = 256, text_col: str = 'text'):
    """Tokenize with truncation (and padding if needed).

    Input : dataset, tokenizer, max_length, text_col
    Output: tokenized dataset with input_ids, attention_mask and label

    VI: Tokenize với truncation (và padding nếu cần).
    Vào : dataset, tokenizer, max_length, text_col
    Ra  : dataset đã tokenize có input_ids, attention_mask và label
    """
    raise NotImplementedError("TODO(Member A): implement tokenize_dataset")
