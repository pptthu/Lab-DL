"""Module `model` (Model) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `model` (Mô hình). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def load_model_and_tokenizer(model_name: str = 'distilbert-base-uncased', num_labels: int = 2) -> tuple:
    """Load a pre-trained model with a classification head and its matching tokenizer.

    Input : model_name, num_labels
    Output: (model, tokenizer)

    VI: Nạp model pre-trained kèm đầu phân loại và tokenizer tương ứng.
    Vào : model_name, num_labels
    Ra  : (model, tokenizer)
    """
    raise NotImplementedError("TODO(Member B): implement load_model_and_tokenizer")
