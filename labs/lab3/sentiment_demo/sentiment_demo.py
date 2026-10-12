"""Module `sentiment_demo` (Exercise 1: pre-trained sentiment analysis) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `sentiment_demo` (Bài 1: phân tích cảm xúc bằng model có sẵn). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def show_tokens(sentence: str, model_name: str = None) -> dict:
    """Tokenize a sample sentence and show tokens and ids.

    Input : sentence, model_name (None = pipeline default)
    Output: dict: tokens (list[str]), input_ids (list[int])

    VI: Tokenize một câu mẫu, hiển thị token và id.
    Vào : sentence, model_name (None = mặc định của pipeline)
    Ra  : dict: tokens (list[str]), input_ids (list[int])
    """
    raise NotImplementedError("TODO(Member A): implement show_tokens")


def analyze_sentences(sentences: list, model_name: str = None) -> list:
    """Run a pre-trained sentiment model from the Hugging Face Hub.

    Input : sentences, model_name
    Output: list of dicts: sentence, label, score

    VI: Chạy model phân tích cảm xúc có sẵn trên Hugging Face Hub.
    Vào : sentences, model_name
    Ra  : danh sách dict: sentence, label, score
    """
    raise NotImplementedError("TODO(Member A): implement analyze_sentences")
