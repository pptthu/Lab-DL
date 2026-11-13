"""Module `sentiment_demo` (Exercise 1: pre-trained sentiment analysis) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `sentiment_demo` (Bài 1: phân tích cảm xúc bằng model có sẵn). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

# Default checkpoint (model_name=None means "pipeline default"): DistilBERT already fine-tuned on SST-2.
# VI: Checkpoint mặc định (model_name=None nghĩa là "mặc định của pipeline"): DistilBERT đã fine-tune sẵn trên SST-2.
# (distilbert-base-uncased gốc chưa có đầu phân loại cảm xúc nên không dùng được cho Bài 1.)
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

_tokenizers: dict = {}
_pipelines: dict = {}


def _get_tokenizer(model_name):
    """Load a tokenizer once and reuse it. / Nạp tokenizer một lần rồi dùng lại."""
    name = model_name or DEFAULT_MODEL
    if name not in _tokenizers:
        from transformers import AutoTokenizer  # imported here: heavy library
        _tokenizers[name] = AutoTokenizer.from_pretrained(name)
    return _tokenizers[name]


def _get_pipeline(model_name):
    """Load a sentiment pipeline once and reuse it. / Nạp pipeline cảm xúc một lần rồi dùng lại."""
    name = model_name or DEFAULT_MODEL
    if name not in _pipelines:
        from transformers import pipeline  # imported here: heavy library
        _pipelines[name] = pipeline("sentiment-analysis", model=name)
    return _pipelines[name]


def show_tokens(sentence: str, model_name: str = None) -> dict:
    """Tokenize a sample sentence and show tokens and ids.

    Special tokens such as [CLS] and [SEP] are included. With model_name=None the tokenizer of the
    same default checkpoint as `analyze_sentences` is used.

    Input : sentence, model_name (None = pipeline default)
    Output: dict: tokens (list[str]), input_ids (list[int])

    VI: Tokenize một câu mẫu, hiển thị token và id.
    Có cả token đặc biệt như [CLS], [SEP]. Với model_name=None dùng tokenizer của cùng checkpoint mặc định
    với `analyze_sentences`.
    Vào : sentence, model_name (None = mặc định của pipeline)
    Ra  : dict: tokens (list[str]), input_ids (list[int])
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a str / sentence phải là str")
    tokenizer = _get_tokenizer(model_name)
    input_ids = list(tokenizer(sentence)["input_ids"])
    return {"tokens": tokenizer.convert_ids_to_tokens(input_ids), "input_ids": input_ids}


def analyze_sentences(sentences: list, model_name: str = None) -> list:
    """Run a pre-trained sentiment model from the Hugging Face Hub.

    This is the zero-shot baseline (no extra training). Long sentences are truncated by the pipeline.

    Input : sentences, model_name
    Output: list of dicts: sentence, label, score

    VI: Chạy model phân tích cảm xúc có sẵn trên Hugging Face Hub.
    Đây là baseline không huấn luyện thêm. Câu quá dài được pipeline cắt bớt.
    Vào : sentences, model_name
    Ra  : danh sách dict: sentence, label, score
    """
    if isinstance(sentences, str):
        sentences = [sentences]
    sentences = list(sentences)
    results = _get_pipeline(model_name)(sentences, truncation=True, batch_size=16)
    return [{"sentence": s, "label": r["label"], "score": float(r["score"])} for s, r in zip(sentences, results)]
