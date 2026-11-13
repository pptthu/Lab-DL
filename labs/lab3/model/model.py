"""Model loading for Lab 3 (Member B). `transformers` is imported inside the function so tests need no network.

VI: Tải model cho Lab 3 (Member B). `transformers` được import bên trong hàm để test không cần mạng.
"""


def load_model_and_tokenizer(model_name: str = "distilbert-base-uncased", num_labels: int = 2):
    """Load a sequence-classification model and its tokenizer from the SAME checkpoint.

    The classification head is newly initialised (random); Hugging Face prints a warning about it.
    That warning is expected: the head is learned during fine-tuning.
    For two labels the config gets id2label {0: NEGATIVE, 1: POSITIVE} (same convention as the pipeline).

    Input : model_name (checkpoint), num_labels
    Output: (model, tokenizer)

    VI: Tải model phân loại chuỗi và tokenizer từ CÙNG một checkpoint.
    Đầu phân loại được khởi tạo ngẫu nhiên; Hugging Face sẽ in cảnh báo. Cảnh báo này bình thường:
    đầu phân loại được học trong lúc fine-tune. Với 2 nhãn, config có id2label {0: NEGATIVE, 1: POSITIVE}.
    Vào : model_name (checkpoint), num_labels
    Ra  : (model, tokenizer)
    """
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(f"num_labels must be an int >= 2, got {num_labels!r} / num_labels phải là số nguyên >= 2")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"num_labels": num_labels}
    if num_labels == 2:
        kwargs["id2label"] = {0: "NEGATIVE", 1: "POSITIVE"}
        kwargs["label2id"] = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    return model, tokenizer
