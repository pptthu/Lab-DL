"""Exercise 2.3: load the pre-trained DistilBERT and its tokenizer, with a 2-class head.

VI: Bài 2.3 — nạp distilbert-base-uncased + tokenizer cùng checkpoint, đầu phân loại 2 lớp.
Desc: labs/lab3/docs/phases/model_v1.md

Run from the repo root / Chạy từ gốc repo:
    python -m labs.lab3.model.model     # nạp thật, in kiến trúc đầu phân loại và số param
"""
from __future__ import annotations

from labs.lab3 import config

ID2LABEL = dict(config.LABEL_NAMES)                 # {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def load_tokenizer(checkpoint: str = config.CHECKPOINT):
    """Tokenizer of the same checkpoint as the model. / Tokenizer cùng checkpoint với model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint)


def build_model(seed: int = config.SEED, checkpoint: str = config.CHECKPOINT, model_config=None):
    """DistilBERT backbone (pre-trained) + new classification head (random init, depends on seed).

    VI: Đặt seed TRƯỚC khi tạo model: đầu phân loại khởi tạo ngẫu nhiên theo seed (Dodge et al. 2020).
    model_config (chỉ dùng trong test): tạo model nhỏ từ config, không tải trọng số.
    """
    from transformers import AutoModelForSequenceClassification, set_seed

    set_seed(seed)
    if model_config is not None:
        model_config.num_labels = 2
        model_config.id2label, model_config.label2id = ID2LABEL, LABEL2ID
        return AutoModelForSequenceClassification.from_config(model_config)
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )


def count_parameters(model) -> dict:
    """Total / trainable / backbone / head parameters. / Đếm param theo nhóm (điền model card)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone = sum(p.numel() for p in model.base_model.parameters())
    return {"total": total, "trainable": trainable, "backbone": backbone, "head": total - backbone}


def describe_head(model) -> str:
    """Text of the layers after the backbone (pre_classifier, dropout, classifier).

    VI: In các lớp của đầu phân loại để kiểm với tài liệu HF (model_v1.md).
    """
    names = [n for n, _ in model.named_children() if n != model.base_model_prefix]
    return "\n".join(f"{n}: {getattr(model, n)}" for n in names)


def main() -> None:
    model = build_model()
    print(model.config.__class__.__name__, {k: getattr(model.config, k) for k in
          ("n_layers", "dim", "n_heads", "hidden_dim", "max_position_embeddings", "dropout",
           "seq_classif_dropout", "vocab_size")})
    print(describe_head(model))
    print(count_parameters(model))
    print("revision:", getattr(model.config, "_commit_hash", None))


if __name__ == "__main__":
    main()
