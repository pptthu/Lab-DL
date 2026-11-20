"""Tests for labs/lab3/model (offline: tiny DistilBERT built from a config, no download)."""
import torch

from labs.lab3.model.model import ID2LABEL, build_model, count_parameters, describe_head


def tiny_config():
    from transformers import DistilBertConfig

    return DistilBertConfig(vocab_size=40, dim=16, n_layers=1, n_heads=2, hidden_dim=32,
                            max_position_embeddings=64)


def test_build_model_outputs_two_logits():
    model = build_model(seed=0, model_config=tiny_config())
    out = model(input_ids=torch.tensor([[2, 5, 6, 3]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
    assert out.logits.shape == (1, 2)
    assert model.config.id2label == ID2LABEL


def test_same_seed_same_head_different_seed_different_head():
    a = build_model(seed=1, model_config=tiny_config()).classifier.weight
    b = build_model(seed=1, model_config=tiny_config()).classifier.weight
    c = build_model(seed=2, model_config=tiny_config()).classifier.weight
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_count_parameters_splits_backbone_and_head():
    model = build_model(seed=0, model_config=tiny_config())
    n = count_parameters(model)
    assert n["total"] == n["backbone"] + n["head"] == n["trainable"]
    # đầu phân loại DistilBERT: pre_classifier dim->dim + classifier dim->2 (dim=16)
    assert n["head"] == (16 * 16 + 16) + (16 * 2 + 2)
    assert "pre_classifier" in describe_head(model) and "classifier" in describe_head(model)
