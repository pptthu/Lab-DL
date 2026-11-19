"""Interface contract + smoke tests for lab3.

The contract test fails if someone renames a function that other modules depend on.
VI: Test hợp đồng giao diện: fail nếu ai đó đổi tên hàm mà module khác đang dùng.
"""
import importlib

import pytest

SLUG = "lab3"
EXPECTED = {'sentiment_demo': ['show_tokens', 'analyze_sentences'], 'eda': ['run_eda'], 'preprocessing': ['load_text_dataset', 'clean_text'], 'features': ['tokenize_dataset'], 'model': ['load_model_and_tokenizer'], 'train': ['build_training_args', 'train_with_trainer'], 'evaluate': ['compute_metrics', 'evaluate_on_test', 'predict_texts']}


@pytest.mark.parametrize("module,names", list(EXPECTED.items()))
def test_module_interface(module, names):
    mod = importlib.import_module(f"labs.{SLUG}.{module}.{module}")
    for name in names:
        assert hasattr(mod, name), f"{module}.{name} is missing / thiếu {module}.{name}"


def test_main_importable():
    main = importlib.import_module(f"labs.{SLUG}.main")
    assert callable(main.main)
