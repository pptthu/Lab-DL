"""Interface contract + smoke tests for lab2.

The contract test fails if someone renames a function that other modules depend on.
VI: Test hợp đồng giao diện: fail nếu ai đó đổi tên hàm mà module khác đang dùng.
"""
import importlib

import pytest

SLUG = "lab2"
EXPECTED = {'eda': ['run_eda'], 'preprocessing': ['load_raw_dataset', 'make_split', 'save_split', 'load_split'], 'features': ['build_transforms', 'make_loaders'], 'model': ['build_model', 'describe_model', 'unfreeze_last_blocks'], 'train': ['train_model', 'save_checkpoint', 'load_checkpoint'], 'evaluate': ['evaluate', 'confusion_matrix', 'top_confusions'], 'experiment': ['run_comparison', 'save_results']}


@pytest.mark.parametrize("module,names", list(EXPECTED.items()))
def test_module_interface(module, names):
    mod = importlib.import_module(f"labs.{SLUG}.{module}.{module}")
    for name in names:
        assert hasattr(mod, name), f"{module}.{name} is missing / thiếu {module}.{name}"


def test_main_importable():
    main = importlib.import_module(f"labs.{SLUG}.main")
    assert callable(main.main)
