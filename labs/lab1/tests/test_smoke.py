"""Interface contract + smoke tests for lab1.

The contract test fails if someone renames a function that other modules depend on.
VI: Test hợp đồng giao diện: fail nếu ai đó đổi tên hàm mà module khác đang dùng.
"""
import importlib

import pytest

SLUG = "lab1"
EXPECTED = {'eda': ['run_eda'], 'preprocessing': ['load_csv', 'make_split', 'save_split', 'load_split'], 'features': ['compute_mean_std', 'build_transform', 'FashionDataset', 'make_loaders'], 'model': ['MLP', 'count_parameters'], 'train': ['train_model', 'save_checkpoint', 'load_checkpoint'], 'evaluate': ['evaluate', 'confusion_matrix', 'per_class_accuracy'], 'visualize': ['plot_loss_curves', 'show_predictions', 'plot_confusion_matrix'], 'experiment': ['run_experiments', 'save_results']}


@pytest.mark.parametrize("module,names", list(EXPECTED.items()))
def test_module_interface(module, names):
    mod = importlib.import_module(f"labs.{SLUG}.{module}.{module}")
    for name in names:
        assert hasattr(mod, name), f"{module}.{name} is missing / thiếu {module}.{name}"


def test_main_importable():
    main = importlib.import_module(f"labs.{SLUG}.main")
    assert callable(main.main)
