"""Test suite cho Module Preprocessing (CIFAR-10 thật)."""
from pathlib import Path
import pytest
import sys
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shared.paths import raw_data_dir, artifact_dir
from shared.seed import set_seed
from labs.lab2.preprocessing.preprocessing import (
    load_raw_dataset,
    make_split,
    save_split,
    load_split,
)


def test_preprocessing_pipeline():
    set_seed(42)
    data_root = str(raw_data_dir("lab2"))
    splits_dir = artifact_dir("lab2", "splits")
    split_path = str(splits_dir / "train_val_split.json")

    # 1. Test load_raw_dataset
    train_set, test_set = load_raw_dataset(data_root=data_root, download=False)
    assert len(train_set) == 50000, f"Train set kỳ vọng 50000, nhận được: {len(train_set)}"
    assert len(test_set) == 10000, f"Test set kỳ vọng 10000, nhận được: {len(test_set)}"
    assert train_set.transform is None, "Dataset thô chưa được áp dụng transform"

    # 2. Test make_split với tỷ lệ 10% validation
    n_samples = len(train_set)
    train_idx, val_idx = make_split(n_samples=n_samples, val_ratio=0.1, seed=42)

    assert len(train_idx) == 45000, f"Train idx kỳ vọng 45000, nhận được: {len(train_idx)}"
    assert len(val_idx) == 5000, f"Val idx kỳ vọng 5000, nhận được: {len(val_idx)}"
    assert len(train_idx) + len(val_idx) == n_samples

    # Đảm bảo 2 tập không trùng lặp và bao phủ toàn bộ 50000 mẫu
    set_train, set_val = set(train_idx), set(val_idx)
    assert set_train.isdisjoint(set_val), "Rò rỉ dữ liệu: train_idx và val_idx có phần tử trùng nhau!"
    assert (set_train | set_val) == set(range(n_samples))

    # Đảm bảo tính tái lập (cùng seed ra cùng kết quả)
    t2, v2 = make_split(n_samples=n_samples, val_ratio=0.1, seed=42)
    assert train_idx == t2
    assert val_idx == v2

    # 3. Test save_split và load_split (Milestone bàn giao cho Member B)
    save_split(train_idx, val_idx, split_path)
    assert Path(split_path).is_file(), f"File split JSON chưa được tạo tại {split_path}"

    loaded_train, loaded_val = load_split(split_path)
    assert loaded_train == train_idx
    assert loaded_val == val_idx


if __name__ == "__main__":
    test_preprocessing_pipeline()
    print(">>> [PASSED] Test Preprocessing thành công!")