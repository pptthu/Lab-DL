"""Test suite cho Module Features (CIFAR-10 thật)."""
from pathlib import Path
import pytest
import sys
import torch
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shared.paths import raw_data_dir, artifact_dir
from shared.seed import set_seed
from labs.lab2.preprocessing.preprocessing import load_raw_dataset, load_split
from labs.lab2.features.features import build_transforms, make_loaders


def test_features_pipeline():
    set_seed(42)
    data_root = str(raw_data_dir("lab2"))
    split_path = str(artifact_dir("lab2", "splits") / "train_val_split.json")

    # 1. Nạp dataset và split đã tạo từ step trước
    train_set, test_set = load_raw_dataset(data_root=data_root, download=False)
    train_idx, val_idx = load_split(split_path)

    # 2. Test build_transforms
    IMAGE_SIZE = 32
    train_tf, eval_tf = build_transforms(image_size=IMAGE_SIZE, augment=False)

    sample_img, _ = train_set[0]
    tensor_img = eval_tf(sample_img)
    assert isinstance(tensor_img, torch.Tensor), "Output của transform phải là torch.Tensor"
    assert tensor_img.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Shape sai: {tensor_img.shape}"
    assert tensor_img.dtype == torch.float32

    # 3. Test make_loaders
    BATCH_SIZE = 64
    train_loader, val_loader, test_loader = make_loaders(
        train_set=train_set,
        test_set=test_set,
        train_idx=train_idx,
        val_idx=val_idx,
        train_tf=train_tf,
        eval_tf=eval_tf,
        batch_size=BATCH_SIZE,
    )

    # Kiểm tra quy tắc num_workers = 0 (an toàn trên Windows)
    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0
    assert test_loader.num_workers == 0

    # Kiểm tra số lượng batch
    assert len(train_loader) == 704
    assert len(val_loader) == 79
    assert len(test_loader) == 157

    # 4. Kiểm tra Milestone Handoff: batch thật [B, 3, H, W] và nhãn 0-9
    batch_images, batch_labels = next(iter(train_loader))
    assert batch_images.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), f"Batch shape sai: {batch_images.shape}"
    assert batch_labels.shape == (BATCH_SIZE,), f"Batch labels shape sai: {batch_labels.shape}"
    assert batch_labels.min() >= 0 and batch_labels.max() <= 9, "Nhãn vượt ngoài phạm vi 0-9"


if __name__ == "__main__":
    test_features_pipeline()
    print(">>> [PASSED] Test Features thành công!")