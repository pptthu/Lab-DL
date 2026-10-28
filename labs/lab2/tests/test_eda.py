"""Test suite cho Module EDA (CIFAR-10 thật)."""
import os
import sys
from pathlib import Path
import pytest
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shared.paths import raw_data_dir, artifact_dir
from labs.lab2.eda.eda import run_eda


def test_run_eda():
    data_root = str(raw_data_dir("lab2"))
    out_dir = str(artifact_dir("lab2", "figures"))

    # Gọi hàm từ module labs.lab2.eda.eda
    stats = run_eda(data_root=data_root, out_dir=out_dir)

    # 1. Kiểm tra cấu trúc dict trả về
    expected_keys = {"n_train", "n_test", "class_counts", "image_size", "channel_mean", "channel_std"}
    assert set(stats.keys()) == expected_keys, f"Thiếu key trong dict EDA: {expected_keys - set(stats.keys())}"

    # 2. Kiểm tra số lượng ảnh CIFAR-10
    assert stats["n_train"] == 50000, f"n_train kỳ vọng 50000, nhận được: {stats['n_train']}"
    assert stats["n_test"] == 10000, f"n_test kỳ vọng 10000, nhận được: {stats['n_test']}"
    assert stats["image_size"] == (32, 32, 3), f"image_size kỳ vọng (32, 32, 3), nhận được: {stats['image_size']}"

    # 3. Kiểm tra phân bố đều 10 lớp
    assert len(stats["class_counts"]) == 10
    for cls_name, count in stats["class_counts"].items():
        assert count == 5000, f"Lớp {cls_name} có {count} mẫu (kỳ vọng 5000)"

    # 4. Kiểm tra channel mean và std trong khoảng [0, 1]
    assert len(stats["channel_mean"]) == 3
    assert len(stats["channel_std"]) == 3
    for m in stats["channel_mean"]:
        assert 0.0 < m < 1.0, f"Mean bất thường: {m}"
    for s in stats["channel_std"]:
        assert 0.0 < s < 1.0, f"Std bất thường: {s}"

    # 5. Kiểm tra 2 file biểu đồ PNG đã được tạo ra và không rỗng
    dist_png = Path(out_dir) / "cifar10_class_distribution.png"
    grid_png = Path(out_dir) / "cifar10_sample_grid.png"
    assert dist_png.is_file(), f"Không tìm thấy {dist_png}"
    assert grid_png.is_file(), f"Không tìm thấy {grid_png}"
    assert dist_png.stat().st_size > 0, "File distribution PNG bị rỗng"
    assert grid_png.stat().st_size > 0, "File sample grid PNG bị rỗng"


if __name__ == "__main__":
    test_run_eda()
    print(">>> [PASSED] Test EDA thành công!")