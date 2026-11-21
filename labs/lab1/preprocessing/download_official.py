"""Download the official Fashion-MNIST (Zalando GitHub) and write it as the lab's CSV format.

torchvision.datasets.FashionMNIST downloads the 4 original Zalando IDX files and checks
their MD5. This script writes them as label + 784 pixel columns (pixel1..pixel784), with
the same file names that load_csv, run_eda, main.py and the notebook already read.
Run: python -m labs.lab1.preprocessing.download_official

VI: Tải Fashion-MNIST chính thức (GitHub của Zalando) và ghi ra đúng định dạng CSV của lab.
torchvision.datasets.FashionMNIST tải 4 file IDX gốc của Zalando và tự kiểm MD5. Script ghi
chúng thành cột label + 784 cột pixel (pixel1..pixel784), cùng tên file mà load_csv, run_eda,
main.py và notebook đang đọc.
Chạy: python -m labs.lab1.preprocessing.download_official
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from torchvision.datasets import FashionMNIST

from labs.lab1.preprocessing.preprocessing import (
    LABEL_COL, N_PIXELS, TEST_CSV_NAME, TRAIN_CSV_NAME, load_csv,
)
from shared.paths import data_dir

# Official sizes: 60,000 train / 10,000 test, 10 balanced classes.
# VI: Kích thước chính thức: 60.000 train / 10.000 test, 10 lớp cân bằng.
EXPECTED = {True: (TRAIN_CSV_NAME, 60_000), False: (TEST_CSV_NAME, 10_000)}


def main() -> None:
    root = data_dir("lab1", "raw")
    for train, (csv_name, n_rows) in EXPECTED.items():
        ds = FashionMNIST(root=str(root), train=train, download=True)
        pixels = ds.data.numpy().reshape(-1, N_PIXELS)
        df = pd.DataFrame(pixels, columns=[f"pixel{i}" for i in range(1, N_PIXELS + 1)])
        df.insert(0, LABEL_COL, ds.targets.numpy())
        out = root / csv_name
        df.to_csv(out, index=False)

        # Self-check: read back with load_csv and compare with the source arrays.
        # VI: Tự kiểm: đọc lại bằng load_csv và so với mảng gốc.
        images, labels = load_csv(str(out))
        assert images.shape == (n_rows, 28, 28), images.shape
        assert np.array_equal(images, ds.data.numpy())
        assert np.array_equal(labels, ds.targets.numpy())
        counts = np.bincount(labels, minlength=10)
        assert (counts == n_rows // 10).all(), counts
        print(out.name, images.shape, "per class", counts.tolist())


if __name__ == "__main__":
    main()
