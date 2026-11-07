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
# Input / Output of this script
#   Input : 4 IDX files from https://github.com/zalandoresearch/fashion-mnist (downloaded by torchvision,
#           MD5 checked) into labs/lab1/raw_data/FashionMNIST/raw/
#   Output: labs/lab1/raw_data/fashion-mnist_train.csv (60,000 rows) and fashion-mnist_test.csv (10,000 rows),
#           columns label, pixel1..pixel784; NOT committed (raw_data/ is in .gitignore);
#           printed line per file: name, shape, images per class
# VI: Input / Output của script
#   Vào : 4 file IDX từ https://github.com/zalandoresearch/fashion-mnist (torchvision tải, kiểm MD5)
#         vào labs/lab1/raw_data/FashionMNIST/raw/
#   Ra  : labs/lab1/raw_data/fashion-mnist_train.csv (60.000 dòng) và fashion-mnist_test.csv (10.000 dòng),
#         cột label, pixel1..pixel784; KHÔNG commit (raw_data/ nằm trong .gitignore);
#         in ra một dòng mỗi file: tên, shape, số ảnh mỗi lớp

# Type hints are not evaluated at import time (works on older Python versions too).
# VI: Không tính type hint lúc import (chạy được cả trên Python cũ).
from __future__ import annotations

# numpy: comparisons and counts; pandas: write the CSV;
# FashionMNIST: official loader that downloads the IDX files and checks their MD5.
# VI: numpy: so sánh và đếm; pandas: ghi CSV;
# FashionMNIST: bộ tải chính thức, tải file IDX và kiểm MD5.
import numpy as np
import pandas as pd
from torchvision.datasets import FashionMNIST

# Same constants and reader as the rest of the lab, so names and format cannot drift.
# VI: Dùng chung hằng số và hàm đọc với phần còn lại của lab, nên tên và định dạng không lệch.
from labs.lab1.preprocessing.preprocessing import (
    LABEL_COL, N_PIXELS, TEST_CSV_NAME, TRAIN_CSV_NAME, load_csv,
)
# raw_data_dir("lab1") -> labs/lab1/raw_data/ (never a hand-written path).
# VI: raw_data_dir("lab1") -> labs/lab1/raw_data/ (không viết tay đường dẫn).
from shared.paths import raw_data_dir

# Official sizes: 60,000 train / 10,000 test, 10 balanced classes.
# VI: Kích thước chính thức: 60.000 train / 10.000 test, 10 lớp cân bằng.
# Key = the `train` argument of FashionMNIST; value = (CSV name, expected rows).
# VI: Khóa = tham số `train` của FashionMNIST; giá trị = (tên CSV, số dòng mong đợi).
EXPECTED = {True: (TRAIN_CSV_NAME, 60_000), False: (TEST_CSV_NAME, 10_000)}


def main() -> None:
    root = raw_data_dir("lab1")
    # Two passes: train=True then train=False.
    # VI: Hai lượt: train=True rồi train=False.
    for train, (csv_name, n_rows) in EXPECTED.items():
        # Downloads once into raw_data/FashionMNIST/ and checks MD5; later runs reuse the files.
        # VI: Tải một lần vào raw_data/FashionMNIST/ và kiểm MD5; các lần sau dùng lại file đã có.
        ds = FashionMNIST(root=str(root), train=train, download=True)
        # ds.data is uint8 [N,28,28]; flatten each image to one row of 784 values.
        # VI: ds.data là uint8 [N,28,28]; làm phẳng mỗi ảnh thành một dòng 784 giá trị.
        pixels = ds.data.numpy().reshape(-1, N_PIXELS)
        # Column names pixel1..pixel784, then the label as the first column = the lab's CSV format.
        # VI: Tên cột pixel1..pixel784, rồi chèn nhãn làm cột đầu = đúng định dạng CSV của lab.
        df = pd.DataFrame(pixels, columns=[f"pixel{i}" for i in range(1, N_PIXELS + 1)])
        df.insert(0, LABEL_COL, ds.targets.numpy())
        out = root / csv_name
        # index=False: do not write the pandas row index as an extra column.
        # VI: index=False: không ghi chỉ số dòng của pandas thành một cột thừa.
        df.to_csv(out, index=False)

        # Self-check: read back with load_csv and compare with the source arrays.
        # VI: Tự kiểm: đọc lại bằng load_csv và so với mảng gốc.
        images, labels = load_csv(str(out))
        # Right number of images, then identical pixels and labels to the IDX source.
        # VI: Đúng số ảnh, rồi khớp từng pixel và từng nhãn với file IDX gốc.
        assert images.shape == (n_rows, 28, 28), images.shape
        assert np.array_equal(images, ds.data.numpy())
        assert np.array_equal(labels, ds.targets.numpy())
        # Every class has exactly N/10 images (6,000 train, 1,000 test).
        # VI: Mỗi lớp đúng N/10 ảnh (6.000 train, 1.000 test).
        counts = np.bincount(labels, minlength=10)
        assert (counts == n_rows // 10).all(), counts
        # Visible evidence to record in progress_status.md.
        # VI: Bằng chứng hiện ra để ghi vào progress_status.md.
        print(out.name, images.shape, "per class", counts.tolist())


if __name__ == "__main__":
    main()
