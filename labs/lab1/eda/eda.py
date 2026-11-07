"""Module `eda` (EDA) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `eda` (EDA (khám phá dữ liệu)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
# Input / Output of this module
#   Input : labs/lab1/raw_data/fashion-mnist_train.csv (written by preprocessing/download_official.py; not committed)
#   Output: 1) dict of 6 statistics, returned to the caller (main.py stage 1 prints it; numbers recorded in
#              labs/lab1/docs/eda/STATUS.md and progress_status.md)
#           2) labs/lab1/artifacts/figures/class_counts.png and sample_grid.png
#              (main.py passes out_dir = artifact_dir("lab1", "figures"); not committed)
# VI: Input / Output của module
#   Vào : labs/lab1/raw_data/fashion-mnist_train.csv (do preprocessing/download_official.py ghi ra; không commit)
#   Ra  : 1) dict 6 thống kê, trả về cho nơi gọi (main.py stage 1 in ra; số được ghi vào
#            labs/lab1/docs/eda/STATUS.md và progress_status.md)
#         2) labs/lab1/artifacts/figures/class_counts.png và sample_grid.png
#            (main.py truyền out_dir = artifact_dir("lab1", "figures"); không commit)

# Type hints are not evaluated at import time (works on older Python versions too).
# VI: Không tính type hint lúc import (chạy được cả trên Python cũ).
from __future__ import annotations

# Path: build output paths with `/` instead of joining strings.
# VI: Path: ghép đường dẫn bằng `/` thay vì cộng chuỗi.
from pathlib import Path

# pandas reads the CSV; Figure draws without pyplot (no GUI window, no global figure left in memory).
# VI: pandas đọc CSV; Figure vẽ không qua pyplot (không mở cửa sổ, không để hình tồn trong bộ nhớ).
import pandas as pd
from matplotlib.figure import Figure

# 28 x 28 image -> 784 pixel columns; named constants instead of magic numbers.
# VI: Ảnh 28 x 28 -> 784 cột pixel; dùng hằng số có tên thay cho số cứng.
IMAGE_SIDE = 28
N_PIXELS = IMAGE_SIDE * IMAGE_SIDE
LABEL_COL = "label"
# The 10 class names in label order 0-9 (used on the charts).
# VI: 10 tên lớp theo thứ tự nhãn 0-9 (dùng để ghi lên biểu đồ).
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def run_eda(train_csv: str, out_dir: str) -> dict:
    """Explore the training CSV (read-only, never modify data) and save figures.

    Input : train_csv: path to the training CSV (official Zalando GitHub data); out_dir: folder for figures
    Output: dict with n_rows, class_counts, pixel_min, pixel_max, n_missing, n_duplicates; PNG files (class bar chart, sample image grid) in out_dir

    VI: Khám phá file CSV train (chỉ đọc, không sửa dữ liệu) và lưu biểu đồ.
    Vào : train_csv: đường dẫn CSV train (dữ liệu chính thức từ GitHub của Zalando); out_dir: thư mục lưu hình
    Ra  : dict gồm n_rows, class_counts, pixel_min, pixel_max, n_missing, n_duplicates; file PNG (biểu đồ số lượng mỗi lớp, lưới ảnh mẫu) trong out_dir
    """
    # Read the whole CSV: 60,000 rows x 785 columns (label + 784 pixels).
    # VI: Đọc cả file CSV: 60.000 dòng x 785 cột (nhãn + 784 pixel).
    df = pd.read_csv(train_csv)
    # Fail early on a wrong format, before any statistic is computed.
    # VI: Sai định dạng thì báo lỗi ngay, trước khi tính bất kỳ thống kê nào.
    _check_columns(df)

    # Pixel columns only, so min/max are not mixed with the labels.
    # VI: Chỉ lấy cột pixel để min/max không lẫn với nhãn.
    pixels = df.drop(columns=LABEL_COL)
    # value_counts() counts each label, sort_index() orders them 0-9;
    # int() turns numpy numbers into plain Python ints (printable, JSON-safe).
    # VI: value_counts() đếm từng nhãn, sort_index() xếp theo 0-9;
    # int() đổi số numpy thành int Python thường (in được, ghi JSON được).
    class_counts = {
        int(label): int(count)
        for label, count in df[LABEL_COL].value_counts().sort_index().items()
    }
    stats = {
        # Number of images (expected 60,000).
        # VI: Số ảnh (mong đợi 60.000).
        "n_rows": int(len(df)),
        # Class balance (expected 6,000 per class).
        # VI: Độ cân bằng lớp (mong đợi 6.000 mỗi lớp).
        "class_counts": class_counts,
        # min/max skip NaN, so missing cells do not break the statistics.
        # VI: min/max bỏ qua NaN nên ô trống không làm hỏng thống kê.
        # First min per column, then min over the columns = range of the whole table (expected 0-255).
        # VI: Min từng cột rồi min của các cột = khoảng giá trị của cả bảng (mong đợi 0-255).
        "pixel_min": int(pixels.min().min()),
        "pixel_max": int(pixels.max().max()),
        # isna() marks empty cells; the two sums count them over rows and columns.
        # VI: isna() đánh dấu ô trống; hai lần sum đếm theo dòng rồi theo cột.
        "n_missing": int(df.isna().sum().sum()),
        # A duplicate = a row identical to an earlier row (label + all pixels).
        # VI: Dòng trùng = giống hệt một dòng phía trước (cả nhãn và pixel).
        # The first copy is not counted: an image seen 3 times counts 2.
        # VI: Bản đầu tiên không bị đếm: ảnh xuất hiện 3 lần thì đếm 2.
        "n_duplicates": int(df.duplicated().sum()),
    }

    # Create the figure folder if missing; no error if it already exists.
    # VI: Tạo thư mục hình nếu chưa có; có rồi thì không báo lỗi.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Saved files: <out_dir>/class_counts.png and <out_dir>/sample_grid.png (in main.py: labs/lab1/artifacts/figures/).
    # VI: File được lưu: <out_dir>/class_counts.png và <out_dir>/sample_grid.png (trong main.py: labs/lab1/artifacts/figures/).
    _plot_class_counts(class_counts, out / "class_counts.png")
    _plot_sample_grid(df, out / "sample_grid.png")
    # The caller (main.py, notebook) prints and records these numbers.
    # VI: Nơi gọi (main.py, notebook) in ra và ghi lại các số này.
    return stats


def _check_columns(df: pd.DataFrame) -> None:
    """Fail early if the CSV is not label + 784 pixels. / Báo lỗi sớm nếu CSV sai định dạng."""
    # No label column -> not a Fashion-MNIST CSV.
    # VI: Không có cột nhãn -> không phải CSV Fashion-MNIST.
    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV must have a '{LABEL_COL}' column")
    # Total columns minus the label column must be 784.
    # VI: Tổng số cột trừ cột nhãn phải bằng 784.
    n_pixel_cols = df.shape[1] - 1
    if n_pixel_cols != N_PIXELS:
        raise ValueError(f"expected {N_PIXELS} pixel columns, got {n_pixel_cols}")


def _plot_class_counts(class_counts: dict, out_path: Path) -> None:
    """Bar chart: number of samples per class. / Biểu đồ cột số mẫu mỗi lớp."""
    # One wide axis for 10 bars.
    # VI: Một trục rộng cho 10 cột.
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    # x-axis text = label number and class name on two lines.
    # VI: Chữ trục x = số nhãn và tên lớp, xuống 2 dòng.
    labels = list(class_counts)
    names = [f"{i}\n{CLASS_NAMES[i]}" for i in labels]
    # Bar height = number of images of the class.
    # VI: Chiều cao cột = số ảnh của lớp.
    ax.bar(names, list(class_counts.values()), color="#4C72B0")
    ax.set_title("Samples per class (train)")
    ax.set_ylabel("Count")
    # tight_layout keeps the labels inside the image; dpi=100 -> 1000 x 400 px PNG.
    # VI: tight_layout giữ chữ không bị cắt; dpi=100 -> ảnh PNG 1000 x 400 px.
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)


def _plot_sample_grid(df: pd.DataFrame, out_path: Path) -> None:
    """Grid 2x5: the first image of each class. / Lưới 2x5: ảnh đầu tiên của mỗi lớp."""
    # 2 x 5 grid = one cell per class.
    # VI: Lưới 2 x 5 = mỗi lớp một ô.
    fig = Figure(figsize=(10, 5))
    axes = fig.subplots(2, 5)
    # axes.flat walks the grid row by row, so cell i shows class i.
    # VI: axes.flat duyệt lưới theo từng hàng, nên ô thứ i hiện lớp i.
    for i, ax in enumerate(axes.flat):
        # No ticks or frame around an image.
        # VI: Ẩn trục và khung quanh ảnh.
        ax.axis("off")
        # Rows of class i; skip safely if the class is missing (small fake data in tests).
        # VI: Các dòng của lớp i; bỏ qua nếu lớp không có (dữ liệu giả nhỏ trong test).
        rows = df[df[LABEL_COL] == i]
        if rows.empty:
            continue
        # First image of the class: drop the label, 784 values -> 28 x 28.
        # VI: Ảnh đầu tiên của lớp: bỏ nhãn, 784 số -> 28 x 28.
        img = rows.iloc[0].drop(LABEL_COL).to_numpy(dtype=float).reshape(IMAGE_SIDE, IMAGE_SIDE)
        # Fixed 0-255 scale so brightness is comparable between cells.
        # VI: Cố định thang 0-255 để độ sáng giữa các ô so sánh được.
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{i}: {CLASS_NAMES[i]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
