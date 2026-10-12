"""Module `eda` (EDA) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `eda` (EDA (khám phá dữ liệu)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def run_eda(train_csv: str, out_dir: str) -> dict:
    """Explore the training CSV (read-only, never modify data) and save figures.

    Input : train_csv: path to the Kaggle training CSV; out_dir: folder for figures
    Output: dict with n_rows, class_counts, pixel_min, pixel_max, n_missing, n_duplicates; PNG files (class bar chart, sample image grid) in out_dir

    VI: Khám phá file CSV train (chỉ đọc, không sửa dữ liệu) và lưu biểu đồ.
    Vào : train_csv: đường dẫn CSV train của Kaggle; out_dir: thư mục lưu hình
    Ra  : dict gồm n_rows, class_counts, pixel_min, pixel_max, n_missing, n_duplicates; file PNG (biểu đồ số lượng mỗi lớp, lưới ảnh mẫu) trong out_dir
    """
    raise NotImplementedError("TODO(Member A): implement run_eda")
