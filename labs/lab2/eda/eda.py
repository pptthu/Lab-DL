"""Module `eda` (EDA) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `eda` (EDA (khám phá dữ liệu)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def run_eda(data_root: str, out_dir: str) -> dict:
    """Explore the image dataset (read-only) and save figures.

    Input : data_root: dataset folder (downloaded by torchvision); out_dir
    Output: dict: n_train, n_test, class_counts, image_size, channel_mean, channel_std; PNG files in out_dir

    VI: Khám phá bộ dữ liệu ảnh (chỉ đọc) và lưu biểu đồ.
    Vào : data_root: thư mục dữ liệu (torchvision tải về); out_dir
    Ra  : dict: n_train, n_test, class_counts, image_size, channel_mean, channel_std; file PNG trong out_dir
    """
    raise NotImplementedError("TODO(Member A): implement run_eda")
