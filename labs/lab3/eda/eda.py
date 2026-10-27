"""Module `eda` (EDA (text)) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `eda` (EDA (văn bản)). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def run_eda(dataset, out_dir: str, text_col: str = 'text', label_col: str = 'label') -> dict:
    """Explore the text dataset (read-only) and save figures.

    Input : dataset (train split), out_dir, column names
    Output: dict: n_samples, label_counts, length_stats (words); PNG files (label bar chart, length histogram)

    VI: Khám phá dữ liệu văn bản (chỉ đọc) và lưu biểu đồ.
    Vào : dataset (tập train), out_dir, tên cột
    Ra  : dict: n_samples, label_counts, length_stats (số từ); file PNG (biểu đồ nhãn, histogram độ dài)
    """
    raise NotImplementedError("TODO(Member A): implement run_eda")
