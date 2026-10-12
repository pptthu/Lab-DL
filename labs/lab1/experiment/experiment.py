"""Module `experiment` (Experiments) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `experiment` (Thí nghiệm). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def run_experiments(configs: list, loaders: tuple, device) -> list:
    """Train one model per config (e.g. hidden sizes, lr, dropout, optimizer) and compare on VALIDATION data.

    Input : configs: list of dicts; loaders: (train, val, test); device
    Output: list of result dicts: config, best_val_acc, n_params, train_seconds

    VI: Huấn luyện mỗi config một model (hidden sizes, lr, dropout, optimizer...) và so sánh trên VALIDATION.
    Vào : configs: danh sách dict; loaders: (train, val, test); device
    Ra  : danh sách dict kết quả: config, best_val_acc, n_params, train_seconds
    """
    raise NotImplementedError("TODO(Member B): implement run_experiments")


def save_results(results: list, csv_path: str) -> None:
    """Save the comparison table.

    Input : results, csv_path
    Output: CSV file

    VI: Lưu bảng so sánh.
    Vào : results, csv_path
    Ra  : file CSV
    """
    raise NotImplementedError("TODO(Member B): implement save_results")
