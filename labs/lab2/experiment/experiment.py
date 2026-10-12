"""Module `experiment` (Experiments) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `experiment` (Thí nghiệm). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def run_comparison(model_names: list, hparam_grid: list, loaders: tuple, device, log_root: str) -> list:
    """Train/fine-tune several pre-trained models and hyperparameter settings; one TensorBoard run per setting.

    Input : model_names (e.g. ['resnet18','vgg16']), hparam_grid (list of dicts: lr, batch_size, freeze...), loaders, device, log_root
    Output: list of result dicts: model, hparams, best_val_acc, train_seconds; TensorBoard runs under log_root

    VI: Huấn luyện/fine-tune nhiều model pre-trained và nhiều bộ siêu tham số; mỗi bộ một run TensorBoard.
    Vào : model_names (vd ['resnet18','vgg16']), hparam_grid (danh sách dict: lr, batch_size, freeze...), loaders, device, log_root
    Ra  : danh sách dict kết quả: model, hparams, best_val_acc, train_seconds; run TensorBoard trong log_root
    """
    raise NotImplementedError("TODO(Member B): implement run_comparison")


def save_results(results: list, csv_path: str) -> None:
    """Save the comparison table.

    Input : results, csv_path
    Output: CSV file

    VI: Lưu bảng so sánh.
    Vào : results, csv_path
    Ra  : file CSV
    """
    raise NotImplementedError("TODO(Member B): implement save_results")
