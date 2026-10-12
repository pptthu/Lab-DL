"""Module `train` (Training with Trainer) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `train` (Huấn luyện bằng Trainer). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def build_training_args(output_dir: str, epochs: int = 2, lr: float = 2e-5, batch_size: int = 16):
    """Create TrainingArguments. Argument names differ between transformers versions: check the installed version.

    Input : output_dir, epochs, lr, batch_size
    Output: TrainingArguments object

    VI: Tạo TrainingArguments. Tên tham số khác nhau giữa các phiên bản transformers: hãy kiểm tra phiên bản đã cài.
    Vào : output_dir, epochs, lr, batch_size
    Ra  : đối tượng TrainingArguments
    """
    raise NotImplementedError("TODO(Member B): implement build_training_args")


def train_with_trainer(model, tokenized_train, tokenized_val, args, compute_metrics) -> object:
    """Create a Trainer, fine-tune and save the model.

    Input : model, tokenized train/val, args, compute_metrics
    Output: trained Trainer; model saved under args.output_dir

    VI: Tạo Trainer, fine-tune và lưu model.
    Vào : model, tập train/val đã tokenize, args, compute_metrics
    Ra  : Trainer đã huấn luyện; model lưu trong args.output_dir
    """
    raise NotImplementedError("TODO(Member B): implement train_with_trainer")
