"""Module `preprocessing` (Preprocessing) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `preprocessing` (Tiền xử lý). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

def load_text_dataset(name: str = 'imdb', n_train: int = 2000, n_val: int = 500, n_test: int = 500, seed: int = 42) -> tuple:
    """Load a binary text-classification dataset and take small reproducible subsets (CPU friendly).

    Input : name, subset sizes, seed
    Output: (train, val, test) dataset objects with columns text and label (0/1)

    VI: Nạp dataset phân loại văn bản nhị phân và lấy tập con nhỏ, tái lập được (nhẹ cho CPU).
    Vào : name, kích thước tập con, seed
    Ra  : (train, val, test) với cột text và label (0/1)
    """
    raise NotImplementedError("TODO(Member A): implement load_text_dataset")


def clean_text(text: str) -> str:
    """Basic cleaning (e.g. strip HTML tags, extra spaces). Keep it minimal and documented.

    Input : text
    Output: cleaned text

    VI: Làm sạch cơ bản (bỏ thẻ HTML, khoảng trắng thừa). Giữ tối giản và ghi rõ.
    Vào : text
    Ra  : văn bản đã làm sạch
    """
    raise NotImplementedError("TODO(Member A): implement clean_text")
