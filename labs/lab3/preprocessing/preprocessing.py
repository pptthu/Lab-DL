"""Module `preprocessing` (Preprocessing) of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `preprocessing` (Tiền xử lý). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

import html
import re
from collections import Counter

from shared.paths import raw_data_dir

SLUG = "lab3"
# Short name -> Hugging Face Hub id. / Tên ngắn -> id trên Hugging Face Hub.
HUB_IDS = {"imdb": "stanfordnlp/imdb"}

_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")   # real HTML tags only; "<3" or "a < b" are kept
_SPACE_RE = re.compile(r"\s+")


def label_counts(dataset, label_col: str = "label") -> dict:
    """Count samples per label, e.g. {0: 1003, 1: 997}. Used to check class balance.

    Input : dataset, label column name
    Output: dict {label: count}, sorted by label

    VI: Đếm số mẫu mỗi nhãn, ví dụ {0: 1003, 1: 997}. Dùng để kiểm tra độ cân bằng nhãn.
    Vào : dataset, tên cột nhãn
    Ra  : dict {nhãn: số mẫu}, sắp theo nhãn
    """
    return dict(sorted(Counter(int(x) for x in dataset[label_col]).items()))


def load_text_dataset(name: str = 'imdb', n_train: int = 2000, n_val: int = 500, n_test: int = 500, seed: int = 42) -> tuple:
    """Load a binary text-classification dataset and take small reproducible subsets (CPU friendly).

    The Hub splits of IMDB are ordered by label, so each pool is shuffled with `seed` BEFORE
    taking the subset. Validation is cut from the original train split; test comes from the
    original test split, so the three subsets never overlap. Text is cleaned with `clean_text`.

    Input : name, subset sizes, seed
    Output: (train, val, test) dataset objects with columns text and label (0/1)

    VI: Nạp dataset phân loại văn bản nhị phân và lấy tập con nhỏ, tái lập được (nhẹ cho CPU).
    Tập IMDB trên Hub xếp theo nhãn nên mỗi tập gốc được xáo trộn theo `seed` TRƯỚC khi lấy tập con.
    Validation cắt từ train gốc; test lấy từ test gốc nên ba tập con không trùng nhau.
    Văn bản được làm sạch bằng `clean_text`.
    Vào : name, kích thước tập con, seed
    Ra  : (train, val, test) với cột text và label (0/1)
    """
    from datasets import load_dataset  # imported here: heavy library, needs internet the first time

    cache_dir = raw_data_dir(SLUG)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(HUB_IDS.get(name, name), cache_dir=str(cache_dir))

    train_pool = raw["train"].shuffle(seed=seed)   # shuffle BEFORE select
    test_pool = raw["test"].shuffle(seed=seed)
    if n_train + n_val > len(train_pool):
        raise ValueError(f"n_train + n_val = {n_train + n_val} but the train split has {len(train_pool)} rows")
    if n_test > len(test_pool):
        raise ValueError(f"n_test = {n_test} but the test split has {len(test_pool)} rows")

    subsets = {
        "train": train_pool.select(range(n_train)),
        "val": train_pool.select(range(n_train, n_train + n_val)),
        "test": test_pool.select(range(n_test)),
    }
    for split, ds in subsets.items():
        missing = {"text", "label"} - set(ds.column_names)
        if missing:
            raise ValueError(f"{split} split is missing columns {sorted(missing)}")
        subsets[split] = ds.map(lambda batch: {"text": [clean_text(t) for t in batch["text"]]}, batched=True)
        bad = set(label_counts(subsets[split])) - {0, 1}
        if bad:
            raise ValueError(f"{split} split has labels outside {{0, 1}}: {sorted(bad)}")
    return subsets["train"], subsets["val"], subsets["test"]


def clean_text(text: str) -> str:
    """Basic cleaning (e.g. strip HTML tags, extra spaces). Keep it minimal and documented.

    Steps: decode HTML entities (&amp; -> &), replace HTML tags such as <br /> with a space,
    collapse whitespace, strip the ends. It does NOT lowercase and does NOT remove punctuation
    or stop words: the uncased DistilBERT tokenizer lowercases by itself and marks like "!" carry sentiment.

    Input : text
    Output: cleaned text

    VI: Làm sạch cơ bản (bỏ thẻ HTML, khoảng trắng thừa). Giữ tối giản và ghi rõ.
    Các bước: giải mã thực thể HTML (&amp; -> &), thay thẻ HTML như <br /> bằng khoảng trắng,
    gộp khoảng trắng, cắt hai đầu. KHÔNG hạ chữ thường và KHÔNG bỏ dấu câu hay stop word:
    tokenizer uncased của DistilBERT tự hạ chữ thường và các dấu như "!" mang thông tin cảm xúc.
    Vào : text
    Ra  : văn bản đã làm sạch
    """
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = _TAG_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()
