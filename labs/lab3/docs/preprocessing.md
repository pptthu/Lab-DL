# Lab 3 / preprocessing: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/preprocessing/preprocessing.py` |
| Owner | Member A (Muội), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only, 2026-10-25) |
| Files you may edit | `labs/lab3/preprocessing/preprocessing.py`, `labs/lab3/tests/test_preprocessing.py`, this guide |
| Files you must not edit | other modules, the dataset cache in `labs/lab3/raw_data/`, other owners' notebook cells |
| Agreed signatures | `load_text_dataset(name: str = 'imdb', n_train: int = 2000, n_val: int = 500, n_test: int = 500, seed: int = 42) -> tuple`; `clean_text(text: str) -> str` (same as `project_description.md`). Extra helper in the code: `label_counts(dataset, label_col='label') -> dict` |
| Depends on / used by | `datasets`, `shared.paths.raw_data_dir` / `eda`, `features`, notebook |
| Prohibited | taking validation from the test split; selecting before shuffling; lowercasing or removing punctuation in `clean_text` without the team leader; committing the dataset; a download inside unit tests |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_preprocessing.py -q` (fake dataset, no network). Smoke: one real `load_text_dataset()` call |
| Done when | unit tests pass; smoke output recorded in `progress_status.md`; this guide updated |

## Current task (edited ONLY by the team leader)

- None.

Module owners and agents never assign work to themselves here; they ask the team leader.

## What the brief requires

- Brief requirement (Exercise 2): load a simple binary text dataset and preprocess it for the model.
- Team choice: IMDB from the Hugging Face Hub (`stanfordnlp/imdb`; official page https://ai.stanford.edu/~amaas/data/sentiment/). The link list came from another group, not from the lecturer, so IMDB is a team choice. Small subsets for CPU.

## How the module works (current code)

- `load_dataset(HUB_IDS.get(name, name))` with the cache in `raw_data_dir("lab3")` (never `./data/raw`).
- The Hub splits are ordered by label, so each pool is **shuffled with `seed` before selecting**.
- Train and validation come from the original train split (`0..n_train`, then `n_train..n_train+n_val`); test comes from the original test split. The three subsets never overlap.
- `clean_text` is applied to every subset; labels are checked to be in {0, 1}; missing `text`/`label` columns raise `ValueError`.
- `clean_text`: HTML entities decoded, HTML tags such as `<br />` replaced by a space, whitespace collapsed. No lowercasing, no punctuation or stop-word removal (the uncased tokenizer lowercases; "!" carries sentiment).

## Inputs, outputs and pitfalls

- Output: three `datasets.Dataset` objects with columns `text` and `label` (0/1).
- `n_train + n_val` larger than the train split, or `n_test` larger than the test split, raises `ValueError`.
- The first call needs internet; later calls use the cache.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_preprocessing.py -q` | all pass, no network |
| Smoke test | `python -c "from labs.lab3.preprocessing.preprocessing import load_text_dataset, label_counts; tr,va,te=load_text_dataset(); print(len(tr), label_counts(tr), len(va), label_counts(va), len(te), label_counts(te), '<br' in ' '.join(tr['text']))"` | recorded: `2000 {0: 1000, 1: 1000} 500 {0: 243, 1: 257} 500 {0: 254, 1: 246} False` (Reported only) |
| Real integration run | only at an approved milestone | as above |

## Open questions / limitations

- IMDB and the subset sizes are team choices, not confirmed by the lecturer.
- The README says "put the dataset in `labs/lab3/raw_data/`"; in the code the dataset is downloaded automatically into that folder as a cache.

---

**Tiếng Việt**

- **Dành cho AI:** người phụ trách Member A (Muội), do nhóm trưởng giao. Được sửa `preprocessing.py`, `tests/test_preprocessing.py`, file này. Chữ ký `load_text_dataset(name='imdb', n_train=2000, n_val=500, n_test=500, seed=42)` và `clean_text(text)`, khớp `project_description.md`; có thêm hàm phụ `label_counts`. Cấm lấy val từ test, cấm chọn trước khi xáo, cấm hạ chữ thường hay bỏ dấu câu khi chưa hỏi nhóm trưởng, cấm commit data, cấm tải data trong unit test.
- **Task hiện tại:** chỉ nhóm trưởng ghi ở mục "Current task".
- **Đề:** nạp dataset văn bản nhị phân và tiền xử lý. IMDB (`stanfordnlp/imdb`) là lựa chọn của nhóm; danh sách link do nhóm khác tổng hợp, không phải của cô.
- **Cách hoạt động:** cache trong `raw_data_dir("lab3")`; xáo theo `seed` **trước khi** chọn vì dữ liệu gốc xếp theo nhãn; train và val lấy từ train gốc, test từ test gốc, không trùng nhau; `clean_text` chỉ bỏ thẻ HTML và khoảng trắng thừa.
- **Kiểm chứng:** số đã ghi 2000/500/500, nhãn {0: 1000, 1: 1000} ở train, không còn `<br` (chỉ có trong báo cáo).
- **Câu hỏi mở:** IMDB và kích thước tập con chưa được cô xác nhận; README ghi "đặt dataset vào raw_data" nhưng code tự tải vào đó.
