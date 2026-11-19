# Lab 3 / preprocessing: guide for AI agents

## Purpose

- Load the IMDB movie reviews and cut **small, reproducible** subsets (train / validation / test) that run on a CPU.
- Clean the review text a little (remove HTML such as `<br />`) without losing sentiment signals.

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/preprocessing/preprocessing.py` |
| Owner | Member A (Muội), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only, 2026-10-25) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab3/preprocessing/preprocessing.py`, `labs/lab3/tests/test_preprocessing.py`, this guide |
| Files you must not edit | other modules, the dataset cache in `labs/lab3/raw_data/`, other owners' notebook cells |
| Agreed signatures | `load_text_dataset(name: str = 'imdb', n_train: int = 2000, n_val: int = 500, n_test: int = 500, seed: int = 42) -> tuple`; `clean_text(text: str) -> str` (same as `project_description.md`). Extra helper in the code: `label_counts(dataset, label_col='label') -> dict` |
| Depends on / used by | `datasets`, `shared.paths.raw_data_dir` / `eda`, `features`, notebook |
| Prohibited | taking validation from the test split; selecting before shuffling; lowercasing or removing punctuation in `clean_text` without the team leader; committing the dataset; a download inside unit tests |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_preprocessing.py -q` (fake dataset, no network). Smoke: one real `load_text_dataset()` call |
| Done when | unit tests pass; smoke output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement (Exercise 2): load a simple binary text dataset and preprocess it for the model.
- Team implementation choice: IMDB from the Hugging Face Hub (`stanfordnlp/imdb`).
- Not yet confirmed: the dataset link list came from another group, not from the lecturer.
- Team implementation choice: small subsets (2000 / 500 / 500) for CPU.

## How the module works (current code)

- Loads the dataset with `load_dataset(HUB_IDS.get(name, name), cache_dir=...)`; the cache is `raw_data_dir("lab3")`, i.e. `labs/lab3/raw_data/` (never `./data/raw`).
- The Hub splits are ordered by label, so each pool is **shuffled with `seed` before selecting**.
- Train and validation both come from the original train split (`0..n_train`, then `n_train..n_train+n_val`); test comes from the original test split. The three subsets never overlap.
- Applies `clean_text` to every subset.
- Checks that labels are in {0, 1}; a missing `text` or `label` column raises `ValueError`.
- `clean_text`:
  - decodes HTML entities (`&amp;` becomes `&`).
  - replaces HTML tags such as `<br />` with a space.
  - collapses whitespace and strips the ends.
  - does **not** lowercase and does **not** remove punctuation or stop words (the uncased tokenizer lowercases by itself; "!" carries sentiment).

## Inputs, outputs and pitfalls

- Input: `name`, `n_train`, `n_val`, `n_test`, `seed`.
- Output: a tuple of three `datasets.Dataset` objects with columns `text` and `label` (0/1).
- Where data goes: the downloaded dataset is a cache in `labs/lab3/raw_data/`; it is not committed.
- Pitfalls:
  - `n_train + n_val` larger than the train split, or `n_test` larger than the test split, raises `ValueError`.
  - the first call needs internet; later calls use the cache.
  - selecting before shuffling would give a one-label subset, because the source is sorted by label.

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

## Mục đích

- Nạp các review phim IMDB và cắt các tập con **nhỏ, tái lập được** (train / validation / test) để chạy được trên CPU.
- Làm sạch văn bản một chút (bỏ HTML như `<br />`) mà không làm mất tín hiệu cảm xúc.

## Dành cho AI (đọc trước)

- Người phụ trách: Member A (Muội), do nhóm trưởng giao. Trạng thái: xong (chỉ có trong báo cáo, 25/10).
- Việc được giao: xem GitHub Issue và `progress_status.md`; không tự giao việc cho mình, hỏi nhóm trưởng.
- Được sửa: `preprocessing.py`, `tests/test_preprocessing.py`, file này.
- Cấm sửa: module khác, cache dataset trong `labs/lab3/raw_data/`, ô notebook của người khác.
- Chữ ký: `load_text_dataset(name='imdb', n_train=2000, n_val=500, n_test=500, seed=42) -> tuple` và `clean_text(text) -> str`, khớp `project_description.md`; có thêm hàm phụ `label_counts`.
- Cấm: lấy validation từ test; chọn mẫu trước khi xáo; hạ chữ thường hoặc bỏ dấu câu trong `clean_text` khi chưa hỏi nhóm trưởng; commit dataset; tải dữ liệu trong unit test.
- Kiểm tra: unit test `pytest labs/lab3/tests/test_preprocessing.py` (dataset giả, không cần mạng); smoke test gọi `load_text_dataset()` thật một lần.
- Xong khi: unit test qua, kết quả smoke ghi vào `progress_status.md`, đã cập nhật file này.

## Đề yêu cầu gì

- Đề (Bài 2): nạp một dataset văn bản nhị phân đơn giản và tiền xử lý cho model.
- Lựa chọn của nhóm: IMDB từ Hugging Face Hub (`stanfordnlp/imdb`).
- Chưa xác nhận: danh sách link dataset do một nhóm khác tổng hợp, không phải của cô.
- Lựa chọn của nhóm: tập con nhỏ (2000 / 500 / 500) để chạy trên CPU.

## Cách hoạt động

- Nạp bằng `load_dataset(...)`, cache ở `raw_data_dir("lab3")` tức `labs/lab3/raw_data/` (không dùng `./data/raw`).
- Dữ liệu gốc trên Hub xếp theo nhãn nên mỗi tập được **xáo theo `seed` trước khi chọn**.
- Train và validation cùng lấy từ tập train gốc (`0..n_train`, rồi `n_train..n_train+n_val`); test lấy từ tập test gốc. Ba tập không trùng nhau.
- Áp dụng `clean_text` cho mọi tập.
- Kiểm tra nhãn thuộc {0, 1}; thiếu cột `text` hoặc `label` thì báo `ValueError`.
- `clean_text`:
  - giải mã thực thể HTML (`&amp;` thành `&`).
  - thay thẻ HTML như `<br />` bằng khoảng trắng.
  - gộp khoảng trắng và cắt hai đầu.
  - **không** hạ chữ thường, **không** bỏ dấu câu hay stop word (tokenizer uncased tự hạ chữ thường; dấu "!" mang thông tin cảm xúc).

## Đầu vào, đầu ra và lỗi hay gặp

- Đầu vào: `name`, `n_train`, `n_val`, `n_test`, `seed`.
- Đầu ra: bộ ba `datasets.Dataset` có cột `text` và `label` (0/1).
- Dữ liệu nằm đâu: dataset tải về là cache trong `labs/lab3/raw_data/`; không commit.
- Lưu ý:
  - `n_train + n_val` lớn hơn tập train gốc, hoặc `n_test` lớn hơn tập test gốc, sẽ báo `ValueError`.
  - lần gọi đầu cần mạng; các lần sau dùng cache.
  - chọn trước khi xáo sẽ ra tập chỉ có một nhãn vì dữ liệu gốc xếp theo nhãn.

## Cách kiểm chứng

- Unit test: `python -m pytest labs/lab3/tests/test_preprocessing.py -q`, kỳ vọng tất cả qua, không cần mạng.
- Smoke test: lệnh trong bảng tiếng Anh ở trên. Đã ghi (chỉ có trong báo cáo): `2000 {0: 1000, 1: 1000} 500 {0: 243, 1: 257} 500 {0: 254, 1: 246} False`.
- Chạy tích hợp thật: chỉ ở mốc nhóm đã duyệt.

## Câu hỏi mở

- IMDB và kích thước tập con là lựa chọn của nhóm, cô chưa xác nhận.
- README ghi "đặt dataset vào `labs/lab3/raw_data/`"; trong code dataset tự tải về thư mục đó làm cache.
