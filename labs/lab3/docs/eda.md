# Lab 3 / eda: guide for AI agents

## Purpose

- Look at the **training data before modelling** (read-only; the data is never changed).
- Answer two simple questions: are the two labels balanced, and how long are the reviews?
- The answers help choose `max_length` for tokenization (see `features.md`).

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/eda/eda.py` |
| Owner | Member A (Muội), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab3/eda/eda.py`, `labs/lab3/tests/test_eda.py`, this guide |
| Files you must not edit | other modules, raw data, saved figures of real runs, other owners' notebook cells |
| Agreed signatures | `run_eda(dataset, out_dir: str, text_col: str = 'text', label_col: str = 'label') -> dict` (same as `project_description.md`) |
| Depends on / used by | output of `preprocessing.load_text_dataset` (train subset) / notebook and README EDA section |
| Prohibited | modifying the dataset; running EDA on validation or test to make decisions; writing figures to a real folder from tests (use `tmp_path`) |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_eda.py -q`. Smoke: `run_eda` on the real train subset into a temporary folder |
| Done when | unit tests pass; smoke output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Not yet confirmed: the Practice 3 brief does not list an EDA step.
- Team implementation choice: a read-only look at the train subset to justify `max_length` and check label balance.

## How the module works (current code)

- Reads `text_col` and `label_col`; counts samples per label.
- Measures length in **words** (split on whitespace), not in tokens.
- Computes `length_stats`: mean, std, min, median, p95, max.
- Saves three figures into `out_dir`: `label_distribution.png`, `length_distribution.png` (median and p95 lines), `length_by_label.png` (extra box plot).
- Matplotlib uses the `Agg` backend, so no window opens.

## Inputs, outputs and pitfalls

- Input:
  - `dataset`: the **train** subset with a text column and a label column.
  - `out_dir`: the folder for the PNG files, created if missing.
- Output:
  - a dict with `n_samples`, `label_counts`, `length_stats`.
  - 3 PNG files in `out_dir`.
- Where figures go (team implementation choice, from `progress_status.md`):
  - real runs: `labs/lab3/artifacts/figures/`, created with `artifact_dir("lab3", "figures")` in `main.py` (intended not committed).
  - tests and smoke tests: a temporary folder only.
- Pitfalls:
  - words are fewer than tokens, so p95 in words is only a guide for `max_length`.
  - an empty dataset raises `ValueError`.
  - use only the train subset for decisions.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_eda.py -q` | all pass |
| Smoke test | `python -c "import tempfile; from labs.lab3.preprocessing.preprocessing import load_text_dataset; from labs.lab3.eda.eda import run_eda; tr,_,_=load_text_dataset(); print(run_eda(tr, tempfile.mkdtemp()))"` | `n_samples` 2000, two labels, 3 PNG files (the first call downloads the dataset) |
| Real integration run | only at an approved milestone | stats recorded in `progress_status.md` |

- Recorded (Reported only): n=2000, 3 PNG saved.

## Open questions / limitations

- Whether the lecturer expects an EDA section is not confirmed.

---

**Tiếng Việt**

## Mục đích

- Xem qua **dữ liệu train trước khi làm model** (chỉ đọc, không sửa dữ liệu).
- Trả lời hai câu hỏi đơn giản: hai nhãn có cân bằng không, review dài khoảng bao nhiêu?
- Kết quả giúp chọn `max_length` khi tokenize (xem `features.md`).

## Dành cho AI (đọc trước)

- Người phụ trách: Member A (Muội), do nhóm trưởng giao. Trạng thái: xong (chỉ có trong báo cáo).
- Việc được giao: xem GitHub Issue và `progress_status.md`; không tự giao việc cho mình, hỏi nhóm trưởng.
- Được sửa: `eda.py`, `tests/test_eda.py`, file này.
- Cấm sửa: module khác, data gốc, hình của lần chạy thật, ô notebook của người khác.
- Chữ ký: `run_eda(dataset, out_dir, text_col='text', label_col='label') -> dict`, khớp `project_description.md`.
- Cấm: sửa dataset; dùng validation/test để ra quyết định; để test ghi hình vào thư mục thật (dùng `tmp_path`).
- Kiểm tra: unit test `pytest labs/lab3/tests/test_eda.py`; smoke test chạy trên tập train thật, lưu hình vào thư mục tạm.
- Xong khi: unit test qua, kết quả smoke ghi vào `progress_status.md`, đã cập nhật file này.

## Đề yêu cầu gì

- Chưa xác nhận: đề Practice 3 không nêu bước EDA.
- Lựa chọn của nhóm: xem tập train (chỉ đọc) để chọn `max_length` và kiểm tra cân bằng nhãn.

## Cách hoạt động

- Đọc cột text và cột nhãn, đếm số mẫu mỗi nhãn.
- Đo độ dài theo **số từ** (tách theo khoảng trắng), không phải số token.
- Tính `length_stats`: mean, std, min, median, p95, max.
- Lưu 3 hình vào `out_dir`: `label_distribution.png`, `length_distribution.png` (có đường median và p95), `length_by_label.png` (boxplot thêm).
- Dùng backend `Agg` của matplotlib nên không mở cửa sổ.

## Đầu vào, đầu ra và lỗi hay gặp

- Đầu vào:
  - `dataset`: tập **train** có cột text và cột nhãn.
  - `out_dir`: thư mục lưu PNG, tự tạo nếu chưa có.
- Đầu ra:
  - dict gồm `n_samples`, `label_counts`, `length_stats`.
  - 3 file PNG trong `out_dir`.
- Hình lưu ở đâu (lựa chọn của nhóm, theo `progress_status.md`):
  - chạy thật: `labs/lab3/artifacts/figures/`, tạo bằng `artifact_dir("lab3", "figures")` trong `main.py` (ý định là không commit).
  - test và smoke test: chỉ thư mục tạm.
- Lưu ý:
  - số từ ít hơn số token nên p95 theo từ chỉ là gợi ý cho `max_length`.
  - dataset rỗng báo `ValueError`.
  - chỉ dùng tập train để ra quyết định.

## Cách kiểm chứng

- Unit test: `python -m pytest labs/lab3/tests/test_eda.py -q`, kỳ vọng tất cả qua.
- Smoke test: lệnh trong bảng tiếng Anh ở trên, kỳ vọng `n_samples` 2000, hai nhãn, 3 file PNG (lần đầu cần mạng để tải dataset).
- Chạy tích hợp thật: chỉ ở mốc nhóm đã duyệt.
- Đã ghi (chỉ có trong báo cáo): n=2000, lưu 3 hình.

## Câu hỏi mở

- Chưa xác nhận cô có yêu cầu phần EDA hay không.
