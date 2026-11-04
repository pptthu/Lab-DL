# Lab 3 / eda: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/eda/eda.py` |
| Owner | Member A (Muội), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only) |
| Files you may edit | `labs/lab3/eda/eda.py`, `labs/lab3/tests/test_eda.py`, this guide |
| Files you must not edit | other modules, raw data, saved figures of real runs, other owners' notebook cells |
| Agreed signatures | `run_eda(dataset, out_dir: str, text_col: str = 'text', label_col: str = 'label') -> dict` (same as `project_description.md`) |
| Depends on / used by | output of `preprocessing.load_text_dataset` (train subset) / notebook and README EDA section |
| Prohibited | modifying the dataset; running EDA on validation or test to make decisions; writing figures to a real folder from tests (use `tmp_path`) |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_eda.py -q`. Smoke: `run_eda` on the real train subset into a temporary folder |
| Done when | unit tests pass; smoke output recorded in `progress_status.md`; this guide updated |

## Current task (edited ONLY by the team leader)

- None.

Module owners and agents never assign work to themselves here; they ask the team leader.

## What the brief requires

- The Practice 3 brief does not list an EDA step (not yet confirmed as required). Team choice: a read-only look at the train subset to justify `max_length` and check label balance.

## How the module works (current code)

- Reads `text_col` and `label_col`, counts labels, measures length in **words** (whitespace split).
- Returns `n_samples`, `label_counts`, `length_stats` (mean, std, min, median, p95, max).
- Saves `label_distribution.png`, `length_distribution.png` (median and p95 lines), and an extra `length_by_label.png` into `out_dir`. Matplotlib uses the `Agg` backend (no window).

## Inputs, outputs and pitfalls

- Length is in words, not tokens: the tokenizer produces more tokens than words, so p95 in words is only a guide for `max_length`.
- Raises `ValueError` on an empty dataset.
- Use only the train subset for decisions.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_eda.py -q` | all pass |
| Smoke test | `python -c "import tempfile; from labs.lab3.preprocessing.preprocessing import load_text_dataset; from labs.lab3.eda.eda import run_eda; tr,_,_=load_text_dataset(); print(run_eda(tr, tempfile.mkdtemp()))"` | `n_samples` 2000, two labels, 3 PNG files (needs the dataset download the first time) |
| Real integration run | only at an approved milestone | stats recorded in `progress_status.md` |

Recorded (Reported only): n=2000, 3 PNG saved.

## Open questions / limitations

- Whether the lecturer expects an EDA section is not confirmed.

---

**Tiếng Việt**

- **Dành cho AI:** người phụ trách Member A (Muội), do nhóm trưởng giao. Được sửa `eda.py`, `tests/test_eda.py`, file này; cấm sửa module khác, data, hình của lần chạy thật. Chữ ký `run_eda(dataset, out_dir, text_col='text', label_col='label') -> dict`, khớp `project_description.md`. Cấm sửa dataset, cấm dùng val/test để ra quyết định, test phải ghi hình vào `tmp_path`.
- **Task hiện tại:** chỉ nhóm trưởng ghi ở mục "Current task".
- **Đề:** slide Practice 3 không nêu bước EDA (chưa xác nhận là bắt buộc); nhóm chọn làm để chọn `max_length` và kiểm tra cân bằng nhãn.
- **Cách hoạt động:** đếm nhãn, đo độ dài theo **số từ**, trả về thống kê, lưu 3 hình.
- **Lưu ý:** số từ khác số token; dataset rỗng báo `ValueError`; chỉ dùng tập train.
- **Kiểm chứng:** unit test như bảng trên; smoke test lưu hình vào thư mục tạm. Đã ghi: n=2000, 3 hình (chỉ có trong báo cáo).
