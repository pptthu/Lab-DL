# EDA_STATUS — Lab 1 / eda (progress)

- **Motivation/Background**: The spec says what `eda` must do; this file says what is done, with evidence, so nobody has to re-read the spec or re-run the code to know the state.
- **Purpose**: Track the status of `eda`: checklist with evidence (command, output path, commit), dated log, versions, workflow steps, blockers, decisions, next step.
- **Overview Pipeline**: update after every meaningful change -> mark a task `[x]` only with evidence from a real run -> team-wide table stays in `labs/lab1/progress_status.md`.
- **Detailed Plan**: Header; Checklist; Log; Version history; Workflow status; Blockers; Decisions; Next step; Links; Vietnamese version.
- **References**: [eda.md](eda.md), [AUDIT.md](AUDIT.md), [PROMPTS.md](PROMPTS.md), `labs/lab1/progress_status.md`.
- **Created**: 2026-11-07T00:00:00+07:00
- **Last Updated**: 2026-11-07T00:00:00+07:00

---

## Table of Contents

- [Header](#header)
- [Checklist](#checklist)
- [Log](#log)
- [Version history](#version-history)
- [Workflow status](#workflow-status)
- [Blockers](#blockers)
- [Decisions](#decisions)
- [Next step](#next-step)
- [Links](#links)
- [Tiếng Việt](#tiếng-việt)

---

## Header

- **Title:** Exploratory Data Analysis
- **Owner:** Member A (Toàn)
- **Description:** Tracks `run_eda`, its tests and the EDA numbers on the official data.
- **Status:** Done (code) · doc v3.0 waiting for approval
- **Phase doc:** [eda.md](eda.md)

## Checklist

| Task | Done | Evidence (command / output / commit) |
|---|---|---|
| `run_eda` with column check, 6 statistics, 2 figures | [x] | commit `4bcdd45` (2026-10-13) |
| Unit tests (3) | [x] | `python -m pytest labs/lab1/tests/test_eda.py -q` -> pass (2026-11-07) |
| Run on the official data | [x] | `python -m labs.lab1.main` stage 1 -> "EDA completed: 60000 rows, 10 classes, 0 duplicates." (2026-11-04); re-run 2026-11-07 identical |
| Figures saved | [x] | `labs/lab1/artifacts/figures/class_counts.png`, `sample_grid.png` (not committed) |
| Docstring names the official source | [x] | commit `202f702` (2026-11-02) |
| Line-by-line comments in `eda.py` | [x] | 2026-11-07; AST unchanged, `pytest labs/lab1/tests` -> 52 passed |
| Doc v3.0 approved by the team leader | [ ] | pending |

## Log

- 2026-10-13: `run_eda` + tests written, pytest passes; run on the old Kaggle CSV -> 43 duplicate rows.
- 2026-11-02: switched to the official data -> 0 duplicate rows; identical images across splits train-val 0, train-test 0, val-test 0 (Kaggle CSV: 9, 9, 1).
- 2026-11-04 (Run): `n_rows` 60000, 6000 per class, pixels 0-255, `n_missing` 0, `n_duplicates` 0.
- 2026-11-07: doc rewritten (spec / STATUS / PROMPTS / AUDIT); line-by-line comments added to `eda.py`; audit done.

## Version history

| Version | Date | Author | Change | Approved |
|---|---|---|---|---|
| v1.0 | 2026-10-13 | Toàn | `run_eda` + tests (Kaggle CSV) | team leader (PR) |
| v2.0 | 2026-11-02 | Toàn | official Zalando data; docstring updated | team leader (PR) |
| v3.0 | 2026-11-07 | Toàn | doc rewritten with the new template; split into spec/STATUS/PROMPTS/AUDIT; line-by-line comments in `eda.py` | pending |

## Workflow status

| Step | Status |
|---|---|
| 1. Description (outline, [PROMPTS.md](PROMPTS.md)) | draft by the AI agent; Toàn to review |
| 2. Discussion | v3.0: pending |
| 3. Detail description ([eda.md](eda.md)) | done (v3.0) |
| 4. Approve | v3.0: pending |
| 5. Implementation | done (v1.0, v2.0) |
| 6. Smoke run | done 2026-11-07 |
| 7. Normal run | done 2026-11-04 |
| 8. Audit ([AUDIT.md](AUDIT.md)) | done 2026-11-07 |

## Blockers

- None.

## Decisions

- Duplicate rows are counted, never deleted — keep the data as released.
- No EDA on the test file — the test set is kept for the final measurement.

## Next step

- Toàn reviews the outline in [PROMPTS.md](PROMPTS.md), then asks the team leader to approve v3.0.

## Links

- Spec: [eda.md](eda.md) · Audit: [AUDIT.md](AUDIT.md) · Prompts: [PROMPTS.md](PROMPTS.md)
- Next module: [../preprocessing/STATUS.md](../preprocessing/STATUS.md)

---

## Tiếng Việt

### Header

- **Tên:** Khám phá dữ liệu (EDA)
- **Người phụ trách:** Member A (Toàn)
- **Mô tả:** theo dõi `run_eda`, test và các số EDA trên dữ liệu chính thức.
- **Trạng thái:** Xong (code) · doc v3.0 chờ duyệt
- **File spec:** [eda.md](eda.md)

### Checklist

| Việc | Xong | Bằng chứng (lệnh / output / commit) |
|---|---|---|
| `run_eda` gồm kiểm cột, 6 thống kê, 2 hình | [x] | commit `4bcdd45` (13/10/2026) |
| Unit test (3) | [x] | `python -m pytest labs/lab1/tests/test_eda.py -q` -> đạt (07/11/2026) |
| Chạy trên dữ liệu chính thức | [x] | `python -m labs.lab1.main` stage 1 -> "EDA completed: 60000 rows, 10 classes, 0 duplicates." (04/11/2026); chạy lại 07/11/2026 giống hệt |
| Lưu hình | [x] | `labs/lab1/artifacts/figures/class_counts.png`, `sample_grid.png` (không commit) |
| Docstring ghi nguồn chính thức | [x] | commit `202f702` (02/11/2026) |
| Comment từng dòng trong `eda.py` | [x] | 07/11/2026; AST không đổi, `pytest labs/lab1/tests` -> 52 test đạt |
| Nhóm trưởng duyệt doc v3.0 | [ ] | chờ duyệt |

### Nhật ký

- 13/10/2026: viết `run_eda` + test, pytest đạt; chạy trên CSV Kaggle cũ -> 43 dòng trùng.
- 02/11/2026: đổi sang dữ liệu chính thức -> 0 dòng trùng; ảnh giống hệt giữa các tập train-val 0, train-test 0, val-test 0 (CSV Kaggle: 9, 9, 1).
- 04/11/2026 (đã chạy): `n_rows` 60000, 6000 mỗi lớp, pixel 0–255, `n_missing` 0, `n_duplicates` 0.
- 07/11/2026: viết lại doc (spec / STATUS / PROMPTS / AUDIT); thêm comment từng dòng vào `eda.py`; audit xong.

### Lịch sử phiên bản

| Phiên bản | Ngày | Người viết | Thay đổi | Duyệt |
|---|---|---|---|---|
| v1.0 | 13/10/2026 | Toàn | `run_eda` + test (CSV Kaggle) | nhóm trưởng (PR) |
| v2.0 | 02/11/2026 | Toàn | dữ liệu chính thức Zalando; sửa docstring | nhóm trưởng (PR) |
| v3.0 | 07/11/2026 | Toàn | viết lại doc theo mẫu mới; tách spec/STATUS/PROMPTS/AUDIT; comment từng dòng trong `eda.py` | chờ duyệt |

### Trạng thái quy trình

| Bước | Trạng thái |
|---|---|
| 1. Mô tả (dàn ý, [PROMPTS.md](PROMPTS.md)) | bản nháp do AI soạn; Toàn cần đọc lại |
| 2. Thảo luận | v3.0: đang chờ |
| 3. Mô tả chi tiết ([eda.md](eda.md)) | xong (v3.0) |
| 4. Duyệt | v3.0: chờ duyệt |
| 5. Thực thi | xong (v1.0, v2.0) |
| 6. Smoke run | xong 07/11/2026 |
| 7. Normal run | xong 04/11/2026 |
| 8. Audit ([AUDIT.md](AUDIT.md)) | xong 07/11/2026 |

### Vướng mắc

- Không có.

### Quyết định

- Dòng trùng chỉ đếm, không xóa — giữ dữ liệu như bản phát hành.
- Không EDA trên file test — tập test để dành cho lần đo cuối.

### Bước tiếp theo

- Toàn đọc lại dàn ý trong [PROMPTS.md](PROMPTS.md), rồi nhờ nhóm trưởng duyệt v3.0.

### Liên kết

- Spec: [eda.md](eda.md) · Audit: [AUDIT.md](AUDIT.md) · Prompt: [PROMPTS.md](PROMPTS.md)
- Module tiếp theo: [../preprocessing/STATUS.md](../preprocessing/STATUS.md)
