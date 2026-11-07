# PREPROCESSING_STATUS — Lab 1 / preprocessing (progress)

- **Motivation/Background**: The spec says what `preprocessing` must do; this file says what is done, with evidence, so nobody has to re-read the spec or re-run the code to know the state.
- **Purpose**: Track the status of `preprocessing` and `download_official.py`: checklist with evidence (command, output path, commit), dated log, versions, workflow steps, blockers, decisions, next step.
- **Overview Pipeline**: update after every meaningful change -> mark a task `[x]` only with evidence from a real run -> team-wide table stays in `labs/lab1/progress_status.md`.
- **Detailed Plan**: Header; Checklist; Log; Version history; Workflow status; Blockers; Decisions; Open questions; Next step; Links; Vietnamese version.
- **References**: [preprocessing.md](preprocessing.md), [AUDIT.md](AUDIT.md), [PROMPTS.md](PROMPTS.md), `labs/lab1/progress_status.md`.
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
- [Open questions](#open-questions)
- [Next step](#next-step)
- [Links](#links)
- [Tiếng Việt](#tiếng-việt)

---

## Header

- **Title:** Data Source, Preprocessing & Split
- **Owner:** Member A (Toàn)
- **Description:** Tracks the official data download, `load_csv`, the stratified split and the shared split file.
- **Status:** Done (code) · doc v3.0 waiting for approval · 1 open question
- **Phase doc:** [preprocessing.md](preprocessing.md)

## Checklist

| Task | Done | Evidence (command / output / commit) |
|---|---|---|
| `load_csv`, `make_split`, `save_split`, `load_split` | [x] | commit `c218ab1` (2026-10-13) |
| Shared split file committed | [x] | `labs/lab1/artifacts/splits/train_val_split.json`, commit `b58869f` (2026-10-13) |
| `download_official.py` (official data, MD5) and split recreated | [x] | commit `2709e70` (2026-11-02) |
| Unit tests (6) | [x] | `python -m pytest labs/lab1/tests/test_preprocessing.py -q` -> pass (2026-11-07) |
| CSV MD5 | [x] | `md5sum` (2026-11-07): train `8ff2cced1c624159b21dac4f735b4215`, test `78e18be2098f1256a07ce6f91a4d436e` |
| Split reproducible | [x] | `make_split(labels, 0.1, 42)` vs `load_split(...)` -> identical, 54,000 / 6,000, 600 per class (2026-11-07) |
| Line-by-line comments in both `.py` files | [x] | 2026-11-07; AST unchanged, `pytest labs/lab1/tests` -> 52 passed |
| JSON stores `seed` and `val_ratio` | [ ] | open question (team leader) |
| Doc v3.0 approved by the team leader | [ ] | pending |

## Log

- 2026-10-13: 4 functions + tests written, pytest passes; shared split created (54,000 / 6,000).
- 2026-11-02: `download_official.py` added (official data, MD5 check); Kaggle CSV renamed `*_kaggle.csv`; split recreated; 50 passed.
- 2026-11-04 (Run): train 60,000 rows (6,000 per class), test 10,000 rows (1,000 per class); MD5 as above.
- 2026-11-04 (Verified): regenerated split identical to the committed JSON.
- 2026-11-07: doc rewritten (spec / STATUS / PROMPTS / AUDIT); line-by-line comments added; audit re-run, all checks identical.

## Version history

| Version | Date | Author | Change | Approved |
|---|---|---|---|---|
| v1.0 | 2026-10-13 | Toàn | 4 functions + tests, shared split (Kaggle CSV) | team leader (PR) |
| v2.0 | 2026-11-02 | Toàn | `download_official.py`, official data, split recreated | team leader (PR) |
| v3.0 | 2026-11-07 | Toàn | doc rewritten with the new template; split into spec/STATUS/PROMPTS/AUDIT; line-by-line comments in the `.py` files | pending |

## Workflow status

| Step | Status |
|---|---|
| 1. Description (outline, [PROMPTS.md](PROMPTS.md)) | draft by the AI agent; Toàn to review |
| 2. Discussion | v3.0: pending (open question below) |
| 3. Detail description ([preprocessing.md](preprocessing.md)) | done (v3.0) |
| 4. Approve | v3.0: pending |
| 5. Implementation | done (v1.0, v2.0) |
| 6. Smoke run | done 2026-11-07 |
| 7. Normal run | done 2026-11-04 |
| 8. Audit ([AUDIT.md](AUDIT.md)) | done 2026-11-07 |

## Blockers

- None for the code. The open question below needs the team leader.

## Decisions

- Official Zalando data instead of the Kaggle CSV — the Kaggle test file did not follow the official split.
- Stratified, `val_ratio=0.1`, `seed=42` — validation as balanced as train.
- Single split policy: JSON committed and read-only — every experiment compares on the same split.

## Open questions

- The JSON stores only `train_idx`, `val_idx`; the reference submission also stores `seed` and `val_ratio`. Changing the format touches `save_split`, `load_split`, the tests and the committed file -> team leader decision.

## Next step

- Ask the team leader about the open question; Toàn reviews the outline in [PROMPTS.md](PROMPTS.md).

## Links

- Spec: [preprocessing.md](preprocessing.md) · Audit: [AUDIT.md](AUDIT.md) · Prompts: [PROMPTS.md](PROMPTS.md)
- Previous / next module: [../eda/STATUS.md](../eda/STATUS.md) / [../features/STATUS.md](../features/STATUS.md)

---

## Tiếng Việt

### Header

- **Tên:** Nguồn dữ liệu, tiền xử lý và chia dữ liệu
- **Người phụ trách:** Member A (Toàn)
- **Mô tả:** theo dõi việc tải dữ liệu chính thức, `load_csv`, split phân tầng và file split dùng chung.
- **Trạng thái:** Xong (code) · doc v3.0 chờ duyệt · 1 câu hỏi mở
- **File spec:** [preprocessing.md](preprocessing.md)

### Checklist

| Việc | Xong | Bằng chứng (lệnh / output / commit) |
|---|---|---|
| `load_csv`, `make_split`, `save_split`, `load_split` | [x] | commit `c218ab1` (13/10/2026) |
| Commit file split dùng chung | [x] | `labs/lab1/artifacts/splits/train_val_split.json`, commit `b58869f` (13/10/2026) |
| `download_official.py` (dữ liệu chính thức, MD5) và tạo lại split | [x] | commit `2709e70` (02/11/2026) |
| Unit test (6) | [x] | `python -m pytest labs/lab1/tests/test_preprocessing.py -q` -> đạt (07/11/2026) |
| MD5 của CSV | [x] | `md5sum` (07/11/2026): train `8ff2cced1c624159b21dac4f735b4215`, test `78e18be2098f1256a07ce6f91a4d436e` |
| Split tái lập được | [x] | so `make_split(labels, 0.1, 42)` với `load_split(...)` -> giống hệt, 54.000 / 6.000, 600 mỗi lớp (07/11/2026) |
| Comment từng dòng trong 2 file `.py` | [x] | 07/11/2026; AST không đổi, `pytest labs/lab1/tests` -> 52 test đạt |
| JSON lưu `seed` và `val_ratio` | [ ] | câu hỏi mở (nhóm trưởng) |
| Nhóm trưởng duyệt doc v3.0 | [ ] | chờ duyệt |

### Nhật ký

- 13/10/2026: viết 4 hàm + test, pytest đạt; tạo split dùng chung (54.000 / 6.000).
- 02/11/2026: thêm `download_official.py` (dữ liệu chính thức, kiểm MD5); đổi tên CSV Kaggle thành `*_kaggle.csv`; tạo lại split; 50 test đạt.
- 04/11/2026 (đã chạy): train 60.000 dòng (6.000 mỗi lớp), test 10.000 dòng (1.000 mỗi lớp); MD5 như trên.
- 04/11/2026 (đã kiểm): split tạo lại giống hệt JSON đã commit.
- 07/11/2026: viết lại doc (spec / STATUS / PROMPTS / AUDIT); thêm comment từng dòng; chạy lại audit, mọi số giống hệt.

### Lịch sử phiên bản

| Phiên bản | Ngày | Người viết | Thay đổi | Duyệt |
|---|---|---|---|---|
| v1.0 | 13/10/2026 | Toàn | 4 hàm + test, split dùng chung (CSV Kaggle) | nhóm trưởng (PR) |
| v2.0 | 02/11/2026 | Toàn | `download_official.py`, dữ liệu chính thức, tạo lại split | nhóm trưởng (PR) |
| v3.0 | 07/11/2026 | Toàn | viết lại doc theo mẫu mới; tách spec/STATUS/PROMPTS/AUDIT; comment từng dòng trong file `.py` | chờ duyệt |

### Trạng thái quy trình

| Bước | Trạng thái |
|---|---|
| 1. Mô tả (dàn ý, [PROMPTS.md](PROMPTS.md)) | bản nháp do AI soạn; Toàn cần đọc lại |
| 2. Thảo luận | v3.0: đang chờ (câu hỏi mở bên dưới) |
| 3. Mô tả chi tiết ([preprocessing.md](preprocessing.md)) | xong (v3.0) |
| 4. Duyệt | v3.0: chờ duyệt |
| 5. Thực thi | xong (v1.0, v2.0) |
| 6. Smoke run | xong 07/11/2026 |
| 7. Normal run | xong 04/11/2026 |
| 8. Audit ([AUDIT.md](AUDIT.md)) | xong 07/11/2026 |

### Vướng mắc

- Code không vướng. Câu hỏi mở bên dưới cần nhóm trưởng.

### Quyết định

- Dùng dữ liệu chính thức Zalando thay CSV Kaggle — file test Kaggle không theo split chính thức.
- Phân tầng, `val_ratio=0.1`, `seed=42` — validation cân bằng như train.
- Single split policy: JSON có commit, chỉ đọc — mọi thí nghiệm so trên cùng một split.

### Câu hỏi mở

- JSON chỉ lưu `train_idx`, `val_idx`; bài tham khảo lưu thêm `seed` và `val_ratio`. Đổi format sẽ chạm `save_split`, `load_split`, test và file đã commit -> nhóm trưởng quyết.

### Bước tiếp theo

- Hỏi nhóm trưởng câu hỏi mở; Toàn đọc lại dàn ý trong [PROMPTS.md](PROMPTS.md).

### Liên kết

- Spec: [preprocessing.md](preprocessing.md) · Audit: [AUDIT.md](AUDIT.md) · Prompt: [PROMPTS.md](PROMPTS.md)
- Module trước / sau: [../eda/STATUS.md](../eda/STATUS.md) / [../features/STATUS.md](../features/STATUS.md)
