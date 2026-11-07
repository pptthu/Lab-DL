# FEATURES_STATUS — Lab 1 / features (progress)

- **Motivation/Background**: The spec says what `features` must do; this file says what is done, with evidence, so nobody has to re-read the spec or re-run the code to know the state.
- **Purpose**: Track the status of `features`: checklist with evidence (command, output path, commit), dated log, versions, workflow steps, blockers, decisions, next step.
- **Overview Pipeline**: update after every meaningful change -> mark a task `[x]` only with evidence from a real run -> team-wide table stays in `labs/lab1/progress_status.md`.
- **Detailed Plan**: Header; Checklist; Log; Version history; Workflow status; Blockers; Decisions; Next step; Links; Vietnamese version.
- **References**: [features.md](features.md), [AUDIT.md](AUDIT.md), [PROMPTS.md](PROMPTS.md), `labs/lab1/progress_status.md`.
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

- **Title:** Normalization & DataLoaders
- **Owner:** Member A (Toàn)
- **Description:** Tracks train-only mean/std, the transform, `FashionDataset` and the 3 DataLoaders.
- **Status:** Done (code) · doc v3.0 waiting for approval
- **Phase doc:** [features.md](features.md)

## Checklist

| Task | Done | Evidence (command / output / commit) |
|---|---|---|
| `compute_mean_std`, `build_transform`, `FashionDataset`, `make_loaders` | [x] | commit `87ba5a3` (2026-10-13) |
| Unit tests (6) | [x] | `python -m pytest labs/lab1/tests/test_features.py -q` -> pass (2026-11-07) |
| Train-only statistics on the official data | [x] | `python -m labs.lab1.main` stage 3 -> "mean=0.2859, std=0.3530" (2026-11-04); re-run 2026-11-07 identical |
| Loaders | [x] | re-run 2026-11-07: batch `[64,1,28,28]` float32, labels `[64]` int64; 844 / 94 / 157 batches |
| No file written | [x] | by design (in memory only) |
| Line-by-line comments in `features.py` | [x] | 2026-11-07; AST unchanged, `pytest labs/lab1/tests` -> 52 passed |
| Doc v3.0 approved by the team leader | [ ] | pending |

## Log

- 2026-10-13: 4 functions/classes + tests written, pytest passes; real batch checked.
- 2026-11-02: notebook part A re-run on the official data -> mean 0.2859, std 0.3530 (Kaggle CSV: 0.2860 / 0.3527).
- 2026-11-04 (Run): `main.py` stage 3 prints mean 0.2859, std 0.3530.
- 2026-11-07: doc rewritten (spec / STATUS / PROMPTS / AUDIT); line-by-line comments added; audit re-run, all numbers identical.

## Version history

| Version | Date | Author | Change | Approved |
|---|---|---|---|---|
| v1.0 | 2026-10-13 | Toàn | 4 functions/classes + tests | team leader (PR) |
| v2.0 | 2026-11-02 | Toàn | re-run on official data (mean 0.2859, std 0.3530) | team leader (PR) |
| v3.0 | 2026-11-07 | Toàn | doc rewritten with the new template; split into spec/STATUS/PROMPTS/AUDIT; line-by-line comments in `features.py` | pending |

## Workflow status

| Step | Status |
|---|---|
| 1. Description (outline, [PROMPTS.md](PROMPTS.md)) | draft by the AI agent; Toàn to review |
| 2. Discussion | v3.0: pending |
| 3. Detail description ([features.md](features.md)) | done (v3.0) |
| 4. Approve | v3.0: pending |
| 5. Implementation | done (v1.0, v2.0) |
| 6. Smoke run | done 2026-11-07 |
| 7. Normal run | done 2026-11-04 |
| 8. Audit ([AUDIT.md](AUDIT.md)) | done 2026-11-07 |

## Blockers

- None.

## Decisions

- One global mean/std from the training split only — no leakage; grayscale = 1 channel.
- Batch 64, shuffle train only — validation/test keep their order so predictions stay aligned with images.
- No augmentation — the brief does not ask for it.

## Next step

- Toàn reviews the outline in [PROMPTS.md](PROMPTS.md), then asks the team leader to approve v3.0.

## Links

- Spec: [features.md](features.md) · Audit: [AUDIT.md](AUDIT.md) · Prompts: [PROMPTS.md](PROMPTS.md)
- Previous / next module: [../preprocessing/STATUS.md](../preprocessing/STATUS.md) / [../model/STATUS.md](../model/STATUS.md)

---

## Tiếng Việt

### Header

- **Tên:** Chuẩn hóa và DataLoader
- **Người phụ trách:** Member A (Toàn)
- **Mô tả:** theo dõi mean/std chỉ trên train, transform, `FashionDataset` và 3 DataLoader.
- **Trạng thái:** Xong (code) · doc v3.0 chờ duyệt
- **File spec:** [features.md](features.md)

### Checklist

| Việc | Xong | Bằng chứng (lệnh / output / commit) |
|---|---|---|
| `compute_mean_std`, `build_transform`, `FashionDataset`, `make_loaders` | [x] | commit `87ba5a3` (13/10/2026) |
| Unit test (6) | [x] | `python -m pytest labs/lab1/tests/test_features.py -q` -> đạt (07/11/2026) |
| Thống kê chỉ trên train, dữ liệu chính thức | [x] | `python -m labs.lab1.main` stage 3 -> "mean=0.2859, std=0.3530" (04/11/2026); chạy lại 07/11/2026 giống hệt |
| Loader | [x] | chạy lại 07/11/2026: batch `[64,1,28,28]` float32, nhãn `[64]` int64; 844 / 94 / 157 batch |
| Không ghi file | [x] | theo thiết kế (chỉ trong bộ nhớ) |
| Comment từng dòng trong `features.py` | [x] | 07/11/2026; AST không đổi, `pytest labs/lab1/tests` -> 52 test đạt |
| Nhóm trưởng duyệt doc v3.0 | [ ] | chờ duyệt |

### Nhật ký

- 13/10/2026: viết 4 hàm/lớp + test, pytest đạt; đã kiểm batch thật.
- 02/11/2026: chạy lại notebook phần A trên dữ liệu chính thức -> mean 0,2859, std 0,3530 (CSV Kaggle: 0,2860 / 0,3527).
- 04/11/2026 (đã chạy): `main.py` stage 3 in mean 0,2859, std 0,3530.
- 07/11/2026: viết lại doc (spec / STATUS / PROMPTS / AUDIT); thêm comment từng dòng; chạy lại audit, mọi số giống hệt.

### Lịch sử phiên bản

| Phiên bản | Ngày | Người viết | Thay đổi | Duyệt |
|---|---|---|---|---|
| v1.0 | 13/10/2026 | Toàn | 4 hàm/lớp + test | nhóm trưởng (PR) |
| v2.0 | 02/11/2026 | Toàn | chạy lại trên dữ liệu chính thức (mean 0,2859, std 0,3530) | nhóm trưởng (PR) |
| v3.0 | 07/11/2026 | Toàn | viết lại doc theo mẫu mới; tách spec/STATUS/PROMPTS/AUDIT; comment từng dòng trong `features.py` | chờ duyệt |

### Trạng thái quy trình

| Bước | Trạng thái |
|---|---|
| 1. Mô tả (dàn ý, [PROMPTS.md](PROMPTS.md)) | bản nháp do AI soạn; Toàn cần đọc lại |
| 2. Thảo luận | v3.0: đang chờ |
| 3. Mô tả chi tiết ([features.md](features.md)) | xong (v3.0) |
| 4. Duyệt | v3.0: chờ duyệt |
| 5. Thực thi | xong (v1.0, v2.0) |
| 6. Smoke run | xong 07/11/2026 |
| 7. Normal run | xong 04/11/2026 |
| 8. Audit ([AUDIT.md](AUDIT.md)) | xong 07/11/2026 |

### Vướng mắc

- Không có.

### Quyết định

- Một mean/std toàn cục, chỉ tính trên train — không rò rỉ; ảnh xám 1 kênh.
- Batch 64, chỉ xáo train — val/test giữ thứ tự để dự đoán khớp đúng ảnh.
- Không augmentation — đề không yêu cầu.

### Bước tiếp theo

- Toàn đọc lại dàn ý trong [PROMPTS.md](PROMPTS.md), rồi nhờ nhóm trưởng duyệt v3.0.

### Liên kết

- Spec: [features.md](features.md) · Audit: [AUDIT.md](AUDIT.md) · Prompt: [PROMPTS.md](PROMPTS.md)
- Module trước / sau: [../preprocessing/STATUS.md](../preprocessing/STATUS.md) / [../model/STATUS.md](../model/STATUS.md)
