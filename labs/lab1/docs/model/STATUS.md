# MODEL_STATUS — Lab 1 / model (progress)

- **Motivation/Background**: The spec says what the model must be; this file says what is done, with evidence, so nobody has to re-read the spec or re-run the code to know the state.
- **Purpose**: Track the status of `model`: checklist with evidence (command, output path, commit), dated log, versions, workflow steps, blockers, decisions, open questions for Member B, next step.
- **Overview Pipeline**: update after every meaningful change -> mark a task `[x]` only with evidence from a real run -> team-wide table stays in `labs/lab1/progress_status.md`.
- **Detailed Plan**: Header; Checklist; Log; Version history; Workflow status; Blockers; Decisions; Open questions; Next step; Links; Vietnamese version.
- **References**: [model.md](model.md), [AUDIT.md](AUDIT.md), [PROMPTS.md](PROMPTS.md), `labs/lab1/progress_status.md`, [../train.md](../train.md).
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

- **Title:** Initial Model (MLP baseline)
- **Owner:** Member A (Toàn); training by Member B (Thư)
- **Description:** Tracks `MLP`, `count_parameters`, the hand-off to `train` and the trained checkpoint.
- **Status:** Done (code, trained 2026-11-04) · doc v3.0 waiting for approval · lecturer's training-log and stopping rules not met (Member B)
- **Phase doc:** [model.md](model.md)

## Checklist

| Task | Done | Evidence (command / output / commit) |
|---|---|---|
| `MLP` (logits, no softmax) and `count_parameters` | [x] | commit `1c4a7c7` (2026-10-13) |
| Unit tests (6, incl. overfit one batch) | [x] | `python -m pytest labs/lab1/tests/test_model.py -q` -> pass (2026-11-07) |
| Parameter count | [x] | `count_parameters(MLP())` -> 235146; `MLP()(torch.zeros(4,1,28,28)).shape` -> `[4, 10]` (2026-11-07) |
| Trained by Member B | [x] | `python -m labs.lab1.main` (2026-11-04): best epoch 5, val loss 0.3018, val acc 89.05%, test 88.36%; `labs/lab1/artifacts/models/mlp_baseline.pt` (not committed) |
| Line-by-line comments in `model.py` | [x] | 2026-11-07; AST unchanged, `pytest labs/lab1/tests` -> 52 passed |
| Date-time named training log (params, architecture, per-epoch train loss/acc) | [ ] | not in `train.py` (Member B) |
| Stopping rule (e.g. 10 epochs without improvement) | [ ] | not in `train.py` (Member B) |
| Doc v3.0 approved by the team leader | [ ] | pending |

## Log

- 2026-10-13: `MLP`, `count_parameters` + tests written, pytest passes; forward on a real batch.
- 2026-11-02: notebook cell `[5]` re-run on the official data, 0 errors.
- 2026-11-04 (Run, Member B): trained on the official data; second run gives the same `mlp_baseline.pt` (same MD5).
- 2026-11-07: doc rewritten with the lecturer's Model.md template (spec / STATUS / PROMPTS / AUDIT); line-by-line comments added; audit done.

## Version history

| Version | Date | Author | Change | Approved |
|---|---|---|---|---|
| v1.0 | 2026-10-13 | Toàn | `MLP`, `count_parameters` + tests | team leader (PR) |
| v2.0 | 2026-11-04 | Thư | trained on official data (`mlp_baseline.pt`); source unchanged | team leader |
| v3.0 | 2026-11-07 | Toàn | doc rewritten with the lecturer's Model.md template; split into spec/STATUS/PROMPTS/AUDIT; line-by-line comments in `model.py` | pending |

## Workflow status

| Step | Status |
|---|---|
| 1. Description (outline, [PROMPTS.md](PROMPTS.md)) | draft by the AI agent; Toàn to review |
| 2. Discussion | v3.0: pending (open questions below) |
| 3. Detail description ([model.md](model.md)) | done (v3.0) |
| 4. Approve | v3.0: pending |
| 5. Implementation | done (v1.0); training done by Member B (v2.0) |
| 6. Smoke run | done 2026-11-07 |
| 7. Normal run | done 2026-11-04 |
| 8. Audit ([AUDIT.md](AUDIT.md)) | done 2026-11-07 |

## Blockers

- The lecturer's training rules (log file, stopping rule, train accuracy, `training/` or `model_present/` folder) depend on Member B's `train.py` and `main.py`.

## Decisions

- Baseline `(256, 128)`, `dropout=0.0`, logits — `CrossEntropyLoss` needs logits.
- Seed set again right before building the model — notebook and `main.py` start from the same weights.
- MLP only, no CNN (team decision 2026-11-04).

## Open questions

- For Member B and the team leader: add a date-time named training log (parameters, architecture, per-epoch train loss and train accuracy), a stopping rule, and the `training/` or `model_present/` folder named by the new rules? Today: fixed 5 epochs, `history` has `train_loss`, `val_loss`, `val_acc` only, checkpoints in `artifacts/models/`.

## Next step

- Send the open question to Member B and the team leader; Toàn reviews the outline in [PROMPTS.md](PROMPTS.md).

## Links

- Spec: [model.md](model.md) · Audit: [AUDIT.md](AUDIT.md) · Prompts: [PROMPTS.md](PROMPTS.md)
- Previous module: [../features/STATUS.md](../features/STATUS.md) · Training (Member B): [../train.md](../train.md)

---

## Tiếng Việt

### Header

- **Tên:** Model ban đầu (MLP baseline)
- **Người phụ trách:** Member A (Toàn); train do Member B (Thư)
- **Mô tả:** theo dõi `MLP`, `count_parameters`, bàn giao cho `train` và checkpoint đã train.
- **Trạng thái:** Xong (code, đã train 04/11/2026) · doc v3.0 chờ duyệt · quy định về file log và điều kiện dừng của cô chưa đạt (phần của Thư)
- **File spec:** [model.md](model.md)

### Checklist

| Việc | Xong | Bằng chứng (lệnh / output / commit) |
|---|---|---|
| `MLP` (logits, không softmax) và `count_parameters` | [x] | commit `1c4a7c7` (13/10/2026) |
| Unit test (6, có overfit 1 batch) | [x] | `python -m pytest labs/lab1/tests/test_model.py -q` -> đạt (07/11/2026) |
| Số tham số | [x] | `count_parameters(MLP())` -> 235146; `MLP()(torch.zeros(4,1,28,28)).shape` -> `[4, 10]` (07/11/2026) |
| Thư đã train | [x] | `python -m labs.lab1.main` (04/11/2026): epoch tốt nhất 5, val loss 0,3018, val acc 89,05%, test 88,36%; `labs/lab1/artifacts/models/mlp_baseline.pt` (không commit) |
| Comment từng dòng trong `model.py` | [x] | 07/11/2026; AST không đổi, `pytest labs/lab1/tests` -> 52 test đạt |
| File log train đặt tên theo ngày giờ (tham số, kiến trúc, train loss/acc từng epoch) | [ ] | chưa có trong `train.py` (phần của Thư) |
| Điều kiện dừng (ví dụ 10 epoch không cải thiện) | [ ] | chưa có trong `train.py` (phần của Thư) |
| Nhóm trưởng duyệt doc v3.0 | [ ] | chờ duyệt |

### Nhật ký

- 13/10/2026: viết `MLP`, `count_parameters` + test, pytest đạt; forward trên batch thật.
- 02/11/2026: chạy lại ô `[5]` của notebook trên dữ liệu chính thức, 0 lỗi.
- 04/11/2026 (đã chạy, Thư): train trên dữ liệu chính thức; chạy lần hai ra đúng `mlp_baseline.pt` (cùng MD5).
- 07/11/2026: viết lại doc theo mẫu Model.md của cô (spec / STATUS / PROMPTS / AUDIT); thêm comment từng dòng; audit xong.

### Lịch sử phiên bản

| Phiên bản | Ngày | Người viết | Thay đổi | Duyệt |
|---|---|---|---|---|
| v1.0 | 13/10/2026 | Toàn | `MLP`, `count_parameters` + test | nhóm trưởng (PR) |
| v2.0 | 04/11/2026 | Thư | train trên dữ liệu chính thức (`mlp_baseline.pt`); source không đổi | nhóm trưởng |
| v3.0 | 07/11/2026 | Toàn | viết lại doc theo mẫu Model.md của cô; tách spec/STATUS/PROMPTS/AUDIT; comment từng dòng trong `model.py` | chờ duyệt |

### Trạng thái quy trình

| Bước | Trạng thái |
|---|---|
| 1. Mô tả (dàn ý, [PROMPTS.md](PROMPTS.md)) | bản nháp do AI soạn; Toàn cần đọc lại |
| 2. Thảo luận | v3.0: đang chờ (câu hỏi mở bên dưới) |
| 3. Mô tả chi tiết ([model.md](model.md)) | xong (v3.0) |
| 4. Duyệt | v3.0: chờ duyệt |
| 5. Thực thi | xong (v1.0); Thư đã train (v2.0) |
| 6. Smoke run | xong 07/11/2026 |
| 7. Normal run | xong 04/11/2026 |
| 8. Audit ([AUDIT.md](AUDIT.md)) | xong 07/11/2026 |

### Vướng mắc

- Các quy định train của cô (file log, điều kiện dừng, train accuracy, thư mục `training/` hoặc `model_present/`) phụ thuộc `train.py` và `main.py` của Thư.

### Quyết định

- Baseline `(256, 128)`, `dropout=0.0`, trả logits — `CrossEntropyLoss` cần logits.
- Đặt lại seed ngay trước khi tạo model — notebook và `main.py` cùng trọng số ban đầu.
- Chỉ làm MLP, không CNN (nhóm chốt 04/11/2026).

### Câu hỏi mở

- Cho Thư và nhóm trưởng: có thêm file log train đặt tên theo ngày giờ (tham số, kiến trúc, train loss và train accuracy từng epoch), điều kiện dừng, và thư mục `training/` hoặc `model_present/` theo quy định mới không? Hiện tại: cố định 5 epoch, `history` chỉ có `train_loss`, `val_loss`, `val_acc`, checkpoint ở `artifacts/models/`.

### Bước tiếp theo

- Gửi câu hỏi mở cho Thư và nhóm trưởng; Toàn đọc lại dàn ý trong [PROMPTS.md](PROMPTS.md).

### Liên kết

- Spec: [model.md](model.md) · Audit: [AUDIT.md](AUDIT.md) · Prompt: [PROMPTS.md](PROMPTS.md)
- Module trước: [../features/STATUS.md](../features/STATUS.md) · Train (Thư): [../train.md](../train.md)
