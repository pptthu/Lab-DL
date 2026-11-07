# Lab 1 / model: audit (code vs spec)

- **Motivation/Background**: Step 8 of the workflow: the code is checked against the spec before the version is approved.
- **Purpose**: Record each claim of [model.md](../model/model.md), the evidence in the code or a run, and the result.
- **Overview Pipeline**: spec claim -> code / command -> OK or not done (open question).
- **Detailed Plan**: Audit table; Vietnamese version.
- **References**: [model.md](../model/model.md), `labs/lab1/tests/`.
- **Created**: 2026-11-07T00:00:00+07:00
- **Last Updated**: 2026-11-07T00:00:00+07:00

---

## Table of Contents

- [Audit (code vs description)](#audit-code-vs-description)
- [Tiếng Việt](#tiếng-việt)

---

## Audit (code vs description)

Checked on 2026-11-07 with `.venv`: `pytest` of the 4 data/model test files + `test_smoke.py` -> 30 passed; after adding the line-by-line comments to the `.py` files, `pytest labs/lab1/tests` -> 52 passed and the code AST is unchanged.

| Description says | Code / evidence | Result |
|---|---|---|
| logits `[B,10]`, no softmax | `MLP.__init__` (last `Linear`), `MLP.forward`; `test_output_is_logits_not_probabilities`; `MLP()(zeros(4,1,28,28))` -> `[4, 10]` (2026-11-07) | OK |
| 3 `Linear` layers, 784-256-128-10 | `MLP.__init__` | OK |
| 235,146 parameters | `count_parameters(MLP())` = 235146 (2026-11-07) | OK |
| overfit one batch | `test_can_overfit_one_batch` | OK |
| built from scratch, no pretrained weights | only `torch.nn` layers | OK |
| checkpoint path, training rule | `main.py`: `artifacts/models/mlp_baseline.pt`, 5 epochs, best val loss | OK (as described) |
| new rules: `training/` or `model_present/`, date-time log, train accuracy, stopping rule | not in `train.py` / `main.py` (Member B code) | Not done, open question |

---

## Tiếng Việt

### Audit (đối chiếu code với mô tả)

Kiểm ngày 07/11/2026 bằng `.venv`: `pytest` 4 file test data/model + `test_smoke.py` → 30 test đạt; sau khi thêm comment từng dòng vào file `.py`, `pytest labs/lab1/tests` → 52 test đạt và AST của code không đổi.

| Mô tả ghi | Code / bằng chứng | Kết quả |
|---|---|---|
| logits `[B,10]`, không softmax | `MLP.__init__` (lớp `Linear` cuối), `MLP.forward`; test `test_output_is_logits_not_probabilities`; `MLP()(zeros(4,1,28,28))` → `[4, 10]` (07/11/2026) | Đạt |
| 3 lớp `Linear`, 784-256-128-10 | `MLP.__init__` | Đạt |
| 235.146 tham số | `count_parameters(MLP())` = 235146 (07/11/2026) | Đạt |
| overfit 1 batch | test `test_can_overfit_one_batch` | Đạt |
| tự xây, không pretrained | chỉ dùng lớp của `torch.nn` | Đạt |
| nơi lưu checkpoint, quy tắc train | `main.py`: `artifacts/models/mlp_baseline.pt`, 5 epoch, val loss thấp nhất | Đạt (đúng như mô tả) |
| quy định mới: `training/` hoặc `model_present/`, log theo ngày giờ, train accuracy, điều kiện dừng | chưa có trong `train.py` / `main.py` (code của Thư) | Chưa, câu hỏi mở |
