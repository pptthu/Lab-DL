# Lab 1 / model: outline and small prompts

- **Motivation/Background**: The lecturer's workflow: outline -> AI writes the detail -> small prompts, one step each, each followed by its check.
- **Purpose**: Keep the outline (draft) and the small prompts for `model`.
- **Overview Pipeline**: outline -> discussion -> spec -> one prompt per step -> check -> next prompt.
- **Detailed Plan**: Scaffold (owner's outline); Small prompts (one step each); Vietnamese version.
- **References**: [model.md](../model/model.md), [STATUS.md](STATUS.md).
- **Created**: 2026-11-07T00:00:00+07:00
- **Last Updated**: 2026-11-07T00:00:00+07:00

---

## Table of Contents

- [Scaffold (owner's outline)](#scaffold-owners-outline)
- [Small prompts (one step each)](#small-prompts-one-step-each)
- [Tiếng Việt](#tiếng-việt)

---

## Scaffold (owner's outline)

Draft written by the AI agent from the owner's decisions (`project_description.md`, `progress_status.md`); **Toàn must review or rewrite it**. The spec file is the detail written from this outline.

- Name, author, dates.
- Purpose: MLP baseline, logits `[B,10]`, MLP only.
- Initial model: backbone (none), number of layers, architecture drawing, input/output, source file, builder.
- Trained model: who trained it, when it finished, results, source changes, saved path, note.
- Evaluation: accuracy (balanced classes), val loss for selection, test once.
- Success: shape, logits, 235,146 parameters, overfit one batch.

## Small prompts (one step each)

Use one prompt at a time; run the check of that step before the next prompt.

| # | Prompt | Check after it |
|---|---|---|
| 1 | Read `labs/lab1/docs/model.md`. Do not write code. Restate the architecture, IO and success criteria; list what is unclear. | summary matches the scaffold |
| 2 | Implement `MLP` exactly as drawn in `model.md` (no softmax), plus `count_parameters`, with tests. | `pytest labs/lab1/tests/test_model.py` |
| 3 | Add the overfit-one-batch test described in `model.md`. | same pytest |
| 4 | Smoke: print `count_parameters(MLP())` and the output shape for `torch.zeros(4,1,28,28)`. | `235146`, `[4, 10]` |
| 5 | After Member B's run: fill section 4 (who, when, results, saved path) from `progress_status.md` only. | numbers have a date and a source |
| 6 | Audit `model.py` against `model.md`; report only. | Audit table updated |

---

## Tiếng Việt

### Dàn ý (người phụ trách tự viết)

Bản nháp do AI soạn từ các quyết định của người phụ trách (`project_description.md`, `progress_status.md`); **Toàn cần đọc lại hoặc viết lại**. File spec là phần chi tiết viết từ dàn ý này.

- Tên, tác giả, ngày.
- Mục đích: MLP baseline, logits `[B,10]`, chỉ làm MLP.
- Model ban đầu: backbone (không có), số lớp, hình kiến trúc, input/output, file nguồn, builder.
- Model đã train: ai train, xong khi nào, kết quả, thay đổi source, nơi lưu, ghi chú.
- Đánh giá: accuracy (lớp cân bằng), val loss để chọn model, test đo một lần.
- Thành công: shape, logits, 235.146 tham số, overfit 1 batch.

### Prompt nhỏ (mỗi prompt một bước)

Mỗi lần dùng một prompt; chạy phần kiểm tra của bước đó xong mới sang prompt sau.

| # | Prompt | Kiểm tra sau đó |
|---|---|---|
| 1 | Đọc `labs/lab1/docs/model.md`. Chưa viết code. Nói lại kiến trúc, IO, tiêu chí; liệt kê chỗ chưa rõ. | tóm tắt khớp dàn ý |
| 2 | Viết `MLP` đúng như hình trong `model.md` (không softmax) và `count_parameters`, kèm test. | `pytest labs/lab1/tests/test_model.py` |
| 3 | Thêm test overfit 1 batch như mô tả trong `model.md`. | pytest như trên |
| 4 | Smoke: in `count_parameters(MLP())` và shape đầu ra với `torch.zeros(4,1,28,28)`. | `235146`, `[4, 10]` |
| 5 | Sau khi Thư chạy: điền mục 4 (ai, khi nào, kết quả, nơi lưu) chỉ từ `progress_status.md`. | số có ngày và nguồn |
| 6 | Audit `model.py` so với `model.md`; chỉ báo cáo. | cập nhật bảng Audit |
