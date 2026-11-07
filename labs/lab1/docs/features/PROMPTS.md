# Lab 1 / features: outline and small prompts

- **Motivation/Background**: The lecturer's workflow: outline -> AI writes the detail -> small prompts, one step each, each followed by its check.
- **Purpose**: Keep the outline (draft) and the small prompts for `features`.
- **Overview Pipeline**: outline -> discussion -> spec -> one prompt per step -> check -> next prompt.
- **Detailed Plan**: Scaffold (owner's outline); Small prompts (one step each); Vietnamese version.
- **References**: [features.md](../features/features.md), [STATUS.md](STATUS.md).
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

- Purpose: normalized tensors in batches, statistics from train only.
- Input: arrays + split indices from `preprocessing`. Output: 3 DataLoaders in memory, batch `[64,1,28,28]` float32.
- Steps: mean/std on train -> one transform -> 3 datasets -> loaders (shuffle train only).
- Success: mean 0.2859, std 0.3530; mean ~0, std ~1 after normalization; right shape/dtype. If not: check that val/test were not used.
- Not allowed: statistics on val/test, shuffling val/test, augmentation.

## Small prompts (one step each)

Use one prompt at a time; run the check of that step before the next prompt.

| # | Prompt | Check after it |
|---|---|---|
| 1 | Read `labs/lab1/docs/features.md`. Do not write code. Restate purpose, IO, success criteria; list what is unclear. | summary matches the scaffold |
| 2 | Implement `compute_mean_std` and `build_transform` as described, with tests (0-1 scale, `std <= 0` error, mean 0 / std 1). | `pytest labs/lab1/tests/test_features.py` |
| 3 | Implement `FashionDataset` (shared arrays, `indices`) with a test. | same pytest |
| 4 | Implement `make_loaders` (shuffle train only) with a test. | same pytest |
| 5 | Smoke: build the loaders from the real data, print mean/std and one batch shape. | 0.2859 / 0.3530, `[64,1,28,28]` float32 |
| 6 | Audit `features.py` against `features.md`; report only. | Audit table updated |

---

## Tiếng Việt

### Dàn ý (người phụ trách tự viết)

Bản nháp do AI soạn từ các quyết định của người phụ trách (`project_description.md`, `progress_status.md`); **Toàn cần đọc lại hoặc viết lại**. File spec là phần chi tiết viết từ dàn ý này.

- Mục đích: tensor đã chuẩn hóa theo batch, thống kê chỉ lấy từ train.
- Input: mảng + chỉ số split từ `preprocessing`. Output: 3 DataLoader trong bộ nhớ, batch `[64,1,28,28]` float32.
- Các bước: mean/std trên train → một transform → 3 dataset → loader (chỉ xáo train).
- Thành công: mean 0,2859, std 0,3530; sau chuẩn hóa mean ≈ 0, std ≈ 1; đúng shape/dtype. Không đạt: kiểm có lỡ dùng val/test không.
- Cấm: thống kê trên val/test, xáo val/test, augmentation.

### Prompt nhỏ (mỗi prompt một bước)

Mỗi lần dùng một prompt; chạy phần kiểm tra của bước đó xong mới sang prompt sau.

| # | Prompt | Kiểm tra sau đó |
|---|---|---|
| 1 | Đọc `labs/lab1/docs/features.md`. Chưa viết code. Nói lại mục đích, IO, tiêu chí; liệt kê chỗ chưa rõ. | tóm tắt khớp dàn ý |
| 2 | Viết `compute_mean_std`, `build_transform` theo mô tả, kèm test (thang 0–1, lỗi khi `std <= 0`, mean 0 / std 1). | `pytest labs/lab1/tests/test_features.py` |
| 3 | Viết `FashionDataset` (dùng chung mảng, `indices`) kèm test. | pytest như trên |
| 4 | Viết `make_loaders` (chỉ xáo train) kèm test. | pytest như trên |
| 5 | Smoke: tạo loader từ dữ liệu thật, in mean/std và shape một batch. | 0,2859 / 0,3530, `[64,1,28,28]` float32 |
| 6 | Audit `features.py` so với `features.md`; chỉ báo cáo. | cập nhật bảng Audit |
