# Lab 1 / eda: outline and small prompts

- **Motivation/Background**: The lecturer's workflow: outline -> AI writes the detail -> small prompts, one step each, each followed by its check.
- **Purpose**: Keep the outline (draft) and the small prompts for `eda`.
- **Overview Pipeline**: outline -> discussion -> spec -> one prompt per step -> check -> next prompt.
- **Detailed Plan**: Scaffold (owner's outline); Small prompts (one step each); Vietnamese version.
- **References**: [eda.md](../eda/eda.md), [STATUS.md](STATUS.md).
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

- Purpose: check the training data before training; read-only.
- Input: `raw_data/fashion-mnist_train.csv`. Output: 6 statistics + 2 figures in `artifacts/figures/`.
- Steps: check columns -> 6 statistics -> 2 figures.
- Success: 60,000 rows, 6,000 per class, 0-255, 0 missing, 0 duplicates. If not: record, do not fix the data.
- Not allowed: modifying the CSV, EDA on test.

## Small prompts (one step each)

Use one prompt at a time; run the check of that step before the next prompt.

| # | Prompt | Check after it |
|---|---|---|
| 1 | Read `labs/lab1/docs/eda.md`. Do not write code. Restate purpose, IO, success criteria and list what is unclear. | your summary matches the scaffold |
| 2 | Implement only `_check_columns` as described in `eda.md`. Add its test in `tests/test_eda.py`. | `pytest labs/lab1/tests/test_eda.py` |
| 3 | Implement `run_eda` statistics (6 keys) as described. No plotting yet. | unit test on a fake CSV |
| 4 | Add the 2 figure helpers as described (`Figure`, `vmin=0, vmax=255`). | figures exist in `tmp_path` |
| 5 | Run the smoke command of `eda.md` and report the numbers. | numbers equal the success criteria |
| 6 | Audit: compare `eda.py` with `eda.md` line by line; report mismatches only, change nothing. | Audit table updated |

---

## Tiếng Việt

### Dàn ý (người phụ trách tự viết)

Bản nháp do AI soạn từ các quyết định của người phụ trách (`project_description.md`, `progress_status.md`); **Toàn cần đọc lại hoặc viết lại**. File spec là phần chi tiết viết từ dàn ý này.

- Mục đích: kiểm dữ liệu train trước khi train; chỉ đọc.
- Input: `raw_data/fashion-mnist_train.csv`. Output: 6 thống kê + 2 hình trong `artifacts/figures/`.
- Các bước: kiểm cột → 6 thống kê → 2 hình.
- Thành công: 60.000 dòng, 6.000 mỗi lớp, 0–255, 0 ô thiếu, 0 dòng trùng. Không đạt: ghi lại, không sửa dữ liệu.
- Cấm: sửa CSV, EDA trên test.

### Prompt nhỏ (mỗi prompt một bước)

Mỗi lần dùng một prompt; chạy phần kiểm tra của bước đó xong mới sang prompt sau.

| # | Prompt | Kiểm tra sau đó |
|---|---|---|
| 1 | Đọc `labs/lab1/docs/eda.md`. Chưa viết code. Nói lại mục đích, IO, tiêu chí thành công và liệt kê chỗ chưa rõ. | phần tóm tắt khớp dàn ý |
| 2 | Chỉ viết `_check_columns` theo `eda.md`, thêm test vào `tests/test_eda.py`. | `pytest labs/lab1/tests/test_eda.py` |
| 3 | Viết phần thống kê của `run_eda` (6 khóa) theo mô tả. Chưa vẽ hình. | unit test trên CSV giả |
| 4 | Thêm 2 hàm vẽ theo mô tả (`Figure`, `vmin=0, vmax=255`). | 2 hình có trong `tmp_path` |
| 5 | Chạy lệnh smoke trong `eda.md` và báo các con số. | số khớp tiêu chí thành công |
| 6 | Audit: đối chiếu `eda.py` với `eda.md` từng dòng; chỉ báo chỗ lệch, không sửa gì. | cập nhật bảng Audit |
