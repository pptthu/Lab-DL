# Lab 1 / preprocessing: outline and small prompts

- **Motivation/Background**: The lecturer's workflow: outline -> AI writes the detail -> small prompts, one step each, each followed by its check.
- **Purpose**: Keep the outline (draft) and the small prompts for `preprocessing`.
- **Overview Pipeline**: outline -> discussion -> spec -> one prompt per step -> check -> next prompt.
- **Detailed Plan**: Scaffold (owner's outline); Small prompts (one step each); Vietnamese version.
- **References**: [preprocessing.md](../preprocessing/preprocessing.md), [STATUS.md](STATUS.md).
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

- Purpose: official data, checked arrays, one shared split saved to a file.
- Input: 4 IDX files (torchvision). Output: 2 CSV in `raw_data/` (not committed), `artifacts/splits/train_val_split.json` (committed).
- Steps: download + self-check -> `load_csv` -> stratified `make_split` (10%, seed 42) -> `save_split`; later runs only `load_split`.
- Success: CSV = IDX, MD5 as recorded; 54,000 / 6,000, 600 per class; regenerated split = committed file. If not: never overwrite the JSON.
- Not allowed: validation from test; changing `val_ratio`/`seed` without the team leader; downloads in unit tests.

## Small prompts (one step each)

Use one prompt at a time; run the check of that step before the next prompt.

| # | Prompt | Check after it |
|---|---|---|
| 1 | Read `labs/lab1/docs/preprocessing.md`. Do not write code. Restate purpose, IO, success criteria; list what is unclear. | summary matches the scaffold |
| 2 | Implement only `load_csv` as described (check 0-255 before `uint8`), with tests on fake CSV. | `pytest labs/lab1/tests/test_preprocessing.py` |
| 3 | Implement `make_split` (stratified, seeded) and its tests: disjoint, complete, reproducible, 600 per class. | same pytest |
| 4 | Implement `save_split` / `load_split` with a round-trip test in `tmp_path`. | same pytest |
| 5 | Smoke: regenerate the split from the real CSV and compare with the committed JSON; do not write the JSON. | identical, 54,000 / 6,000 |
| 6 | Audit `preprocessing.py` and `download_official.py` against `preprocessing.md`; report only. | Audit table updated |

---

## Tiếng Việt

### Dàn ý (người phụ trách tự viết)

Bản nháp do AI soạn từ các quyết định của người phụ trách (`project_description.md`, `progress_status.md`); **Toàn cần đọc lại hoặc viết lại**. File spec là phần chi tiết viết từ dàn ý này.

- Mục đích: dữ liệu chính thức, mảng đã kiểm, một split dùng chung lưu ra file.
- Input: 4 file IDX (torchvision). Output: 2 CSV trong `raw_data/` (không commit), `artifacts/splits/train_val_split.json` (có commit).
- Các bước: tải + tự kiểm → `load_csv` → `make_split` phân tầng (10%, seed 42) → `save_split`; các lần sau chỉ `load_split`.
- Thành công: CSV = IDX, MD5 như đã ghi; 54.000 / 6.000, 600 mỗi lớp; split tạo lại = file đã commit. Không đạt: không bao giờ ghi đè JSON.
- Cấm: lấy validation từ test; đổi `val_ratio`/`seed` khi chưa hỏi nhóm trưởng; tải dữ liệu trong unit test.

### Prompt nhỏ (mỗi prompt một bước)

Mỗi lần dùng một prompt; chạy phần kiểm tra của bước đó xong mới sang prompt sau.

| # | Prompt | Kiểm tra sau đó |
|---|---|---|
| 1 | Đọc `labs/lab1/docs/preprocessing.md`. Chưa viết code. Nói lại mục đích, IO, tiêu chí; liệt kê chỗ chưa rõ. | tóm tắt khớp dàn ý |
| 2 | Chỉ viết `load_csv` theo mô tả (kiểm 0–255 trước khi ép `uint8`), kèm test trên CSV giả. | `pytest labs/lab1/tests/test_preprocessing.py` |
| 3 | Viết `make_split` (phân tầng, có seed) và test: không giao nhau, đủ, tái lập, 600 mỗi lớp. | pytest như trên |
| 4 | Viết `save_split` / `load_split` với test ghi rồi đọc lại trong `tmp_path`. | pytest như trên |
| 5 | Smoke: tạo lại split từ CSV thật và so với JSON đã commit; không ghi JSON. | giống hệt, 54.000 / 6.000 |
| 6 | Audit `preprocessing.py`, `download_official.py` so với `preprocessing.md`; chỉ báo cáo. | cập nhật bảng Audit |
