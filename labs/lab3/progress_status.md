# Tiến độ Lab 3

- **Created**: 2026-11-20T01:25:13+07:00 (pptthu)
- **Last Updated**: 2026-11-20T01:25:13+07:00 (pptthu)

Cập nhật file này sau MỖI bước.
Bảng dưới đây là nguồn thông tin duy nhất về việc ai đang làm gì. Bài làm lại độc lập trên nhánh `feat/lab3-redo` (plan: `docs/plans/detail_plan_lab3_md_v1.md`).

| Module | Phụ trách | Trạng thái (chưa làm / đang làm / xong) | Test (đạt / lỗi / chưa chạy) | Chạy trên dữ liệu thật (đạt / lỗi / chưa chạy) | Ghi chú |
|---|---|---|---|---|---|
| `sentiment_demo` | Thư | chưa làm | chưa chạy | chưa chạy | Bài 1; 6 câu kiểm soát theo CheckList |
| `preprocessing` | Thư | chưa làm | chưa chạy | chưa chạy | Tải aclImdb gốc; val 10% chia theo phim |
| `eda` | Phúc (ngohongphuc2805ag-hash) | chưa làm | chưa chạy | chưa chạy | Giao ngày 2026-11-20: code + test + chạy full data + đề xuất `max_length` cho Thư chốt; làm sau khi `preprocessing` xong; desc: `docs/phases/eda_v1.md` |
| `features` | Thư | chưa làm | chưa chạy | chưa chạy | |
| `model` | Thư | chưa làm | chưa chạy | chưa chạy | |
| `train` | Thư | chưa làm | chưa chạy | chưa chạy | Chạy trên Kaggle GPU |
| `evaluate` | Thư | chưa làm | chưa chạy | chưa chạy | |
| `experiment` | Thư | chưa làm | chưa chạy | chưa chạy | Pilot → 3 seed → `max_length` |
| `visualize` | Thư | chưa làm | chưa chạy | chưa chạy | |
| `main.py`, README, notebook, Kaggle | Thư | chưa làm | chưa chạy | chưa chạy | |

## Nhật ký (mỗi bước một dòng, có ngày)
- 2026-11-20 Thư: xong tài liệu (plan, rules, folder structure, templates, desc lab, mục lục, 9 file `phases/`); giao module `eda` cho Phúc.
- 2026-11-19 Thư: tạo nhánh `feat/lab3-redo` từ `main` (`a648057`), bỏ bài Lab 3 cũ trên nhánh này để làm lại độc lập.
