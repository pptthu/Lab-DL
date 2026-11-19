# Tiến độ Lab 3

- **Created**: 2026-11-20T01:25:13+07:00 (pptthu)
- **Last Updated**: 2026-11-20T01:37:02+07:00 (pptthu)

Cập nhật file này sau MỖI bước.
Bảng dưới đây là nguồn thông tin duy nhất về việc ai đang làm gì. Bài làm lại độc lập trên nhánh `feat/lab3-redo` (plan: `docs/plans/detail_plan_lab3_md_v1.md`).

## Phân công (Thư chốt 2026-11-20)
- **Member A — Thư:** `sentiment_demo` (Bài 1), `preprocessing`, `eda`, `features`, `model` (Bài 2.1–2.4); tích hợp `main.py`, README, notebook trình bày
- **Member B — Phúc (ngohongphuc2805ag-hash):** `train`, `evaluate`, `experiment`, `visualize` (Bài 2.5–2.7, mở rộng); notebook Kaggle và các lần chạy trên Kaggle
- Theo cách chia của Lab 1 và Lab 3 cũ (A: dữ liệu → model; B: train → đánh giá)

| Module | Phụ trách | Trạng thái (chưa làm / đang làm / xong) | Test (đạt / lỗi / chưa chạy) | Chạy trên dữ liệu thật (đạt / lỗi / chưa chạy) | Ghi chú |
|---|---|---|---|---|---|
| `sentiment_demo` | Member A (Thư) | chưa làm | chưa chạy | chưa chạy | Bài 1; 6 câu kiểm soát theo CheckList |
| `preprocessing` | Member A (Thư) | chưa làm | chưa chạy | chưa chạy | Tải aclImdb gốc; val 10% chia theo phim |
| `eda` | Member A (Thư) | chưa làm | chưa chạy | chưa chạy | Ra số liệu để chốt `max_length` |
| `features` | Member A (Thư) | chưa làm | chưa chạy | chưa chạy | |
| `model` | Member A (Thư) | chưa làm | chưa chạy | chưa chạy | |
| `train` | Member B (Phúc) | chưa làm | chưa chạy | chưa chạy | Chạy trên Kaggle GPU |
| `evaluate` | Member B (Phúc) | chưa làm | chưa chạy | chưa chạy | |
| `experiment` | Member B (Phúc) | chưa làm | chưa chạy | chưa chạy | Pilot → 3 seed → `max_length` |
| `visualize` | Member B (Phúc) | chưa làm | chưa chạy | chưa chạy | |
| Notebook Kaggle, các lần chạy Kaggle | Member B (Phúc) | chưa làm | – | chưa chạy | |
| `main.py`, README, notebook trình bày | Member A (Thư) | chưa làm | chưa chạy | chưa chạy | Tích hợp |

## Handoff A → B (làm tuần tự)
Phúc bắt đầu code khi TẤT CẢ các mục sau đúng:
- [ ] Dataset đã tokenize có `input_ids`, `attention_mask`, `label` (3 split)
- [ ] `max_length` đã chốt sau EDA, ghi trong `docs/experiments/experiment_log.md`
- [ ] `model.build_model(seed)` và `model.load_tokenizer()` chạy được
- [ ] `pytest labs/lab3/tests` đạt cho các module của Member A
- [ ] Code của Member A đã push lên `feat/lab3-redo`; Thư gửi tin bàn giao (đã làm gì, lệnh chạy, vấn đề còn lại)

Thư duyệt ở các cổng: smoke run, pilot (chốt trần epoch, patience), audit.

## Nhật ký (mỗi bước một dòng, có ngày)
- 2026-11-20 Thư: chia việc Member A (Thư) / Member B (Phúc); `eda` chuyển về Member A.
- 2026-11-20 Thư: xong tài liệu (plan, rules, folder structure, templates, desc lab, mục lục, 9 file `phases/`).
- 2026-11-19 Thư: tạo nhánh `feat/lab3-redo` từ `main` (`a648057`), bỏ bài Lab 3 cũ trên nhánh này để làm lại độc lập.
