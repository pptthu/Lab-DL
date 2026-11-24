# Tài liệu Lab 3 — đọc file này trước (v1)

- Tác giả: Thư | Tạo: 2026-11-19T17:14:08+07:00 | Cập nhật: Thư, 2026-11-24T17:01:03+07:00

## Quy tắc version
- Tên file có hậu tố `_vN`. Luôn dùng bản có **số lớn nhất**; bản cũ giữ nguyên.
- Muốn đổi nội dung: hỏi Thư sửa đè hay tạo `_vN+1` (xem `rules/RULES_v1.md`).

## Desc hiện hành
- Desc của Lab 3 là **`lab3_description_v1.md`**. AI sinh báo cáo, slide từ file này và các file nó trỏ tới.
- `../project_description.md`: bản tổng quan để trình bày với cô (viết sau khi có kết quả). `../progress_status.md`: tiến độ và bằng chứng chạy.
- Bài làm lại độc lập trên nhánh `feat/lab3-redo`; không dùng code, số liệu, tài liệu của Lab 3 cũ.

## Mục lục

| File / thư mục | Nội dung |
|---|---|
| `lab3_description_v1.md` | Desc Lab 3: Bài 1, Bài 2, mở rộng, success criteria, các bước |
| `plans/detail_plan_lab3_md_v1.md` | Plan đã duyệt: quyết định L1–L17 kèm căn cứ, trình tự S0–S5 |
| `rules/RULES_v1.md` | Quy trình làm việc, luật chung, phân vai người và AI |
| `rules/FOLDER_STRUCTURE_v1.md` | Cây thư mục, kết quả nào lưu ở đâu, file nào commit |
| `templates/DESC_TEMPLATE_v1.md` | Khuôn desc cả lab |
| `templates/DESC_TEMPLATE_v3.md` | Khuôn desc từng module (`phases/`) |
| `templates/LOG_TEMPLATE_v1.md` | Khuôn log mỗi lần chạy, quy tắc chọn checkpoint và dừng |
| `templates/MODEL_CARD_TEMPLATE_v1.md` | Khuôn model card (Initial / Trained model) |
| `phases/` | Desc 9 module: `sentiment_demo`, `eda`, `preprocessing`, `features`, `model`, `train`, `evaluate`, `experiment`, `visualize` (`<module>_v1.md`) |
| `experiments/` | Sổ ghi thí nghiệm `experiment_log.md`: pilot, 3 seed, `max_length` (tạo khi chạy pilot); `overfit_calibration_v1.md`: overfit theo loss, bằng chứng chọn checkpoint theo val acc, độ tự tin (ECE), các cách chặn overfit |
| `models/` | Model card DistilBERT fine-tune (tạo khi viết xong code `model/`) |
| `bugs/` | Chưa có; tạo khi có lỗi đầu tiên |

## Thứ tự đọc
1. File này → 2. `rules/RULES_v1.md` → 3. `rules/FOLDER_STRUCTURE_v1.md` → 4. `lab3_description_v1.md` → 5. `plans/detail_plan_lab3_md_v1.md` → 6. `phases/<module>_v1.md` của module đang làm → 7. template cần dùng.
