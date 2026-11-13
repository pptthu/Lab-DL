# Tài liệu Lab 1 — đọc file này trước (v1)

- Tác giả: Thư | Tạo: 2026-11-13T11:22:05+07:00 | Cập nhật: Thư, 2026-11-13T11:22:05+07:00

## Quy tắc version
- Tên file có hậu tố `_vN`. Luôn dùng bản có **số lớn nhất**; bản cũ giữ nguyên, không sửa đè.
- Muốn đổi nội dung: tạo bản `_vN+1` sau khi leader duyệt (xem `rules/RULES_v1.md`).

## Desc hiện hành
- Desc của Lab 1 là **`lab1_description_v1.md`**. AI sinh báo cáo, slide từ file này và các file nó trỏ tới.
- `../project_description.md` (bản dài cũ) và `../progress_status.md` ở gốc lab giữ nguyên chỗ, không còn là desc hiện hành.

## Mục lục

| File / thư mục | Nội dung |
|---|---|
| `lab1_description_v1.md` | Desc Lab 1: 5 cấu hình, success criteria, các bước |
| `rules/RULES_v1.md` | Quy trình làm việc, luật chung, phân vai người và AI |
| `rules/FOLDER_STRUCTURE_v1.md` | Cây thư mục đích, kết quả nào lưu ở đâu |
| `templates/DESC_TEMPLATE_v1.md` | Khuôn desc khoảng 20 dòng |
| `templates/LOG_TEMPLATE_v1.md` | Khuôn log mỗi lần chạy và quy tắc dừng |
| `templates/MODEL_CARD_TEMPLATE_v1.md` | Khuôn model card (Initial / Trained model) |
| `plans/` | Detail plan có version (`detail_plan_lab1_md_v0.md`, `_v1.md`) |
| `phases/`, `models/`, `experiments/`, `bugs/` | Chưa có; tạo khi có file đầu tiên |
| `demo.md`, `eda.md`, `evaluate.md`, `experiment.md`, `features.md`, `model.md`, `preprocessing.md`, `train.md`, `visualize.md` | Hướng dẫn module bản cũ; giai đoạn sau viết lại theo khuôn desc và chuyển vào `phases/` |

## Thứ tự đọc
1. File này → 2. `rules/RULES_v1.md` → 3. `rules/FOLDER_STRUCTURE_v1.md` → 4. `lab1_description_v1.md` → 5. template cần dùng.
