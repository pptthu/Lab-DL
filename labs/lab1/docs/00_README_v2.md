# Tài liệu Lab 1 — đọc file này trước (v2)

- Tác giả: Thư | Tạo: 2026-11-14T00:29:40+07:00 | Cập nhật: Thư, 2026-11-14T01:43:35+07:00

## Quy tắc version
- Tên file có hậu tố `_vN`. Luôn dùng bản có **số lớn nhất**; bản cũ giữ nguyên, không sửa đè.
- Muốn đổi nội dung: tạo bản `_vN+1` sau khi Thư duyệt (xem `rules/RULES_v1.md`).

## Desc hiện hành
- Desc của Lab 1 là **`lab1_description_v1.md`**. AI sinh báo cáo, slide từ file này và các file nó trỏ tới.
- `../project_description.md` (bản dài cũ) và `../progress_status.md` ở gốc lab giữ nguyên chỗ, không còn là desc hiện hành.

## Mục lục

| File / thư mục | Nội dung |
|---|---|
| `lab1_description_v1.md` | Desc Lab 1: 5 cấu hình, success criteria, các bước |
| `rules/RULES_v1.md` | Quy trình làm việc, luật chung, phân vai người và AI |
| `rules/FOLDER_STRUCTURE_v2.md` | Cây thư mục đích, kết quả nào lưu ở đâu |
| `templates/DESC_TEMPLATE_v1.md` | Khuôn desc cả lab, khoảng 20 dòng |
| `templates/DESC_TEMPLATE_v3.md` | Khuôn desc từng module (theo phase của M) — bản hiện hành cho `phases/` |
| `templates/DESC_TEMPLATE_v2.md` | Bản cũ của khuôn module, giữ theo luật version, không dùng nữa |
| `templates/LOG_TEMPLATE_v1.md` | Khuôn log mỗi lần chạy và quy tắc dừng |
| `templates/MODEL_CARD_TEMPLATE_v1.md` | Khuôn model card (Initial / Trained model) |
| `plans/` | Detail plan có version (`detail_plan_lab1_md_v0.md`, `_v1.md`, `_v2.md`) |
| `phases/` | Desc 9 module: `eda_v1.md`, `preprocessing_v1.md`, `features_v1.md`, `model_v1.md`, `train_v1.md`, `evaluate_v1.md`, `visualize_v1.md`, `experiment_v1.md`, `demo_v1.md` |
| `experiments/` | Sổ ghi thí nghiệm `experiment_log.md` (trước là `logbook/`) |
| `models/` | Model card: `mlp_baseline_v1.md`, `mlp_larger_network_v1.md` (cấu hình được chọn) |
| `bugs/` | Chưa có; tạo khi có file đầu tiên |

## Thứ tự đọc
1. File này → 2. `rules/RULES_v1.md` → 3. `rules/FOLDER_STRUCTURE_v2.md` → 4. `lab1_description_v1.md` → 5. `phases/<module>_v1.md` của module đang làm → 6. template cần dùng.
