# Tài liệu Lab 3 — đọc file này trước (v1)

- Tác giả: Member B | Tạo: 2026-11-18T13:00:00+07:00 | Cập nhật: Member B, 2026-11-18T13:00:00+07:00

## Quy tắc version
- Tên file có hậu tố `_vN`. Luôn dùng bản có **số lớn nhất**; bản cũ giữ nguyên, không sửa đè.
- Muốn đổi nội dung: tạo bản `_vN+1` sau khi nhóm trưởng duyệt.
- Luật làm việc và các khuôn dùng chung với Lab 1 (chưa có bản riêng cho Lab 3): [../../lab1/docs/rules/RULES_v1.md](../../lab1/docs/rules/RULES_v1.md), [../../lab1/docs/templates/DESC_TEMPLATE_v3.md](../../lab1/docs/templates/DESC_TEMPLATE_v3.md), [../../lab1/docs/templates/MODEL_CARD_TEMPLATE_v1.md](../../lab1/docs/templates/MODEL_CARD_TEMPLATE_v1.md).

## Ngôn ngữ
- Tài liệu trong `labs/lab3/docs/` viết tiếng Việt, giữ thuật ngữ tiếng Anh, không song ngữ, giống Lab 1 (`../../lab1/docs/rules/RULES_v1.md` mục 2). Đây là ngoại lệ so với `AGENTS.md` dòng 6 ("English first, Vietnamese below") [cần chốt: nhóm trưởng xác nhận cho Lab 3].

## Desc hiện hành
- Desc của Lab 3 là **`lab3_description_v1.md`**. AI sinh báo cáo, slide từ file này và các file nó trỏ tới.
- `../project_description.md` là bản tổng quan (Define + Design); `../progress_status.md` ghi tiến độ và bằng chứng chạy; `../README.md` là báo cáo ngắn.

## Mục lục

| File / thư mục | Nội dung |
|---|---|
| `lab3_description_v1.md` | Desc Lab 3: mục tiêu, success criteria, các bước, quyết định |
| `phases/sentiment_demo_v1.md` | Bài 1: model cảm xúc có sẵn và token (Member A) |
| `phases/preprocessing_v1.md` | Nạp IMDB, cắt tập con, làm sạch (Member A) |
| `phases/eda_v1.md` | Khám phá dữ liệu train (Member A) |
| `phases/features_v1.md` | Tokenize, cắt `max_length` (Member A) |
| `phases/model_v1.md` | Nạp DistilBERT pre-trained (Member B) |
| `phases/train_v1.md` | Fine-tune bằng Trainer, nơi lưu model (Member B) |
| `phases/evaluate_v1.md` | accuracy, F1, test một lần, so zero-shot (Member B) |
| `phases/demo_v1.md` | Chạy bằng `main.py` và notebook, nơi lưu kết quả, vấn đáp (nhóm trưởng) |
| `models/distilbert_imdb_v1.md` | Model card của model đã fine-tune |
| `../logbook/` | Bug report và ghi chú thí nghiệm của Lab 3 (hiện chưa có file) |

## Các file cũ
- Bảy file `docs/<module>.md` (`demo.md`, `eda.md`, `evaluate.md`, `features.md`, `model.md`, `preprocessing.md`, `train.md`) là bản trước: song ngữ, viết cho AI agent. Theo luật version chúng được giữ nguyên; bản hiện hành là các file `_v1` ở trên.
- `demo.md` cũ gồm hai phần, nay tách thành `phases/sentiment_demo_v1.md` (module `sentiment_demo`) và `phases/demo_v1.md` (cách chạy và trình bày).

## Thứ tự đọc
1. File này → 2. `../../lab1/docs/rules/RULES_v1.md` → 3. `lab3_description_v1.md` → 4. `phases/<module>_v1.md` của module đang làm → 5. khuôn cần dùng.
