# Lab 3 — Phân tích cảm xúc và fine-tuning với Hugging Face (v1)
- Tác giả: Member B | Tạo: 2026-11-18T13:00:00+07:00 | Cập nhật: Member B, 2026-11-18T13:00:00+07:00
- Trạng thái: In Progress (7 module đã `done` trong `../progress_status.md`; `main.py`, README và notebook mục 11, 12 của nhóm trưởng còn `todo`)
## Background
Đề Practice 3: Bài 1: chạy một model cảm xúc có sẵn và xem tokenization. Bài 2: fine-tune một transformer pre-trained cho phân loại văn bản nhị phân bằng Trainer API. Slide môn học không có chương Hugging Face, nên nhóm dựa vào các link trong slide Practice 3.
Slide không nêu sản phẩm nộp. Nhóm trưởng chốt ngày 2026-11-04: toàn bộ source code của ba lab và một báo cáo PDF viết sau.
## Purpose
- Làm:
  - Bài 1: `show_tokens` và `analyze_sentences` với model `distilbert-base-uncased-finetuned-sst-2-english`, chạy trên 3 câu mẫu
  - Bài 2: fine-tune `distilbert-base-uncased` trên IMDB, tập con 2000 / 500 / 500 (train / validation / test), seed 42, chọn epoch theo F1 validation, đánh giá test đúng một lần
  - So sánh model fine-tune với baseline zero-shot của Bài 1 trên cùng 500 review test
- Không làm: LoRA, cleanlab, ensemble (ngoài phạm vi, notebook mục 12); họ model khác ngoài DistilBERT.
## Input / Output
- Input: IMDB (`stanfordnlp/imdb`) tải tự động vào `labs/lab3/raw_data/` (không commit, lần đầu cần mạng); 3 câu mẫu cho Bài 1.
- Output: `labs/lab3/artifacts/{figures,models,logs}/` (hình EDA, model fine-tune, metric test; không commit); model card ở `docs/models/`.
- Code: `labs/lab3/main.py` gọi 7 module `labs/lab3/<module>/<module>.py`; notebook `labs/lab3/notebook/lab3_presentation.ipynb`.
## Success criteria
- `python -m pytest labs/lab3/tests -q` đạt.
- Bài 1: in token (đầu `[CLS]`, cuối `[SEP]`) và nhãn của 3 câu mẫu.
- Bài 2: fine-tune nhỏ chạy hết không lỗi; model lưu vào `artifacts/models/` và nạp lại được; accuracy và F1 trên validation và trên test; tập test chỉ dùng một lần.
- Có đủ: hình EDA, bảng so sánh zero-shot và fine-tuned trên cùng 500 review test, model card, báo cáo ngắn trong `../README.md`.
- Chưa đặt ngưỡng accuracy cho lab này [cần chốt: hỏi cô xem có mức đạt không].
## How-to-do
1. Bài 1: chạy model cảm xúc có sẵn và xem token (`sentiment_demo`).
2. Nạp IMDB, cắt tập con, làm sạch; xem phân bố nhãn và độ dài (`preprocessing`, `eda`).
3. Nạp `distilbert-base-uncased` và tokenizer; tokenize ba tập (`model`, `features`).
4. Fine-tune rất ngắn để kiểm tra, rồi fine-tune nhỏ đầy đủ; đo metric trên validation (`train`, `evaluate`).
5. Đánh giá test một lần; so với zero-shot; viết báo cáo ngắn (`evaluate`, `main.py`, notebook).
6. Điền model card; audit code và tài liệu so với desc này.
## Quyết định (người ghi)
- 2026-11-04 (nhóm trưởng): bài nộp là toàn bộ source code của ba lab và một báo cáo PDF viết sau.
- 2026-10-31: lần fine-tune đã ghi dùng cấu hình nhỏ — `max_length` 128, 1 epoch, batch 8 — vì RAM 7,8 GB và 2 nhân CPU (`../README.md`); mặc định của `main.py` là 256, 2 và 16.
- [cần chốt] Hỏi cô: IMDB, `distilbert-base-uncased`, tập con 2000 / 500 / 500, và việc có làm EDA hay không (slide không nêu). Danh sách link dataset do một nhóm khác tổng hợp, không phải của cô.
- [cần chốt] Nhóm trưởng chọn một tên thư mục lưu model: `distilbert_imdb` (notebook) hay `distilbert_sentiment` (`main.py`).
- [cần chốt] Nhóm trưởng xác nhận tài liệu trong `labs/lab3/docs/` viết tiếng Việt, ngoại lệ so với `AGENTS.md` dòng 6, như Lab 1 đã làm (`../../lab1/docs/rules/RULES_v1.md` mục 2).
