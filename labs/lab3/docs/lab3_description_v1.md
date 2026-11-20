# Lab 3 — Phân tích cảm xúc và fine-tuning với Hugging Face (v1)
- Tác giả: Thư | Tạo: 2026-11-19T17:02:59+07:00 | Cập nhật: Thư, 2026-11-21T01:42:29+07:00
- Trạng thái: Done (đã merge vào `main` qua PR #84, 2026-11-20)
## Background
Đề Practice 3. Bài 1: dùng model cảm xúc pre-trained từ Hugging Face Hub, tokenize và phân tích cảm xúc câu mẫu. Bài 2: fine-tune model pre-trained cho phân loại văn bản nhị phân bằng `transformers`, `datasets`, `evaluate` và Trainer (7 bước). Bài làm lại độc lập, không dùng code hay số liệu Lab 3 cũ.
## Purpose
- Làm:
  - Bài 1: pipeline `distilbert-base-uncased-finetuned-sst-2-english` — in token và nhãn + điểm cho câu mẫu; đo zero-shot trên 25k test IMDB làm baseline
  - Bài 2: fine-tune `distilbert-base-uncased` trên IMDB full (train 22.499 / val 2.501 tách theo phim — không ép tròn 2.500 vì chia nguyên phim, Thư chốt 2026-11-20 / test 25.000), metric bằng thư viện `evaluate`, 3 seed
  - Mở rộng: đường loss/acc, thí nghiệm `max_length` 128/256/512, confusion matrix zero-shot vs fine-tuned, ví dụ dự đoán sai theo nhóm
- Không làm: ROC/PR, head+tail, model khác (BERT-base, RoBERTa), LoRA, cleanlab, Optuna, 50k review không nhãn
## Input / Output
- Input: `data/raw/aclImdb/` (bản gốc Stanford `aclImdb_v1.tar.gz`, 25k train + 25k test, cân bằng nhãn; không commit)
- Một mẫu: 1 file `.txt` = một review + nhãn 0/1 (thư mục `neg`/`pos`); kèm rating (tên file `<id>_<rating>.txt`) và mã phim (`urls_[pos|neg].txt`)
- Vào model: tối đa 512 token (`input_ids`, `attention_mask`), pad theo batch
- Model trả: 2 logits; softmax → xác suất khen (`prob_pos`)
- Loss: cross-entropy (model tự tính khi có `labels`); metric: accuracy (chính) + F1 bằng thư viện `evaluate`
- Output: checkpoint tốt nhất (`outputs/models/`, trên Kaggle, không commit); file dự đoán test (`uid, label, pred, prob_pos`); `test_summary.csv`; log mỗi lần chạy, hình, CSV Bài 1 (`outputs/{logs,figures,results,demo}/`, commit); `data/splits/train_val_split.json`
- Code: `labs/lab3/main.py` gọi 9 module `labs/lab3/<module>/<module>.py`; chạy chính trên Kaggle (`labs/lab3/kaggle/`); trình bày ở `notebook/lab3_presentation.ipynb` (chỉ đọc `outputs/`)
## Success criteria
- `python -m pytest labs/lab3/tests -q` đạt; smoke run chạy hết không lỗi
- Bài 1: token bắt đầu `[CLS]`, kết thúc `[SEP]`; nhãn + điểm cho mỗi câu mẫu (`outputs/demo/`)
- Bài 2: test accuracy trung bình 3 seed ≥ 91% trên 25k test — DistilBERT paper (Sanh et al. 2019, Table 2) báo 92,82; model card cộng đồng 91,2–93,1
- Test 25k dùng đúng một lần cho mỗi model; checkpoint nạp lại dự đoán được
- Có đủ: hình EDA, đường huấn luyện, bảng zero-shot vs fine-tuned (mean ± std), bảng `max_length`, confusion matrix, ví dụ sai, model card
## How-to-do
1. Bài 1 (`sentiment_demo`)
2. Tải + chia dữ liệu theo phim (`preprocessing`), EDA độ dài token (`eda`) → chốt `max_length`
3. Model + tokenizer (`model`), tokenize (`features`)
4. Smoke run trên máy Thư
5. Pilot trên Kaggle (`train`) → chốt trần epoch, patience
6. Lần chạy chính 3 seed (`train`, `experiment`) → std theo seed
7. Thí nghiệm `max_length` theo std (`experiment`)
8. Test một lần + zero-shot (`evaluate`); hình (`visualize`); model card, README, notebook
## Quyết định (người ghi)
- 2026-11-19 (Thư): toàn bộ L1–L17 trong `plans/detail_plan_lab3_md_v1.md` §2 — căn cứ ghi ở cột "Căn cứ" của plan
- 2026-11-20 (Thư): `max_length` lần chạy chính = 512 — 512 cắt 13% review, 256 cắt 41%; zero-shot cũng cắt ở 512 (`experiments/experiment_log.md` S1)
- [cần chốt sau pilot] trần epoch, patience, threshold
- [cần chốt sau 3 seed] số seed của thí nghiệm `max_length`
