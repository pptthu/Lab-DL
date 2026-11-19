# sentiment_demo — Bài 1: phân tích cảm xúc bằng model pre-trained (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T02:32:51+07:00 (Thư)
- **Người phụ trách**: Thư
- **Status**: In Progress — code, test, smoke và normal run xong (2026-11-20); chờ Thư audit

---

## Name
Bài 1 — nạp pipeline cảm xúc pre-trained, tokenize câu mẫu, dự đoán nhãn + điểm

## Background
Bài 1 của đề (1.1–1.4): cài `transformers`, dùng model cảm xúc pre-trained từ Hub, tokenize một câu mẫu, phân tích cảm xúc câu đó. Cùng model này được `evaluate` dùng làm baseline zero-shot trên 25k test (plan L6).

## Goals / Purpose
- `tokenize_sentence(text)` trả `tokens`, `input_ids`, `attention_mask`; token đầu `[CLS]`, cuối `[SEP]`
- `predict_sentiment(texts)` trả nhãn (`POSITIVE`/`NEGATIVE`) và điểm cho mỗi câu
- `run_demo()` ghi `outputs/demo/sentiment_demo.csv` (câu, năng lực kiểm, nhãn kỳ vọng, nhãn dự đoán, điểm, đúng/sai) và `outputs/demo/sentiment_demo_tokens.csv` (token + id của câu 1); in bảng ra màn hình
- `CONTROL_SENTENCES`: 6 câu kiểm soát theo CheckList, dùng lại ở `evaluate` cho model sau fine-tune
- Không làm:
  - Đo zero-shot trên 25k test — thuộc `evaluate` (dùng chung `predict_sentiment`)
  - Fine-tune — thuộc `train` (Bài 2)

## Input / Output
- **Input:** danh sách câu mẫu (hằng số trong module); model `distilbert-base-uncased-finetuned-sst-2-english` từ Hub (lần đầu cần mạng)
- **Output:** `outputs/demo/sentiment_demo.csv`; smoke run: `outputs/_smoke/demo/`

## How to do it (general plan)
1. Nạp tokenizer + pipeline `sentiment-analysis` với model ở trên; ghi `revision` (commit hash của model trên Hub) vào CSV để chạy lại ra cùng kết quả
2. `tokenize_sentence`: `tokenizer(text)` + `convert_ids_to_tokens`
3. `predict_sentiment`: gọi pipeline với `truncation=True`, theo batch
4. `run_demo`: chạy trên câu mẫu, ghi CSV
5. Test với tokenizer/pipeline giả (không cần mạng)
6. Smoke run → kiểm CSV có đủ cột, token đầu/cuối đúng
7. Normal run (máy Thư được, Bài 1 nhẹ)

## Pipeline
```
câu mẫu → tokenize_sentence → tokens / ids
        → predict_sentiment (pipeline SST-2) → nhãn + điểm → outputs/demo/sentiment_demo.csv
                                                          → evaluate (zero-shot trên 25k test)
```

## Detailed plan
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english` — model ví dụ cho text classification trong tài liệu DistilBERT của HF; ghi rõ tên thay vì để pipeline tự chọn mặc định, để không đổi theo phiên bản thư viện
- **Câu mẫu (Thư duyệt 2026-11-20):** 6 câu, mỗi câu kiểm một năng lực theo CheckList (Ribeiro et al. 2020 — model cảm xúc, kể cả thương mại, hay sai ở phủ định): (1) rõ tích cực, (2) rõ tiêu cực, (3) phủ định câu tiêu cực ("not bad"), (4) phủ định câu tích cực ("not good"), (5) tương phản có "but", (6) trung tính. Mỗi câu ghi nhãn kỳ vọng; câu (6) không có nhãn kỳ vọng — cho thấy model nhị phân vẫn buộc chọn POSITIVE/NEGATIVE
- **Tokenize:** đề ghi "a sample sentence" → in token chi tiết cho câu (1); cả 6 câu đều dự đoán
- **Truncation:** `truncation=True` (DistilBERT nhận tối đa 512 vị trí, `max_position_embeddings` = 512 theo `DistilBertConfig`) — cần khi dùng lại cho review IMDB dài ở `evaluate`
- **Thiết bị:** GPU nếu có (`device=0`), không thì CPU
- **Gotchas:**
  - Model học trên SST-2 (câu ngắn) còn IMDB là review dài → baseline zero-shot có thể lệch domain (plan §5)
  - Pipeline trả nhãn chữ; `evaluate` phải đổi sang 1/0 đúng với nhãn IMDB (pos = 1, neg = 0)

## Triển khai (2026-11-20)
- **Hàm** (`labs/lab3/sentiment_demo/sentiment_demo.py`): `CONTROL_SENTENCES` (`:22`); `load_sentiment_pipeline(model_name=config.SST2_MODEL, device=None)` (`:37`); `tokenize_sentence(text, tokenizer) -> dict` (`:51`); `predict_sentiment(texts, pipe, batch_size=32) -> list[dict]` (`:64`); `hub_revision(model_name) -> str` (`:73`); `run_demo(out_dir, pipe=None, sentences=None) -> dict` (`:96`). Chạy: `python -m labs.lab3.sentiment_demo.sentiment_demo [--smoke]`
- **Chỗ code khác plan:** thêm `hub_revision` — transformers 5.22 không còn ghi `config._commit_hash`, nên commit của model đọc từ thư mục cache Hugging Face (`snapshots/<hash>/`). CSV có thêm cột `model`, `revision`. Câu (5) tương phản: kỳ vọng NEGATIVE vì vế sau "but" chê cả cốt truyện lẫn kết thúc
- **Kết quả normal run** (máy Thư, CPU; model commit `714eb0fa89d2f80546fda750413ed43d93601a13`): 5/5 câu có nhãn kỳ vọng đoán đúng (cả 2 câu phủ định và câu "but", điểm 0,998–0,9999); câu trung tính → NEGATIVE, điểm 0,9955. Câu 1 tokenize thành 15 token, `[CLS]` = 101, `[SEP]` = 102. File: `outputs/demo/sentiment_demo.csv`, `sentiment_demo_tokens.csv`
- **Test / smoke:** `pytest labs/lab3/tests/test_sentiment_demo.py` 4 passed (pipeline giả, không cần mạng); smoke ghi `outputs/_smoke/demo/` không lỗi

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_sentiment_demo.py` đạt: token đầu/cuối, đủ cột CSV, nhãn thuộc {POSITIVE, NEGATIVE}, điểm trong [0, 1]
- **Smoke run:** `outputs/_smoke/demo/sentiment_demo.csv` có đủ số câu mẫu
- **Normal run:** `outputs/demo/sentiment_demo.csv` (6 dòng) và `sentiment_demo_tokens.csv`; câu (1), (2) ra đúng nhãn; kết quả câu (3)–(5) ghi đúng/sai như đo được, không sửa câu cho đẹp

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [evaluate_v1.md](evaluate_v1.md)
- Nguồn: Ribeiro et al. 2020, CheckList https://arxiv.org/abs/2005.04118
- Code: `labs/lab3/sentiment_demo/sentiment_demo.py` | Test: `labs/lab3/tests/test_sentiment_demo.py`
