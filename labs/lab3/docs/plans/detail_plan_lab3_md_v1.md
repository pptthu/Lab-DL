# Detail plan v1 — Làm lại Lab 3 (Hugging Face: sentiment analysis + fine-tuning)

- **Tác giả:** Thư duyệt; AI soạn theo thảo luận 2026-11-19
- **Tạo:** 2026-11-19T15:30:56+07:00 | **Cập nhật:** 2026-11-19T15:30:56+07:00 (v1)
- **Trạng thái:** chờ Thư duyệt
- **Nhánh:** `lab3/redo` (tách từ `main` tại `a648057`). PR vào `main` khi Thư bảo. Bài của các bạn trên `main` và các nhánh `lab3/*` giữ nguyên; nhánh này là bản làm độc lập để dự phòng
- **Chuẩn:** quy trình và tài liệu theo Lab 1 (`labs/lab1/docs/rules/RULES_v1.md`). Nội dung, thí nghiệm, ngưỡng đạt lấy căn cứ riêng cho Lab 3 (§2), không chép bài Lab 3 cũ

## 0. Vì sao làm lại

Kết quả Lab 3 cũ (`labs/lab3/README.md` dòng 49–56): fine-tuned test acc **0,8220** / F1 0,8195, thấp hơn pipeline zero-shot (acc 0,8920) 7 điểm. Cấu hình: 2000 review train, 1 epoch, `max_length` 128, CPU, test 500 review. Số liệu cũ không dùng lại trong bài mới.

## 1. Đề bài (slide Practice 3) → module

| Bước của đề | Module |
|---|---|
| Bài 1.1 cài `transformers` | `requirements.txt` |
| 1.2 model cảm xúc pre-trained từ Hub; 1.3 tokenize câu mẫu; 1.4 phân tích cảm xúc | `sentiment_demo/` |
| Bài 2.1 cài `transformers`, `datasets`, `evaluate` | `requirements.txt`; metric dùng thư viện `evaluate` |
| 2.2 nạp dataset nhị phân | `preprocessing/` (+ `eda/`) |
| 2.3 nạp model và tokenizer pre-trained | `model/` |
| 2.4 tiền xử lý cho model | `features/` |
| 2.5 TrainingArguments; 2.6 Trainer và fine-tune | `train/` |
| 2.7 đánh giá model đã fine-tune | `evaluate/` |
| Mở rộng (§5) | `experiment/`, `visualize/` |

## 2. Quyết định đã chốt (ngày 2026-11-19, Thư)

| # | Quyết định | Căn cứ |
|---|---|---|
| L1 | Dataset IMDB (Maas et al. 2011): 25k train / 25k test chính thức, cân bằng nhãn. Không dùng 50k review không nhãn | Trang Stanford: *"25,000 for training and 25,000 for testing"* |
| L2 | **Dùng full data.** Test 25k chỉ dùng một lần ở cuối | So sánh được với DistilBERT paper (đo trên test chính thức) |
| L3 | Validation = **10% train (2.500)**, cân bằng nhãn, **chia theo phim** nếu file gốc có ID phim; train còn 22.500 | README aclImdb: train và test là hai tập phim không trùng nhau → val cũng nên tách phim. Sai số chuẩn acc ở ~92% với 2.500 mẫu ≈ 0,54 điểm |
| L4 | Tải bản gốc `aclImdb_v1.tar.gz` từ Stanford (có URL phim cho từng review), không dùng `load_dataset("imdb")` | Cần ID phim cho L3. `[cần kiểm khi tải]` file `urls_*.txt` |
| L5 | Model `distilbert-base-uncased` | Đề yêu cầu model pre-trained; DistilBERT paper (Sanh et al. 2019, Table 2) báo IMDb test acc 92,82 |
| L6 | Bài 1 dùng pipeline `distilbert-base-uncased-finetuned-sst-2-english`; đo zero-shot trên cùng 25k test làm baseline | So sánh công bằng với model fine-tune |
| L7 | Metric bằng thư viện `evaluate` (`accuracy`, `f1`) | Đề bước 2.1; tutorial chính thức HF |
| L8 | **Chọn checkpoint theo val accuracy**; `val_loss` chỉ theo dõi overfit (khác Lab 1, 2) | Khi fine-tune, `val_loss` có thể tăng trong khi accuracy vẫn tăng (HF course ch.3.5, HF forum) |
| L9 | lr 2e-5, AdamW, warmup 10% rồi linear decay, batch 16, fp16 | BERT gốc (lr chọn trong {5e-5,…,2e-5}, batch 16/32); Mosbach et al. 2021 (lr 2e-5, warmup 10%, linear decay) |
| L10 | Số epoch **không đặt cứng**: chạy pilot (§4 bước S2) rồi chọn | Mosbach et al. 2021 cho thấy train ngắn có thể chưa đủ; Niophy thấy overfit nhẹ ở epoch 3 → phải đo trên chính Lab 3 |
| L11 | Đánh giá val **mỗi 1/4 epoch** | Dodge et al. 2020: hiệu năng giữa epoch 1 đã tương quan cao với kết quả cuối |
| L12 | Lần chạy chính **3 seed**; độ lệch chuẩn đo được quyết định cách làm thí nghiệm `max_length` (§4 S4) | Dodge et al. 2020, Table 3: std theo seed SST 0,0028, tập nhỏ 0,06–0,10 |
| L13 | Ngưỡng đạt: **test acc ≥ 91%** (trung bình 3 seed); kỳ vọng 92–93% | DistilBERT paper 92,82; model card cộng đồng 91,2–93,1 |
| L14 | Chạy train trên **Kaggle GPU**; máy Thư chỉ chạy test, smoke, EDA | Máy: i5-1240P, RAM 7,7 GB, không CUDA (đo 2026-11-19) |
| L15 | Cấu trúc thư mục như Lab 1: `data/`, `outputs/`; bỏ `raw_data/`, `artifacts/`, `logbook/` | Thư chốt; `labs/lab1/docs/rules/FOLDER_STRUCTURE_v2.md` |
| L16 | Tài liệu tiếng Việt, có Created/Last Updated đủ giờ phút giây, file có `_vN` | Luật Lab 1 (`RULES_v1.md` §2) |
| L17 | Mở rộng chỉ gồm §5; bỏ ROC/PR, head+tail, so sánh model khác, LoRA, cleanlab | Thư: mở rộng quá nhiều thì khó vấn đáp |

## 3. Cấu trúc `labs/lab3/` (đích)

    labs/lab3/
    ├── sentiment_demo/ eda/ preprocessing/ features/ model/ train/ evaluate/ experiment/ visualize/
    ├── data/
    │   ├── raw/        # aclImdb (không commit)
    │   └── splits/     # train_val_split.json (ID review + ID phim của val)
    ├── outputs/
    │   ├── models/     # checkpoint tốt nhất (không commit; trên Kaggle)
    │   ├── logs/       # <ts>_<run>.log, _history.csv, _config.json
    │   ├── figures/  results/  demo/  _smoke/
    ├── docs/  (rules/ templates/ plans/ phases/ models/ experiments/ bugs/)
    ├── notebook/  tests/  kaggle/   # kaggle/: notebook chạy main.py trên Kaggle
    └── main.py  README.md  requirements.txt  progress_status.md  project_description.md

Code module cũ trên nhánh này viết lại từ đầu (không chép). File cũ chỉ bị thay trên nhánh `lab3/redo`.

## 4. Trình tự chạy (mỗi bước xong mới chốt số của bước sau)

| Bước | Việc | Chạy ở | Chốt sau bước |
|---|---|---|---|
| S0 | Code + `pytest` + smoke run (vài chục review, vài step) | Máy Thư | Pipeline chạy hết, log đủ |
| S1 | EDA trên full data: phân bố nhãn, độ dài token (median, P90, % > 128/256/512) | Máy Thư (chỉ tokenize) | `max_length` cho lần chạy chính `[cần chốt sau S1]` |
| S2 | **Pilot**: 1 seed, trần epoch trên mức 2–4 của BERT (số cụ thể theo thời gian/epoch trên Kaggle), không dừng sớm, log val mỗi 1/4 epoch | Kaggle | Số epoch, patience, ngưỡng cải thiện `[cần chốt sau S2]` |
| S3 | **Lần chạy chính**: 3 seed, giữ checkpoint val acc cao nhất, early stopping theo S2 | Kaggle | std theo seed |
| S4 | Thí nghiệm `max_length` 128/256/512: std nhỏ (≈ ≤ 0,3 điểm) → 1 seed/giá trị; std lớn → 3 seed/giá trị. So sánh trên **val** | Kaggle | Kết luận ảnh hưởng của việc cắt review |
| S5 | Test 25k: model chính (3 seed) + zero-shot, mỗi model đúng một lần | Kaggle | Kết quả báo cáo |

**Quy tắc kiểm tra trần epoch:** nếu checkpoint tốt nhất rơi vào lần đánh giá cuối cùng → trần quá thấp, tăng trần và chạy lại.

## 5. Phần mở rộng (mỗi mục có lý do riêng của Lab 3)

- **Đường loss/acc theo mỗi lần đánh giá** (train loss, val loss, val acc) — bằng chứng chọn điểm dừng (L10, L11)
- **Thí nghiệm `max_length`** — review IMDB dài; Sun et al. 2019 cho thấy cách cắt văn bản ảnh hưởng acc trên IMDb; nối với EDA S1
- **Confusion matrix zero-shot và fine-tuned** — zero-shot học trên SST-2 (câu ngắn), IMDB là review dài: xem mỗi model lệch FP hay FN
- **Ví dụ dự đoán sai, chia nhóm** — review bị cắt / không bị cắt (nối với `max_length`); phủ định, mỉa mai

## 6. Success criteria

- `python -m pytest labs/lab3/tests -q` đạt; smoke run chạy hết không lỗi
- Bài 1: in token (đầu `[CLS]`, cuối `[SEP]`) và nhãn + điểm cho câu mẫu
- Bài 2: test acc trung bình 3 seed ≥ 91%; báo mean ± std acc và F1; test 25k dùng đúng một lần cho mỗi model
- Checkpoint nạp lại dự đoán được; mọi số liệu trong tài liệu trỏ tới file trong `outputs/` của một lần chạy có log
- Có đủ: hình EDA, đường huấn luyện, bảng so sánh zero-shot / fine-tuned, bảng `max_length`, confusion matrix, ví dụ sai, model card

## 7. Không làm

ROC/PR, head+tail truncation, so sánh model khác (BERT-base, RoBERTa), LoRA, cleanlab, Optuna, dùng 50k review không nhãn.

## 8. Tài liệu sẽ tạo (theo khuôn Lab 1)

`docs/00_README_v1.md`, `docs/lab3_description_v1.md`, `docs/rules/RULES_v1.md`, `docs/rules/FOLDER_STRUCTURE_v1.md`, `docs/templates/` (copy khuôn Lab 1: DESC_TEMPLATE_v3, LOG_TEMPLATE_v1, MODEL_CARD_TEMPLATE_v1), `docs/phases/<module>_v1.md` (9 module), `docs/models/distilbert_imdb_v1.md`, `docs/experiments/experiment_log.md`. `README.md`, `progress_status.md`, `project_description.md` viết lại tiếng Việt.

## 9. Bước tiếp theo

1. Thư duyệt plan này
2. Tài liệu: rules → FOLDER_STRUCTURE → templates → lab3_description → phases; mỗi lượt một file, Thư nói "lưu"
3. Code từng module (small prompt), mỗi module một commit trên `lab3/redo`, kèm test
4. S0 → S5 theo §4; cập nhật `progress_status.md`, `AI_usage.md`
5. PR vào `main` theo `CONTRIBUTING.md` khi Thư bảo

## 10. Nguồn

- Maas et al. 2011, IMDB: https://ai.stanford.edu/~amaas/data/sentiment/ ; README aclImdb (bản sao): https://huggingface.co/datasets/ngthanhtinqn/aclImdb/blob/main/README
- Sanh et al. 2019, DistilBERT: https://arxiv.org/abs/1910.01108
- Mosbach et al. 2021, On the Stability of Fine-tuning BERT: https://arxiv.org/abs/2006.04884
- Dodge et al. 2020, Fine-Tuning Pretrained Language Models: https://arxiv.org/abs/2002.06305
- Sun et al. 2019, How to Fine-Tune BERT for Text Classification?: https://arxiv.org/abs/1905.05583
- HF Text classification: https://huggingface.co/docs/transformers/tasks/sequence_classification
- HF Callbacks (EarlyStoppingCallback): https://huggingface.co/docs/transformers/main_classes/callback
- HF LLM course ch.3.5: https://huggingface.co/learn/llm-course/chapter3/5
- Model card cộng đồng: https://huggingface.co/nickapch/distilbert-base-uncased-finetuned-imdb , https://huggingface.co/NikkeS/imdb-distilbert , https://huggingface.co/Niophy/distilbert-imdb-sentiment
