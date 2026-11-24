# Lab 3: Phân tích cảm xúc và fine-tuning với Hugging Face

- **Created**: 2026-11-20T14:29:08+07:00 (pptthu)
- **Last Updated**: 2026-11-24T16:51:29+07:00 (pptthu, AI soạn)

Số liệu lấy từ các lần chạy ngày 20/11/2026: Stage 0–5 trên máy Thư (CPU), Stage 6–10 trên Kaggle Tesla T4 (tài khoản Kaggle của Thư). Mỗi lần chạy có log trong `outputs/logs/`; nhật ký đầy đủ ở [`docs/experiments/experiment_log.md`](docs/experiments/experiment_log.md). Desc: [`docs/lab3_description_v1.md`](docs/lab3_description_v1.md).

## 1. Tổng quan dự án (Project overview)
Lab 3 làm đề Practice 3 "Get started with Hugging Face" gồm 2 bài. Bài 1 dùng model cảm xúc có sẵn trên Hugging Face Hub. Bài 2 fine-tune `distilbert-base-uncased` để phân loại review phim IMDB là khen hay chê. Model fine-tune đạt test accuracy **0,9315 ± 0,0020** (3 seed, 25.000 review test), cao hơn model có sẵn của Bài 1 (zero-shot, 0,8922). Tổng quan Define + Design: [`project_description.md`](project_description.md).

## 2. Bài toán (Problem statement)
**Đề.** Bài 1: cài `transformers`, nạp model cảm xúc pre-trained từ Hub, tokenize câu mẫu, phân tích cảm xúc. Bài 2: dùng `transformers`, `datasets`, `evaluate`; nạp dataset nhị phân, model pre-trained và tokenizer; tiền xử lý; `TrainingArguments`; `Trainer` fine-tune; đánh giá.

**Dữ liệu.** IMDB gốc (Maas et al. 2011): 25.000 review train, 25.000 review test, nhãn cân bằng. Tách 10% train làm validation, **chia theo phim** để không phim nào nằm ở cả train và val → 22.499 / 2.501. Bỏ thẻ `<br />`; cắt review ở 512 token (13% review dài hơn).

Nguồn: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz (84 MB). Lần chạy đầu, `preprocessing.download_imdb()` tự tải và giải nén vào `labs/lab3/data/raw/` (không commit). Dùng bản gốc thay vì `load_dataset("imdb")` vì chỉ bản gốc có mã phim của từng review; dữ liệu sau đó được chuyển thành `datasets.DatasetDict`. File chia val `data/splits/train_val_split.json` đã commit.

**Đầu vào / đầu ra.** Một mẫu là một review (file `.txt`) với nhãn 0 (chê) hoặc 1 (khen). Model nhận tối đa 512 token, trả 2 logits; softmax ra xác suất review là khen. Loss cross-entropy; chỉ số chính là accuracy, báo kèm F1.

**Khó khăn.** Review dài (13% bị cắt ở 512 token), nhiều review vừa khen vừa chê, có phủ định ("not bad").

**Tiêu chí đạt** (chốt trước khi chạy). Test accuracy trung bình 3 seed ≥ 91% (bài báo DistilBERT báo 92,82% trên IMDb); test 25k chạy một lần cho mỗi model.

## 3. Tính năng chính (Main features)
- **Bài 1 (`sentiment_demo`).** Pipeline `distilbert-base-uncased-finetuned-sst-2-english`: in token (`[CLS]` … `[SEP]`), nhãn và điểm cho 6 câu kiểm soát (khen rõ, chê rõ, phủ định, "but", câu trung tính).
- **Dữ liệu (`preprocessing`, `eda`).** Tải IMDB gốc, làm sạch, chia train/val theo phim và gộp review trùng nội dung vào cùng phía; EDA độ dài token, review trùng, rating → chốt `max_length` 512.
- **Model (`features`, `model`).** Tokenize, pad theo batch; DistilBERT + đầu phân loại 2 lớp.
- **Huấn luyện (`train`, `experiment`).** `TrainingArguments` + `Trainer`; đánh giá val mỗi 1/4 epoch; chọn checkpoint theo val accuracy; early stopping; pilot để chốt số epoch; 3 seed; thí nghiệm `max_length` 128 / 256 / 512.
- **Đánh giá (`evaluate`, `visualize`).** Metric bằng thư viện `evaluate`; test 25k chạy một lần cho mỗi model (có cờ chặn chạy lần hai); so với zero-shot; confusion matrix; phân tích lỗi theo rating và theo việc bị cắt; đường huấn luyện.
- **Chạy trên GPU (`kaggle/`).** Đưa code lên Kaggle và chạy từng bước bằng CLI; mỗi lần chạy ghi log, lịch sử đánh giá và cấu hình.
- **Kiểm tra.** 48 test trên dữ liệu giả; smoke run toàn pipeline.

## 4. Cấu trúc hệ thống
Luồng chạy: `preprocessing` → `eda` (chốt `max_length`) → `features` + `model` → `experiment` gọi `train` (pilot, 3 seed, `max_length`) → `evaluate` (test một lần, zero-shot) → `visualize` → `outputs/` → notebook (chỉ đọc `outputs/`). `main.py` gọi tất cả (`--smoke`, `--stage`). Sơ đồ kiến trúc, interface từng module, rủi ro kỹ thuật: [`project_description.md`](project_description.md) mục 2.

```text
labs/lab3/
├── README.md                # báo cáo ngắn (file này)
├── project_description.md   # tổng quan Define + Design
├── progress_status.md       # tiến độ, phân công, nhật ký
├── config.py                # mọi tham số ở một chỗ
├── main.py                  # chạy pipeline (--smoke, --stage)
├── sentiment_demo/ preprocessing/ eda/ features/ model/                # Member A (Thư)
├── train/ evaluate/ experiment/ visualize/                             # Member B (Phúc)
├── kaggle/                  # đưa code lên Kaggle và chạy từng bước bằng CLI
├── notebook/lab3_presentation.ipynb    # trình bày, chỉ đọc outputs/
├── data/                    # raw/ (không commit) | splits/ (có commit)
├── outputs/                 # logs, results, figures, demo (có commit) | models, _smoke (không commit)
├── docs/                    # tài liệu, đọc docs/00_README_v1.md trước
└── tests/                   # test từng module (dữ liệu giả, không cần mạng)
```
Chi tiết: [`docs/rules/FOLDER_STRUCTURE_v1.md`](docs/rules/FOLDER_STRUCTURE_v1.md). Desc từng module: [`docs/phases/`](docs/phases/).

## 5. Cách chạy
```bash
pip install -r labs/lab3/requirements.txt
python -m pytest labs/lab3/tests            # 48 test trên dữ liệu giả
python -m labs.lab3.main --smoke            # chạy thử, ghi vào outputs/_smoke/ (phần A ~40 giây; phần B trên CPU > 15 phút)
python -m labs.lab3.main                    # phần A trên dữ liệu thật (Bài 1, EDA, split, tokenize, model)
python -m labs.lab3.main --stage pilot      # một bước GPU; các bước: timing, pilot, main, maxlen, test, figures
```
Các bước GPU (pilot, 3 seed, `max_length`, test) chạy trên Kaggle qua [`kaggle/README.md`](kaggle/README.md); checkpoint nằm trong output của kernel Kaggle, không commit. Notebook chỉ đọc `outputs/`, không train.

Môi trường của kết quả: Python 3.13, transformers 5.22.1, datasets 5.1.0, evaluate 0.4.6; Kaggle: torch 2.11.0 + CUDA, Tesla T4, 1 GPU, fp16. GPU không tất định nên chạy lại có thể lệch vài phần nghìn.

## 6. Kết quả
**Model.** `distilbert-base-uncased` (66,96 triệu tham số) + đầu phân loại 2 lớp mới (0,59 triệu), fine-tune toàn bộ.

**Huấn luyện.** AdamW lr 2e-5, warmup 10% rồi giảm tuyến tính, batch 16, fp16, weight decay 0,01. Đánh giá val mỗi 1/4 epoch, giữ checkpoint có **val accuracy** cao nhất. Quy tắc dừng chốt từ pilot (6 epoch, không dừng sớm): trần 6 epoch, patience 8 lần đánh giá. Chạy 3 seed (42, 43, 44).

| Seed | Epoch tốt nhất | Dừng | val acc | val F1 | test acc | test F1 |
|---|---:|:---:|---:|---:|---:|---:|
| 42 | 3,0 | early stopping, epoch 5,0 | 0,9288 | 0,9290 | 0,9300 | 0,9304 |
| 43 | 5,5 | hết trần, epoch 6,0 | 0,9308 | 0,9316 | 0,9309 | 0,9319 |
| 44 | 2,75 | early stopping, epoch 4,75 | 0,9312 | 0,9321 | 0,9338 | 0,9345 |
| **Trung bình ± std** | | | **0,9303 ± 0,0013** | | **0,9315 ± 0,0020** (≥ 91% ✔) | **0,9323** |

| Model (test 25k, mỗi model chạy một lần) | test acc | test F1 | test acc khi bỏ 123 review trùng train |
|---|---:|---:|---:|
| Zero-shot SST-2 (Bài 1) | 0,8922 | 0,8892 | 0,8921 |
| Fine-tuned (trung bình 3 seed) | **0,9315** | **0,9323** | 0,9312 |

| `max_length` (seed 42, val) | 128 | 256 | 512 |
|---|---:|---:|---:|
| val acc | 0,8729 | 0,9152 | **0,9288** |

**Nhận xét.**
- Fine-tuned đạt 0,9315 ± 0,0020, vượt ngưỡng 91% và gần mức 92,82% của DistilBERT trên IMDb trong bài báo gốc.
- Fine-tune hơn zero-shot 3,93 điểm. Zero-shot bỏ sót 13,5% review tích cực (1.690 FN), fine-tuned còn 685.
- Cắt review càng ngắn càng mất accuracy: 512 hơn 256 1,36 điểm, 256 hơn 128 4,24 điểm, đều lớn hơn 2 × std giữa các seed.
- Model có overfit theo loss: train loss về 0,02–0,04, val loss thấp nhất ở epoch 1–2 (0,20–0,21) rồi tăng lên 0,38–0,42, trong khi val acc còn tăng tới epoch 3–5. Phần bị overfit là độ tự tin: model đúng 93% nhưng tự tin trung bình 99% (ECE 0,05–0,06 trên test, chỉ báo cáo). Nếu chọn theo val loss, cả pilot và 3 seed đều dừng ở epoch 1–2 với val acc thấp hơn 0,3–1 điểm, nên chọn checkpoint theo val acc. Chi tiết: [`docs/experiments/overfit_calibration_v1.md`](docs/experiments/overfit_calibration_v1.md).
- Cùng seed 42 nhưng pilot (0,9308) và lần chạy chính (0,9288) khác nhau: GPU không tất định, nên báo trung bình ± std của 3 seed.
- Fine-tuned sai cả 2 câu phủ định kiểm soát ("not bad at all", "would not say … good") mà zero-shot đúng.
- Lỗi tập trung ở rating sát ranh giới (rating 4 sai 17,2%, rating 7 sai 11,7%) và ở review bị cắt (sai 10,05% so với 6,14%). Chia theo độ dài: tỉ lệ sai đứng ở 5,9–6,8% cho tới 512 token rồi nhảy lên 9,8–10,5% ngay sau mốc 512, nên lỗi tăng là do bị cắt chứ không chỉ do review dài (seed 44, `experiment_log.md` S5).

**Giới hạn.**
- 3 seed, một cách chia val, một loại GPU. `max_length` so sánh với 1 seed mỗi giá trị.
- Xác suất `prob_pos` tự tin quá mức, không dùng làm độ tin cậy được; muốn dùng thì phải calibrate (ví dụ temperature scaling), bài này chỉ dùng nhãn nên không làm.
- Không thêm regularization (tăng dropout, label smoothing…): nhóm M cùng model, cùng IMDb báo chênh lệch chỉ −0,07 đến +0,29 điểm test (1 seed, so trên test), không lớn hơn độ lệch giữa các seed của nhóm (std 0,20 điểm).
- Không làm: ROC/PR, cắt head + tail, so sánh model khác, LoRA, Optuna (plan §7).

**Hình và bảng.** Notebook [`notebook/lab3_presentation.ipynb`](notebook/lab3_presentation.ipynb); sơ đồ kiến trúc và số liệu model trong [`docs/models/distilbert_imdb_v1.md`](docs/models/distilbert_imdb_v1.md).

## 7. Tài liệu tham khảo
1. A. L. Maas et al. Learning Word Vectors for Sentiment Analysis. ACL 2011.
2. V. Sanh, L. Debut, J. Chaumond, T. Wolf. DistilBERT, a distilled version of BERT. arXiv:1910.01108, 2019.
3. J. Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
4. M. Mosbach, M. Andriushchenko, D. Klakow. On the Stability of Fine-tuning BERT. ICLR 2021.
5. J. Dodge et al. Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. arXiv:2002.06305, 2020.
6. C. Sun, X. Qiu, Y. Xu, X. Huang. How to Fine-Tune BERT for Text Classification? CCL 2019.
7. M. T. Ribeiro et al. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL 2020.
8. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML 2017.
9. S. Desai, G. Durrett. Calibration of Pre-trained Transformers. EMNLP 2020.
10. Tài liệu Hugging Face Transformers: https://huggingface.co/docs/transformers
