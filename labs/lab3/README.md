# Lab 3: Phân tích cảm xúc và fine-tuning với Hugging Face

- **Created**: 2026-11-20T14:29:08+07:00 (pptthu)
- **Last Updated**: 2026-11-20T14:29:08+07:00 (pptthu)

Số liệu lấy từ các lần chạy ngày 20/11/2026: Stage 0–5 trên máy Thư (CPU), Stage 6–10 trên Kaggle Tesla T4 (tài khoản Kaggle của Thư). Mỗi lần chạy có log trong `outputs/logs/`; nhật ký đầy đủ ở [`docs/experiments/experiment_log.md`](docs/experiments/experiment_log.md). Desc: [`docs/lab3_description_v1.md`](docs/lab3_description_v1.md).

## Báo cáo ngắn
**Bài toán.** Bài 1: dùng model cảm xúc pre-trained từ Hugging Face Hub, tokenize câu mẫu và dự đoán cảm xúc. Bài 2: fine-tune model pre-trained để phân loại review phim là tích cực hay tiêu cực, bằng `transformers`, `datasets`, `evaluate` và `Trainer`.

**Dữ liệu.** IMDB gốc (Maas et al. 2011): 25.000 review train, 25.000 review test, nhãn cân bằng. Tách 10% train làm validation, **chia theo phim** để không phim nào nằm ở cả train và val → 22.499 / 2.501. Bỏ thẻ `<br />`; cắt review ở 512 token (13% review dài hơn).

**Model.** `distilbert-base-uncased` (66,96 triệu tham số) + đầu phân loại 2 lớp mới (0,59 triệu), fine-tune toàn bộ.

**Huấn luyện.** AdamW lr 2e-5, warmup 10% rồi giảm tuyến tính, batch 16, fp16, weight decay 0,01. Đánh giá val mỗi 1/4 epoch, giữ checkpoint có **val accuracy** cao nhất. Quy tắc dừng chốt từ pilot (6 epoch, không dừng sớm): trần 6 epoch, patience 8 lần đánh giá. Chạy 3 seed (42, 43, 44).

**Kết quả.**

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
- Val loss thấp nhất ở epoch 1 rồi tăng, trong khi val acc còn tăng tới epoch 3–5: chọn checkpoint theo val acc thay vì val loss là đúng với bài này.
- Cùng seed 42 nhưng pilot (0,9308) và lần chạy chính (0,9288) khác nhau: GPU không tất định, nên báo trung bình ± std của 3 seed.
- Fine-tuned sai cả 2 câu phủ định kiểm soát ("not bad at all", "would not say … good") mà zero-shot đúng.
- Lỗi tập trung ở rating sát ranh giới (rating 4 sai 17,2%, rating 7 sai 11,7%) và ở review bị cắt (sai 10,05% so với 6,14%).

**Giới hạn.** 3 seed, một cách chia val, một loại GPU. `max_length` so sánh với 1 seed mỗi giá trị. Không làm: ROC/PR, cắt head + tail, so sánh model khác, LoRA, Optuna (plan §7).

**Hình và bảng.** Notebook [`notebook/lab3_presentation.ipynb`](notebook/lab3_presentation.ipynb); sơ đồ kiến trúc và số liệu model trong [`docs/models/distilbert_imdb_v1.md`](docs/models/distilbert_imdb_v1.md).

### Tài liệu tham khảo
1. A. L. Maas et al. Learning Word Vectors for Sentiment Analysis. ACL 2011.
2. V. Sanh, L. Debut, J. Chaumond, T. Wolf. DistilBERT, a distilled version of BERT. arXiv:1910.01108, 2019.
3. J. Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
4. M. Mosbach, M. Andriushchenko, D. Klakow. On the Stability of Fine-tuning BERT. ICLR 2021.
5. J. Dodge et al. Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. arXiv:2002.06305, 2020.
6. C. Sun, X. Qiu, Y. Xu, X. Huang. How to Fine-Tune BERT for Text Classification? CCL 2019.
7. M. T. Ribeiro et al. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL 2020.
8. Tài liệu Hugging Face Transformers: https://huggingface.co/docs/transformers

## Dữ liệu
IMDB gốc của Stanford (https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz, 84 MB). Lần chạy đầu, `preprocessing.download_imdb()` tự tải và giải nén vào `labs/lab3/data/raw/` (không commit). Dùng bản gốc thay vì `load_dataset("imdb")` vì chỉ bản gốc có mã phim của từng review. File chia val `data/splits/train_val_split.json` đã commit.

## Cấu trúc thư mục
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

## Cách chạy
```bash
pip install -r labs/lab3/requirements.txt
python -m pytest labs/lab3/tests            # 48 test trên dữ liệu giả
python -m labs.lab3.main --smoke            # chạy thử, ghi vào outputs/_smoke/ (phần A ~40 giây; phần B trên CPU > 15 phút)
python -m labs.lab3.main                    # phần A trên dữ liệu thật (Bài 1, EDA, split, tokenize, model)
python -m labs.lab3.main --stage pilot      # một bước GPU; các bước: timing, pilot, main, maxlen, test, figures
```
Các bước GPU (pilot, 3 seed, `max_length`, test) chạy trên Kaggle qua [`kaggle/README.md`](kaggle/README.md); checkpoint nằm trong output của kernel Kaggle, không commit. Notebook chỉ đọc `outputs/`, không train.

Môi trường của kết quả trên: Python 3.13, transformers 5.22.1, datasets 5.1.0, evaluate 0.4.6; Kaggle: torch 2.11.0 + CUDA, Tesla T4, 1 GPU, fp16. GPU không tất định nên chạy lại có thể lệch vài phần nghìn.
