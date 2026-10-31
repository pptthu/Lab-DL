# Lab 3: Hugging Face sentiment analysis and fine-tuning

## Brief report (about 200 words or fewer)

**Problem.** Exercise 1: run a pre-trained sentiment model and inspect tokenization. Exercise 2: fine-tune a pre-trained transformer (`distilbert-base-uncased`) for binary sentiment classification with the Trainer API.

**Data.** IMDB movie reviews (Maas et al., ACL 2011), binary labels, small subsets with seed 42: 2000 train, 500 validation, 500 test. Train labels are balanced; review length is right-skewed (median 170 words) and 40.1% of train reviews reach the 256-token limit.

**Method.** Preprocessing and tokenization are described in the Data section. Fine-tuning with the Trainer: 1 epochs, learning rate 2e-5, batch size 8, `max_length` 128, on CPU; the epoch with the best validation F1 is kept.

**Results.** The pre-trained sentiment pipeline classified 3 sample sentences as POSITIVE, NEGATIVE and POSITIVE (scores above 0.99), including the negation "Not bad at all". On the same 500 test reviews, the fine-tuned model reached accuracy 0.8220 and F1 0.8195; the pre-trained pipeline (Exercise 1) reached accuracy 0.8920 and F1 0.8889.

**Conclusion.** The accuracy difference is -7.0 points, larger than the sampling error (about 1.7 points): the pre-trained pipeline scored higher on this test subset. One run, one seed; the cause was not tested.

## Data
- **Source:** IMDB (Large Movie Review Dataset), Maas, Daly, Pham, Huang, Ng and Potts, ACL 2011. Official page: https://ai.stanford.edu/~amaas/data/sentiment/ ; Hugging Face Hub: `stanfordnlp/imdb`.
- **Size:** 50,000 labeled reviews (25,000 train, 25,000 test). We use small, CPU-friendly subsets: 2000 / 500 / 500 (train / validation / test), seed 42.
- **Splits:** the Hub data is ordered by label, so each split is shuffled with the seed before taking the subset (otherwise a subset could contain one label only). Validation is cut from the original train split and test comes from the original test split, so the three subsets do not overlap.
- **Labels:** train {0: 1000, 1: 1000}, validation {0: 243, 1: 257}, test {0: 254, 1: 246}. Validation and test are random samples, so they are close to balanced but not exactly.
- **Cleaning (`clean_text`):** minimal. HTML entities are decoded, HTML tags such as `<br />` are replaced by a space and whitespace is collapsed. Text is not lowercased and punctuation is kept: the uncased DistilBERT tokenizer lowercases by itself and marks like "!" can carry sentiment.
- **Tokenization (`tokenize_dataset`):** DistilBERT tokenizer, texts truncated to `max_length=256` tokens, no fixed padding (padding is done per batch by the Trainer). The result has the columns `input_ids`, `attention_mask` and `label`.
- **Files:** downloaded automatically on the first run into `labs/lab3/raw_data/` (not committed); internet is needed the first time.

### Exploratory data analysis (train subset, 2000 reviews)

| Statistic (words per review) | Value |
|---|---|
| mean | 226.2 |
| median | 170 |
| 95th percentile | 585 |
| min / max | 12 / 999 |

The length distribution is right-skewed (mean above median): a few very long reviews pull the mean up. Counted in tokens, 802 of 2000 train reviews (40.1%) reach `max_length=256` and are truncated (or have exactly 256 tokens). The effect of this truncation on accuracy was not measured. `run_eda` also saves three figures (label distribution, length histogram, length by label) to `labs/lab3/artifacts/figures/` (not committed).

## Exercise 1: pre-trained sentiment analysis

`analyze_sentences` runs the Hugging Face pipeline with `distilbert-base-uncased-finetuned-sst-2-english` (DistilBERT already fine-tuned on SST-2); the plain `distilbert-base-uncased` has no sentiment head, so it cannot be used here. `show_tokens` uses the tokenizer of the same checkpoint.

| Sentence | Token ids | Label | Score |
|---|---|---|---|
| I love this movie | 101, 1045, 2293, 2023, 3185, 102 | POSITIVE | 0.9999 |
| The plot was boring | 101, 1996, 5436, 2001, 11771, 102 | NEGATIVE | 0.9998 |
| Not bad at all | 101, 2025, 2919, 2012, 2035, 102 | POSITIVE | 0.9991 |

Tokens are lowercase (uncased tokenizer); every sentence starts with `[CLS]` (id 101) and ends with `[SEP]` (id 102). The third sentence has a negation and is read correctly, but only 3 sentences were tested, so this does not show how the model handles negation in general.

## Exercise 2: fine-tuning

**Setup.** Model `distilbert-base-uncased` (2-label classification head added), `max_length` 128, 1 epochs, learning rate 2e-5, batch size 8, seed 42, device: CPU, training time: about 17 minutes. The Trainer evaluates on the validation set after every epoch and keeps the epoch with the best F1 (epoch 1).

| Model | Validation accuracy | Validation F1 | Test accuracy | Test F1 |
|---|---|---|---|---|
| Fine-tuned (Exercise 2) | 0.8340 | 0.8337 | 0.8220 | 0.8195 |
| Pre-trained pipeline (Exercise 1) | not measured | not measured | 0.8920 | 0.8889 |

The Exercise 1 baseline was measured on the same 500 test reviews (positive -> 1, negative -> 0). The validation numbers come from the epoch chosen on the validation set, so they are slightly optimistic; the test set was used once, at the end. With 500 reviews the standard error of an accuracy is about 1.7 points, so differences below about 2.5 to 3 points are within sampling error. The pipeline truncates reviews at 512 tokens, the fine-tuned model at 128 tokens. One run, one seed, one subset. Note: the EDA and tokenization above are described with `max_length=256`, but because of the RAM limit (7.8 GB, 2 CPU cores) the fine-tuning run used the smaller settings given in this section.

## Folder structure
```text
labs/lab3/
├── README.md                # brief report (EN + VI) / báo cáo ngắn
├── project_description.md   # Define + Design (feed this to the AI agent)
├── progress_status.md       # progress, audit, dated log / tiến độ
├── main.py                  # runs the whole pipeline (leader)
├── requirements.txt
├── sentiment_demo/sentiment_demo.py# module: sentiment_demo
├── eda/eda.py                  # module: eda
├── preprocessing/preprocessing.py# module: preprocessing
├── features/features.py        # module: features
├── model/model.py              # module: model
├── train/train.py              # module: train
├── evaluate/evaluate.py        # module: evaluate
├── raw_data/                # dataset files (NOT committed)
├── artifacts/               # splits/ (committed) | models, figures, logs, runs (NOT committed)
├── logbook/                 # bug reports and experiment notes (docs/templates)
└── tests/test_smoke.py      # interface contract + smoke tests
```

## How to run
```bash
pip install -r labs/lab3/requirements.txt
# put the dataset in labs/lab3/raw_data/ (see project_description.md)
pytest labs/lab3/tests          # interface + smoke tests
python -m labs.lab3.main        # full pipeline (after modules are wired)
```

---

**Tiếng Việt**

# Lab 3: Phân tích cảm xúc và fine-tuning với Hugging Face

## Báo cáo ngắn (khoảng 200 từ trở xuống)

**Bài toán.** Bài 1: chạy model cảm xúc có sẵn và xem tokenization. Bài 2: fine-tune transformer pre-trained (`distilbert-base-uncased`) cho phân loại cảm xúc nhị phân bằng Trainer.

**Dữ liệu.** IMDB (Maas và cộng sự, ACL 2011), nhãn nhị phân, tập con nhỏ seed 42: 2000 train, 500 validation, 500 test. Nhãn train cân bằng; độ dài lệch phải (trung vị 170 từ) và 40,1% review train đạt giới hạn 256 token.

**Phương pháp.** Tiền xử lý và tokenize xem ở phần Dữ liệu. Fine-tune bằng Trainer: 1 epoch, learning rate 2e-5, batch size 8, `max_length` 128, trên CPU; giữ epoch có F1 validation tốt nhất.

**Kết quả.** Pipeline cảm xúc có sẵn phân loại 3 câu mẫu là POSITIVE, NEGATIVE và POSITIVE (điểm trên 0,99), kể cả câu phủ định "Not bad at all". Trên cùng 500 review test, model đã fine-tune đạt accuracy 0,8220 và F1 0,8195; pipeline có sẵn (Bài 1) đạt accuracy 0,8920 và F1 0,8889.

**Kết luận.** Chênh lệch accuracy là -7,0 điểm %, lớn hơn sai số lấy mẫu (khoảng 1,7 điểm): pipeline có sẵn đạt điểm cao hơn trên tập test này. Một lần chạy, một seed; nguyên nhân chưa được kiểm chứng.

## Dữ liệu

- **Nguồn:** IMDB (Large Movie Review Dataset), Maas, Daly, Pham, Huang, Ng và Potts, ACL 2011. Trang chính thức: https://ai.stanford.edu/~amaas/data/sentiment/ ; Hugging Face Hub: `stanfordnlp/imdb`.
- **Kích thước:** 50.000 review có nhãn (25.000 train, 25.000 test). Dùng tập con nhỏ, nhẹ cho CPU: 2000 / 500 / 500 (train / validation / test), seed 42.
- **Chia tập:** dữ liệu trên Hub xếp theo nhãn nên mỗi tập gốc được xáo trộn theo seed trước khi lấy tập con (nếu không, tập con có thể chỉ có một nhãn). Validation cắt từ train gốc, test lấy từ test gốc nên ba tập con không trùng nhau.
- **Nhãn:** train {0: 1000, 1: 1000}, validation {0: 243, 1: 257}, test {0: 254, 1: 246}. Validation và test là mẫu ngẫu nhiên nên gần cân bằng chứ không tuyệt đối.
- **Làm sạch (`clean_text`):** tối giản. Giải mã ký tự HTML, thay thẻ HTML như `<br />` bằng khoảng trắng, gộp khoảng trắng. Không hạ chữ thường và giữ dấu câu: tokenizer uncased của DistilBERT tự hạ chữ thường và các dấu như "!" có thể mang thông tin cảm xúc.
- **Tokenize (`tokenize_dataset`):** tokenizer DistilBERT, cắt văn bản còn `max_length=256` token, không padding cố định (Trainer padding theo từng batch). Kết quả có các cột `input_ids`, `attention_mask` và `label`.
- **File:** tải tự động ở lần chạy đầu vào `labs/lab3/raw_data/` (không commit); lần đầu cần có mạng.
- **Giả định cần hỏi lại cô:** IMDB và `distilbert-base-uncased`.

### Khám phá dữ liệu (tập train, 2000 review)

| Thống kê (số từ mỗi review) | Giá trị |
|---|---|
| trung bình | 226,2 |
| trung vị | 170 |
| phân vị 95 | 585 |
| nhỏ nhất / lớn nhất | 12 / 999 |

Phân bố độ dài lệch phải (trung bình lớn hơn trung vị): một vài review rất dài kéo trung bình lên. Tính theo token, 802 trên 2000 review train (40,1%) đạt `max_length=256` và bị cắt (hoặc đúng 256 token). Ảnh hưởng của việc cắt này tới accuracy chưa được đo. `run_eda` cũng lưu ba biểu đồ (phân bố nhãn, histogram độ dài, độ dài theo nhãn) vào `labs/lab3/artifacts/figures/` (không commit).

## Bài 1: phân tích cảm xúc bằng model có sẵn

`analyze_sentences` chạy pipeline của Hugging Face với `distilbert-base-uncased-finetuned-sst-2-english` (DistilBERT đã fine-tune sẵn trên SST-2); bản `distilbert-base-uncased` gốc chưa có đầu phân loại cảm xúc nên không dùng được ở đây. `show_tokens` dùng tokenizer của cùng checkpoint.

Kết quả 3 câu mẫu xem bảng ở phần tiếng Anh phía trên. Token viết thường (tokenizer uncased); mỗi câu bắt đầu bằng `[CLS]` (id 101) và kết thúc bằng `[SEP]` (id 102). Câu thứ ba có phủ định và được đọc đúng, nhưng chỉ có 3 câu nên chưa thể kết luận model xử lý phủ định tốt nói chung.

## Bài 2: fine-tuning

**Thiết lập.** Model `distilbert-base-uncased` (thêm đầu phân loại 2 nhãn), `max_length` 128, 1 epoch, learning rate 2e-5, batch size 8, seed 42, thiết bị: CPU, thời gian huấn luyện: khoảng 17 phút. Trainer đánh giá trên validation sau mỗi epoch và giữ epoch có F1 tốt nhất (epoch 1).

| Model | Accuracy validation | F1 validation | Accuracy test | F1 test |
|---|---|---|---|---|
| Fine-tune (Bài 2) | 0,8340 | 0,8337 | 0,8220 | 0,8195 |
| Pipeline có sẵn (Bài 1) | không đo | không đo | 0,8920 | 0,8889 |

Baseline của Bài 1 được đo trên cùng 500 review test (positive -> 1, negative -> 0). Số liệu validation lấy từ epoch được chọn bằng chính tập validation nên hơi lạc quan; tập test chỉ dùng một lần ở cuối. Với 500 review, sai số chuẩn của accuracy khoảng 1,7 điểm %, nên chênh lệch dưới khoảng 2,5 đến 3 điểm nằm trong sai số lấy mẫu. Pipeline cắt review ở 512 token còn model fine-tune cắt ở 128 token. Một lần chạy, một seed, một tập con. Lưu ý: phần EDA và tokenize ở trên mô tả với `max_length=256`, nhưng do giới hạn RAM (7,8 GB, 2 nhân CPU) nên lần fine-tune thực tế dùng cấu hình nhỏ hơn nêu trong mục này.

## Cấu trúc thư mục

Xem cây thư mục ở phần tiếng Anh phía trên (tên module giống nhau).

## Cách chạy

Các lệnh giống phần tiếng Anh phía trên.
