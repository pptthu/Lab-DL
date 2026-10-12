# Lab 3: Hugging Face sentiment analysis and fine-tuning: project description (Define + Design)

## Define
- **Problem:** Exercise 1: run a pre-trained sentiment model and inspect tokenization. Exercise 2: fine-tune a pre-trained transformer for binary text classification with the Trainer API.
- **Target users:** the lecturer (grading) and the group (learning).
- **Input / Output:** Input: sentences / a binary text dataset. Output: sentiment predictions, fine-tuned model, metrics (accuracy, F1).
- **Data:** ASSUMPTION (confirm with the lecturer): IMDB movie reviews (binary), small subsets; model `distilbert-base-uncased`.
- **Core features:**
  1. Ex1: use a pre-trained sentiment model from the Hugging Face Hub
  2. Ex1: tokenize a sample sentence
  3. Ex2: load a simple binary text dataset
  4. Ex2: load a pre-trained model and its tokenizer
  5. Ex2: preprocess the dataset for the model
  6. Ex2: define training arguments, create a Trainer and fine-tune
  7. Ex2: evaluate the fine-tuned model
- **Technology constraints:** transformers, datasets, evaluate, accelerate; internet is needed to download models and data. CPU: use small subsets and short max_length.
- **Evaluation criteria:** Deliverables are not stated on the Practice 3 slide: confirm with the lecturer (assumed: code, short report). Confirm also whether `.py` or notebooks are accepted. Oral Q&A.

## Design
- **Data flow:** Ex1: sentiment_demo ; Ex2: eda -> preprocessing -> model (load) -> features (tokenize) -> train (Trainer) -> evaluate
- **Modules and owners** (each module = one sub-folder with one script):
- Member A: `sentiment_demo`, `eda`, `preprocessing`, `features`
- Member B: `model`, `train`, `evaluate`
- The team leader owns `main.py`, `README.md`, `progress_status.md`, `AI_usage.md` and PR review.
- **Module applicability rule:** if a module is not needed in this lab, mark it `N/A` in `progress_status.md` with a one-line reason. Do not delete it silently.
- **Interfaces** (names and signatures are a contract; the test in `tests/test_smoke.py` checks the names):

| Module | Owner | Function / class | Input | Output |
|---|---|---|---|---|
| `sentiment_demo` | Member A | `show_tokens(sentence: str, model_name: str = None)` | sentence, model_name (None = pipeline default) | dict: tokens (list[str]), input_ids (list[int]) |
| `sentiment_demo` | Member A | `analyze_sentences(sentences: list, model_name: str = None)` | sentences, model_name | list of dicts: sentence, label, score |
| `eda` | Member A | `run_eda(dataset, out_dir: str, text_col: str = 'text', label_col: str = 'label')` | dataset (train split), out_dir, column names | dict: n_samples, label_counts, length_stats (words); PNG files (label bar chart, length histogram) |
| `preprocessing` | Member A | `load_text_dataset(name: str = 'imdb', n_train: int = 2000, n_val: int = 500, n_test: int = 500, seed: int = 42)` | name, subset sizes, seed | (train, val, test) dataset objects with columns text and label (0/1) |
| `preprocessing` | Member A | `clean_text(text: str)` | text | cleaned text |
| `features` | Member A | `tokenize_dataset(dataset, tokenizer, max_length: int = 256, text_col: str = 'text')` | dataset, tokenizer, max_length, text_col | tokenized dataset with input_ids, attention_mask and label |
| `model` | Member B | `load_model_and_tokenizer(model_name: str = 'distilbert-base-uncased', num_labels: int = 2)` | model_name, num_labels | (model, tokenizer) |
| `train` | Member B | `build_training_args(output_dir: str, epochs: int = 2, lr: float = 2e-5, batch_size: int = 16)` | output_dir, epochs, lr, batch_size | TrainingArguments object |
| `train` | Member B | `train_with_trainer(model, tokenized_train, tokenized_val, args, compute_metrics)` | model, tokenized train/val, args, compute_metrics | trained Trainer; model saved under args.output_dir |
| `evaluate` | Member B | `compute_metrics(eval_pred)` | eval_pred: (logits, labels) | dict: accuracy, f1 |
| `evaluate` | Member B | `evaluate_on_test(trainer, tokenized_test)` | trainer, tokenized_test | dict of metrics |
| `evaluate` | Member B | `predict_texts(model, tokenizer, texts: list)` | model, tokenizer, texts | list of dicts: text, label, score |

- **Folder structure:**
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
- **Milestones:**
  - M1: Ex1 prints tokens and sentiment for 3 sample sentences
  - M2: tokenized dataset has input_ids, attention_mask, label
  - M3: Trainer runs a tiny fine-tune (few steps) without error
  - M4: compute_metrics returns accuracy and F1 on validation
  - M5: full small run + test evaluation + short report
- **Technical risks:**
  - Tokenizer and model must come from the same checkpoint
  - Argument names of TrainingArguments differ between versions: check the installed version
  - Memory/time: small subsets, short max_length
  - Labels must be integers 0/1 and the column named as expected by the Trainer
  - No Hugging Face chapter in the course slides: rely on the links of the Practice 3 slide

---

**Tiếng Việt**

# Lab 3: Phân tích cảm xúc và fine-tuning với Hugging Face: mô tả dự án (Define + Design)

## Define
- **Bài toán:** Bài 1: chạy model cảm xúc có sẵn và xem tokenization. Bài 2: fine-tune transformer pre-trained cho phân loại văn bản nhị phân bằng Trainer.
- **Người dùng:** cô (chấm bài) và nhóm (học).
- **Đầu vào / Đầu ra:** Đầu vào: câu văn / dataset văn bản nhị phân. Đầu ra: dự đoán cảm xúc, model đã fine-tune, metric (accuracy, F1).
- **Dữ liệu:** GIẢ ĐỊNH (cần hỏi lại cô): IMDB (nhị phân), dùng tập con nhỏ; model `distilbert-base-uncased`.
- **Tính năng chính:**
  1. Bài 1: dùng model cảm xúc có sẵn trên Hugging Face Hub
  2. Bài 1: tokenize một câu mẫu
  3. Bài 2: nạp dataset văn bản nhị phân đơn giản
  4. Bài 2: nạp model pre-trained và tokenizer
  5. Bài 2: tiền xử lý dataset cho model
  6. Bài 2: định nghĩa training arguments, tạo Trainer và fine-tune
  7. Bài 2: đánh giá model đã fine-tune
- **Ràng buộc công nghệ:** transformers, datasets, evaluate, accelerate; cần internet để tải model và dữ liệu. CPU: dùng tập con nhỏ và max_length ngắn.
- **Tiêu chí đánh giá:** Slide Practice 3 không nêu sản phẩm nộp: hỏi lại cô (giả định: code, báo cáo ngắn). Hỏi thêm cô nhận `.py` hay notebook. Vấn đáp.

## Design
- **Luồng dữ liệu:** Ex1: sentiment_demo ; Ex2: eda -> preprocessing -> model (load) -> features (tokenize) -> train (Trainer) -> evaluate
- **Module và người phụ trách** (mỗi module = một sub-folder + một script):
- Member A: `sentiment_demo`, `eda`, `preprocessing`, `features`
- Member B: `model`, `train`, `evaluate`
- Nhóm trưởng phụ trách `main.py`, `README.md`, `progress_status.md`, `AI_usage.md` và duyệt PR.
- **Quy tắc linh hoạt module:** nếu lab này không cần một module, ghi `N/A` trong `progress_status.md` kèm lý do một dòng. Không xóa im lặng.
- **Interface** (tên hàm và chữ ký là hợp đồng; `tests/test_smoke.py` kiểm tra tên hàm):

| Module | Phụ trách | Hàm / lớp | Đầu vào | Đầu ra |
|---|---|---|---|---|
| `sentiment_demo` | Member A | `show_tokens(sentence: str, model_name: str = None)` | sentence, model_name (None = mặc định của pipeline) | dict: tokens (list[str]), input_ids (list[int]) |
| `sentiment_demo` | Member A | `analyze_sentences(sentences: list, model_name: str = None)` | sentences, model_name | danh sách dict: sentence, label, score |
| `eda` | Member A | `run_eda(dataset, out_dir: str, text_col: str = 'text', label_col: str = 'label')` | dataset (tập train), out_dir, tên cột | dict: n_samples, label_counts, length_stats (số từ); file PNG (biểu đồ nhãn, histogram độ dài) |
| `preprocessing` | Member A | `load_text_dataset(name: str = 'imdb', n_train: int = 2000, n_val: int = 500, n_test: int = 500, seed: int = 42)` | name, kích thước tập con, seed | (train, val, test) với cột text và label (0/1) |
| `preprocessing` | Member A | `clean_text(text: str)` | text | văn bản đã làm sạch |
| `features` | Member A | `tokenize_dataset(dataset, tokenizer, max_length: int = 256, text_col: str = 'text')` | dataset, tokenizer, max_length, text_col | dataset đã tokenize có input_ids, attention_mask và label |
| `model` | Member B | `load_model_and_tokenizer(model_name: str = 'distilbert-base-uncased', num_labels: int = 2)` | model_name, num_labels | (model, tokenizer) |
| `train` | Member B | `build_training_args(output_dir: str, epochs: int = 2, lr: float = 2e-5, batch_size: int = 16)` | output_dir, epochs, lr, batch_size | đối tượng TrainingArguments |
| `train` | Member B | `train_with_trainer(model, tokenized_train, tokenized_val, args, compute_metrics)` | model, tập train/val đã tokenize, args, compute_metrics | Trainer đã huấn luyện; model lưu trong args.output_dir |
| `evaluate` | Member B | `compute_metrics(eval_pred)` | eval_pred: (logits, labels) | dict: accuracy, f1 |
| `evaluate` | Member B | `evaluate_on_test(trainer, tokenized_test)` | trainer, tokenized_test | dict các metric |
| `evaluate` | Member B | `predict_texts(model, tokenizer, texts: list)` | model, tokenizer, texts | danh sách dict: text, label, score |

- **Mốc (milestone):**
  - M1: Bài 1 in token và cảm xúc của 3 câu mẫu
  - M2: dataset đã tokenize có input_ids, attention_mask, label
  - M3: Trainer chạy fine-tune rất ngắn (vài bước) không lỗi
  - M4: compute_metrics trả accuracy và F1 trên validation
  - M5: chạy nhỏ đầy đủ + đánh giá test + báo cáo ngắn
- **Rủi ro kỹ thuật:**
  - Tokenizer và model phải cùng một checkpoint
  - Tên tham số của TrainingArguments khác nhau giữa các phiên bản: kiểm tra phiên bản đã cài
  - Bộ nhớ/thời gian: dùng tập con nhỏ, max_length ngắn
  - Nhãn phải là số nguyên 0/1 và tên cột đúng như Trainer mong đợi
  - Slide môn học không có chương Hugging Face: dựa vào các link trong slide Practice 3
