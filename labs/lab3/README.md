# Lab 3: Hugging Face sentiment analysis and fine-tuning

## Brief report (about 200 words or fewer)
_To be written after the experiments: problem, data, method, results (accuracy, figures), conclusion._

## Data
ASSUMPTION (confirm with the lecturer): IMDB movie reviews (binary), small subsets; model `distilbert-base-uncased`.
Put the files in `labs/lab3/raw_data/` (this folder is not committed).

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
_Viết sau khi chạy thí nghiệm: bài toán, dữ liệu, phương pháp, kết quả (accuracy, hình), kết luận._

## Dữ liệu
GIẢ ĐỊNH (cần hỏi lại cô): IMDB (nhị phân), dùng tập con nhỏ; model `distilbert-base-uncased`.
Đặt file vào `labs/lab3/raw_data/` (thư mục này không được commit).

## Cấu trúc thư mục
Xem cây thư mục ở phần tiếng Anh phía trên (tên module giống nhau).

## Cách chạy
Các lệnh giống phần tiếng Anh. Đặt dataset vào `labs/lab3/raw_data/` (xem `project_description.md`).
