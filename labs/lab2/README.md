# Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch)

## Brief report (about 200 words or fewer)
_To be written after the experiments: problem, data, method, results (accuracy, figures), conclusion._

## Data
ASSUMPTION (confirm with the lecturer): CIFAR-10 via `torchvision.datasets` (the slide uses it as the example).
Put the files in `labs/lab2/raw_data/` (this folder is not committed).

## Folder structure
```text
labs/lab2/
├── README.md                # brief report (EN + VI) / báo cáo ngắn
├── project_description.md   # Define + Design (feed this to the AI agent)
├── progress_status.md       # progress, audit, dated log / tiến độ
├── main.py                  # runs the whole pipeline (leader)
├── requirements.txt
├── eda/eda.py                  # module: eda
├── preprocessing/preprocessing.py# module: preprocessing
├── features/features.py        # module: features
├── model/model.py              # module: model
├── train/train.py              # module: train
├── evaluate/evaluate.py        # module: evaluate
├── experiment/experiment.py    # module: experiment
├── raw_data/                # dataset files (NOT committed)
├── artifacts/               # splits/ (committed) | models, figures, logs, runs (NOT committed)
├── logbook/                 # bug reports and experiment notes (docs/templates)
└── tests/test_smoke.py      # interface contract + smoke tests
```

## How to run
```bash
pip install -r labs/lab2/requirements.txt
# put the dataset in labs/lab2/raw_data/ (see project_description.md)
pytest labs/lab2/tests          # interface + smoke tests
python -m labs.lab2.main        # full pipeline (after modules are wired)
```

---

**Tiếng Việt**

# Lab 2: CNN pre-trained và fine-tuning trên CIFAR-10 (PyTorch)

## Báo cáo ngắn (khoảng 200 từ trở xuống)
_Viết sau khi chạy thí nghiệm: bài toán, dữ liệu, phương pháp, kết quả (accuracy, hình), kết luận._

## Dữ liệu
GIẢ ĐỊNH (cần hỏi lại cô): CIFAR-10 qua `torchvision.datasets` (slide dùng làm ví dụ).
Đặt file vào `labs/lab2/raw_data/` (thư mục này không được commit).

## Cấu trúc thư mục
Xem cây thư mục ở phần tiếng Anh phía trên (tên module giống nhau).

## Cách chạy
Các lệnh giống phần tiếng Anh. Đặt dataset vào `labs/lab2/raw_data/` (xem `project_description.md`).
