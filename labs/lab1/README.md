# Lab 1: Fashion-MNIST classification with an MLP (PyTorch)

## Brief report (about 200 words or fewer)
_To be written after the experiments: problem, data, method, results (accuracy, figures), conclusion._

## Data
Kaggle `zalando-research/fashionmnist` (CSV: `label` + 784 pixel columns, 60,000 train rows). Check the real file names after download.
Put the files in `labs/lab1/raw_data/` (this folder is not committed).

## Folder structure
```text
labs/lab1/
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
├── visualize/visualize.py      # module: visualize
├── experiment/experiment.py    # module: experiment
├── raw_data/                # dataset files (NOT committed)
├── artifacts/               # splits/ (committed) | models, figures, logs, runs (NOT committed)
├── logbook/                 # bug reports and experiment notes (docs/templates)
└── tests/test_smoke.py      # interface contract + smoke tests
```

## How to run
```bash
pip install -r labs/lab1/requirements.txt
# put the dataset in labs/lab1/raw_data/ (see project_description.md)
pytest labs/lab1/tests          # interface + smoke tests
python -m labs.lab1.main        # full pipeline (after modules are wired)
```

---

**Tiếng Việt**

# Lab 1: Phân loại Fashion-MNIST bằng MLP (PyTorch)

## Báo cáo ngắn (khoảng 200 từ trở xuống)
_Viết sau khi chạy thí nghiệm: bài toán, dữ liệu, phương pháp, kết quả (accuracy, hình), kết luận._

## Dữ liệu
Kaggle `zalando-research/fashionmnist` (CSV: cột `label` + 784 cột pixel, 60.000 dòng train). Kiểm tra tên file thật sau khi tải.
Đặt file vào `labs/lab1/raw_data/` (thư mục này không được commit).

## Cấu trúc thư mục
Xem cây thư mục ở phần tiếng Anh phía trên (tên module giống nhau).

## Cách chạy
Các lệnh giống phần tiếng Anh. Đặt dataset vào `labs/lab1/raw_data/` (xem `project_description.md`).
