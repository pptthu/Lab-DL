# Lab 1: Fashion-MNIST classification with an MLP (PyTorch)

## Brief report
**Task.** Classify 28x28 grayscale clothing images into 10 classes with a multilayer perceptron (MLP).

**Data.** Kaggle Fashion-MNIST CSVs: 60,000 train, 10,000 test. Stratified 90/10 split of the training file (seed 42):
54,000 train / 6,000 validation (600 per class). Normalization (mean 0.2860, std 0.3527) is computed on the training split only.

**Model and training.** MLP 784 -> 256 -> 128 -> 10 (235,146 parameters), ReLU, outputs logits for CrossEntropyLoss.
Adam (lr 1e-3), batch size 64, 5 epochs; the checkpoint with the lowest validation loss is kept (epoch 4).

**Results.** Validation accuracy 88.77% (loss 0.3194); test accuracy 88.32% (loss 0.3212, 8,832/10,000).
Shirt is the hardest class (recall 71.00%); its errors go mostly to Pullover, Coat and T-shirt/top.

**Experiments.** Four configurations (baseline, SGD, larger network, dropout) reached 88.10-89.05% best validation accuracy,
one seed each. The same baseline gave 88.87% and 89.05% in two runs, so these differences are within run-to-run noise:
no configuration is shown to be better.

**Figures.** Loss curves, predicted-vs-actual grid and confusion matrix: shown in the notebook and saved by `main.py` to
`artifacts/figures/` (not committed).

## Data
Kaggle `zalando-research/fashionmnist`: `fashion-mnist_train.csv` and `fashion-mnist_test.csv` (`label` + 784 pixel columns).
Put them in `labs/lab1/raw_data/` (this folder is not committed).

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
├── notebook/lab1_presentation.ipynb # presentation notebook
├── raw_data/                # dataset files (NOT committed)
├── artifacts/               # splits/ (committed) | models, figures, logs, runs (NOT committed)
├── logbook/                 # bug reports and experiment notes (docs/templates)
└── tests/                   # interface contract + module tests (fake data)
```

## How to run
```bash
pip install -r labs/lab1/requirements.txt
python -m pytest labs/lab1/tests        # tests on fake data
python -m labs.lab1.main                # full pipeline (needs the CSVs in raw_data/)
```
Presentation: open `labs/lab1/notebook/lab1_presentation.ipynb` and run all cells.

---

**Tiếng Việt**

# Lab 1: Phân loại Fashion-MNIST bằng MLP (PyTorch)

## Báo cáo ngắn
**Bài toán.** Phân loại ảnh quần áo xám 28x28 thành 10 lớp bằng mạng nơ-ron nhiều lớp (MLP).

**Dữ liệu.** File CSV Fashion-MNIST trên Kaggle: 60.000 train, 10.000 test. Chia phân tầng 90/10 file train (seed 42):
54.000 train / 6.000 validation (600 mẫu mỗi lớp). Chuẩn hóa (mean 0,2860, std 0,3527) chỉ tính trên tập train.

**Mô hình và huấn luyện.** MLP 784 -> 256 -> 128 -> 10 (235.146 tham số), ReLU, xuất logits cho CrossEntropyLoss.
Adam (lr 1e-3), batch 64, 5 epoch; giữ checkpoint có validation loss thấp nhất (epoch 4).

**Kết quả.** Accuracy validation 88,77% (loss 0,3194); test 88,32% (loss 0,3212, 8.832/10.000).
Shirt là lớp khó nhất (recall 71,00%); lỗi chủ yếu bị nhầm sang Pullover, Coat và T-shirt/top.

**Thí nghiệm.** Bốn cấu hình (baseline, SGD, mạng lớn hơn, dropout) đạt accuracy validation tốt nhất 88,10-89,05%,
mỗi cấu hình một seed. Cùng cấu hình baseline cho 88,87% và 89,05% ở hai lần chạy, nên các chênh lệch này nằm trong
sai số giữa các lần chạy: chưa cấu hình nào được chứng minh là tốt hơn.

**Hình.** Đường loss, lưới ảnh dự đoán so với thực tế và ma trận nhầm lẫn: hiển thị trong notebook và được `main.py` lưu vào
`artifacts/figures/` (không commit).

## Dữ liệu
Kaggle `zalando-research/fashionmnist`: `fashion-mnist_train.csv` và `fashion-mnist_test.csv` (cột `label` + 784 cột pixel).
Đặt vào `labs/lab1/raw_data/` (thư mục này không được commit).

## Cấu trúc thư mục
Xem cây thư mục ở phần tiếng Anh phía trên (tên module giống nhau).

## Cách chạy
Các lệnh giống phần tiếng Anh. Notebook trình bày: mở `labs/lab1/notebook/lab1_presentation.ipynb` và chạy tất cả các ô.
