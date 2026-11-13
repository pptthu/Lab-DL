# Lab 1: Phân loại Fashion-MNIST bằng MLP (PyTorch)

- **Created**: 2026-10-12T10:27:18+07:00 (pptthu)
- **Last Updated**: 2026-11-14T01:45:00+07:00 (pptthu)

> Số liệu trong file này lấy từ normal run ngày 14/11/2026 theo desc [`docs/lab1_description_v1.md`](docs/lab1_description_v1.md): `python -m labs.lab1.main`, tóm tắt ở `outputs/results/run_summary.json`. Kết quả cũ (04/11, 5 epoch, 7 cấu hình × 5 seed) ghi ở `progress_status.md`, phần lịch sử.

## Báo cáo ngắn
**Bài toán.** Phân loại ảnh quần áo xám 28×28 thành 10 lớp bằng mạng nơ-ron nhiều lớp (MLP).

**Dữ liệu.** Fashion-MNIST chính thức của Zalando: 60.000 train, 10.000 test. Tách 10% train làm validation, phân tầng theo lớp, seed 42 → 54.000 train / 6.000 val. Chuẩn hoá bằng mean/std của phần train.

**Huấn luyện.** 5 cấu hình, mỗi cấu hình khác baseline đúng một yếu tố; seed 42, batch 64, CrossEntropyLoss. Tối đa 30 epoch, dừng sớm khi `val_loss` không giảm ít nhất 1e-4 trong 5 epoch; giữ checkpoint có `val_loss` thấp nhất. Mỗi run ghi 3 file log (`.log`, `_history.csv`, `_config.json`).

**Kết quả.** Baseline (784→256→128→10, Adam lr 1e-3) đạt **88,60%** trên 10.000 ảnh test, vượt mức 87% của MLP trong bài báo gốc Fashion-MNIST. Cấu hình được chọn trên validation là mạng lớn hơn (89,28% val), test 88,46%. Cả 5 cấu hình đều dừng sớm ở epoch 11–15. Lớp khó nhất là Shirt (recall 68,67% trên val).

| Cấu hình | Tham số | best / dừng | val_loss | val_acc | train − val (điểm) |
|---|---:|:---:|---:|---:|---:|
| baseline | 235.146 | 8 / 13 | 0,2943 | 88,97% | 2,59 |
| softmax_regression | 7.850 | 10 / 15 | 0,4243 | 85,15% | 1,07 |
| larger_network | 567.434 | 7 / 12 | 0,3009 | **89,28%** | 1,70 |
| sgd_lr_0.01 | 235.146 | 10 / 15 | 0,2852 | 89,13% | 2,89 |
| dropout_0.2 | 235.146 | 6 / 11 | 0,2996 | 89,00% | −0,24 |

| Model (test chạy một lần) | val_acc | test_acc | test_loss |
|---|---:|---:|---:|
| baseline | 88,97% | **88,60%** (≥ 87% ✔) | 0,3239 |
| larger_network (chọn trên val) | 89,28% | 88,46% | 0,3315 |

**Nhận xét.**
- MLP hơn mô hình tuyến tính (softmax_regression) 3,8 điểm val_acc, cùng chiều với bài báo gốc (0,871 so với 0,842).
- Sau best_epoch, train loss vẫn giảm còn val loss tăng: model bắt đầu overfit, nên dừng sớm và giữ `_best.pt` là cần thiết.
- Ba cấu hình larger_network, sgd_lr_0.01, dropout_0.2 chỉ hơn baseline 0,03–0,31 điểm val với một seed; trên test, mạng lớn hơn còn thấp hơn baseline. Chênh lệch nhỏ như vậy chưa đủ để kết luận cấu hình nào tốt hơn.
- Dropout 0,2 đưa khoảng cách train − val về gần 0 nhưng không tăng val_acc rõ rệt.
- Shirt hay bị nhầm với T-shirt/top (71 ảnh), Pullover (56), Coat (37) trên 600 ảnh val: các lớp áo có hình dáng gần nhau, và MLP làm phẳng ảnh nên bỏ qua cấu trúc không gian.

**Hình và bảng.** Notebook [`notebook/lab1_presentation.ipynb`](notebook/lab1_presentation.ipynb) đọc từ `outputs/`: loss/accuracy curve của 5 run (vẽ từ `_history.csv`), confusion matrix, báo cáo từng lớp, ảnh dự đoán 16 ảnh test. Sơ đồ kiến trúc: [`docs/models/mlp_baseline_v1.md`](docs/models/mlp_baseline_v1.md).

**Giới hạn.** Một seed, một cách chia, một máy CPU; tập test đã được xem ở lần chạy cũ nhưng không dùng để chọn gì. Không làm CNN, Optuna, đa seed, kiểm định thống kê (desc lab, phần "Không làm").

### Tài liệu tham khảo
1. H. Xiao, K. Rasul, R. Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.
2. D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980, 2014.
3. N. Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 15, 2014.
4. Tài liệu PyTorch: https://pytorch.org/docs/stable/

## Dữ liệu
Fashion-MNIST chính thức từ GitHub của Zalando (https://github.com/zalandoresearch/fashion-mnist). Chạy `python -m labs.lab1.preprocessing.download_official`: lệnh này tải 4 file IDX bằng torchvision (tự kiểm MD5) và ghi `fashion-mnist_train.csv`, `fashion-mnist_test.csv` (cột `label` + 784 cột pixel) vào `labs/lab1/data/raw/` (thư mục này không commit).

## Cấu trúc thư mục
```text
labs/lab1/
├── README.md                # báo cáo ngắn (file này)
├── project_description.md   # tổng quan Define + Design
├── progress_status.md       # tiến độ, kiểm chứng, nhật ký theo ngày
├── main.py                  # chạy toàn bộ pipeline (--smoke để chạy thử)
├── requirements.txt
├── eda/ preprocessing/ features/ model/ train/ evaluate/ visualize/ experiment/   # 8 module, mỗi folder 1 script
├── notebook/lab1_presentation.ipynb    # notebook trình bày (chỉ đọc outputs/)
├── data/                    # raw/ (KHÔNG commit) | splits/ (có commit)
├── outputs/                 # models, logs, figures, results, demo, _smoke (KHÔNG commit)
├── docs/                    # tài liệu Lab 1, đọc docs/00_README_v2.md trước
│   ├── lab1_description_v1.md   # desc hiện hành
│   ├── rules/  templates/  plans/
│   ├── phases/              # desc từng module (<module>_v1.md)
│   ├── models/              # model card (mlp_baseline_v1.md, mlp_larger_network_v1.md)
│   └── experiments/         # ghi chép thí nghiệm
└── tests/                   # test hợp đồng interface + test từng module (dữ liệu giả)
```
Giải thích cây thư mục: xem [`docs/rules/FOLDER_STRUCTURE_v2.md`](docs/rules/FOLDER_STRUCTURE_v2.md).

## Desc từng module
Mỗi module có một file trong [`docs/phases/`](docs/phases/): `eda_v1.md`, `preprocessing_v1.md`, `features_v1.md`, `model_v1.md`, `train_v1.md`, `evaluate_v1.md`, `visualize_v1.md`, `experiment_v1.md`, `demo_v1.md`. Mỗi file có mục "Kết quả audit (2026-11-14)".

## Cách chạy
```bash
pip install -r labs/lab1/requirements.txt
python -m pytest labs/lab1/tests tests       # test trên dữ liệu giả
python -m labs.lab1.main --smoke             # smoke run: 1 epoch, 2 batch, ghi vào outputs/_smoke/ (khoảng 20 giây)
python -m labs.lab1.main                     # normal run: 5 cấu hình (khoảng 7 phút trên CPU 4 thread)
```
Notebook trình bày: chạy `main.py` trước (notebook đọc `outputs/results/run_summary.json`), rồi mở `labs/lab1/notebook/lab1_presentation.ipynb` và chạy tất cả các ô. Notebook không train; chạy hết khoảng 30 giây.

Môi trường của kết quả báo cáo (14/11/2026): Python 3.13.7, torch 2.14.1 (CPU), 4 thread. Giữ nguyên `data/splits/train_val_split.json`. Máy khác có thể ra số hơi khác.
