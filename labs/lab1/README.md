# Lab 1: Phân loại Fashion-MNIST bằng MLP (PyTorch)

- **Created**: 2026-10-12T10:27:18+07:00 (pptthu)
- **Last Updated**: 2026-11-14T00:36:56+07:00 (pptthu)

> Kết quả trong file này là lần chạy ngày 04/11/2026 theo yêu cầu cũ (5 epoch, 7 cấu hình × 5 seed). Nhóm sẽ chạy lại theo desc mới [`docs/lab1_description_v1.md`](docs/lab1_description_v1.md) (5 cấu hình, seed 42, tối đa 30 epoch + early stopping) và cập nhật lại báo cáo này.

## Báo cáo ngắn
**Bài toán.** Phân loại ảnh quần áo xám 28x28 thành 10 lớp bằng mạng nơ-ron nhiều lớp (MLP).

**Dữ liệu.** Fashion-MNIST chính thức từ GitHub của Zalando, ghi ra CSV: 60.000 train, 10.000 test, chia 90/10 theo từng lớp (seed 42) thành 54.000 train / 6.000 validation.
Chuẩn hoá chỉ tính trên train. Tập test là 10.000 ảnh test chính thức.

**Mô hình và huấn luyện.** MLP 784 -> 256 -> 128 -> 10 (235.146 tham số), ReLU, CrossEntropyLoss.
Adam (lr 1e-3), batch 64, 5 epoch; giữ checkpoint có validation loss thấp nhất.

**Kết quả.** Accuracy test qua 5 seed: 88,00 ± 0,39%. Mô hình đã lưu (seed 42): 88,36%.
Shirt là lớp khó nhất (recall thấp nhất ở 4/5 seed, trung bình 69,97%).

**Thí nghiệm.** Bảy cấu hình, mỗi cấu hình 5 seed, mỗi cặp đổi một yếu tố. Tác động rõ lên validation accuracy:
bỏ lớp ẩn (-3,80 điểm), SGD thay Adam ở lr 1e-3 (-3,45), SGD lr 0,01 thay 1e-3 (+3,14).
Kích thước mạng, dropout, weight decay chưa có tác động rõ trong 5 epoch.
Ở 20 epoch, baseline overfit; dropout giữ val loss cuối thấp hơn nhưng chưa làm tăng accuracy rõ rệt.

**Hình.** Hiển thị trong notebook, lưu ở `artifacts/figures/`. Đường cong theo epoch của cả 45 lần chạy: notebook Stage 9b và 10b, từ `artifacts/runs/epoch_history.csv` và `extended_epoch_history.csv`. Chi tiết: xem Báo cáo chi tiết.

## Báo cáo chi tiết

### 1. Câu hỏi nghiên cứu
1. MLP nhỏ đạt accuracy bao nhiêu, và dao động giữa các seed thế nào?
2. Lớp ẩn, optimizer, learning rate, kích thước mạng, dropout, weight decay có tác động đo được không?
3. Lỗi tập trung ở đâu, và có ổn định qua các seed không?
4. Huấn luyện lâu hơn có làm đổi kết luận của 5 epoch không?

### 2. Dữ liệu
- File chính thức của Zalando (`zalandoresearch/fashion-mnist`), tải bằng `python -m labs.lab1.preprocessing.download_official` (torchvision, tự kiểm MD5) và ghi ra CSV: 10 lớp cân bằng, không thiếu giá trị, 0 dòng trùng.
- Một cách chia phân tầng (seed 42), dùng cho mọi lần chạy. Mean 0,2859 và std 0,3530 chỉ tính trên train.

### 3. Phương pháp
- Baseline: MLP 256-128, Adam, lr 1e-3, batch 64, 5 epoch, chốt trước khi thí nghiệm. Giữ checkpoint có validation loss thấp nhất.
- Mỗi cặp cấu hình chỉ khác một yếu tố và dùng chung seed 42, 7, 123, 2024, 3407. SGD luôn có momentum 0,9.
- Quy tắc chốt trước khi đọc kết quả: khác biệt rõ khi cả 5 seed cùng chiều và |chênh lệch trung bình| ≥ 2 std. Hai cấu hình của một cặp dùng chung 5 seed.
- So sánh chỉ dùng validation. Tập test chỉ báo cho baseline đã chốt trước.

### 4. Kết quả
| So sánh (validation accuracy) | Chênh lệch trung bình (điểm) | Độ lệch chuẩn của chênh lệch | Số seed cao hơn (trên 5) | Kết luận |
|---|---|---|---|---|
| Bỏ lớp ẩn so với baseline | -3,80 | 0,71 | 0 | Rõ |
| SGD so với Adam, lr 1e-3 | -3,45 | 0,33 | 0 | Rõ |
| SGD lr 0,01 so với 1e-3 | +3,14 | 0,17 | 5 | Rõ |
| Mạng lớn hơn so với baseline | -0,52 | 0,40 | 0 | Chưa rõ |
| Dropout 0,2 so với baseline | -0,46 | 0,35 | 1 | Chưa rõ |
| Weight decay 1e-4 so với baseline | -0,29 | 0,17 | 0 | Chưa rõ |

| Baseline qua 5 seed | Train | Validation | Test |
|---|---|---|---|
| 5 epoch | 90,92% | 88,62 ± 0,36% | 88,00 ± 0,39% |
| 20 epoch (checkpoint tốt nhất) | 92,95% | 89,22 ± 0,26% | 88,62 ± 0,19% |

| 20 epoch, 5 seed | Epoch tốt nhất | Val loss thấp nhất | Val loss cuối |
|---|---|---|---|
| Baseline | 7 đến 10 | 0,2988 | 0,3972 |
| Dropout 0,2 | 9 đến 20 | 0,2916 | 0,3095 |

Shirt có recall thấp nhất ở 4/5 seed với 5 epoch (trung bình 69,97%) và 5/5 seed với 20 epoch (trung bình 69,03%). Các cặp nhầm nhiều nhất qua các seed ở 5 epoch: Shirt -> T-shirt/top, T-shirt/top -> Shirt và Pullover -> Shirt.

### 5. Thảo luận
- Ba tác động rõ đều lớn hơn nhiều so với dao động giữa các seed. Lớp ẩn thêm khoảng 3,80 điểm so với mô hình tuyến tính; mô hình tuyến tính đứng quanh 84% từ epoch 2 và train loss chững ở khoảng 0,41 (Stage 9b).
- Ở lr 1e-3, SGD có momentum học quá chậm trong 5 epoch: vẫn đang tăng ở epoch 5 (Stage 9b). lr 0,01 kéo lại phần lớn khoảng cách, nên kết quả về optimizer phụ thuộc learning rate.
- Mạng lớn hơn, dropout và weight decay đều thấp hơn baseline một chút, nhưng không cặp nào qua quy tắc (chênh lệch nhỏ hơn 2 lần độ lệch chuẩn), nên không kết luận có tác động.
- Ở 20 epoch, baseline overfit: val loss trung bình thấp nhất ở epoch 7 (0,3028) rồi tăng lên 0,3972, trong khi train loss vẫn giảm xuống 0,1240 (Stage 10b). Nhờ chọn checkpoint tốt nhất, accuracy vẫn gần 89%.
- Dropout chưa làm tăng val accuracy rõ rệt ở 20 epoch (+0,36 điểm, 3/5 seed cao hơn), nhưng giữ val loss cuối thấp hơn (0,3095 so với 0,3972, thấp hơn ở cả 5 seed).
- 20 so với 5 epoch: +0,60 điểm trên validation, 4/5 seed cao hơn, nên chưa qua quy tắc.

### 6. Giới hạn
- Một cách chia, một máy, 5 seed. Không qua quy tắc không chứng minh là không có tác động.
- Tập test đã được xem nhiều lần khi phát triển, dù không dùng để chọn gì.
- Nguyên nhân lỗi Shirt chưa kiểm chứng; độ phân giải thấp và việc MLP không khai thác cấu trúc lân cận của ảnh vẫn là giả thuyết.
- Phạm vi: chỉ làm MLP. CNN không thuộc lab này (quyết định của nhóm, 04/11/2026).

### 7. Kết luận
MLP tốt hơn rõ so với mô hình tuyến tính, đạt 88,00 ± 0,39% trên tập test chính thức sau 5 epoch. Lớp ẩn, optimizer và learning rate có tác động rõ; kích thước mạng, dropout, weight decay chưa cho thấy tác động lên accuracy. Lỗi tập trung ở nhóm áo, và huấn luyện lâu hơn gây overfit mà chưa chứng minh được lợi ích về accuracy.

### Tài liệu tham khảo
1. H. Xiao, K. Rasul, R. Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.
2. D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980, 2014.
3. N. Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 15, 2014.
4. Tài liệu PyTorch: https://pytorch.org/docs/stable/

## Dữ liệu
Fashion-MNIST chính thức từ GitHub của Zalando (https://github.com/zalandoresearch/fashion-mnist). Chạy `python -m labs.lab1.preprocessing.download_official`: lệnh này tải 4 file IDX bằng torchvision (tự kiểm MD5) và ghi `fashion-mnist_train.csv`, `fashion-mnist_test.csv` (cột `label` + 784 cột pixel) vào `labs/lab1/raw_data/` (thư mục này không commit).

## Cấu trúc thư mục
```text
labs/lab1/
├── README.md                # báo cáo ngắn
├── project_description.md   # Define + Design (tổng quan)
├── progress_status.md       # tiến độ, kiểm chứng, nhật ký theo ngày
├── main.py                  # chạy toàn bộ pipeline
├── extended_study.py        # thí nghiệm 20 epoch
├── requirements.txt
├── eda/eda.py                          # module: eda
├── preprocessing/preprocessing.py      # module: preprocessing
├── preprocessing/download_official.py  # tải dữ liệu chính thức, ghi ra CSV
├── features/features.py                # module: features
├── model/model.py                      # module: model
├── train/train.py                      # module: train
├── evaluate/evaluate.py                # module: evaluate
├── visualize/visualize.py              # module: visualize
├── experiment/experiment.py            # module: experiment
├── notebook/lab1_presentation.ipynb    # notebook trình bày
├── raw_data/                # file dữ liệu (KHÔNG commit)
├── artifacts/               # splits/ (có commit) | models, figures, logs, runs (KHÔNG commit)
├── docs/                    # tài liệu Lab 1, đọc docs/00_README_v2.md trước
│   ├── lab1_description_v1.md   # desc hiện hành
│   ├── rules/  templates/  plans/
│   ├── phases/              # desc từng module (<module>_v1.md)
│   └── experiments/         # ghi chép thí nghiệm (trước là logbook/)
└── tests/                   # test hợp đồng interface + test từng module (dữ liệu giả)
```
Cây thư mục đích sau khi đổi `artifacts/`, `raw_data/` sang `outputs/`, `data/`: xem [`docs/rules/FOLDER_STRUCTURE_v2.md`](docs/rules/FOLDER_STRUCTURE_v2.md).

## Desc từng module
Mỗi module có một file trong [`docs/phases/`](docs/phases/): `eda_v1.md`, `preprocessing_v1.md`, `features_v1.md`, `model_v1.md`, `train_v1.md`, `evaluate_v1.md`, `visualize_v1.md`, `experiment_v1.md`, `demo_v1.md`.

## Cách chạy
```bash
pip install -r labs/lab1/requirements.txt
python -m pytest labs/lab1/tests        # test trên dữ liệu giả
python -m labs.lab1.main                # toàn bộ pipeline (cần CSV trong raw_data/)
python -m labs.lab1.extended_study      # thí nghiệm 20 epoch, chạy sau main.py
```
Notebook trình bày: chạy `main.py` và `extended_study.py` trước (notebook đọc kết quả của chúng), rồi mở `labs/lab1/notebook/lab1_presentation.ipynb` và chạy tất cả các ô.

Môi trường của kết quả báo cáo (04/11/2026, dữ liệu chính thức): Python 3.13.7, torch 2.14.1 (CPU), 4 thread. Giữ nguyên `artifacts/splits/train_val_split.json`. Máy khác có thể ra số hơi khác.
