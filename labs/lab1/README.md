# Lab 1: Fashion-MNIST classification with an MLP (PyTorch)

## Brief report
**Task.** Classify 28x28 grayscale clothing images into 10 classes with a multilayer perceptron (MLP).

**Data.** Kaggle Fashion-MNIST CSVs: 60,000 train, 10,000 test, split 90/10 by class (seed 42) into 54,000 train / 6,000 validation.
Normalization uses the training split only. **Limitation:** these CSVs differ from the official split, so scores are not comparable with published results.

**Model and training.** MLP 784 -> 256 -> 128 -> 10 (235,146 parameters), ReLU, CrossEntropyLoss.
Adam (lr 1e-3), batch size 64, 5 epochs; the checkpoint with the lowest validation loss is kept.

**Results.** Test accuracy across 5 seeds: 88.42 +/- 0.37%. Saved model (seed 42): 88.97%.
Shirt is the hardest class (lowest recall in 4 of 5 seeds, mean 68.07%).

**Experiments.** Seven configurations, 5 seeds each, changing one factor per pair. Clear effects on validation accuracy:
no hidden layers (-3.75 points), SGD instead of Adam at lr 1e-3 (-3.33), SGD lr 0.01 instead of 1e-3 (+2.81).
Network size, dropout and weight decay show no significant effect within 5 epochs.
At 20 epochs the baseline overfits; dropout limits that but does not raise accuracy.

**Figures.** Shown in the notebook and saved in `artifacts/figures/`. Details: see Detailed report.

## Detailed report

### 1. Research questions
1. What accuracy does a small MLP reach, and how much does it vary between seeds?
2. Do hidden layers, optimizer, learning rate, network size, dropout and weight decay have a measurable effect?
3. Where do the errors concentrate, and is that stable across seeds?
4. Does longer training change the 5-epoch findings?

### 2. Data
- Kaggle `zalando-research/fashionmnist`: 10 balanced classes, no missing values, 43 duplicate rows kept (team decision).
- One stratified split (seed 42), used by every run. Mean 0.2860 and std 0.3527 come from the training split only.
- **Comparison with the official files:** of the 9,999 distinct images in the Kaggle test file, 7,918 come from the official training set, 1,291 from the official test set, and about 790 have no exact match. Of the 59,957 distinct images in the Kaggle train file, 47,259 come from the official training set, 7,885 from the official test set, and about 4,800 have no exact match. The cause is unknown.
- Identical images remain across our splits: train-val 9, train-test 9, val-test 1.

### 3. Method
- Baseline: MLP 256-128, Adam, lr 1e-3, batch 64, 5 epochs, fixed before the experiments. The checkpoint with the lowest validation loss is kept.
- Each pair of configurations differs in one factor and shares seeds 42, 7, 123, 2024, 3407. SGD always uses momentum 0.9.
- Rule fixed before reading the results: a pair shows a clear difference if all 5 seeds point the same way and |mean difference| >= 2 std. Second check: paired t-test with Holm correction [5] and a 95% confidence interval.
- Comparisons use validation data only. The test set is reported only for baselines fixed beforehand.

### 4. Results
| Comparison (validation accuracy) | Mean difference (points) | 95% CI | p (Holm) | Result |
|---|---|---|---|---|
| No hidden layers vs baseline | -3.75 | -4.31 to -3.19 | 0.0003 | Clear |
| SGD vs Adam, lr 1e-3 | -3.33 | -3.75 to -2.91 | 0.0002 | Clear |
| SGD lr 0.01 vs 1e-3 | +2.81 | 2.35 to 3.28 | 0.0003 | Clear |
| Larger network vs baseline | -0.34 | -0.93 to 0.26 | 0.2755 | Not significant |
| Dropout 0.2 vs baseline | -0.68 | -1.25 to -0.10 | 0.0927 | Not significant |
| Weight decay 1e-4 vs baseline | -0.30 | -0.76 to 0.15 | 0.2755 | Not significant |

| Baseline across 5 seeds | Train | Validation | Test |
|---|---|---|---|
| 5 epochs | 90.60% | 88.82 +/- 0.39% | 88.42 +/- 0.37% |
| 20 epochs (best checkpoint) | 92.35% | 89.29 +/- 0.22% | 89.11 +/- 0.17% |

| 20 epochs, 5 seeds | Best epoch | Lowest val loss | Final val loss |
|---|---|---|---|
| Baseline | 5 to 10 | 0.3055 | 0.4020 |
| Dropout 0.2 | 10 to 18 | 0.2999 | 0.3101 |

Shirt has the lowest recall in 4 of 5 seeds at 5 epochs (mean 68.07%) and 5 of 5 at 20 epochs (mean 70.37%). The most frequent confusions across seeds are Shirt -> T-shirt/top, Pullover -> Coat and T-shirt/top -> Shirt.

### 5. Discussion
- The three clear effects are large compared with the variation between seeds. Hidden layers add about 3.75 points over a linear model.
- At lr 1e-3, SGD with momentum learns too slowly in 5 epochs; lr 0.01 recovers most of that gap, so the optimizer result depends on the learning rate.
- At 20 epochs the baseline overfits: val loss ends far above its lowest value. Best-checkpoint selection keeps accuracy near 89%.
- Dropout does not raise val accuracy at 20 epochs (+0.08 points, p = 0.71), but it holds the final val loss near its lowest value.
- 20 versus 5 epochs: +0.46 points on validation, p (Holm) = 0.051, so the gain is not established.

### 6. Limitations
- One split, one machine, 5 seeds. Non-significant results are not proof of no effect.
- The Kaggle CSVs differ from the official split, so no comparison with published results is made.
- The test set was seen several times during development, though no choice used it.
- The causes of the Shirt errors are not tested; low resolution and the lack of built-in local structure in an MLP remain hypotheses.

### 7. Conclusion
The MLP learns the task well above a linear model, reaching 88.42 +/- 0.37% on test in 5 epochs. Hidden layers, optimizer and learning rate have clear effects; network size, dropout and weight decay do not show one on accuracy here. Errors concentrate in upper-body garments, and longer training overfits without a proven accuracy gain.

### References
1. H. Xiao, K. Rasul, R. Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.
2. D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980, 2014.
3. N. Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 15, 2014.
4. PyTorch documentation: https://pytorch.org/docs/stable/
5. S. Holm. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics 6(2), 1979.

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
├── extended_study.py        # 20-epoch study (leader)
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
python -m labs.lab1.extended_study      # 20-epoch study, after main.py
```
Presentation: run `main.py` and `extended_study.py` first (the notebook reads their results), then open `labs/lab1/notebook/lab1_presentation.ipynb` and run all cells.

Environment of the reported results: Python 3.13.7, torch 2.14.1 (CPU), 4 threads. Keep `artifacts/splits/train_val_split.json`. Another machine may give slightly different numbers.

---

**Tiếng Việt**

# Lab 1: Phân loại Fashion-MNIST bằng MLP (PyTorch)

## Báo cáo ngắn
**Bài toán.** Phân loại ảnh quần áo xám 28x28 thành 10 lớp bằng mạng nơ-ron nhiều lớp (MLP).

**Dữ liệu.** File CSV Fashion-MNIST trên Kaggle: 60.000 train, 10.000 test, chia 90/10 theo từng lớp (seed 42) thành 54.000 train / 6.000 validation.
Chuẩn hóa chỉ tính trên train. **Giới hạn:** CSV khác cách chia chính thức, nên không so trực tiếp được với kết quả đã công bố.

**Mô hình và huấn luyện.** MLP 784 -> 256 -> 128 -> 10 (235.146 tham số), ReLU, CrossEntropyLoss.
Adam (lr 1e-3), batch 64, 5 epoch; giữ checkpoint có validation loss thấp nhất.

**Kết quả.** Accuracy test qua 5 seed: 88,42 +/- 0,37%. Mô hình đã lưu (seed 42): 88,97%.
Shirt là lớp khó nhất (recall thấp nhất ở 4/5 seed, trung bình 68,07%).

**Thí nghiệm.** Bảy cấu hình, mỗi cấu hình 5 seed, mỗi cặp đổi một yếu tố. Tác động rõ lên validation accuracy:
bỏ lớp ẩn (-3,75 điểm), SGD thay Adam ở lr 1e-3 (-3,33), SGD lr 0,01 thay 1e-3 (+2,81).
Kích thước mạng, dropout, weight decay chưa có tác động có ý nghĩa trong 5 epoch.
Ở 20 epoch, baseline overfit; dropout hạn chế điều đó nhưng không tăng accuracy.

**Hình.** Hiển thị trong notebook, lưu ở `artifacts/figures/`. Chi tiết: xem Báo cáo chi tiết.

## Báo cáo chi tiết

### 1. Câu hỏi nghiên cứu
1. MLP nhỏ đạt accuracy bao nhiêu, và dao động giữa các seed thế nào?
2. Lớp ẩn, optimizer, learning rate, kích thước mạng, dropout, weight decay có tác động đo được không?
3. Lỗi tập trung ở đâu, và có ổn định qua các seed không?
4. Huấn luyện lâu hơn có làm đổi kết luận của 5 epoch không?

### 2. Dữ liệu
- Kaggle `zalando-research/fashionmnist`: 10 lớp cân bằng, không thiếu giá trị, giữ 43 dòng trùng (quyết định của nhóm).
- Một cách chia phân tầng (seed 42), dùng cho mọi lần chạy. Mean 0,2860 và std 0,3527 chỉ tính trên train.
- **Đối chiếu với file chính thức:** trong 9.999 ảnh khác nhau của file test Kaggle, 7.918 ảnh thuộc train chính thức, 1.291 ảnh thuộc test chính thức, khoảng 790 ảnh không khớp chính xác. Trong 59.957 ảnh khác nhau của file train Kaggle, 47.259 ảnh thuộc train chính thức, 7.885 ảnh thuộc test chính thức, khoảng 4.800 ảnh không khớp chính xác. Chưa rõ nguyên nhân.
- Còn ảnh giống hệt nhau giữa các tập: train-val 9, train-test 9, val-test 1.

### 3. Phương pháp
- Baseline: MLP 256-128, Adam, lr 1e-3, batch 64, 5 epoch, chốt trước khi thí nghiệm. Giữ checkpoint có validation loss thấp nhất.
- Mỗi cặp cấu hình chỉ khác một yếu tố và dùng chung seed 42, 7, 123, 2024, 3407. SGD luôn có momentum 0,9.
- Quy tắc chốt trước khi đọc kết quả: khác biệt rõ khi cả 5 seed cùng chiều và |chênh lệch trung bình| >= 2 std. Kiểm tra thứ hai: kiểm định t theo cặp có hiệu chỉnh Holm [5] và khoảng tin cậy 95%.
- So sánh chỉ dùng validation. Tập test chỉ báo cho baseline đã chốt trước.

### 4. Kết quả
Các bảng số giống phần tiếng Anh ở trên.

- Bỏ lớp ẩn: -3,75 điểm; SGD thay Adam ở lr 1e-3: -3,33 điểm; SGD lr 0,01 thay 1e-3: +2,81 điểm. Cả ba đều rõ.
- Mạng lớn hơn: -0,34; dropout 0,2: -0,68; weight decay 1e-4: -0,30. Cả ba chưa có ý nghĩa sau hiệu chỉnh.
- Baseline 5 epoch: train 90,60%, validation 88,82 +/- 0,39%, test 88,42 +/- 0,37%.
- Baseline 20 epoch (checkpoint tốt nhất): train 92,35%, validation 89,29 +/- 0,22%, test 89,11 +/- 0,17%.
- Val loss cuối ở 20 epoch: baseline 0,4020, dropout 0,3101.
- Shirt có recall thấp nhất ở 4/5 seed với 5 epoch (trung bình 68,07%) và 5/5 seed với 20 epoch (trung bình 70,37%).

### 5. Thảo luận
- Ba tác động rõ đều lớn hơn nhiều so với dao động giữa các seed.
- Ở lr 1e-3, SGD có momentum học quá chậm trong 5 epoch; lr 0,01 kéo lại phần lớn khoảng cách, nên kết quả về optimizer phụ thuộc learning rate.
- Ở 20 epoch, baseline overfit: val loss cuối cao hơn nhiều so với mức thấp nhất. Nhờ chọn checkpoint tốt nhất, accuracy vẫn gần 89%.
- Dropout không tăng val accuracy ở 20 epoch (+0,08 điểm, p = 0,71), nhưng giữ val loss cuối gần mức thấp nhất.
- 20 so với 5 epoch: +0,46 điểm trên validation, p (Holm) = 0,051, nên chưa khẳng định được lợi ích.

### 6. Giới hạn
- Một cách chia, một máy, 5 seed. Kết quả chưa có ý nghĩa không chứng minh là không có tác động.
- CSV Kaggle khác cách chia chính thức, nên không so với kết quả đã công bố.
- Tập test đã được xem nhiều lần khi phát triển, dù không dùng để chọn gì.
- Nguyên nhân lỗi Shirt chưa kiểm chứng; độ phân giải thấp và việc MLP không khai thác sẵn vùng lân cận vẫn là giả thuyết.

### 7. Kết luận
MLP tốt hơn rõ so với mô hình tuyến tính, đạt 88,42 +/- 0,37% trên test sau 5 epoch. Lớp ẩn, optimizer và learning rate có tác động rõ; kích thước mạng, dropout, weight decay chưa cho thấy tác động lên accuracy. Lỗi tập trung ở nhóm áo, và huấn luyện lâu hơn gây overfit mà chưa chứng minh được lợi ích về accuracy.

### Tài liệu tham khảo
Xem danh sách ở phần tiếng Anh.

## Dữ liệu
Kaggle `zalando-research/fashionmnist`: `fashion-mnist_train.csv` và `fashion-mnist_test.csv` (cột `label` + 784 cột pixel).
Đặt vào `labs/lab1/raw_data/` (thư mục này không được commit).

## Cấu trúc thư mục
Xem cây thư mục ở phần tiếng Anh phía trên (tên module giống nhau).

## Cách chạy
Các lệnh giống phần tiếng Anh. Notebook trình bày: chạy `main.py` và `extended_study.py` trước (notebook đọc kết quả của chúng), rồi mở `labs/lab1/notebook/lab1_presentation.ipynb` và chạy tất cả các ô.

Môi trường của kết quả báo cáo: Python 3.13.7, torch 2.14.1 (CPU), 4 thread. Giữ nguyên `artifacts/splits/train_val_split.json`. Máy khác có thể ra số hơi khác.
