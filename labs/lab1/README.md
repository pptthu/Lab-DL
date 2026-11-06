# Lab 1: Fashion-MNIST classification with an MLP (PyTorch)

## Brief report
**Task.** Classify 28x28 grayscale clothing images into 10 classes with a multilayer perceptron (MLP).

**Data.** Official Fashion-MNIST from Zalando GitHub, written as CSV: 60,000 train, 10,000 test, split 90/10 by class (seed 42) into 54,000 train / 6,000 validation.
Normalization uses the training split only. The official 10,000 test images are the test set.

**Model and training.** MLP 784 -> 256 -> 128 -> 10 (235,146 parameters), ReLU, CrossEntropyLoss.
Adam (lr 1e-3), batch size 64, 5 epochs; the checkpoint with the lowest validation loss is kept.

**Results.** Test accuracy across 5 seeds: 88.00 +/- 0.39%. Saved model (seed 42): 88.36%.
Shirt is the hardest class (lowest recall in 4 of 5 seeds, mean 69.97%).

**Experiments.** Seven configurations, 5 seeds each, changing one factor per pair. Clear effects on validation accuracy:
no hidden layers (-3.80 points), SGD instead of Adam at lr 1e-3 (-3.45), SGD lr 0.01 instead of 1e-3 (+3.14).
Network size, dropout and weight decay show no significant effect within 5 epochs.
At 20 epochs the baseline overfits; dropout keeps the final validation loss lower but does not clearly raise accuracy.

**Figures.** Shown in the notebook and saved in `artifacts/figures/`. Per-epoch curves of all 45 runs: notebook Stages 9b and 10b, from `artifacts/runs/epoch_history.csv` and `extended_epoch_history.csv`. Details: see Detailed report.

## Detailed report

### 1. Research questions
1. What accuracy does a small MLP reach, and how much does it vary between seeds?
2. Do hidden layers, optimizer, learning rate, network size, dropout and weight decay have a measurable effect?
3. Where do the errors concentrate, and is that stable across seeds?
4. Does longer training change the 5-epoch findings?

### 2. Data
- Official Zalando files (`zalandoresearch/fashion-mnist`), downloaded by `python -m labs.lab1.preprocessing.download_official` (torchvision, MD5 checked) and written as CSV: 10 balanced classes, no missing values, 0 duplicate rows.
- One stratified split (seed 42), used by every run. Mean 0.2859 and std 0.3530 come from the training split only.

### 3. Method
- Baseline: MLP 256-128, Adam, lr 1e-3, batch 64, 5 epochs, fixed before the experiments. The checkpoint with the lowest validation loss is kept.
- Each pair of configurations differs in one factor and shares seeds 42, 7, 123, 2024, 3407. SGD always uses momentum 0.9.
- Rule fixed before reading the results: a pair shows a clear difference if all 5 seeds point the same way and |mean difference| >= 2 std. Second check: paired t-test with Holm correction [5] and a 95% confidence interval.
- Comparisons use validation data only. The test set is reported only for baselines fixed beforehand.

### 4. Results
| Comparison (validation accuracy) | Mean difference (points) | 95% CI | p (Holm) | Result |
|---|---|---|---|---|
| No hidden layers vs baseline | -3.80 | -4.67 to -2.92 | 0.0011 | Clear |
| SGD vs Adam, lr 1e-3 | -3.45 | -3.86 to -3.04 | 0.0001 | Clear |
| SGD lr 0.01 vs 1e-3 | +3.14 | 2.93 to 3.36 | < 0.0001 | Clear |
| Larger network vs baseline | -0.52 | -1.02 to -0.03 | 0.0844 | Not significant |
| Dropout 0.2 vs baseline | -0.46 | -0.89 to -0.02 | 0.0844 | Not significant |
| Weight decay 1e-4 vs baseline | -0.29 | -0.51 to -0.08 | 0.0587 | Not significant |

| Baseline across 5 seeds | Train | Validation | Test |
|---|---|---|---|
| 5 epochs | 90.92% | 88.62 +/- 0.36% | 88.00 +/- 0.39% |
| 20 epochs (best checkpoint) | 92.95% | 89.22 +/- 0.26% | 88.62 +/- 0.19% |

| 20 epochs, 5 seeds | Best epoch | Lowest val loss | Final val loss |
|---|---|---|---|
| Baseline | 7 to 10 | 0.2988 | 0.3972 |
| Dropout 0.2 | 9 to 20 | 0.2916 | 0.3095 |

Shirt has the lowest recall in 4 of 5 seeds at 5 epochs (mean 69.97%) and 5 of 5 at 20 epochs (mean 69.03%). The most frequent confusions across seeds at 5 epochs are Shirt -> T-shirt/top, T-shirt/top -> Shirt and Pullover -> Shirt.

### 5. Discussion
- The three clear effects are large compared with the variation between seeds. Hidden layers add about 3.80 points over a linear model; the linear model levels off near 84% from epoch 2 and its training loss flattens around 0.41 (Stage 9b).
- At lr 1e-3, SGD with momentum learns too slowly in 5 epochs: it is still rising at epoch 5 (Stage 9b). lr 0.01 recovers most of that gap, so the optimizer result depends on the learning rate.
- The larger network, dropout and weight decay are all slightly lower than the baseline, but none passes the rule or the Holm-corrected test, so no effect is claimed.
- At 20 epochs the baseline overfits: mean val loss is lowest at epoch 7 (0.3028) and rises to 0.3972, while train loss keeps falling to 0.1240 (Stage 10b). Best-checkpoint selection keeps accuracy near 89%.
- Dropout does not clearly raise val accuracy at 20 epochs (+0.36 points, p (Holm) = 0.15), but it keeps the final val loss lower (0.3095 vs 0.3972, lower on all 5 seeds).
- 20 versus 5 epochs: +0.60 points on validation, p (Holm) = 0.15, so the gain is not established.

### 6. Limitations
- One split, one machine, 5 seeds. Non-significant results are not proof of no effect.
- The test set was seen several times during development, though no choice used it.
- The causes of the Shirt errors are not tested; low resolution and the lack of built-in local structure in an MLP remain hypotheses.
- Scope: MLP only. A CNN is not part of this lab (team decision, 2026-11-04).

### 7. Conclusion
The MLP learns the task well above a linear model, reaching 88.00 +/- 0.39% on the official test set in 5 epochs. Hidden layers, optimizer and learning rate have clear effects; network size, dropout and weight decay do not show one on accuracy here. Errors concentrate in upper-body garments, and longer training overfits without a proven accuracy gain.

### References
1. H. Xiao, K. Rasul, R. Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.
2. D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980, 2014.
3. N. Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 15, 2014.
4. PyTorch documentation: https://pytorch.org/docs/stable/
5. S. Holm. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics 6(2), 1979.

## Data
Official Fashion-MNIST from Zalando GitHub (https://github.com/zalandoresearch/fashion-mnist). Run `python -m labs.lab1.preprocessing.download_official`: it downloads the 4 IDX files with torchvision (MD5 checked) and writes `fashion-mnist_train.csv` and `fashion-mnist_test.csv` (`label` + 784 pixel columns) into `labs/lab1/raw_data/` (this folder is not committed).

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
├── preprocessing/download_official.py # downloads the official data as CSV
├── features/features.py        # module: features
├── model/model.py              # module: model
├── train/train.py              # module: train
├── evaluate/evaluate.py        # module: evaluate
├── visualize/visualize.py      # module: visualize
├── experiment/experiment.py    # module: experiment
├── notebook/lab1_presentation.ipynb # presentation notebook
├── raw_data/                # dataset files (NOT committed)
├── artifacts/               # splits/ (committed) | models, figures, logs, runs (NOT committed)
├── docs/                    # per-module guides for AI agents (eda.md ... experiment.md, demo.md)
├── logbook/                 # bug reports and experiment notes (docs/templates)
└── tests/                   # interface contract + module tests (fake data)
```

## Module guides
One guide per module for AI agents and teammates, in `labs/lab1/docs/`: `eda.md`, `preprocessing.md`, `features.md`, `model.md`, `train.md`, `evaluate.md`, `visualize.md`, `experiment.md`, and `demo.md` (how to run and present the lab).

## How to run
```bash
pip install -r labs/lab1/requirements.txt
python -m pytest labs/lab1/tests        # tests on fake data
python -m labs.lab1.main                # full pipeline (needs the CSVs in raw_data/)
python -m labs.lab1.extended_study      # 20-epoch study, after main.py
```
Presentation: run `main.py` and `extended_study.py` first (the notebook reads their results), then open `labs/lab1/notebook/lab1_presentation.ipynb` and run all cells.

Environment of the reported results (2026-11-04, official data): Python 3.13.7, torch 2.14.1 (CPU), 4 threads. Keep `artifacts/splits/train_val_split.json`. Another machine may give slightly different numbers.

---

**Tiếng Việt**

# Lab 1: Phân loại Fashion-MNIST bằng MLP (PyTorch)

## Báo cáo ngắn
**Bài toán.** Phân loại ảnh quần áo xám 28x28 thành 10 lớp bằng mạng nơ-ron nhiều lớp (MLP).

**Dữ liệu.** Fashion-MNIST chính thức từ GitHub của Zalando, ghi ra CSV: 60.000 train, 10.000 test, chia 90/10 theo từng lớp (seed 42) thành 54.000 train / 6.000 validation.
Chuẩn hóa chỉ tính trên train. Tập test là 10.000 ảnh test chính thức.

**Mô hình và huấn luyện.** MLP 784 -> 256 -> 128 -> 10 (235.146 tham số), ReLU, CrossEntropyLoss.
Adam (lr 1e-3), batch 64, 5 epoch; giữ checkpoint có validation loss thấp nhất.

**Kết quả.** Accuracy test qua 5 seed: 88,00 +/- 0,39%. Mô hình đã lưu (seed 42): 88,36%.
Shirt là lớp khó nhất (recall thấp nhất ở 4/5 seed, trung bình 69,97%).

**Thí nghiệm.** Bảy cấu hình, mỗi cấu hình 5 seed, mỗi cặp đổi một yếu tố. Tác động rõ lên validation accuracy:
bỏ lớp ẩn (-3,80 điểm), SGD thay Adam ở lr 1e-3 (-3,45), SGD lr 0,01 thay 1e-3 (+3,14).
Kích thước mạng, dropout, weight decay chưa có tác động có ý nghĩa trong 5 epoch.
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
- Quy tắc chốt trước khi đọc kết quả: khác biệt rõ khi cả 5 seed cùng chiều và |chênh lệch trung bình| >= 2 std. Kiểm tra thứ hai: kiểm định t theo cặp có hiệu chỉnh Holm [5] và khoảng tin cậy 95%.
- So sánh chỉ dùng validation. Tập test chỉ báo cho baseline đã chốt trước.

### 4. Kết quả
Các bảng số giống phần tiếng Anh ở trên.

- Bỏ lớp ẩn: -3,80 điểm; SGD thay Adam ở lr 1e-3: -3,45 điểm; SGD lr 0,01 thay 1e-3: +3,14 điểm. Cả ba đều rõ.
- Mạng lớn hơn: -0,52; dropout 0,2: -0,46; weight decay 1e-4: -0,29. Cả ba chưa có ý nghĩa sau hiệu chỉnh.
- Baseline 5 epoch: train 90,92%, validation 88,62 +/- 0,36%, test 88,00 +/- 0,39%.
- Baseline 20 epoch (checkpoint tốt nhất): train 92,95%, validation 89,22 +/- 0,26%, test 88,62 +/- 0,19%.
- Val loss cuối ở 20 epoch: baseline 0,3972, dropout 0,3095.
- Shirt có recall thấp nhất ở 4/5 seed với 5 epoch (trung bình 69,97%) và 5/5 seed với 20 epoch (trung bình 69,03%).

### 5. Thảo luận
- Ba tác động rõ đều lớn hơn nhiều so với dao động giữa các seed. Mô hình tuyến tính đứng quanh 84% từ epoch 2 và train loss chững ở khoảng 0,41 (Stage 9b).
- Ở lr 1e-3, SGD có momentum học quá chậm trong 5 epoch: vẫn đang tăng ở epoch 5 (Stage 9b). lr 0,01 kéo lại phần lớn khoảng cách, nên kết quả về optimizer phụ thuộc learning rate.
- Mạng lớn hơn, dropout và weight decay đều thấp hơn baseline một chút, nhưng không cặp nào qua quy tắc hay kiểm định đã hiệu chỉnh Holm, nên không kết luận có tác động.
- Ở 20 epoch, baseline overfit: val loss trung bình thấp nhất ở epoch 7 (0,3028) rồi tăng lên 0,3972, trong khi train loss vẫn giảm xuống 0,1240 (Stage 10b). Nhờ chọn checkpoint tốt nhất, accuracy vẫn gần 89%.
- Dropout chưa làm tăng val accuracy rõ rệt ở 20 epoch (+0,36 điểm, p (Holm) = 0,15), nhưng giữ val loss cuối thấp hơn (0,3095 so với 0,3972, thấp hơn ở cả 5 seed).
- 20 so với 5 epoch: +0,60 điểm trên validation, p (Holm) = 0,15, nên chưa khẳng định được lợi ích.

### 6. Giới hạn
- Một cách chia, một máy, 5 seed. Kết quả chưa có ý nghĩa không chứng minh là không có tác động.
- Tập test đã được xem nhiều lần khi phát triển, dù không dùng để chọn gì.
- Nguyên nhân lỗi Shirt chưa kiểm chứng; độ phân giải thấp và việc MLP không khai thác sẵn vùng lân cận vẫn là giả thuyết.
- Phạm vi: chỉ làm MLP. CNN không thuộc lab này (quyết định của nhóm, 04/11/2026).

### 7. Kết luận
MLP tốt hơn rõ so với mô hình tuyến tính, đạt 88,00 +/- 0,39% trên tập test chính thức sau 5 epoch. Lớp ẩn, optimizer và learning rate có tác động rõ; kích thước mạng, dropout, weight decay chưa cho thấy tác động lên accuracy. Lỗi tập trung ở nhóm áo, và huấn luyện lâu hơn gây overfit mà chưa chứng minh được lợi ích về accuracy.

### Tài liệu tham khảo
Xem danh sách ở phần tiếng Anh.

## Dữ liệu
Fashion-MNIST chính thức từ GitHub của Zalando (https://github.com/zalandoresearch/fashion-mnist). Chạy `python -m labs.lab1.preprocessing.download_official`: lệnh này tải 4 file IDX bằng torchvision (tự kiểm MD5) và ghi `fashion-mnist_train.csv`, `fashion-mnist_test.csv` (cột `label` + 784 cột pixel) vào `labs/lab1/raw_data/` (thư mục này không được commit).

## Cấu trúc thư mục
Xem cây thư mục ở phần tiếng Anh phía trên (tên module giống nhau).

## Hướng dẫn từng module
Mỗi module có một file hướng dẫn cho AI và thành viên trong `labs/lab1/docs/`, cùng `demo.md` (cách chạy và trình bày lab).

## Cách chạy
Các lệnh giống phần tiếng Anh. Notebook trình bày: chạy `main.py` và `extended_study.py` trước (notebook đọc kết quả của chúng), rồi mở `labs/lab1/notebook/lab1_presentation.ipynb` và chạy tất cả các ô.

Môi trường của kết quả báo cáo (04/11/2026, data chính thức): Python 3.13.7, torch 2.14.1 (CPU), 4 thread. Giữ nguyên `artifacts/splits/train_val_split.json`. Máy khác có thể ra số hơi khác.
