# Tiến độ Lab 1

- **Created**: 2026-10-12T10:27:18+07:00 (pptthu)
- **Last Updated**: 2026-11-14T00:15:45+07:00 (pptthu)

Cập nhật file này sau MỖI bước.
Bảng dưới đây là nguồn thông tin duy nhất về việc ai đang làm gì.

| Module | Phụ trách | Trạng thái (chưa làm / đang làm / xong) | Test hợp đồng (đạt / lỗi / chưa chạy) | Chạy thử trên dữ liệu thật (đạt / lỗi / chưa chạy) | Ghi chú |
|---|---|---|---|---|---|
| `eda` | Member A (Toàn) | xong | đạt (3 test) | đạt | Dữ liệu chính thức của Zalando (2026-11-02): 60.000 dòng, 10 lớp cân bằng, 0 dòng trùng, 0 giá trị thiếu |
| `preprocessing` | Member A (Toàn) | xong | đạt (6 test) | đạt | Chia phân tầng 90/10, lưu tại artifacts/splits/train_val_split.json |
| `features` | Member A (Toàn) | xong | đạt (6 test) | đạt | Train mean 0,2859, std 0,3530 (dữ liệu chính thức); loader trả [64,1,28,28] float32 |
| `model` | Member A (Toàn) | xong | đạt (6 test) | đạt | MLP 784->256->128->10: 235.146 tham số huấn luyện được |
| `train` | Member B (Thư) | xong | đạt (dữ liệu giả, 4 test) | đạt (data chính thức, VAL 6.000: loss 0,3018, acc 89,05%) | 5 epoch, Adam lr=1e-3, checkpoint tốt nhất lưu tại artifacts/models/mlp_baseline.pt |
| `evaluate` | Member B (Thư) | xong | đạt (dữ liệu giả, 5 test) | đạt (data chính thức, VAL 6.000: loss 0,3018, acc 89,05%; TEST 88,36%) | Trouser cao nhất 98,17%, Shirt thấp nhất 70,17%, ROC-AUC 0,9925; báo cáo từng lớp lưu ra CSV |
| `visualize` | Member B (Thư) | xong | đạt (dữ liệu giả, 3 test) | đạt (lưu 3 hình trong artifacts/figures/) | loss_curves.png, predictions_grid.png, confusion_matrix.png; hình của lần chạy 2026-11-04 chưa được xem lại bằng mắt |
| `experiment` | Member B (Thư) | xong | đạt (dữ liệu giả, 10 test) | đạt (35 lần chạy, lưu tại artifacts/runs/experiment_results.csv) | 7 cấu hình, mỗi cấu hình 5 seed; test của baseline 88,00 +/- 0,39% (data chính thức) |
| `main.py`, README, tích hợp | Nhóm trưởng (Thư) | xong | đạt (chạy thử 1 epoch) | đạt (toàn bộ pipeline) | Báo cáo ngắn và báo cáo chi tiết trong README, `extended_study.py`, lab1_presentation.ipynb (26 ô), hướng dẫn từng module trong `docs/`; chạy lại toàn bộ trên data chính thức ngày 2026-11-04; 54 test đều đạt |

## Bàn giao A -> B (tuần tự, không xen kẽ)
Member B chỉ bắt đầu code khi TẤT CẢ các điều kiện sau đều đúng:
- [x] file JSON chia dữ liệu đã commit trong artifacts/splits/ (chỉ số train/val): `train_val_split.json`
- [x] make_loaders trả batch shape [B,1,28,28] với nhãn 0-9
- [x] MLP forward trên batch giả trả logits [B,10]
- [x] mọi module của Member A đều `xong` trong file này và `pytest labs/lab1/tests` đạt
- [x] các PR của Member A đã merge vào `main`; Member B tạo nhánh từ `main` mới nhất
- [x] Member A gửi tin bàn giao (đã làm gì, lệnh chạy chính xác, vấn đề đã biết)
Sau bàn giao, Member A chỉ trả lời câu hỏi và review PR, không sửa module của Member B.

## Kiểm chứng: lưu lịch sử theo epoch (2026-11-06, Python 3.13.7, torch 2.14.1 CPU, 4 thread)
- `python -m pytest labs/lab1/tests tests -q` -> đạt, 56 test (thêm 2 test cho `history` và `save_history`)
- `python -m labs.lab1.main` -> đạt, 10 phút 26 giây; seed 42: val 89,05% (loss 0,3018), test 88,36% (loss 0,3338); baseline 5 seed: test 88,00 +/- 0,39%
- `python -m labs.lab1.extended_study` -> đạt, 11 phút 42 giây
- So với lần chạy 04/11: 12 file CSV trong `artifacts/runs/` giống hệt (trừ cột thời gian); `epoch_history.csv` (175 dòng) và `extended_epoch_history.csv` (200 dòng) khớp epoch tốt nhất, giá trị tốt nhất và cuối của cả 45 lần chạy (0 sai lệch)
- Notebook chạy lại bằng `python -m jupyter nbconvert --execute` -> 0 lỗi; thêm Stage 9b (đường val accuracy và train loss của 7 cấu hình) và Stage 10b (đường loss 20 epoch, bảng recall 20 epoch)
- Đọc từ đường cong: mô hình tuyến tính đứng quanh 84% từ epoch 2 (train loss khoảng 0,41); SGD lr 1e-3 vẫn tăng ở epoch 5 (+0,70 điểm từ epoch 4 sang 5); baseline 20 epoch: val loss trung bình thấp nhất ở epoch 7 (0,3028), cuối 0,3972, train loss giảm xuống 0,1240; dropout: val loss 0,2984 đến 0,3100 từ epoch 7 đến 20
- Bỏ các số so sánh CSV Kaggle khỏi README, notebook, `project_description.md` và `docs/` (code không còn dùng Kaggle, và không có code tính lại các số đó); câu "0 ảnh giống hệt nhau giữa train/val/test" cũng bỏ vì không có code kiểm tra; giữ "0 dòng trùng trong CSV train" (do `run_eda` đo)

## Kiểm chứng: đổi sang dữ liệu chính thức của Zalando (2026-11-02, Python 3.13.7, torch 2.13.0 CPU, torchvision 0.29.1)
Dữ liệu và split dưới đây thay thế CSV Kaggle. Kết quả huấn luyện trên dữ liệu này nằm ở mục kế tiếp (2026-11-04).
- `python -m labs.lab1.preprocessing.download_official` -> đạt: torchvision tải 4 file IDX gốc và kiểm MD5; ghi `fashion-mnist_train.csv` (60.000 dòng, 6.000 mỗi lớp) và `fashion-mnist_test.csv` (10.000 dòng, 1.000 mỗi lớp); `load_csv` đọc lại khớp từng pixel và nhãn với dữ liệu gốc
- CSV Kaggle cũ (`raw_data/fashion-mnist_*_kaggle.csv`, không commit) từng giữ lại để đối chiếu, đã xóa ngày 2026-11-08 vì code không dùng; các số so sánh Kaggle trong file này không còn kiểm chứng lại được
- `python -m labs.lab1.preprocessing.preprocessing` -> đạt: `make_split(labels, val_ratio=0.1, seed=42)`, train 54.000 / val 6.000 (600 mỗi lớp), ghi lại `artifacts/splits/train_val_split.json`
- `run_eda` trên CSV train mới -> đạt: n_rows 60000, 6000 mỗi lớp, pixel 0-255, n_missing 0, n_duplicates 0 (CSV Kaggle: 43)
- `compute_mean_std` trên 54.000 ảnh train -> mean 0,2859, std 0,3530 (CSV Kaggle: 0,2860 / 0,3527)
- Ảnh giống hệt nhau giữa các tập -> train-val 0, train-test 0, val-test 0 (CSV Kaggle: 9, 9, 1); test có 0 dòng trùng
- `python -m pytest labs/lab1/tests -q` -> 50 passed; `python -m pytest tests -q` -> 4 passed

## Kiểm chứng: kết quả trên dữ liệu chính thức (2026-11-04, Python 3.13.7, torch 2.14.1 CPU, 4 thread)
Các số này thay thế mọi kết quả trên CSV Kaggle; README, notebook và `docs/` dùng đúng các số này.
- CSV tạo lại trên máy nhóm trưởng bằng `python -m labs.lab1.preprocessing.download_official`: MD5 train `8ff2cced1c624159b21dac4f735b4215`, test `78e18be2098f1256a07ce6f91a4d436e`; `make_split(labels, 0.1, 42)` ra đúng y `train_val_split.json` đã commit (54.000 / 6.000); mean 0,2859, std 0,3530
- `python -m labs.lab1.main` -> đạt, 35 lần chạy thí nghiệm, 7.401 giây; seed 42: val 89,05% (loss 0,3018), test 88,36% (loss 0,3338, 8.836/10.000), epoch tốt nhất là 5
- Chạy `main.py` lần hai (2026-11-04, 1.145 giây) -> cả 5 file kết quả (`experiment_results`, `seed_results`, `recall_by_class`, `top_confusions`, `classification_report`) y hệt lần một (bỏ cột thời gian), MD5 của 35 checkpoint thí nghiệm và `mlp_baseline.pt` y hệt
- Thời gian: phần lớn mỗi lần chạy 55-140 giây; riêng `softmax_regression` seed 3407 mất 5.445 giây và seed 42 mất 388 giây (máy không vào sleep, chưa rõ nguyên nhân); accuracy không phụ thuộc thời gian chạy
- Notebook chạy lại toàn bộ bằng `jupyter nbconvert --execute` sau `main.py` và `extended_study.py` -> 0 lỗi, cùng val 89,05% và test 88,36% với `main.py`; ROC-AUC macro 0,9925, PR-AUC macro 0,9479 (validation)
- Baseline qua 5 seed: test 88,00 +/- 0,39%; val 88,62 +/- 0,36%; train 90,92%
- Epoch tốt nhất của 5 seed baseline: 5, 4, 5, 5, 5
- So sánh từng cặp trên validation (quy tắc chốt trước: cả 5 seed cùng chiều và |trung bình| >= 2 std):
  - Khác biệt rõ: bỏ lớp ẩn -3,80 +/- 0,71; SGD so với Adam ở lr 1e-3 -3,45 +/- 0,33; SGD lr 0,01 so với 1e-3 +3,14 +/- 0,17
  - Chưa thấy khác biệt rõ: mạng lớn hơn -0,52 +/- 0,40; dropout 0,2 -0,46 +/- 0,35; weight decay 1e-4 -0,29 +/- 0,17
- Kiểm định t theo cặp có hiệu chỉnh Holm: 3 cặp khác biệt rõ có p sau hiệu chỉnh <= 0,0011; mạng lớn hơn 0,0844; dropout 0,0844; weight decay 0,0587 (trước hiệu chỉnh: 0,042; 0,044; 0,020)
- Recall qua các seed (validation): Shirt thấp nhất ở 4/5 seed (trung bình 69,97%), Coat thấp nhất ở 1 seed; nhầm nhiều nhất: Shirt -> T-shirt/top (419 lần qua 5 seed)
- `python -m labs.lab1.extended_study` -> đạt, 10 lần chạy 20 epoch, 1.440 giây: baseline val 89,22 +/- 0,26%, test 88,62 +/- 0,19%, train 92,95%, epoch tốt nhất 7-10, val loss thấp nhất 0,2988, cuối 0,3972; dropout 0,2 val 89,58 +/- 0,29%, epoch tốt nhất 9-20, val loss cuối 0,3095
- Ở 20 epoch: dropout so với baseline +0,36 điểm val accuracy, p sau hiệu chỉnh 0,15; val loss cuối thấp hơn 0,088 ở cả 5 seed (p sau hiệu chỉnh 0,0001); 20 so với 5 epoch +0,60 điểm, p sau hiệu chỉnh 0,15
- `python -m pytest labs/lab1/tests tests -q` -> 54 passed; md5 của `mlp_baseline.pt` không đổi sau khi chạy test
- Kết quả và notebook của lần chạy trên CSV Kaggle từng được sao lưu ngoài repo (`lab1_kaggle_artifacts_backup_2026-11-04`, không commit), đã xóa ngày 2026-11-08

## Kiểm chứng: kết quả trên CSV Kaggle (2026-11-01, đã thay thế, giữ làm lịch sử)
- `python -m pytest labs/lab1/tests tests -q` -> 54 passed; md5 của mọi file `.pt` không đổi sau khi chạy test
- `python -m labs.lab1.main` -> đạt, 35 lần chạy thí nghiệm; seed 42: val 89,38% (loss 0,3031), test 88,97% (loss 0,3005, 8.897/10.000), epoch tốt nhất là 5
- Chạy `main.py` lần hai -> trọng số baseline y hệt, cả 35 kết quả thí nghiệm y hệt
- Notebook chạy lại sau `main.py` -> cùng loss và accuracy trên val và test với `main.py`, 0 lỗi
- Baseline qua 5 seed: test 88,42 +/- 0,37%; val 88,82 +/- 0,39%; train 90,60%
- Epoch tốt nhất của 5 seed baseline: 5, 5, 5, 5, 4
- So sánh từng cặp trên validation (quy tắc chốt trước: cả 5 seed cùng chiều và |trung bình| >= 2 std):
  - Khác biệt rõ: bỏ lớp ẩn -3,75 +/- 0,45; SGD so với Adam ở lr 1e-3 -3,33 +/- 0,34; SGD lr 0,01 so với 1e-3 +2,81 +/- 0,38
  - Chưa thấy khác biệt rõ: mạng lớn hơn -0,34 +/- 0,48; dropout 0,2 -0,68 +/- 0,46; weight decay 1e-4 -0,30 +/- 0,37
- Recall qua các seed (validation): Shirt thấp nhất ở 4/5 seed (trung bình 68,07%), T-shirt/top thấp nhất ở 1 seed
- Kiểm định t theo cặp có hiệu chỉnh Holm: 3 cặp khác biệt rõ có p sau hiệu chỉnh <= 0,0003; mạng lớn hơn 0,2755; dropout 0,0927; weight decay 0,2755
- `python -m labs.lab1.extended_study` -> đạt, 10 lần chạy 20 epoch: baseline val 89,29 +/- 0,22%, test 89,11 +/- 0,17%, train 92,35%, val loss thấp nhất 0,3055, cuối 0,4020; dropout 0,2 val 89,37 +/- 0,40%, val loss cuối 0,3101
- Dropout so với baseline ở 20 epoch: val accuracy +0,08 điểm, p = 0,71; 20 so với 5 epoch: +0,46 điểm, p sau hiệu chỉnh 0,051
- Ảnh giống hệt nhau giữa các tập, đo ngày 2026-10-31: train-val 9, train-test 9, val-test 1
- So với file chính thức của Zalando (2026-11-01): trong 9.999 ảnh khác nhau của file test Kaggle, 7.918 ảnh thuộc train chính thức, 1.291 ảnh thuộc test chính thức, khoảng 790 ảnh không khớp chính xác

## Kiểm chứng: các lần chạy trước (đã thay thế, giữ làm lịch sử)
- `python -m pytest labs/lab1/tests tests -q` -> đạt, 46 passed trong 16,72 giây (42 test lab1 + 4 test dùng chung; Python 3.13.7, torch 2.x, 2026-10-17)
- `run_eda(raw_data/fashion-mnist_train.csv, artifacts/figures)` -> đạt: n_rows 60000, 6000 mỗi lớp, pixel 0-255, n_missing 0, n_duplicates 43; tạo `class_counts.png`, `sample_grid.png`
- `python -m labs.lab1.preprocessing.preprocessing` -> đạt: images (60000, 28, 28) uint8 0-255, labels int64 0-9, train 54000 / val 6000, tạo `artifacts/splits/train_val_split.json`
- load_csv trên `fashion-mnist_test.csv` -> đạt: (10000, 28, 28) uint8, 1000 mỗi lớp
- Loader trên dữ liệu thật -> đạt: train 54000 / val 6000 / test 10000; batch x [64,1,28,28] float32, y int64 trong 0-9
- `MLP()` forward trên batch train thật -> đạt: logits [64,10], CrossEntropyLoss chưa huấn luyện 2,2931 (ln 10 = 2,3026)
- `pytest labs/lab1/tests/test_train.py` -> đạt: 4 passed
- `train_model` chạy baseline (5 epoch, Adam lr=1e-3, 54k train / 6k val) -> đạt, khoảng 21 giây trên CPU:
  - Epoch 1: train_loss=0,4785, val_loss=0,3744, val_acc=0,8710
  - Epoch 2: train_loss=0,3532, val_loss=0,3481, val_acc=0,8772
  - Epoch 3: train_loss=0,3151, val_loss=0,3266, val_acc=0,8887
  - Epoch 4: train_loss=0,2898, val_loss=0,3194, val_acc=0,8877
  - Epoch 5: train_loss=0,2705, val_loss=0,3310, val_acc=0,8860
  - Checkpoint tốt nhất lưu tại `artifacts/models/mlp_baseline.pt`
- `pytest labs/lab1/tests/test_evaluate.py` -> đạt: 4 passed
- `evaluate` trên VAL (6.000 mẫu) với `mlp_baseline.pt` -> đạt:
  - Validation loss 0,3194, validation accuracy 88,77% (5326/6000)
  - Ma trận nhầm lẫn: shape (10, 10), tổng 6000
  - Báo cáo từng lớp: Trouser cao nhất (recall 98,50%, F1 98,75%), Shirt thấp nhất (recall 71,00%, F1 71,84%)
  - Lỗi của Shirt (hàng cm[6], bỏ đường chéo): 174/600 ảnh sai; Pullover 63, Coat 48, T-shirt/top 42
  - Macro ROC-AUC 0,9913, Macro PR-AUC 0,9428
- `pytest labs/lab1/tests/test_visualize.py` -> đạt: 3 passed
- `visualize` chạy thật -> đạt: lưu 3 hình vào `artifacts/figures/`:
  - `loss_curves.png`: loss train và val qua 5 epoch
  - `predictions_grid.png`: lưới 4x4 có nhãn dự đoán và nhãn thật, ảnh sai màu đỏ
  - `confusion_matrix.png`: bảng màu 10x10 có số và tên lớp
- `pytest labs/lab1/tests/test_experiment.py` -> đạt: 1 passed
- `run_experiments` chạy thật (4 cấu hình, 5 epoch mỗi cấu hình) -> đạt:
  - Baseline (256-128, Adam 1e-3, dropout 0,0): val_acc 89,05%, 235k tham số, độ trễ 0,18 ms
  - SGD (256-128, SGD 0,01, momentum 0,9): val_acc 88,57%, 235k tham số, độ trễ 0,27 ms
  - Mạng lớn hơn (512-256-128, Adam 1e-3): val_acc 88,10%, 567k tham số, độ trễ 0,43 ms
  - Dropout (256-128, Adam 1e-3, dropout 0,2, wd 1e-4): val_acc 88,47%, 235k tham số, độ trễ 0,22 ms
  - Kết quả lưu tại `artifacts/runs/experiment_results.csv`
  - Baseline trên tập TEST (10.000 mẫu; báo cáo cuối, không chọn bằng thí nghiệm): loss 0,3212, accuracy 88,32% (8832/10000)
- Chạy `main.py` -> đạt: toàn bộ các bước chạy từ đầu đến cuối.
- `lab1_presentation.ipynb` -> 26 ô (12 ô code) chạy từ đầu đến cuối, 0 lỗi, 2026-10-30; phần giải thích bằng tiếng Anh, chỉ ô Data Source có hai ngôn ngữ.

## Nhật ký (ghi ngày, mỗi bước một dòng)
- <ngày> <tên>: tạo khung dự án; thống nhất interface -> kiểm tra bằng `pytest labs/lab1/tests`
- 2026-10-13 Toàn: `eda`: viết `run_eda` + `tests/test_eda.py` -> pytest đạt; chạy thật trên 60.000 dòng
- 2026-10-13 Toàn: `preprocessing`: viết `load_csv`, `make_split`, `save_split`, `load_split` + test; tạo file JSON chia dữ liệu dùng chung -> pytest đạt; chạy thật trên CSV train và test
- 2026-10-13 Toàn: `features`: viết `compute_mean_std`, `build_transform`, `FashionDataset`, `make_loaders` + test -> pytest đạt; đã kiểm tra batch thật
- 2026-10-13 Toàn: `model`: viết `MLP`, `count_parameters` + test -> pytest đạt; forward trên batch thật
- 2026-10-15 Thư: `train`: viết `train_model`, `save_checkpoint`, `load_checkpoint` + `tests/test_train.py` -> 4 test đạt; chạy xong baseline (lưu `mlp_baseline.pt`)
- 2026-10-15 Thư: `evaluate`: viết `evaluate`, `confusion_matrix`, `per_class_accuracy` + `tests/test_evaluate.py` -> 4 test đạt; đánh giá checkpoint đã lưu trên VAL
- 2026-10-15 Thư: `visualize`: viết `plot_loss_curves`, `show_predictions`, `plot_confusion_matrix` + `tests/test_visualize.py` -> 3 test đạt; tạo và xem bằng mắt 3 hình trong artifacts/figures/
- 2026-10-15 Thư: `experiment`: viết `run_experiments`, `save_results` + `tests/test_experiment.py` -> test đạt; chạy 4 cấu hình trên dữ liệu thật; lưu bảng tại `artifacts/runs/experiment_results.csv`
- 2026-10-16 Thư: `tích hợp`: thống nhất số theo lần chạy chuẩn (val 88,77%, test 88,32%), bổ sung phân tích lỗi Shirt, đường ROC/PR, đo độ trễ vào notebook, rút gọn README (<200 từ), cập nhật khai báo sử dụng AI.
- 2026-10-17 Thư: `tích hợp`: đối chiếu tài liệu và notebook với output thật; chuyển notebook sang tiếng Anh, phạm vi 3 tầng (bỏ độ trễ, đưa ROC/PR vào phụ lục); ghi nhận 46 test, ảnh hưởng thứ tự RNG/randn và khác biệt giữa các chỉ số.
- 2026-10-17 Thư: `notebook` + `README`: phân tích lỗi Shirt không còn tính dự đoán đúng (Pullover 63, Coat 48, T-shirt/top 42); chạy lại notebook, các output khác không đổi; viết lại README sau khi PowerShell làm hỏng file (số có $, ký tự thoát `a/`b) -> kiểm tra bằng cách chạy lại notebook và `pytest` (46 passed).
- 2026-10-30 Thư: `notebook`: thêm ô Stage 0 Data Source; sửa các chỗ nói quá (cấu hình không đổi từng yếu tố, cách nói về `torch.no_grad`, checkpoint, recall Sandal đúng 95,00%, ghi chú tập test, overfit) -> chạy lại notebook, so output code với bản trước: chỉ khác tên bảng thí nghiệm.
- 2026-10-31 Thư: `project_description`: cập nhật Design với cấu trúc hệ thống, cây thư mục hiện tại và 2 rủi ro kỹ thuật (thứ tự RNG, ảnh trùng giữa các tập); phần tiếng Việt trỏ tới cây thư mục; interface không đổi -> chỉ sửa tài liệu, không chạy code.
- 2026-11-01 Thư: `train`, `evaluate`, `experiment`, `main`, notebook: notebook khớp `main.py` (`torch.zeros`, cố định số thread); `best_val_acc` báo đúng checkpoint được giữ; test không còn ghi đè checkpoint thật; thêm `save_classification_report`, `analyze_seed_runs`; thiết kế lại thí nghiệm thành 7 cấu hình, 5 seed; xóa BOM; cập nhật tài liệu từ một lần chạy -> 50 passed, đã đối chiếu notebook và `main.py`.
- 2026-11-01 Thư: `progress_status`: chuyển toàn bộ file sang tiếng Việt theo quyết định của nhóm trưởng, giữ nguyên nội dung -> chỉ sửa tài liệu, không chạy code.
- 2026-11-02 Toàn: `preprocessing`, `eda`: đổi sang dữ liệu chính thức của Zalando bằng `download_official.py` (torchvision, kiểm MD5), cùng tên file và định dạng CSV; đổi tên CSV Kaggle thành `*_kaggle.csv`; tạo lại split; sửa chữ Kaggle trong docstring `eda.py` và `project_description.md` -> 50 passed; 0 dòng trùng, 0 ảnh chung giữa các tập
- 2026-11-02 Toàn: notebook, README: sửa chữ về nguồn data ở ô 0, 1, 3 của notebook; thêm `README_official_data.md` (bản nháp README, chỉ đổi các đoạn về data) để nhóm trưởng so với `README.md` -> chỉ sửa tài liệu, không chạy code
- 2026-11-02 Toàn: notebook: chạy lại Phần A (cell 0-10: EDA, split, features, model) trên dữ liệu chính thức; cell 11-25 và phần chữ giữ nguyên, chờ Thư chạy lại -> 0 lỗi; Duplicate rows 0, mean 0,2859, std 0,3530
- 2026-11-07 Thư: `train`, `evaluate`, `visualize`, `experiment`, `main.py`: thêm chú thích tiếng Anh và tiếng Việt (`# VI:`) cho các dòng code quan trọng; chỉ thêm comment, không đổi code -> pytest 56 đạt
- 2026-11-07 Thư: notebook: bảng recall bỏ `recall_min`, `recall_max`; bảng cặp hay nhầm bỏ `mean_per_seed` (= total / 5); Summary viết gọn lại, số liệu không đổi -> notebook chạy lại 0 lỗi
- 2026-11-07 Thư: notebook, README, `docs/experiment.md`: bảng so sánh chỉ hiện các cột của quy tắc (`mean_diff`, `std_diff`, `seeds_higher`, `rule_clear`), bỏ p-value, Holm, khoảng tin cậy; kết luận không đổi -> notebook chạy lại 0 lỗi
- 2026-11-06 Thư: `experiment` (thêm `history`, `save_history`), `main.py`, `extended_study.py`, notebook (Stage 9b, 10b), README, `docs/`: lưu lịch sử theo epoch, chạy lại, kết quả giống hệt 04/11; bỏ số liệu Kaggle
- 2026-11-04 Thư: `main.py`, `extended_study.py`, notebook, README, `docs/`: tạo lại CSV chính thức trên máy nhóm trưởng; chạy lại `main.py`, `extended_study.py`, notebook trên data chính thức; gộp `README_official_data.md` của Toàn vào `README.md` rồi xóa bản nháp; thay mọi số trong README, chữ trong notebook và file này; thêm 9 hướng dẫn module trong `labs/lab1/docs/`; ghi phạm vi chỉ làm MLP -> 54 passed, notebook 0 lỗi và khớp `main.py`
- 2026-11-01 Thư: `experiment`, `extended_study`, notebook, README: thêm train accuracy, kiểm định t có hiệu chỉnh Holm, khoảng tin cậy 95%, nghiên cứu 20 epoch; đối chiếu CSV Kaggle với file chính thức; viết báo cáo chi tiết trong README; bỏ CNN vì thiếu thời gian -> 54 passed, chạy `main.py` hai lần ra kết quả y hệt.
- 2026-11-13 Thư: `docs/`: sửa tài liệu theo nhận xét của cô; thêm `rules/`, `templates/`, `plans/`, `lab1_description_v1.md`, `00_README_v1.md`; chuyển 9 hướng dẫn module vào `docs/phases/<module>_v1.md` (giữ lịch sử) rồi viết lại theo khuôn phase của M; trạng thái module ghi "Done (theo yêu cầu cũ) — chờ sửa theo yêu cầu mới" -> chỉ sửa tài liệu, không đổi code; `pytest labs/lab1/tests` 52 passed
- 2026-11-13 Thư: `README`, `docs/experiments/experiment_log.md`, notebook: chuyển sang tiếng Việt (dịch từ phần tiếng Anh vì đầy đủ hơn); notebook chỉ đổi 16 ô markdown, ô code và output giữ nguyên -> chỉ sửa tài liệu, không chạy lại notebook

## Vấn đề mở
- Đã xử lý: tham số `val_fraction` bị thêm nhầm vào `preprocessing.py` của Member A và đã xóa ở commit `e14d58e`; `main.py` và notebook dùng trực tiếp `val_ratio=0.1` để giữ nguyên module và docstring của Member A.
- Đã xử lý 2026-11-01: `best_val_acc` lấy từ epoch có validation loss thấp nhất, cùng epoch với checkpoint được giữ.
- Đã xử lý 2026-11-01: `torch.randn` trong notebook làm đổi thứ tự xáo dữ liệu; nay dùng `torch.zeros`, và `main.py`, notebook đều dùng 4 thread CPU.
- Đã xử lý 2026-11-02: đổi sang dữ liệu chính thức của Zalando; 43 dòng trùng và ảnh chung giữa các tập (9, 9, 1) của CSV Kaggle không còn (0 dòng trùng, 0 ảnh chung).
- Đã xử lý 2026-11-04: chạy lại `main.py`, notebook, `extended_study.py` trên dữ liệu mới và cập nhật README.
- Chưa xong: mỗi thành viên tự điền phần của mình trong `AI_usage.md`.
- Chưa kiểm chứng: giả thuyết về nguyên nhân lỗi của Shirt, và nguyên nhân CSV Kaggle khác file chính thức.
- Chưa làm: chạy trên máy khác. Trên cùng máy, `main.py` chạy hai lần cho kết quả y hệt (2026-11-04).
- Ngoài phạm vi (quyết định của nhóm, 2026-11-04): CNN; Lab 1 chỉ làm MLP.
- Chưa làm (2026-11-13): sửa code theo desc mới `docs/lab1_description_v1.md` (5 cấu hình, seed 42, tối đa 30 epoch + early stopping, log 3 file, `outputs/`); từng chỗ lệch ghi trong `docs/phases/<module>_v1.md`, mục "Chỗ code chưa khớp plan".
