# Tiến độ Lab 1

Cập nhật file này sau MỖI bước.
Bảng dưới đây là nguồn thông tin duy nhất về việc ai đang làm gì.

| Module | Phụ trách | Trạng thái (chưa làm / đang làm / xong) | Test hợp đồng (đạt / lỗi / chưa chạy) | Chạy thử trên dữ liệu thật (đạt / lỗi / chưa chạy) | Ghi chú |
|---|---|---|---|---|---|
| `eda` | Member A (Toàn) | xong | đạt (3 test) | đạt | 60.000 dòng, 10 lớp cân bằng, giữ 43 dòng trùng, 0 giá trị thiếu |
| `preprocessing` | Member A (Toàn) | xong | đạt (6 test) | đạt | Chia phân tầng 90/10, lưu tại artifacts/splits/train_val_split.json |
| `features` | Member A (Toàn) | xong | đạt (6 test) | đạt | Train mean 0,2860, std 0,3527; loader trả [64,1,28,28] float32 |
| `model` | Member A (Toàn) | xong | đạt (6 test) | đạt | MLP 784->256->128->10: 235.146 tham số huấn luyện được |
| `train` | Member B (Thư) | xong | đạt (dữ liệu giả, 4 test) | đạt (VAL 6.000: loss 0,3031, acc 89,38%) | 5 epoch, Adam lr=1e-3, checkpoint tốt nhất lưu tại artifacts/models/mlp_baseline.pt |
| `evaluate` | Member B (Thư) | xong | đạt (dữ liệu giả, 5 test) | đạt (VAL 6.000: loss 0,3031, acc 89,38%) | Trouser cao nhất 98,33%, Shirt thấp nhất 66,33%, ROC-AUC 0,9923; báo cáo từng lớp lưu ra CSV |
| `visualize` | Member B (Thư) | xong | đạt (dữ liệu giả, 3 test) | đạt (lưu 3 hình trong artifacts/figures/) | loss_curves.png, predictions_grid.png, confusion_matrix.png; hình của lần chạy 2026-11-01 chưa được xem lại bằng mắt |
| `experiment` | Member B (Thư) | xong | đạt (dữ liệu giả, 8 test) | đạt (35 lần chạy, lưu tại artifacts/runs/experiment_results.csv) | 7 cấu hình, mỗi cấu hình 5 seed; test của baseline 88,42 +/- 0,37% |
| `main.py`, README, tích hợp | Nhóm trưởng (Thư) | xong | đạt (chạy thử 1 epoch) | đạt (toàn bộ pipeline) | Báo cáo ngắn và báo cáo chi tiết trong README, `extended_study.py`, lab1_presentation.ipynb (26 ô), 54 test đều đạt (python -m pytest labs/lab1/tests tests -q -> 54 passed) |

## Bàn giao A -> B (tuần tự, không xen kẽ)
Member B chỉ bắt đầu code khi TẤT CẢ các điều kiện sau đều đúng:
- [x] file JSON chia dữ liệu đã commit trong artifacts/splits/ (chỉ số train/val): `train_val_split.json`
- [x] make_loaders trả batch shape [B,1,28,28] với nhãn 0-9
- [x] MLP forward trên batch giả trả logits [B,10]
- [x] mọi module của Member A đều `xong` trong file này và `pytest labs/lab1/tests` đạt
- [x] các PR của Member A đã merge vào `main`; Member B tạo nhánh từ `main` mới nhất
- [x] Member A gửi tin bàn giao (đã làm gì, lệnh chạy chính xác, vấn đề đã biết)
Sau bàn giao, Member A chỉ trả lời câu hỏi và review PR, không sửa module của Member B.

## Kiểm chứng: kết quả hiện tại (2026-11-01, Python 3.13.7, torch 2.14.1 CPU, 4 thread)
Các số này thay thế mọi kết quả trong phần kiểm chứng cũ bên dưới.
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
- 2026-11-01 Thư: `experiment`, `extended_study`, notebook, README: thêm train accuracy, kiểm định t có hiệu chỉnh Holm, khoảng tin cậy 95%, nghiên cứu 20 epoch; đối chiếu CSV Kaggle với file chính thức; viết báo cáo chi tiết trong README; bỏ CNN vì thiếu thời gian -> 54 passed, chạy `main.py` hai lần ra kết quả y hệt.

## Vấn đề mở
- Đã xử lý: tham số `val_fraction` bị thêm nhầm vào `preprocessing.py` của Member A và đã xóa ở commit `e14d58e`; `main.py` và notebook dùng trực tiếp `val_ratio=0.1` để giữ nguyên module và docstring của Member A.
- Đã xử lý 2026-11-01: `best_val_acc` lấy từ epoch có validation loss thấp nhất, cùng epoch với checkpoint được giữ.
- Đã xử lý 2026-11-01: `torch.randn` trong notebook làm đổi thứ tự xáo dữ liệu; nay dùng `torch.zeros`, và `main.py`, notebook đều dùng 4 thread CPU.
- Giữ 43 dòng trùng trong CSV train của Kaggle (quyết định của nhóm). Ảnh giống hệt nhau giữa các tập: train-val 9, train-test 9, val-test 1 (đã ghi trong README).
- Chưa xong: mỗi thành viên tự điền phần của mình trong `AI_usage.md`.
- Chưa kiểm chứng: giả thuyết về nguyên nhân lỗi của Shirt, và nguyên nhân CSV Kaggle khác file chính thức.
- Chưa làm: chạy trên máy khác; CNN (bỏ vì thiếu thời gian).
