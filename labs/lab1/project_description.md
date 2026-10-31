# Lab 1: Fashion-MNIST classification with an MLP (PyTorch): project description (Define + Design)

## Define
- **Problem:** Classify 28x28 grayscale clothing images into 10 classes with a neural network trained in PyTorch.
- **Target users:** the lecturer (grading) and the group (learning).
- **Input / Output:** Input: CSV rows (label + 784 pixels, values 0-255). Output: trained model, accuracy, loss curves, predicted-vs-actual images, experiment table.
- **Data:** Kaggle `zalando-research/fashionmnist` (CSV: `label` + 784 pixel columns, 60,000 train rows). Check the real file names after download.
- **Core features:**
  1. Load the dataset and apply transforms
  2. Build a neural network (MLP)
  3. Training loop: forward, loss, backward, optimize
  4. Evaluate model accuracy
  5. Save and load the model
  6. Experiment with network/hyperparameters
  7. Visualize loss; display predicted vs actual images
- **Technology constraints:** Python 3, PyTorch, matplotlib, numpy, pandas; CPU is enough. Raw data is not committed. Group scope = MLP + experiments (CNN is optional extra work of the leader).
- **Evaluation criteria:** Practice 1 deliverables: Python code, brief report (README-style, about 200 words or fewer), loss graphs, image displays. Oral Q&A.

## Design
- **System architecture** (three layers, two entry points):
  - Data layer: `eda`, `preprocessing`, `features` (CSV -> checked arrays -> shared split -> normalized DataLoaders)
  - Model and training layer: `model`, `train`, `experiment` (network, training loop, best checkpoint, comparison of configurations)
  - Analysis layer: `evaluate`, `visualize` (metrics, confusion matrix, figures)
  - Entry points: `main.py` runs the whole pipeline; `notebook/lab1_presentation.ipynb` calls the same modules for the presentation. Outputs go to `artifacts/` (splits, models, figures, runs).
- **Data flow:** eda -> preprocessing -> features -> model -> train -> evaluate -> visualize ; experiment wraps model + train
- **Modules and owners** (each module = one sub-folder with one script):
- Member A: `eda`, `preprocessing`, `features`, `model`
- Member B: `train`, `evaluate`, `visualize`, `experiment`
- The team leader owns `main.py`, `README.md`, `progress_status.md`, `AI_usage.md` and PR review.
- **Module applicability rule:** if a module is not needed in this lab, mark it `N/A` in `progress_status.md` with a one-line reason. Do not delete it silently.
- **Interfaces** (names and signatures are a contract; the test in `tests/test_smoke.py` checks the names):

| Module | Owner | Function / class | Input | Output |
|---|---|---|---|---|
| `eda` | Member A | `run_eda(train_csv: str, out_dir: str)` | train_csv: path to the Kaggle training CSV; out_dir: folder for figures | dict with n_rows, class_counts, pixel_min, pixel_max, n_missing, n_duplicates; PNG files (class bar chart, sample image grid) in out_dir |
| `preprocessing` | Member A | `load_csv(csv_path: str)` | csv_path: path to a CSV file | (images: uint8 array [N,28,28], labels: int64 array [N]) |
| `preprocessing` | Member A | `make_split(labels, val_ratio: float = 0.1, seed: int = 42)` | labels: int64 array [N]; val_ratio; seed | (train_idx, val_idx): disjoint index arrays covering all N samples |
| `preprocessing` | Member A | `save_split(train_idx, val_idx, path: str)` | train_idx, val_idx, path (.json) | JSON file at path |
| `preprocessing` | Member A | `load_split(path: str)` | path (.json) | (train_idx, val_idx) |
| `features` | Member A | `compute_mean_std(train_images)` | train_images: uint8 array [N,28,28] (train indices only) | (mean, std) as floats on the 0-1 pixel scale |
| `features` | Member A | `build_transform(mean: float, std: float)` | mean, std | callable: uint8 [28,28] -> float32 tensor [1,28,28], normalized |
| `features` | Member A | `class FashionDataset` | __init__(images, labels, indices, transform) | __getitem__(i) -> (tensor [1,28,28], int label); __len__() -> number of samples |
| `features` | Member A | `make_loaders(train_ds, val_ds, test_ds, batch_size: int = 64)` | three Dataset objects; batch_size | (train_loader, val_loader, test_loader) |
| `model` | Member A | `class MLP` | __init__(hidden_sizes=(256,128), num_classes=10, dropout=0.0); forward(x: tensor [B,1,28,28]) | forward returns logits [B,10] (NO softmax: CrossEntropyLoss expects logits) |
| `model` | Member A | `count_parameters(model)` | model | int |
| `train` | Member B | `train_model(model, train_loader, val_loader, epochs: int, lr: float, device, ckpt_path: str, optimizer_name: str = 'adam', weight_decay: float = 0.0)` | model, loaders, epochs, lr, device, ckpt_path, optimizer_name ('adam' or 'sgd'), weight_decay | history dict: train_loss, val_loss, val_acc (one value per epoch); best model saved at ckpt_path |
| `train` | Member B | `save_checkpoint(model, path: str)` | model, path | file at path |
| `train` | Member B | `load_checkpoint(model, path: str, device)` | model (same architecture), path, device | the model with loaded weights, in eval mode |
| `evaluate` | Member B | `evaluate(model, loader, device)` | model, loader, device | (avg_loss: float, accuracy: float in 0-1) |
| `evaluate` | Member B | `confusion_matrix(model, loader, device, num_classes: int = 10)` | model, loader, device, num_classes | array [10,10]; rows = true class, columns = predicted class |
| `evaluate` | Member B | `per_class_accuracy(cm)` | cm: confusion matrix | dict: class index -> accuracy |
| `visualize` | Member B | `plot_loss_curves(history: dict, out_path: str)` | history from train_model; out_path (.png) | PNG file |
| `visualize` | Member B | `show_predictions(model, loader, class_names, device, out_path: str, n: int = 16)` | model, loader, class_names (10 names), device, out_path, n | PNG file |
| `visualize` | Member B | `plot_confusion_matrix(cm, class_names, out_path: str)` | cm, class_names, out_path | PNG file |
| `experiment` | Member B | `run_experiments(configs: list, loaders: tuple, device)` | configs: list of dicts; loaders: (train, val, test); device | list of result dicts: config, best_val_acc, n_params, train_seconds |
| `experiment` | Member B | `save_results(results: list, csv_path: str)` | results, csv_path | CSV file |

- **Folder structure:**
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
└── tests/                   # interface contract (test_smoke.py) + one test file per module
```
- **Milestones:**
  - M1: loaders return a batch of shape [B,1,28,28] with labels 0-9
  - M2: model forward on a dummy batch returns [B,10]
  - M3: train_model runs 1 epoch, loss goes down, checkpoint saved and reloaded
  - M4: evaluate + visualize produce numbers and figures from the saved model
  - M5: experiments (3 or more configs) + brief report
- **Technical risks:**
  - Softmax before CrossEntropyLoss: the model must output logits
  - Test set used for tuning: split validation from train
  - Normalization stats computed on all data: compute on train only
  - Wrong reshape/dtype when reading the CSV: print shapes and value range
  - Large CSV committed to git: keep raw_data/ ignored
  - Extra random numbers change the shuffle order, so the same seed and configuration can give different results: state which run each reported number comes from
  - Identical images in different splits (train-val 9, train-test 9, val-test 1): kept by team decision, disclose as a limitation

---

**Tiếng Việt**

# Lab 1: Phân loại Fashion-MNIST bằng MLP (PyTorch): mô tả dự án (Define + Design)

## Define
- **Bài toán:** Phân loại ảnh quần áo xám 28x28 thành 10 lớp bằng mạng nơ-ron huấn luyện trong PyTorch.
- **Người dùng:** cô (chấm bài) và nhóm (học).
- **Đầu vào / Đầu ra:** Đầu vào: các dòng CSV (label + 784 pixel, giá trị 0-255). Đầu ra: model đã huấn luyện, accuracy, đồ thị loss, ảnh dự đoán vs thực tế, bảng thí nghiệm.
- **Dữ liệu:** Kaggle `zalando-research/fashionmnist` (CSV: cột `label` + 784 cột pixel, 60.000 dòng train). Kiểm tra tên file thật sau khi tải.
- **Tính năng chính:**
  1. Nạp dataset và áp dụng transform
  2. Xây dựng mạng nơ-ron (MLP)
  3. Vòng lặp huấn luyện: forward, loss, backward, optimize
  4. Đánh giá accuracy
  5. Lưu và nạp model
  6. Thử nghiệm mạng/siêu tham số
  7. Vẽ loss; hiển thị ảnh dự đoán vs thực tế
- **Ràng buộc công nghệ:** Python 3, PyTorch, matplotlib, numpy, pandas; CPU là đủ. Không commit dữ liệu thô. Phạm vi nhóm = MLP + thí nghiệm (CNN là phần làm thêm riêng của nhóm trưởng).
- **Tiêu chí đánh giá:** Sản phẩm Practice 1: code Python, báo cáo ngắn (kiểu README, khoảng 200 từ trở xuống), đồ thị loss, ảnh hiển thị. Vấn đáp.

## Design
- **Cấu trúc hệ thống** (ba lớp, hai điểm chạy):
  - Lớp dữ liệu: `eda`, `preprocessing`, `features` (CSV -> mảng đã kiểm tra -> split dùng chung -> DataLoader đã chuẩn hóa)
  - Lớp mô hình và huấn luyện: `model`, `train`, `experiment` (mạng, vòng lặp huấn luyện, checkpoint tốt nhất, so sánh cấu hình)
  - Lớp phân tích: `evaluate`, `visualize` (chỉ số, ma trận nhầm lẫn, hình)
  - Điểm chạy: `main.py` chạy toàn bộ pipeline; `notebook/lab1_presentation.ipynb` gọi cùng các module để trình bày. Kết quả lưu vào `artifacts/` (splits, models, figures, runs).
- **Luồng dữ liệu:** eda -> preprocessing -> features -> model -> train -> evaluate -> visualize ; experiment wraps model + train
- **Module và người phụ trách** (mỗi module = một sub-folder + một script):
- Member A: `eda`, `preprocessing`, `features`, `model`
- Member B: `train`, `evaluate`, `visualize`, `experiment`
- Nhóm trưởng phụ trách `main.py`, `README.md`, `progress_status.md`, `AI_usage.md` và duyệt PR.
- **Quy tắc linh hoạt module:** nếu lab này không cần một module, ghi `N/A` trong `progress_status.md` kèm lý do một dòng. Không xóa im lặng.
- **Interface** (tên hàm và chữ ký là hợp đồng; `tests/test_smoke.py` kiểm tra tên hàm):

| Module | Phụ trách | Hàm / lớp | Đầu vào | Đầu ra |
|---|---|---|---|---|
| `eda` | Member A | `run_eda(train_csv: str, out_dir: str)` | train_csv: đường dẫn CSV train của Kaggle; out_dir: thư mục lưu hình | dict gồm n_rows, class_counts, pixel_min, pixel_max, n_missing, n_duplicates; file PNG (biểu đồ số lượng mỗi lớp, lưới ảnh mẫu) trong out_dir |
| `preprocessing` | Member A | `load_csv(csv_path: str)` | csv_path: đường dẫn file CSV | (images: mảng uint8 [N,28,28], labels: mảng int64 [N]) |
| `preprocessing` | Member A | `make_split(labels, val_ratio: float = 0.1, seed: int = 42)` | labels: mảng int64 [N]; val_ratio; seed | (train_idx, val_idx): hai mảng chỉ số không giao nhau, gộp lại đủ N mẫu |
| `preprocessing` | Member A | `save_split(train_idx, val_idx, path: str)` | train_idx, val_idx, path (.json) | file JSON tại path |
| `preprocessing` | Member A | `load_split(path: str)` | path (.json) | (train_idx, val_idx) |
| `features` | Member A | `compute_mean_std(train_images)` | train_images: mảng uint8 [N,28,28] (chỉ chỉ số train) | (mean, std) dạng số thực trên thang pixel 0-1 |
| `features` | Member A | `build_transform(mean: float, std: float)` | mean, std | hàm: uint8 [28,28] -> tensor float32 [1,28,28] đã chuẩn hóa |
| `features` | Member A | `class FashionDataset` | __init__(images, labels, indices, transform) | __getitem__(i) -> (tensor [1,28,28], nhãn int); __len__() -> số mẫu |
| `features` | Member A | `make_loaders(train_ds, val_ds, test_ds, batch_size: int = 64)` | ba đối tượng Dataset; batch_size | (train_loader, val_loader, test_loader) |
| `model` | Member A | `class MLP` | __init__(hidden_sizes=(256,128), num_classes=10, dropout=0.0); forward(x: tensor [B,1,28,28]) | forward trả logits [B,10] (KHÔNG softmax: CrossEntropyLoss cần logits) |
| `model` | Member A | `count_parameters(model)` | model | int |
| `train` | Member B | `train_model(model, train_loader, val_loader, epochs: int, lr: float, device, ckpt_path: str, optimizer_name: str = 'adam', weight_decay: float = 0.0)` | model, loaders, epochs, lr, device, ckpt_path, optimizer_name ('adam' hoặc 'sgd'), weight_decay | dict history: train_loss, val_loss, val_acc (mỗi epoch một giá trị); model tốt nhất lưu tại ckpt_path |
| `train` | Member B | `save_checkpoint(model, path: str)` | model, path | file tại path |
| `train` | Member B | `load_checkpoint(model, path: str, device)` | model (cùng kiến trúc), path, device | model đã nạp trọng số, ở chế độ eval |
| `evaluate` | Member B | `evaluate(model, loader, device)` | model, loader, device | (avg_loss: float, accuracy: float trong 0-1) |
| `evaluate` | Member B | `confusion_matrix(model, loader, device, num_classes: int = 10)` | model, loader, device, num_classes | mảng [10,10]; hàng = lớp thật, cột = lớp dự đoán |
| `evaluate` | Member B | `per_class_accuracy(cm)` | cm: ma trận nhầm lẫn | dict: chỉ số lớp -> độ chính xác |
| `visualize` | Member B | `plot_loss_curves(history: dict, out_path: str)` | history từ train_model; out_path (.png) | file PNG |
| `visualize` | Member B | `show_predictions(model, loader, class_names, device, out_path: str, n: int = 16)` | model, loader, class_names (10 tên), device, out_path, n | file PNG |
| `visualize` | Member B | `plot_confusion_matrix(cm, class_names, out_path: str)` | cm, class_names, out_path | file PNG |
| `experiment` | Member B | `run_experiments(configs: list, loaders: tuple, device)` | configs: danh sách dict; loaders: (train, val, test); device | danh sách dict kết quả: config, best_val_acc, n_params, train_seconds |
| `experiment` | Member B | `save_results(results: list, csv_path: str)` | results, csv_path | file CSV |

- **Cấu trúc thư mục:** xem cây thư mục ở phần tiếng Anh phía trên (tên file và thư mục giống nhau).
- **Mốc (milestone):**
  - M1: loader trả batch shape [B,1,28,28], nhãn 0-9
  - M2: forward model trên batch giả trả [B,10]
  - M3: train_model chạy 1 epoch, loss giảm, checkpoint lưu và nạp lại được
  - M4: evaluate + visualize tạo ra số liệu và hình từ model đã lưu
  - M5: thí nghiệm (từ 3 config) + báo cáo ngắn
- **Rủi ro kỹ thuật:**
  - Thêm softmax trước CrossEntropyLoss: model phải xuất logits
  - Dùng tập test để chỉnh tham số: phải tách validation từ train
  - Tính mean/std trên toàn bộ dữ liệu: chỉ tính trên train
  - Reshape/dtype sai khi đọc CSV: in shape và khoảng giá trị
  - Commit CSV lớn lên git: giữ raw_data/ trong .gitignore
  - Số ngẫu nhiên phát sinh thêm làm đổi thứ tự xáo, nên cùng seed và cấu hình vẫn có thể ra kết quả khác: ghi rõ mỗi số báo cáo đến từ lần chạy nào
  - Ảnh giống hệt nhau nằm ở các tập khác nhau (train-val 9, train-test 9, val-test 1): nhóm quyết định giữ nguyên, ghi rõ là giới hạn
