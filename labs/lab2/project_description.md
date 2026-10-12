# Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch): project description (Define + Design)

## Define
- **Problem:** Reuse pre-trained torchvision networks (VGG, DenseNet, ResNet...) for a new 10-class image task by replacing the final layer, freezing or unfreezing layers, and tuning hyperparameters.
- **Target users:** the lecturer (grading) and the group (learning).
- **Input / Output:** Input: CIFAR-10 images. Output: fine-tuned model(s), accuracy per model/setting, TensorBoard logs, comparison table.
- **Data:** ASSUMPTION (confirm with the lecturer): CIFAR-10 via `torchvision.datasets` (the slide uses it as the example).
- **Core features:**
  1. Load pre-trained models from torchvision.models and print their architecture
  2. Replace the final layer; freeze (transfer learning) or unfreeze last layers (fine-tuning)
  3. Prepare data with torchvision.transforms and DataLoader
  4. Train with a loss function, optimizer and training loop
  5. Evaluate on a test set
  6. Experiment with different models and hyperparameters
  7. Monitor training with TensorBoard
- **Technology constraints:** PyTorch + torchvision + TensorBoard. Pre-trained models at 224x224 are slow on CPU: use a smaller image size or a data subset and document it.
- **Evaluation criteria:** Deliverables are not stated on the Practice 2 slide: confirm with the lecturer (assumed: code, short report, TensorBoard screenshots). Oral Q&A.

## Design
- **Data flow:** eda -> preprocessing -> features -> model (backbone + header) -> train (+TensorBoard) -> evaluate ; experiment wraps model + train
- **Modules and owners** (each module = one sub-folder with one script):
- Member A: `eda`, `preprocessing`, `features`, `model`
- Member B: `train`, `evaluate`, `experiment`
- The team leader owns `main.py`, `README.md`, `progress_status.md`, `AI_usage.md` and PR review.
- **Module applicability rule:** if a module is not needed in this lab, mark it `N/A` in `progress_status.md` with a one-line reason. Do not delete it silently.
- **Interfaces** (names and signatures are a contract; the test in `tests/test_smoke.py` checks the names):

| Module | Owner | Function / class | Input | Output |
|---|---|---|---|---|
| `eda` | Member A | `run_eda(data_root: str, out_dir: str)` | data_root: dataset folder (downloaded by torchvision); out_dir | dict: n_train, n_test, class_counts, image_size, channel_mean, channel_std; PNG files in out_dir |
| `preprocessing` | Member A | `load_raw_dataset(data_root: str, download: bool = True)` | data_root, download | (train_set, test_set) |
| `preprocessing` | Member A | `make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42)` | n_samples, val_ratio, seed | (train_idx, val_idx) |
| `preprocessing` | Member A | `save_split(train_idx, val_idx, path: str)` | indices + path / path | JSON file / (train_idx, val_idx) |
| `features` | Member A | `build_transforms(image_size: int = 224, augment: bool = False)` | image_size, augment | (train_transform, eval_transform) |
| `features` | Member A | `make_loaders(train_set, test_set, train_idx, val_idx, train_tf, eval_tf, batch_size: int = 64)` | datasets, split indices, transforms, batch_size | (train_loader, val_loader, test_loader) |
| `model` | Member A | `build_model(name: str, num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True)` | name, num_classes, freeze_backbone, pretrained | nn.Module that outputs logits [B,num_classes]; backbone frozen if requested |
| `model` | Member A | `describe_model(model)` | model | dict: total_params, trainable_params, layer names |
| `model` | Member A | `unfreeze_last_blocks(model, n_blocks: int)` | model, n_blocks | model updated in place |
| `train` | Member B | `train_model(model, train_loader, val_loader, epochs: int, lr: float, device, ckpt_path: str, writer=None, optimizer_name: str = 'adam', weight_decay: float = 0.0)` | model, loaders, epochs, lr, device, ckpt_path, writer (SummaryWriter or None), optimizer_name, weight_decay | history dict: train_loss, val_loss, val_acc; checkpoint at ckpt_path; TensorBoard logs |
| `train` | Member B | `save_checkpoint(model, path: str)` | model, path (, device) | file / model in eval mode |
| `evaluate` | Member B | `evaluate(model, loader, device)` | model, loader, device | (avg_loss, accuracy) |
| `evaluate` | Member B | `confusion_matrix(model, loader, device, num_classes: int = 10)` | model, loader, device | array [C,C] (rows = true, columns = predicted) |
| `evaluate` | Member B | `top_confusions(cm, class_names, k: int = 5)` | cm, class_names, k | list of (true_class, predicted_class, count) |
| `experiment` | Member B | `run_comparison(model_names: list, hparam_grid: list, loaders: tuple, device, log_root: str)` | model_names (e.g. ['resnet18','vgg16']), hparam_grid (list of dicts: lr, batch_size, freeze...), loaders, device, log_root | list of result dicts: model, hparams, best_val_acc, train_seconds; TensorBoard runs under log_root |
| `experiment` | Member B | `save_results(results: list, csv_path: str)` | results, csv_path | CSV file |

- **Folder structure:**
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
- **Milestones:**
  - M1: loaders return a batch [B,3,H,W] with labels 0-9
  - M2: build_model returns logits [B,10]; describe_model prints parameter counts
  - M3: train_model runs 1 epoch and writes TensorBoard logs
  - M4: evaluate gives accuracy and confusion matrix
  - M5: comparison of 2 or more models/settings + short report
- **Technical risks:**
  - Slow training on CPU: smaller image_size / subset / fewer epochs
  - Deprecated API: use weights=..., not pretrained=True, in recent torchvision
  - Forgetting to replace the final layer: output size must be 10
  - Wrong normalization: use ImageNet mean/std for ImageNet-pretrained models
  - Dataset download time and disk space: download once, keep raw_data/ ignored

---

**Tiếng Việt**

# Lab 2: CNN pre-trained và fine-tuning trên CIFAR-10 (PyTorch): mô tả dự án (Define + Design)

## Define
- **Bài toán:** Tái sử dụng mạng pre-trained của torchvision (VGG, DenseNet, ResNet...) cho bài toán ảnh 10 lớp bằng cách thay lớp cuối, đóng băng/mở khóa lớp và chỉnh siêu tham số.
- **Người dùng:** cô (chấm bài) và nhóm (học).
- **Đầu vào / Đầu ra:** Đầu vào: ảnh CIFAR-10. Đầu ra: model đã fine-tune, accuracy theo từng model/cấu hình, log TensorBoard, bảng so sánh.
- **Dữ liệu:** GIẢ ĐỊNH (cần hỏi lại cô): CIFAR-10 qua `torchvision.datasets` (slide dùng làm ví dụ).
- **Tính năng chính:**
  1. Nạp model pre-trained từ torchvision.models và in cấu trúc
  2. Thay lớp cuối; đóng băng (transfer learning) hoặc mở khóa các lớp cuối (fine-tuning)
  3. Chuẩn bị dữ liệu bằng torchvision.transforms và DataLoader
  4. Huấn luyện với hàm loss, optimizer và vòng lặp huấn luyện
  5. Đánh giá trên tập test
  6. Thử nhiều model và siêu tham số
  7. Theo dõi huấn luyện bằng TensorBoard
- **Ràng buộc công nghệ:** PyTorch + torchvision + TensorBoard. Model pre-trained ở 224x224 chạy chậm trên CPU: dùng kích thước ảnh nhỏ hơn hoặc tập con dữ liệu và ghi rõ.
- **Tiêu chí đánh giá:** Slide Practice 2 không nêu sản phẩm nộp: hỏi lại cô (giả định: code, báo cáo ngắn, ảnh chụp TensorBoard). Vấn đáp.

## Design
- **Luồng dữ liệu:** eda -> preprocessing -> features -> model (backbone + header) -> train (+TensorBoard) -> evaluate ; experiment wraps model + train
- **Module và người phụ trách** (mỗi module = một sub-folder + một script):
- Member A: `eda`, `preprocessing`, `features`, `model`
- Member B: `train`, `evaluate`, `experiment`
- Nhóm trưởng phụ trách `main.py`, `README.md`, `progress_status.md`, `AI_usage.md` và duyệt PR.
- **Quy tắc linh hoạt module:** nếu lab này không cần một module, ghi `N/A` trong `progress_status.md` kèm lý do một dòng. Không xóa im lặng.
- **Interface** (tên hàm và chữ ký là hợp đồng; `tests/test_smoke.py` kiểm tra tên hàm):

| Module | Phụ trách | Hàm / lớp | Đầu vào | Đầu ra |
|---|---|---|---|---|
| `eda` | Member A | `run_eda(data_root: str, out_dir: str)` | data_root: thư mục dữ liệu (torchvision tải về); out_dir | dict: n_train, n_test, class_counts, image_size, channel_mean, channel_std; file PNG trong out_dir |
| `preprocessing` | Member A | `load_raw_dataset(data_root: str, download: bool = True)` | data_root, download | (train_set, test_set) |
| `preprocessing` | Member A | `make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42)` | n_samples, val_ratio, seed | (train_idx, val_idx) |
| `preprocessing` | Member A | `save_split(train_idx, val_idx, path: str)` | chỉ số + path / path | file JSON / (train_idx, val_idx) |
| `features` | Member A | `build_transforms(image_size: int = 224, augment: bool = False)` | image_size, augment | (train_transform, eval_transform) |
| `features` | Member A | `make_loaders(train_set, test_set, train_idx, val_idx, train_tf, eval_tf, batch_size: int = 64)` | dataset, chỉ số split, transform, batch_size | (train_loader, val_loader, test_loader) |
| `model` | Member A | `build_model(name: str, num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True)` | name, num_classes, freeze_backbone, pretrained | nn.Module xuất logits [B,num_classes]; backbone bị đóng băng nếu được yêu cầu |
| `model` | Member A | `describe_model(model)` | model | dict: total_params, trainable_params, tên các lớp |
| `model` | Member A | `unfreeze_last_blocks(model, n_blocks: int)` | model, n_blocks | model được cập nhật tại chỗ |
| `train` | Member B | `train_model(model, train_loader, val_loader, epochs: int, lr: float, device, ckpt_path: str, writer=None, optimizer_name: str = 'adam', weight_decay: float = 0.0)` | model, loaders, epochs, lr, device, ckpt_path, writer (SummaryWriter hoặc None), optimizer_name, weight_decay | dict history: train_loss, val_loss, val_acc; checkpoint tại ckpt_path; log TensorBoard |
| `train` | Member B | `save_checkpoint(model, path: str)` | model, path (, device) | file / model ở chế độ eval |
| `evaluate` | Member B | `evaluate(model, loader, device)` | model, loader, device | (avg_loss, accuracy) |
| `evaluate` | Member B | `confusion_matrix(model, loader, device, num_classes: int = 10)` | model, loader, device | mảng [C,C] (hàng = thật, cột = dự đoán) |
| `evaluate` | Member B | `top_confusions(cm, class_names, k: int = 5)` | cm, class_names, k | danh sách (lớp thật, lớp dự đoán, số lần) |
| `experiment` | Member B | `run_comparison(model_names: list, hparam_grid: list, loaders: tuple, device, log_root: str)` | model_names (vd ['resnet18','vgg16']), hparam_grid (danh sách dict: lr, batch_size, freeze...), loaders, device, log_root | danh sách dict kết quả: model, hparams, best_val_acc, train_seconds; run TensorBoard trong log_root |
| `experiment` | Member B | `save_results(results: list, csv_path: str)` | results, csv_path | file CSV |

- **Mốc (milestone):**
  - M1: loader trả batch [B,3,H,W], nhãn 0-9
  - M2: build_model trả logits [B,10]; describe_model in số tham số
  - M3: train_model chạy 1 epoch và ghi log TensorBoard
  - M4: evaluate cho accuracy và confusion matrix
  - M5: so sánh từ 2 model/cấu hình + báo cáo ngắn
- **Rủi ro kỹ thuật:**
  - Huấn luyện chậm trên CPU: giảm image_size / dùng tập con / ít epoch
  - API cũ: torchvision mới dùng weights=..., không dùng pretrained=True
  - Quên thay lớp cuối: đầu ra phải là 10
  - Chuẩn hóa sai: dùng mean/std ImageNet cho model pre-trained trên ImageNet
  - Thời gian tải dataset và dung lượng ổ đĩa: tải một lần, giữ raw_data/ trong .gitignore
