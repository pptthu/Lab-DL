# Lab 2 — CNN tiền huấn luyện và fine-tuning trên CIFAR-10  (v1)
- Tác giả: Thư | Tạo: 2026-10-17T00:00:00+07:00 | Cập nhật: Đại, 2026-11-19T13:50:00+07:00
- Trạng thái: Done (normal run 2026-11-04, audit trong `docs/*.md`; kết quả ở `README.md` và `REPORT.md`)

## Background
Đề P2: tái sử dụng mạng CNN tiền huấn luyện (VGG, DenseNet, ResNet...) của torchvision cho bài toán phân loại ảnh 10 lớp bằng cách thay lớp cuối, đóng băng hoặc mở khóa lớp, chỉnh siêu tham số. Dữ liệu: CIFAR-10 qua `torchvision.datasets`.

## Purpose
- Làm: so sánh ResNet-18 và DenseNet-121 ở 2 chế độ (feature extraction — đóng băng backbone; fine-tuning — mở khóa khối cuối), dùng batch 64, seed 42, TensorBoard theo dõi:
  1. **Feature extraction:** `freeze_backbone=True`, chỉ huấn luyện classifier head mới
  2. **Fine-tuning:** mở thêm `n_blocks=1` khối sâu cuối bằng `unfreeze_last_blocks`
  3. So sánh 2 model × 2 chế độ bằng `run_comparison`, lưu kết quả ra CSV
- Không làm: huấn luyện từ đầu (random init), Vision Transformer, đa seed, kiểm định thống kê, test trên ảnh ngoài CIFAR-10, đo tốc độ inference.

## Input / Output
- Input: CIFAR-10 từ `torchvision.datasets.CIFAR10` (tải về `raw_data/`); split: `artifacts/splits/train_val_split.json` (45.000 train / 5.000 val, seed 42).
- Output: checkpoint `.pt` trong `artifacts/models/`; log TensorBoard trong `artifacts/logs/`; bảng so sánh CSV trong `artifacts/runs/`; 2 PNG EDA trong `artifacts/figures/`.
- Code: `labs/lab2/main.py` gọi 7 module `labs/lab2/<module>/<module>.py`.

## Interfaces (hợp đồng — `tests/test_smoke.py` kiểm tên hàm)

| Module | Phụ trách | Hàm / lớp | Đầu vào | Đầu ra |
|---|---|---|---|---|
| `eda` | Member A | `run_eda(data_root: str, out_dir: str)` | data_root, out_dir | dict: n_train, n_test, class_counts, image_size, channel_mean, channel_std; PNG trong out_dir |
| `preprocessing` | Member A | `load_raw_dataset(data_root: str, download: bool = True)` | data_root, download | (train_set, test_set) |
| `preprocessing` | Member A | `make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42)` | n_samples, val_ratio, seed | (train_idx, val_idx) |
| `preprocessing` | Member A | `save_split(train_idx, val_idx, path: str)` | indices + path | JSON file |
| `preprocessing` | Member A | `load_split(path: str)` | path | (train_idx, val_idx) |
| `features` | Member A | `build_transforms(image_size: int = 224, augment: bool = False)` | image_size, augment | (train_transform, eval_transform) |
| `features` | Member A | `make_loaders(train_set, test_set, train_idx, val_idx, train_tf, eval_tf, batch_size: int = 64)` | datasets, split indices, transforms, batch_size | (train_loader, val_loader, test_loader) |
| `model` | Member A | `build_model(name: str, num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True)` | name, num_classes, freeze_backbone, pretrained | nn.Module xuất logits [B, num_classes] |
| `model` | Member A | `describe_model(model)` | model | dict: total_params, trainable_params, layer_names |
| `model` | Member A | `unfreeze_last_blocks(model, n_blocks: int)` | model, n_blocks | model cập nhật tại chỗ |
| `train` | Member B | `train_model(model, train_loader, val_loader, epochs: int, lr: float, device, ckpt_path: str, writer=None, optimizer_name: str = 'adam', weight_decay: float = 0.0)` | model, loaders, epochs, lr, device, ckpt_path, writer, optimizer_name, weight_decay | dict history: train_loss, val_loss, val_acc; checkpoint; TensorBoard log |
| `train` | Member B | `save_checkpoint(model, path: str)` | model, path | file checkpoint |
| `evaluate` | Member B | `evaluate(model, loader, device)` | model, loader, device | (avg_loss, accuracy) |
| `evaluate` | Member B | `confusion_matrix(model, loader, device, num_classes: int = 10)` | model, loader, device | array [C, C] |
| `evaluate` | Member B | `top_confusions(cm, class_names, k: int = 5)` | cm, class_names, k | list of (true_class, pred_class, count) |
| `experiment` | Member B | `run_comparison(model_names: list, hparam_grid: list, loaders: tuple, device, log_root: str)` | model_names, hparam_grid, loaders, device, log_root | list of result dicts; TensorBoard runs dưới log_root |
| `experiment` | Member B | `save_results(results: list, csv_path: str)` | results, csv_path | CSV file |

## Success criteria
- `pytest labs/lab2/tests/` đạt toàn bộ; `ruff check` sạch trên các file đã sửa.
- Loader trả batch `[B, 3, H, W]` float32, nhãn int64 trong 0–9 (Milestone M1).
- `build_model('resnet18')` trả logits `[B, 10]`; `describe_model` in đúng số tham số (M2).
- `train_model` chạy 1 epoch và ghi log TensorBoard (M3).
- `evaluate` trả accuracy và confusion matrix (M4).
- So sánh ≥ 2 model/cấu hình, lưu bảng CSV, có báo cáo ngắn trong `README.md` (M5).
- Kết quả thực tế (2026-11-04): ResNet-18 feature extraction 78,94%, fine-tuning 91,90%; DenseNet-121 feature extraction 83,22%, fine-tuning 92,00%; DenseNet-121 test acc 91,32%.

## How-to-do
1. Chạy `run_eda` kiểm tra CIFAR-10: 50.000 train / 10.000 test, 10 lớp cân bằng, kích thước `(32, 32, 3)` (`eda`).
2. Nạp dataset thô, tạo split 90/10 cố định seed 42, lưu `train_val_split.json` (`preprocessing`).
3. Build transforms ImageNet (resize + ToTensor + chuẩn hoá), tạo 3 DataLoader (`features`).
4. Nạp backbone pre-trained, thay classifier head thành 10 lớp, đóng băng backbone (`model`).
5. Chạy `train_model` 1 epoch smoke run; sau đó normal run với TensorBoard (`train`).
6. `run_comparison` trên 2 model × 2 chế độ, lưu kết quả CSV; `evaluate` test trên model tốt nhất (`evaluate`, `experiment`).
7. Điền báo cáo ngắn `README.md`, kiểm tra notebook `lab2_presentation.ipynb`, audit code so với desc này.

## Cấu trúc thư mục
```text
labs/lab2/
├── README.md                        # báo cáo ngắn (EN + VI)
├── REPORT.md                        # báo cáo chi tiết
├── project_description.md           # file này
├── progress_status.md               # tiến độ, audit, nhật ký
├── main.py                          # chạy toàn bộ pipeline
├── requirements.txt
├── eda/eda.py
├── preprocessing/preprocessing.py
├── features/features.py
├── model/model.py
├── train/train.py
├── evaluate/evaluate.py
├── experiment/experiment.py
├── raw_data/                        # dữ liệu thô (không commit)
├── artifacts/                       # splits/ (commit) | models, figures, logs, runs (không commit)
├── docs/                            # hướng dẫn từng module
├── notebook/lab2_presentation.ipynb
└── tests/                           # test_smoke.py + test từng module
```

## Quyết định (người ghi)
- 2026-10-17 Thư: chốt 7 module, interface và milestone; `test_smoke.py` kiểm tên hàm.
- 2026-10-26 Bảo: Member B tự implement train, evaluate, experiment trên nhánh riêng sau khi nhận bàn giao từ Member A.
- 2026-11-04 Thư: TensorBoard là tuỳ chọn (lazy import, không lỗi khi thiếu); `unfreeze_last_blocks` tách khối VGG-16 theo `MaxPool2d` thay vì lấy 2 module cuối; `train_model` ghi thêm `best_epoch`, `best_val_loss`, `best_val_acc`.
- 2026-11-04 Thư: chọn DenseNet-121 fine-tuning là cấu hình tốt nhất (test 91,32%) — đúng quy tắc chọn trên val; không chạy test nhiều lần.
- 2026-11-04 Thư: `image_size=128` mặc định trong `main.py` thay vì 224 — chạy được trên CPU trong thời gian hợp lý; ghi rõ trong README.
