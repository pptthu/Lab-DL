# Experiment Module Documentation (`labs/lab2/experiment/experiment.py`)

## 1. Overview
The `experiment` module orchestrates comparative benchmarking across multiple pre-trained model architectures (e.g., ResNet-18, DenseNet-121, VGG-16) and hyperparameter combinations (learning rates, frozen vs. unfrozen backbone configurations, optimizers). It automates training runs, records isolated TensorBoard logs per configuration, and summarizes empirical metrics into structured CSV reports.

- **File location:** `labs/lab2/experiment/experiment.py`
- **Assigned owner:** Member B
- **Dependencies:** `torch.utils.data.DataLoader`, `torch.utils.tensorboard.SummaryWriter`, `csv`, `time`, `pathlib`, `os`

---

## 2. Functions & API Contract

### `run_comparison(...) -> List[Dict[str, Any]]`
Conducts multi-model and hyperparameter grid experiments sequentially.

#### Function Signature:
```python
def run_comparison(
    model_names: List[str],
    hparam_grid: List[Dict[str, Any]],
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    device: Any,
    log_root: str,
    build_model_fn: Optional[Any] = None,
    unfreeze_fn: Optional[Any] = None,
    train_model_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]
```

#### Arguments:
- `model_names` (`List[str]`): List of architectures to benchmark (e.g. `["resnet18", "densenet121", "vgg16"]`).
- `hparam_grid` (`List[Dict[str, Any]]`): List of dictionaries specifying hyperparameter configurations. Supported keys include:
  - `lr` (`float`): Learning rate (default: `1e-3`).
  - `epochs` (`int`): Training epochs per run (default: `1`).
  - `optimizer_name` (`str`): `"adam"` or `"sgd"` (default: `"adam"`).
  - `weight_decay` (`float`): Regularization factor (default: `0.0`).
  - `freeze_backbone` (`bool`): Whether backbone is initially frozen (default: `True`).
  - `pretrained` (`bool`): Load pre-trained weights (default: `True`).
  - `unfreeze_blocks` (`int`): Number of deep blocks to unfreeze for fine-tuning (default: `0`).
- `loaders` (`tuple`): `(train_loader, val_loader, test_loader)`.
- `device`: Execution hardware device.
- `log_root` (`str`): Base directory for TensorBoard runs and run checkpoints.
- `build_model_fn`, `unfreeze_fn`, `train_model_fn`: Optional function injection hooks (falls back to importing from `model` and `train` modules).

#### Workflow per Experiment Run:
1. Generates an isolated subfolder in `log_root` named `{model_name}_run{idx}_lr{lr}_freeze{freeze_backbone}`.
2. Initializes a dedicated `SummaryWriter` for that run.
3. Builds the model via `build_model_fn`.
4. If `unfreeze_blocks > 0`, unfreezes deep blocks using `unfreeze_fn`.
5. Trains the model using `train_model_fn` while timing wall-clock duration with `time.perf_counter()`.
6. Extracts `best_val_acc` and records metadata.

#### Returns:
A list of run dictionaries:
```python
[
    {
        "model": "resnet18",
        "hparams": { "lr": 0.001, "epochs": 5, ... },
        "best_val_acc": 0.7850,
        "train_seconds": 124.5,
        "checkpoint": ".../resnet18_run0_best.pt"
    },
    ...
]
```

---

### `save_results(results: List[Dict[str, Any]], csv_path: str) -> None`
Exports benchmark results to a formatted CSV file.

#### Arguments:
- `results` (`List[Dict[str, Any]]`): Output from `run_comparison`.
- `csv_path` (`str`): Target destination CSV file path.

#### CSV Header Format:
Flattened column headers containing:
`model`, `hp_<param1>`, `hp_<param2>`, ..., `best_val_acc`, `train_seconds`, `checkpoint`.

---

## 3. Key Implementation Details & Defense Q&A
- **Q: Why isolate TensorBoard writers into subdirectories?**
  - **A:** TensorBoard compares multiple runs side-by-side when each run has its own subfolder inside `log_root`. Placing all scalar outputs in one folder collides graphs and makes individual configuration tracking impossible.
- **Q: Why use `time.perf_counter()` rather than `time.time()`?**
  - **A:** `time.perf_counter()` uses a monotonic clock with the highest available resolution on the system, unaffected by system clock adjustments or daylight saving changes.
- **Q: How does this module support Dependency Injection?**
  - **A:** The arguments `build_model_fn`, `unfreeze_fn`, and `train_model_fn` allow test suites or alternative modules to mock or inject custom logic without modifying the core experiment runner.

---

**Tiếng Việt**

# Tài liệu Module Thí Nghiệm & So Sánh (`labs/lab2/experiment/experiment.py`)

## 1. Tổng quan
Module `experiment` đảm nhiệm việc tự động hóa các thử nghiệm so sánh giữa nhiều kiến trúc mô hình (ResNet-18, DenseNet-121, VGG-16) và nhiều cấu hình siêu tham số (Learning Rate, đóng băng/mở khóa backbone, Optimizer). Module tạo các thư mục log TensorBoard riêng biệt cho từng lượt chạy và xuất bảng tổng hợp kết quả ra file CSV.

- **Vị trí file:** `labs/lab2/experiment/experiment.py`
- **Người phụ trách:** Member B
- **Thư viện phụ thuộc:** `torch.utils.data.DataLoader`, `torch.utils.tensorboard.SummaryWriter`, `csv`, `time`, `pathlib`, `os`

---

## 2. Các hàm & Giao diện lập trình (API)

### `run_comparison(...) -> List[Dict[str, Any]]`
Thực hiện tuần tự các lượt chạy thử nghiệm dựa trên danh sách mô hình và lưới siêu tham số (hyperparameter grid).

#### Các tham số chính:
- `model_names`: Danh sách tên các mô hình cần so sánh (ví dụ: `["resnet18", "densenet121", "vgg16"]`).
- `hparam_grid`: Danh sách các dictionary chứa cấu hình siêu tham số (`lr`, `epochs`, `optimizer_name`, `freeze_backbone`, `unfreeze_blocks`, v.v.).
- `loaders`: Tuple chứa `(train_loader, val_loader, test_loader)`.
- `device`: Thiết bị chạy.
- `log_root`: Thư mục gốc chứa logs TensorBoard và checkpoint mô hình.

#### Quy trình mỗi lượt chạy:
1. Tạo một thư mục con riêng biệt trong `log_root` mang tên đặc trưng của cấu hình đó.
2. Khởi tạo `SummaryWriter` cho riêng lượt chạy này.
3. Tạo mô hình, áp dụng mở khóa các block sâu nếu `unfreeze_blocks > 0`.
4. Huấn luyện mô hình và đo đạc chính xác thời gian thực thi bằng `time.perf_counter()`.
5. Đóng writer và lưu lại độ chính xác validation cao nhất (`best_val_acc`), thời gian chạy (`train_seconds`), và đường dẫn checkpoint.

---

### `save_results(results: List[Dict[str, Any]], csv_path: str) -> None`
Ghi toàn bộ kết quả so sánh ra file CSV, tự động dàn phẳng (flatten) các siêu tham số thành các cột có tiền tố `hp_`.

---

## 3. Chi tiết kỹ thuật & Câu hỏi vấn đáp thường gặp
- **Hỏi: Tại sao mỗi thử nghiệm cần một thư mục log riêng biệt trong TensorBoard?**
  - **Trả lời:** TensorBoard sẽ tự động nhận diện các thư mục con trong `log_root` là các "Run" độc lập và hiển thị các đường cong biểu đồ (Loss, Accuracy) của các run này trên cùng một khung hình với màu sắc khác nhau, giúp việc so sánh trực quan hiệu quả của từng cấu hình siêu tham số trở nên dễ dàng.
- **Hỏi: Tại sao dùng `time.perf_counter()` thay vì `time.time()` để đo thời gian huấn luyện?**
  - **Trả lời:** `time.perf_counter()` là đồng hồ đơn điệu (monotonic clock) có độ chính xác cao nhất của hệ điều hành, không bị ảnh hưởng nếu đồng hồ hệ thống thay đổi hoặc cập nhật qua mạng (NTP), đảm bảo đo thời lượng huấn luyện chính xác.
