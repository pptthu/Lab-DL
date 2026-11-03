# Train Module Documentation (`labs/lab2/train/train.py`)

## 1. Overview
The `train` module implements the end-to-end model training loop, validation tracking across epochs, TensorBoard scalar telemetry, and best checkpoint saving/loading.

- **File location:** `labs/lab2/train/train.py`
- **Assigned owner:** Member B
- **Dependencies:** `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `pathlib`, `os`

---

## 2. Functions & API Contract

### `train_model(...) -> Dict[str, list]`
Executes the supervised training loop over a given number of epochs.

#### Function Signature:
```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: Any,
    ckpt_path: str,
    writer: Optional[Any] = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
) -> Dict[str, list]
```

#### Arguments:
- `model` (`nn.Module`): Model to be trained.
- `train_loader` (`DataLoader`): DataLoader for the training set.
- `val_loader` (`DataLoader`): DataLoader for the validation set.
- `epochs` (`int`): Number of training epochs (must be > 0).
- `lr` (`float`): Learning rate (must be > 0).
- `device` (`Any`): Hardware device (`"cuda"`, `"cpu"`, or `torch.device`).
- `ckpt_path` (`str`): File path where the best model checkpoint is saved.
- `writer` (`SummaryWriter`, optional): TensorBoard writer instance for scalar logging.
- `optimizer_name` (`str`): `"adam"` or `"sgd"` (SGD with momentum 0.9). Default: `"adam"`.
- `weight_decay` (`float`): L2 regularization weight penalty (default: `0.0`).

#### Training Procedure per Epoch:
1. **Train Phase:**
   - Put model in `model.train()` mode.
   - Zero gradients: `optimizer.zero_grad()`.
   - Forward pass through model and compute loss using `nn.CrossEntropyLoss()`.
   - Backward pass: `loss.backward()`.
   - Optimizer step: `optimizer.step()`.
   - Track sample-weighted cumulative training loss.
2. **Validation Phase:**
   - Put model in `model.eval()` mode with `torch.no_grad()`.
   - Compute validation loss and top-1 classification accuracy.
3. **Telemetry & Checkpointing:**
   - If `writer` is provided:
     - Log `Loss/train`
     - Log `Loss/val`
     - Log `Accuracy/val`
   - Compare `val_epoch_loss` with `best_val_loss`. If improved, save weights to `ckpt_path`.

#### Returns:
A history dictionary:
```python
{
    "train_loss": [float, ...],
    "val_loss": [float, ...],
    "val_acc": [float, ...]
}
```

---

### `save_checkpoint(model: nn.Module, path: str) -> None`
Saves the state dictionary of the model to a specified file path, creating necessary parent directories.

- **Implementation:** `torch.save(model.state_dict(), path)`

---

### `load_checkpoint(model: nn.Module, path: str, device: Any) -> nn.Module`
Loads saved weights into a model on a given device and sets the model to evaluation mode (`model.eval()`).

#### Arguments:
- `model`: Target model instance.
- `path`: Path to checkpoint file.
- `device`: Device to map weights onto (`"cuda"` or `"cpu"`).

#### Implementation:
- Uses `torch.load(path, map_location=device, weights_only=True)` to ensure secure loading without executing arbitrary pickled code.
- Calls `model.load_state_dict(state_dict)` and `model.eval()`.

---

## 3. Key Implementation Details & Defense Q&A
- **Q: Why pass only trainable parameters (`requires_grad=True`) to the optimizer?**
  - **A:** Passing frozen parameters to the optimizer wastes memory (allocating unnecessary momentum buffers in Adam/SGD) and can cause unexpected state tracking for parameters that will never receive gradients.
- **Q: Why use `loss.item() * batch_size` when computing epoch loss?**
  - **A:** DataLoader batches may not all have identical sizes (e.g., the last batch is often smaller). Multiplying by `batch_size` and dividing by total samples guarantees an exact sample-weighted mathematical average.
- **Q: Why call `optimizer.zero_grad()` before `loss.backward()`?**
  - **A:** PyTorch accumulates gradients in `.grad` by default upon each `backward()` call. Failing to zero out gradients causes gradients from previous batches to add up, resulting in incorrect parameter updates.
- **Q: Why select the best checkpoint based on `val_loss` rather than `train_loss`?**
  - **A:** `train_loss` can continuously decrease due to overfitting. Monitoring `val_loss` ensures saving the model state before overfitting impairs generalization performance on unseen data.

---

**Tiếng Việt**

# Tài liệu Module Huấn Luyện (`labs/lab2/train/train.py`)

## 1. Tổng quan
Module `train` triển khai vòng lặp huấn luyện mô hình (training loop), kiểm thử trên tập validation sau mỗi epoch, ghi nhật ký chỉ số lên TensorBoard và tự động lưu/nạp checkpoint tốt nhất của mô hình.

- **Vị trí file:** `labs/lab2/train/train.py`
- **Người phụ trách:** Member B
- **Thư viện phụ thuộc:** `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `pathlib`, `os`

---

## 2. Các hàm & Giao diện lập trình (API)

### `train_model(...) -> Dict[str, list]`
Hàm thực hiện vòng lặp huấn luyện chính qua các epoch.

#### Tham số:
- `model`: Mô hình PyTorch cần huấn luyện.
- `train_loader`, `val_loader`: DataLoader tập train và tập validation.
- `epochs`: Số lượng chu kỳ huấn luyện (> 0).
- `lr`: Tốc độ học (Learning Rate, > 0).
- `device`: Thiết bị tính toán (`"cuda"` hoặc `"cpu"`).
- `ckpt_path`: Đường dẫn lưu file checkpoint của mô hình tốt nhất.
- `writer`: Thể hiện của `SummaryWriter` (TensorBoard) để ghi log.
- `optimizer_name`: Thuật toán tối ưu (`"adam"` hoặc `"sgd"`).
- `weight_decay`: Hệ số phạt L2 Regularization (mặc định 0.0).

#### Tiến trình trong mỗi Epoch:
1. Chuyển sang `model.train()`, xóa gradient cũ bằng `optimizer.zero_grad()`, tính loss và lan truyền ngược `loss.backward()`, cập nhật trọng số bằng `optimizer.step()`.
2. Chuyển sang `model.eval()`, dùng `torch.no_grad()` để tính loss và accuracy trên tập validation.
3. Ghi log các chỉ số `Loss/train`, `Loss/val`, `Accuracy/val` lên TensorBoard nếu có `writer`.
4. Nếu `val_loss` nhỏ hơn kỷ lục trước đó, tự động lưu trọng số mô hình vào file tại `ckpt_path`.

---

### `save_checkpoint(model: nn.Module, path: str) -> None`
Lưu `state_dict` của mô hình vào đường dẫn chỉ định và tự động tạo thư mục nếu chưa tồn tại.

---

### `load_checkpoint(model: nn.Module, path: str, device: Any) -> nn.Module`
Nạp trọng số từ checkpoint vào mô hình theo thiết bị tương ứng và chuyển mô hình về trạng thái `model.eval()`. Sử dụng tham số `weights_only=True` để đảm bảo an toàn bảo mật.

---

## 3. Chi tiết kỹ thuật & Câu hỏi vấn đáp thường gặp
- **Hỏi: Tại sao chỉ truyền các tham số có `requires_grad=True` vào Optimizer?**
  - **Trả lời:** Trong Transfer Learning, phần lớn các tầng đã bị đóng băng. Nếu truyền toàn bộ tham số vào optimizer (nhất là Adam), hệ thống sẽ phải cấp phát thêm bộ nhớ đệm (momentum buffer) cho hàng triệu tham số không cần cập nhật, gây lãng phí RAM/VRAM.
- **Hỏi: Tại sao cần `optimizer.zero_grad()` ở đầu mỗi batch?**
  - **Trả lời:** Mặc định PyTorch sẽ cộng dồn gradient tại thuộc tính `.grad` sau mỗi lần gọi `.backward()`. Nếu không xóa về 0, gradient của batch hiện tại sẽ bị cộng gộp với các batch trước đó, dẫn đến bước cập nhật trọng số bị sai lệch.
- **Hỏi: Tiêu chí lưu checkpoint tốt nhất là gì?**
  - **Trả lời:** Dựa trên `val_loss` thấp nhất. `val_loss` phản ánh trực tiếp nguy cơ quá khớp (overfitting); lưu tại điểm `val_loss` tối ưu giúp giữ lại mô hình có khả năng tổng quát hóa tốt nhất trên dữ liệu mới.
