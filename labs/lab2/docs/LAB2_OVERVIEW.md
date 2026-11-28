# Lab 2 — Tài liệu tổng quan toàn bộ (v1)

- **Người soạn:** Member B (Bảo) · **Ngày:** 2026-11-27
- **Nhánh tham chiếu:** `lab2/train` @ `1486638` (dựa trên `main` @ `7577c0e`)
- **Mục đích:** đọc một lượt là hiểu Lab 2 làm gì, code nằm ở đâu, số liệu đến từ đâu, và trả lời được vấn đáp.

> Mọi con số trong tài liệu này lấy từ output đã chạy trong notebook của `origin/main`
> (`lab2_presentation.ipynb`, lần chạy 2026-11-22) hoặc từ tài liệu nguồn `README.md`, `REPORT.md`.
> Chỗ nào chưa chạy thì ghi rõ "chưa có số".

---

## 1. Lab này giải quyết gì

**Bài toán:** phân loại ảnh CIFAR-10 (10 lớp) bằng cách **tái sử dụng** CNN đã huấn luyện trên ImageNet
(`torchvision.models`) thay vì huấn luyện từ đầu.

**Hai chiến lược được so sánh:**

| Chiến lược | Backbone | Phần được cập nhật |
|---|---|---|
| **Feature Extraction** | Đóng băng toàn bộ (`freeze_backbone=True`) | Chỉ classifier head mới |
| **Fine-Tuning** | Mở thêm khối tích chập cuối (`unfreeze_last_blocks(1)`) | Head + `layer4` (ResNet) / `denseblock4` (DenseNet) |

**Dữ liệu:** CIFAR-10 — 50.000 train / 10.000 test, ảnh gốc 32×32.
**Đầu ra:** checkpoint `.pt`, log TensorBoard, bảng so sánh CSV, hình (đường cong, confusion matrix),
notebook trình bày, và báo cáo `README.md` / `REPORT.md`.

**Nguyên tắc đánh giá:** chia 45.000 train / 5.000 validation (seed 42); **chọn model theo validation**;
tập test chỉ dùng **một lần** cho cấu hình thắng.

---

## 2. Bản đồ code

```text
labs/lab2/
├── main.py                  # chạy toàn pipeline + CLI
├── eda/eda.py               # thống kê dữ liệu + 2 hình PNG
├── preprocessing/           # nạp dataset thô, chia train/val, lưu/nạp split
├── features/features.py     # transform + DataLoader
├── model/model.py           # nạp CNN pre-trained, thay head, freeze/unfreeze
├── train/train.py           # vòng lặp train/val, scheduler, early stopping, checkpoint
├── evaluate/evaluate.py     # loss/accuracy, confusion matrix, top confusions
├── experiment/experiment.py # chạy nhiều cấu hình, xuất CSV
├── notebook/lab2_presentation.ipynb   # 11 mục trình bày
├── tests/                   # 56 test (pytest)
├── artifacts/               # splits (commit), figures/logs/models/runs (không commit)
├── checkpoint/              # checkpoint do notebook sinh ra
└── log/                     # TensorBoard do notebook sinh ra
shared/
├── paths.py                 # đường dẫn tính từ gốc repo
└── seed.py                  # set_seed(42) cho python/numpy/torch
```

**Hợp đồng giao diện** (không đổi tên/chữ ký khi chưa báo nhóm trưởng):

| Module | Hàm | Vào → Ra |
|---|---|---|
| `eda` | `run_eda(data_root, out_dir)` | thư mục dữ liệu, thư mục hình → dict thống kê + PNG |
| `preprocessing` | `load_raw_dataset(data_root, download=True)` | → `(train_set, test_set)` chưa transform |
| | `make_split(n_samples, val_ratio=0.1, seed=42)` | → `(train_idx, val_idx)` |
| | `save_split` / `load_split` | JSON ↔ indices |
| `features` | `build_transforms(image_size=224, augment=False)` | → `(train_tf, eval_tf)` |
| | `make_loaders(...)` | → `(train_loader, val_loader, test_loader)` |
| `model` | `build_model(name, num_classes=10, freeze_backbone=True, pretrained=True, dropout=0.0)` | → `nn.Module` xuất logits `[B,10]` |
| | `describe_model(model)` | → `{total_params, trainable_params, layer_names}` |
| | `unfreeze_last_blocks(model, n_blocks)` | mở khối cuối, tại chỗ |
| `train` | `train_model(model, train_loader, val_loader, epochs, lr, device, ckpt_path, writer=None, optimizer_name="adam", weight_decay=0.0, patience=None, scheduler_name=None)` | → `history` + checkpoint |
| | `save_checkpoint` / `load_checkpoint` | state_dict ↔ model |
| `evaluate` | `evaluate(model, loader, device)` | → `(avg_loss, accuracy)` |
| | `confusion_matrix(model, loader, device, num_classes=10)` | → mảng `[10,10]` |
| | `top_confusions(cm, class_names, k=5)` | → 5 cặp nhầm lẫn nhiều nhất |
| `experiment` | `run_comparison(model_names, hparam_grid, loaders, device, log_root, ...)` | → list kết quả |
| | `save_results(results, csv_path)` | → CSV |

---

## 3. Pipeline 8 bước

```text
CIFAR-10 thô
  → [1] EDA                thống kê 10 lớp, mean/std, 2 hình PNG
  → [2] Preprocessing      load_raw_dataset + split 90/10 (seed 42) → JSON
  → [3] Features           transform + DataLoader (128×128, batch 64)
  → [4] Model              build_model (head 10 lớp), freeze hoặc unfreeze
  → [5] Train              train_model (loss, optimizer, scheduler, early stopping)
  → [6] Evaluate           loss/accuracy + confusion matrix trên validation
  → [7] Experiment         run_comparison nhiều cấu hình → CSV
  → [8] Results            đánh giá test MỘT LẦN cho cấu hình thắng + tổng kết
```

### [1] EDA — `eda/eda.py`
- Đọc CIFAR-10 gốc **không transform**, đếm 50.000/10.000, 5.000 ảnh mỗi lớp.
- Tính mean/std 3 kênh trên tập train: `[0.4914, 0.4822, 0.4465]` / `[0.247, 0.2435, 0.2616]`.
- Xuất `cifar10_class_distribution.png`, `cifar10_sample_grid.png` vào `artifacts/figures/`.

### [2] Preprocessing — `preprocessing/preprocessing.py`
- `load_raw_dataset`: trả dataset PIL, chưa transform (để bước sau tự áp).
- `make_split(50000, val_ratio=0.1, seed=42)`: hoán vị bằng `RandomState(42)` → 45.000 train / 5.000 val.
- `save_split`/`load_split`: lưu `artifacts/splits/train_val_split.json` để **mọi người dùng chung một split**.
- Notebook còn assert `set(train_idx) ∩ set(val_idx) = ∅` → chống data leakage.
- Test 10.000 mẫu **không** bị chia, chỉ dùng ở bước [8].

### [3] Features — `features/features.py`
- `build_transforms(image_size, augment)`:
  - **Eval/Test:** `Resize(128)` → `ToTensor` → `Normalize` (ImageNet mean/std). Không shuffle.
  - **Train (augment=True):** `RandomResizedCrop(128, scale=(0.8, 1.0))` → `RandomHorizontalFlip` →
    `RandomRotation(15°)` → `ColorJitter(0.2, 0.2, 0.2)` → `ToTensor` → `Normalize`.
- `make_loaders`: bọc subset theo index (`TransformedSubset`), `shuffle=True` cho train, `False` cho val/test,
  `num_workers=0`. Batch mặc định 64.

### [4] Model — `model/model.py`
- `build_model(name, ...)`: nạp `resnet18` / `densenet121` / `vgg16` với `weights=DEFAULT`.
- Đóng băng backbone nếu `freeze_backbone=True` (`requires_grad=False`).
- Thay head: ResNet `fc`, DenseNet `classifier`, VGG `classifier[6]` → `Linear(in, 10)`.
  Nếu `dropout > 0` → `Sequential(Dropout(p), Linear(...))`.
- `unfreeze_last_blocks(model, 1)`: mở `layer4` (ResNet) / khối conv cuối (VGG) / denseblock cuối (DenseNet),
  luôn giữ head trainable.
- Số tham số: ResNet-18 tổng 11.181.642; FE chỉ train 5.130 (0,05%); FT mở `layer4` → 8.398.858 (75,11%).

### [5] Train — `train/train.py`
- Loss `nn.CrossEntropyLoss()` trên logits (trên `main` có thêm `label_smoothing=0.05`; nhánh `lab2/train`
  đã bỏ — xem §6).
- Optimizer: Adam (hoặc SGD momentum 0,9) chỉ trên tham số `requires_grad=True`.
- Scheduler tuỳ chọn: `scheduler_name` = `"cosine"` / `"step"` / `"plateau"` / `None`.
- Mỗi epoch: train → validate → ghi history → ghi TensorBoard (`Loss/train`, `Accuracy/train`,
  `Loss/val`, `Accuracy/val`) → lưu checkpoint nếu `val_loss` thấp hơn best.
- **Chọn checkpoint theo val loss nhỏ nhất**; history trả `best_epoch`, `best_val_loss`, `best_val_acc`,
  `stopped_epoch`.
- **Early stopping**: chỉ bật khi truyền `patience`; dừng khi `val_loss` không giảm quá `patience` epoch.
- `save_checkpoint` chỉ lưu `state_dict` (không optimizer); `load_checkpoint` dùng `weights_only=True`,
  đưa model về `eval()`.

### [6] Evaluate — `evaluate/evaluate.py`
- `evaluate`: `model.eval()` + `torch.no_grad()`, cộng loss **theo số mẫu** (không theo batch), trả
  `(avg_loss, accuracy∈[0,1])`.
- `confusion_matrix`: `[10,10]` int64, **hàng = nhãn thật, cột = dự đoán**; nhãn ngoài phạm vi → `ValueError`.
- `top_confusions`: bỏ đường chéo và ô 0, sắp giảm dần, lấy `k` cặp.
- Không ghi file, không tự nạp checkpoint.

### [7] Experiment — `experiment/experiment.py`
- `run_comparison`: duyệt `model_names × hparam_grid`, mỗi ô một run độc lập (checkpoint + thư mục
  TensorBoard riêng), trả list kết quả 8 khoá (`model`, `hparams`, `best_val_acc`, `best_val_loss`,
  `best_epoch`, `train_seconds`, `checkpoint`, `tensorboard_logged`).
- `save_results`: làm phẳng `hparams` thành cột `hp_*`, ghi CSV.
- Lưu ý: `batch_size` trong grid **không có tác dụng** (loader tạo một lần bên ngoài); chạy lại cùng tên run sẽ
  ghi đè file.

### [8] Results — `main.py`
- CLI: `python -m labs.lab2.main` với các cờ
  `--seed`, `--model {resnet18,densenet121,vgg16}`, `--epochs`, `--image-size`, `--batch-size`, `--lr`,
  `--weight-decay`, `--dropout`, `--patience`, `--scheduler {cosine,step,plateau}`, `--unfreeze-blocks`,
  `--no-augment`, `--no-freeze`, `--no-pretrained`, `--no-download`, `--no-experiments`,
  `--comparison-models`.
- Ghi `pipeline_summary.json`, đánh giá val + test, in top confusions.

---

## 4. Notebook trình bày — 11 mục

| Mục | Nội dung | Output chính |
|---|---|---|
| 1 | Setup: clone/checkout nhánh, import, seed, device | commit local == remote (assert) |
| 2 | EDA: thống kê + 2 hình | 50k/10k, mean/std, biểu đồ lớp, lưới ảnh |
| 3 | Preprocessing: split 90/10 + kiểm tra leakage | 45.000 / 5.000, 0 mẫu trùng |
| 4 | Features: transform + loader | shape batch `[64,3,128,128]` |
| 5 | Model: dựng 3 biến thể, in số tham số, forward test | logits `[4,10]` |
| 6 | **Train giai đoạn 1 — Feature Extraction** (ResNet-18, DenseNet-121) | history từng epoch, early stopping |
| 7 | Evaluate FE: val metrics, confusion matrix, top confusions, ảnh mẫu | heatmap 2 model |
| 8 | **Train giai đoạn 2 — Fine-Tuning** (mở `layer4`) | history từng epoch |
| 9 | Evaluate FT: val metrics, confusion matrix, top confusions | heatmap 2 model |
| 10 | Đối chiếu 4 cấu hình: bảng + biểu đồ | bảng CSV tổng hợp |
| 11 | **Đánh giá test một lần** cho cấu hình thắng + tổng kết | test accuracy, lệch val–test |

---

## 5. Kết quả mới nhất (từ notebook `origin/main`, chạy 2026-11-22)

| Run | Epoch đã chạy | Best epoch | Best val loss | Val acc @best | Train acc cuối | Val acc cuối |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 — Feature Extraction | 14 (dừng sớm) | 9 | 0.6193 | 78,94% | 78,91% | 78,64% |
| DenseNet-121 — Feature Extraction | 16 (dừng sớm) | 11 | 0.4716 | 83,22% | 82,10% | 83,04% |
| ResNet-18 — Fine-Tuning | 10 (hết trần) | 10 | 0.4881 | **93,34%** | 93,72% | 93,34% |
| DenseNet-121 — Fine-Tuning | 10 (hết trần) | 10 | 0.5296 | 90,98% | 88,22% | 90,98% |

**Test (một lần, cấu hình thắng ResNet-18 FT):** accuracy **92,95%** (9.295/10.000), lệch val–test **0,39 điểm**.

**Cách đọc:**

- **Best epoch** = epoch có val loss nhỏ nhất, cũng là epoch được lưu checkpoint và đem báo cáo.
- Hai run FT có best epoch **trùng epoch cuối (10)** → trong 10 epoch, val loss **vẫn đang giảm**, early stopping
  chưa kích hoạt. Đây là dấu hiệu "chưa đủ epoch", không phải "đã overfit".
- So sánh FE → FT: 78,94% → 93,34% (ResNet) và 83,22% → 90,98% (DenseNet). FT tốt hơn rõ rệt.
- FE có val ≈ train, thậm chí val cao hơn train (DenseNet) vì train đo dưới augmentation → **không overfit**,
  mà là plateau của đặc trưng đóng băng.

> Các số trên là của cấu hình trong notebook `main` (`EPOCHS_FT=10`, `LR_FT=4e-5`, `WEIGHT_DECAY=3e-4`,
> `DROPOUT_FT=0.4`, không scheduler, có label smoothing 0.05). Cấu hình trên nhánh `lab2/train` khác — xem §6 —
> nên **chưa có kết quả** cho cấu hình đó.

---

## 6. Hai cấu hình Fine-Tuning đang tồn tại

| Tham số | `main` (notebook của nhóm) | `lab2/train` (bản vá của bạn) |
|---|---|---|
| `EPOCHS_FT` | 10 | **30** |
| `LR_FT` | 4e-5 | **1e-4** |
| `WEIGHT_DECAY` | 3e-4 | **2e-4** |
| `DROPOUT_FT` | 0.4 | **0.3** |
| `SCHEDULER_NAME` | `None` | **`"cosine"`** |
| `PATIENCE` | 5 | 5 |
| `label_smoothing` trong loss | 0.05 | **0 (đã bỏ)** |
| Optimizer | Adam | Adam |

Hệ quả: bỏ label smoothing làm **val loss đổi thang đo** → không so trực tiếp val loss giữa hai cấu hình;
chỉ so accuracy, best epoch, và hình dạng đường cong.

---

## 7. Overfitting: hiểu đúng và cách chống

### 7.1 Hai thước đo

- **gap cuối** = `train_acc(epoch cuối) − val_acc(epoch cuối)` — đo trong lúc train (train có augmentation,
  dropout bật).
- **clean gap** = `acc_train(không augmentation, eval mode) − val_acc` trên checkpoint tốt nhất — mới phản ánh
  **học thuộc thật**.

Clean gap thường **lớn hơn** gap cuối vì khi tắt augmentation/dropout, điểm train tăng lên.

### 7.2 Dùng gì để kết luận

| Câu hỏi | Dùng |
|---|---|
| Train và val có tách ra, tách từ epoch nào? | gap cuối + `onset` (epoch đầu tiên val loss vượt best > 0.01) |
| Model có học thuộc train không? | clean gap |
| Chênh val–test nhỏ nghĩa là không overfit? | **Sai** — đó chỉ nói test không khó hơn val |

### 7.3 Bốn biện pháp đang bật

1. **Augmentation**: crop 0.8–1.0, lật ngang, xoay ±15°, đổi màu nhẹ.
2. **Dropout** trước head (0.3–0.4 tuỳ cấu hình), không chèn vào backbone.
3. **LR scheduler** (cosine trên `lab2/train`).
4. **Early stopping** theo val loss + **chọn checkpoint tốt nhất** — quan trọng nhất: các epoch overfit bị
   loại khỏi kết quả cuối.

**Bằng chứng từ kết quả §5:** các run FT dừng ở trần 10 epoch với best epoch = 10 → chưa thấy val loss đảo
chiều; FE dừng sớm ở 14/16 epoch với val loss chỉ dao động. Vì vậy với cấu hình `main`, overfitting **chưa
biểu hiện rõ**. Với `lab2/train` (30 epoch, dropout thấp hơn, LR cao hơn) cần chạy mới biết — đây chính là
mục đích của bản vá.

---

## 8. Cách chạy

### 8.1 Kiểm thử (không cần GPU)

```bash
python -m pytest labs/lab2/tests -q          # 56 test
```

### 8.2 Chạy nhanh bằng CLI

```bash
python -m labs.lab2.main --epochs 1 --image-size 64 --no-experiments
```

### 8.3 Notebook trên Colab (cần GPU)

1. Runtime → GPU (T4).
2. Cell 2 clone/checkout `lab2/train` và assert commit local == remote.
3. Cell 3 `!git switch lab2/train`.
4. Run all. Kết quả ghi vào `labs/lab2/checkpoint/renet18/` (ResNet),
   `labs/lab2/checkpoint/densenet121/` (DenseNet) và log TensorBoard trong `labs/lab2/log/`.

Thời gian tham khảo: FE nhanh (14–16 epoch), FT tốn hơn; tổng 1–3 giờ trên T4 tuỳ early stopping.

### 8.4 Quy tắc đánh giá

- Chọn model theo **validation**.
- Test chỉ chạy **một lần** cho cấu hình thắng (mục 11).
- Không chỉnh tham số dựa trên test.

---

## 9. Câu hỏi vấn đáp thường gặp

1. **Vì sao dùng model pre-trained?** ImageNet có 1,2 triệu ảnh; CIFAR-10 chỉ 50.000 → tận dụng đặc trưng
   cạnh/góc/texture đã học, hội tụ nhanh và chính xác hơn huấn luyện từ đầu.
2. **Feature Extraction khác Fine-Tuning thế nào?** FE đóng băng backbone, chỉ học head (5.130 tham số);
   FT mở `layer4`, học 8.398.858 tham số (75,11%), thích nghi sâu hơn nhưng dễ overfit hơn.
3. **Vì sao thay lớp cuối?** ImageNet có 1.000 lớp, CIFAR-10 có 10 → head mới `Linear(in, 10)`.
4. **Vì sao chuẩn hoá theo ImageNet?** Backbone được huấn luyện với mean/std đó; dùng mean/std CIFAR-10 sẽ lệch
   phân phối đầu vào.
5. **Vì sao chọn checkpoint theo val loss mà báo cáo val acc?** Loss phản ánh độ tự tin/calibration, nhạy hơn
   accuracy; nhưng phải báo accuracy ở **đúng epoch đó**, không lấy `max(val_acc)` của epoch khác.
6. **Vì sao 128×128 thay vì 32?** CNN pre-trained cần kích thước lớn hơn để giữ cấu trúc; 128 là điểm cân bằng
   giữa tốc độ và độ chính xác.
7. **Có rò rỉ dữ liệu không?** Không: split chỉ chia từ train, test tách riêng; notebook assert giao = rỗng;
   test chỉ dùng ở mục 11.
8. **Early stopping dừng ở đâu?** FE: epoch 14 (ResNet) / 16 (DenseNet). FT với cấu hình `main`: **không dừng**,
   chạm trần 10 epoch.
9. **Vì sao val loss tăng mà val acc vẫn tăng?** Model tự tin sai ở một số mẫu (calibration xấu đi) trong khi
   top-1 vẫn cải thiện nhẹ.
10. **Confusion matrix đọc thế nào?** Hàng = nhãn thật, cột = dự đoán; đường chéo = đúng; cặp ngoài đường chéo
    lớn nhất là lỗi phổ biến (ví dụ `cat → dog`).
11. **Vì sao test chỉ chạy một lần?** Tránh "nhìn test rồi chọn model" — làm vậy test không còn là ước lượng
    khách quan.
12. **BatchNorm của backbone đóng băng có cập nhật không?** Có — `model.train()` cập nhật running stats kể cả
    khi `requires_grad=False`; đây là điểm cần nói rõ khi bảo vệ.
13. **`label_smoothing` ảnh hưởng gì?** Làm loss cao hơn và giảm tự tin quá mức; đổi thang đo val loss nên
    không so trực tiếp với cấu hình không dùng nó.

---

## 10. Từ điển nhanh

| Thuật ngữ | Nghĩa trong lab này |
|---|---|
| Feature Extraction | Đóng băng backbone, chỉ train head |
| Fine-Tuning | Mở khối cuối + head, LR nhỏ |
| Head / classifier | Lớp tuyến tính cuối xuất 10 logits |
| Logits | Điểm thô trước softmax |
| Epoch / step | Một lượt qua toàn bộ train / một batch |
| Early stopping | Dừng khi val loss không cải thiện `patience` epoch |
| Best checkpoint | File `.pt` của epoch có val loss nhỏ nhất |
| Clean gap | Chênh accuracy train (không augmentation) và val |
| Cosine scheduler | LR giảm theo cosin từ `lr` xuống gần 0 |

---

## 11. Nguồn

- Mã nguồn: `labs/lab2/**` trên nhánh `lab2/train` @ `1486638`.
- Kết quả: `labs/lab2/notebook/lab2_presentation.ipynb` trên `origin/main` (chạy 2026-11-22),
  `README.md`, `REPORT.md`.
- Tài liệu module: `docs/eda.md`, `preprocessing.md`, `features.md`, `model.md`, `train.md`,
  `evaluate.md`, `experiment.md`, `demo.md`.
- Audit overfitting: `docs/OVERFITTING_AUDIT.md`, `docs/OVERFITTING_EXPERIMENTS.md` (phân tích chi tiết,
  kèm thực nghiệm CPU 6k/30k).
- Tiến độ: `progress_status.md`.
