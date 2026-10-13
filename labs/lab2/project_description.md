# Lab 2: Pre-trained CNNs and Fine-tuning on CIFAR-10 (PyTorch) — Project Description

## ENGLISH

### Problem Definition

**Research question:** How does ImageNet pretraining affect the performance, convergence speed, and computational cost of CNN image classification on CIFAR-10?

**Task:** 10-class image classification on CIFAR-10 using three experimental strategies:
- **Experiment A:** Training from scratch (no pretrained weights).
- **Experiment B:** Feature extraction (pretrained backbone, frozen; only new head trained).
- **Experiment C:** Fine-tuning (pretrained backbone; head warmup, then tail blocks unfrozen with smaller LR).

**Scientific distinction:** Experiments B and C both use a pretrained backbone but differ in whether the backbone parameters are updated. This is the central scientific comparison.

---

### Dataset Assumptions

**ASSUMPTION (unconfirmed from official lecturer materials):** CIFAR-10 from `torchvision.datasets.CIFAR10`.

Evidence for assumption: The project description (`project_description.md`) states "CIFAR-10 qua `torchvision.datasets`" and the official slides use it as a worked example. If the lecturer specifies a different dataset, the pipeline can be adapted with minimal changes to `preprocessing.py`.

CIFAR-10 properties:
- 50,000 training images + 10,000 test images.
- 10 balanced classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- Images: 32×32 × 3 channels (RGB), uint8.
- No class imbalance in the standard split.

---

### Experimental Design

| Experiment | Model | Pretrained | Backbone | Head | Notes |
|------------|-------|-----------|----------|------|-------|
| A — Scratch | Custom CNN | No | Trainable | Trainable | Random init |
| B — Feature Extraction | ResNet-18 | Yes (ImageNet1K_V1) | **Frozen** | Trainable | BN in eval mode |
| C — Fine-tuning | ResNet-18 | Yes (ImageNet1K_V1) | Phase 1: Frozen → Phase 2: Tail unfrozen | Trainable | 2-phase: warmup → unfreeze |

All experiments use:
- The same CIFAR-10 train/val/test split (seed=42).
- The same evaluation procedure (full test set).
- The same metrics (accuracy, macro F1, confusion matrix).

---

### Architecture Choices

#### Scratch CNN (`_ScratchCNN`)
```
Input [B, 3, H, W]
  Conv(3→32, 3×3) + BN + ReLU + MaxPool(2)   # H/2 × W/2
  Conv(32→64, 3×3) + BN + ReLU + MaxPool(2)  # H/4 × W/4
  Conv(64→128, 3×3) + BN + ReLU + AdaptiveAvgPool(4×4)  # → 2048
  Flatten → Dropout(0.4) → Linear(2048→256) + ReLU → Linear(256→10)
```
Parameters: ~550K trainable. Designed for inputs up to 224×224 via AdaptiveAvgPool.

#### ResNet-18 (Experiments B & C)
```
Input [B, 3, H, W]
  Backbone: conv1 → BN → ReLU → MaxPool → layer1 → layer2 → layer3 → layer4 → AvgPool
  Head: Linear(512 → 10)   [replacement for original 1000-class head]
```
Total parameters: ~11.18M. Trainable (feature extraction): 5,130 (fc only).

**BatchNorm handling:** When `freeze_backbone=True`, the `_FrozenBNModel` wrapper ensures the backbone stays in eval mode during training, preventing CIFAR-10 batch statistics from corrupting ImageNet running statistics.

---

### Data Preprocessing

#### Training transform (with augmentation):
1. `Resize(image_size)` — default 64×64 for CPU efficiency.
2. `RandomCrop(image_size, padding=4)` — light spatial augmentation.
3. `RandomHorizontalFlip()` — domain-appropriate augmentation.
4. `ToTensor()` — [0,255] uint8 → [0,1] float32.
5. `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` — ImageNet statistics.

#### Evaluation transform (deterministic):
1–2. `Resize(image_size)` + `ToTensor()` + `Normalize(...)` — same statistics, no augmentation.

**Important:** The eval transform is applied to validation samples even though they come from the training dataset. The `_TransformSubset` wrapper in `features.py` ensures this is done correctly without augmentation bleeding.

**Resolution trade-off:** Using 64×64 instead of 224×224 is a documented deviation. At 64×64, downsampling in ResNet-18 leaves a 2×2 spatial map before AvgPool (rather than 7×7 at 224×224). This affects the pretrained features and should be reported honestly. A full 224×224 run is scientifically more correct but is computationally prohibitive on CPU (each epoch ~2–4× slower).

---

### Train/Validation/Test Split

- Method: Random shuffle with fixed seed (seed=42), reproducible via `make_split`.
- Validation fraction: 10% of training set → 5,000 validation, 45,000 training.
- Split saved to: `artifacts/splits/split_seed42.json` (indices only, not data).
- Test set: Official CIFAR-10 test set (10,000 images), never used for selection.

---

### Optimizer and Scheduler

| Setting | Default |
|---------|---------|
| Optimizer | Adam |
| Learning rate | 1e-3 (head), 1e-4 (backbone in fine-tuning) |
| Weight decay | 1e-4 (full training) |
| Scheduler | StepLR (step_size=7, gamma=0.1) |
| Early stopping | val_acc patience=10 epochs |

---

### Metrics

| Metric | Description |
|--------|-------------|
| Test Accuracy | Top-1 accuracy on full 10,000-sample test set |
| Macro Precision | Mean precision across all 10 classes |
| Macro Recall | Mean recall across all 10 classes |
| Macro F1-score | Harmonic mean of precision and recall, averaged |
| Per-class Accuracy | Accuracy for each of the 10 CIFAR-10 classes |
| Confusion Matrix | 10×10 count matrix (rows=true, cols=predicted) |
| Inference Throughput | Images/second on evaluation hardware |

Model selection uses **validation accuracy** only (never test accuracy).

---

### Reproducibility

- All experiments use `set_seed(42)` from `shared/seed.py` (Python, NumPy, PyTorch).
- Split indices saved in JSON; loading from JSON on subsequent runs.
- Checkpoints saved at best validation epoch.
- Configurations saved as `config.json` in each run directory.

---

### Limitations

1. **CPU-only:** No CUDA on this machine. ResNet-18 at 224×224 is impractical; we use 64×64.
2. **CIFAR-10 assumption:** Dataset choice confirmed from project docs but not official lecture slide text.
3. **Single pretrained architecture:** Only ResNet-18 tested; VGG-16, DenseNet-121 are supported by the model module but not benchmarked.
4. **Network bandwidth:** Dataset download at ~30kB/s makes initial setup slow.

---

### Commands

```bash
# From repo root:

# Run all smoke tests (no real data required):
pytest labs/lab2/tests -v

# Run EDA (requires CIFAR-10 in raw_data/):
python -m labs.lab2.main --eda

# Smoke pipeline (fast, small subset):
python -m labs.lab2.main --mode smoke

# Full experiments:
python -m labs.lab2.main --mode full --experiments scratch,feature_extraction,fine_tuning

# Compare results:
python -m labs.lab2.main --compare

# Custom settings:
python -m labs.lab2.main --mode full --image-size 32 --batch-size 128 --seed 123
```

---

### Expected Artifact Structure

```
labs/lab2/
├── artifacts/
│   ├── splits/
│   │   └── split_seed42.json
│   ├── figures/
│   │   ├── eda/
│   │   │   ├── class_distribution.png
│   │   │   ├── sample_grid.png
│   │   │   ├── pixel_histogram.png
│   │   │   └── eda_summary.json
│   │   ├── training_curves.png
│   │   └── comparison_bar.png
│   └── runs/
│       ├── scratch_sc_rnd_lr1e-03_ep20_<ts>/
│       │   ├── config.json
│       │   ├── history.csv
│       │   ├── summary.json
│       │   └── checkpoint_best.pt
│       ├── resnet18_fe_pt_lr1e-03_ep10_<ts>/
│       │   └── (same structure)
│       └── resnet18_fe_pt_lr1e-03_ep15_<ts>/
│           └── (same structure)
└── logbook/
    └── (bug reports and experiment notes)
```

---

## TIẾNG VIỆT

### Định nghĩa bài toán

**Câu hỏi nghiên cứu:** ImageNet pretraining ảnh hưởng như thế nào đến hiệu suất, tốc độ hội tụ và chi phí tính toán khi phân loại ảnh CIFAR-10?

**GIẢ ĐỊNH (chưa xác nhận từ giảng viên):** Dataset là CIFAR-10 qua `torchvision.datasets`. Bằng chứng: slide môn học và file `project_description.md` đều nhắc đến CIFAR-10.

### Ba thí nghiệm chính

| Thí nghiệm | Mô hình | Pre-trained | Backbone | Head |
|------------|---------|-----------|----------|------|
| A — Scratch | CNN tự thiết kế | Không | Trainable | Trainable |
| B — Feature Extraction | ResNet-18 | Có (ImageNet) | **Đóng băng** | Trainable |
| C — Fine-tuning | ResNet-18 | Có (ImageNet) | Pha 1: Đóng → Pha 2: Mở một phần | Trainable |

### Lựa chọn kỹ thuật và lý do

1. **image_size=64** (thay vì 224): CPU-only, 224×224 quá chậm. Ghi rõ trong báo cáo.
2. **Adam optimizer**: Thích nghi learning rate, phù hợp cho bài toán với ít epoch.
3. **StepLR**: Giảm lr sau mỗi 7 epoch để tinh chỉnh.
4. **seed=42**: Tái lập được mọi kết quả.
5. **`_FrozenBNModel`**: Wrapper giữ BatchNorm của backbone ở eval mode khi freeze.

### Ràng buộc đã biết

- Không có GPU (CUDA=False). Tất cả chạy trên CPU Intel i5-1135G7.
- Tốc độ download dataset chậm (~30kB/s) → lần đầu mất nhiều thời gian.
- image_size=64 là trade-off thực tế, không phải điều kiện lý tưởng khoa học.
