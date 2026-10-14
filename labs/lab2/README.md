# Lab 2: Pre-trained CNNs and Fine-tuning on CIFAR-10 (PyTorch)

## Brief Report (English)

### Problem
Compare three CNN training strategies on CIFAR-10: (A) training from scratch, (B) feature extraction with a pretrained backbone (frozen), and (C) fine-tuning a pretrained backbone. Central question: how does ImageNet pretraining affect accuracy, convergence, and compute?

### Dataset
CIFAR-10 — 50,000 training and 10,000 test images across 10 balanced classes (32×32 RGB). Source: `torchvision.datasets.CIFAR10`. Split: 90% train / 10% validation from the training set (seed=42, stratified). Test set held out until final evaluation.

> **ASSUMPTION:** Dataset confirmed from project description and lecture materials. Lecturer confirmation not obtained before submission.

### Methods

| Exp. | Backbone | Pretrained | Trainable Params | Image Size |
|------|----------|-----------|-----------------|------------|
| A — Scratch | Custom CNN (3 conv blocks) | No | ~550K (all) | 64×64 |
| B — Feature Extraction | ResNet-18 (frozen) | Yes (ImageNet1K) | ~5K (head only) | 64×64 |
| C — Fine-tuning | ResNet-18 (tail unfrozen) | Yes (ImageNet1K) | ~2.5M+ (head + last 2 blocks) | 64×64 |

All experiments use the same train/val/test split, Adam optimizer, and evaluation protocol.

### Experimental Setup
- Hardware: Intel i5-1135G7 (8 core), 11 GB RAM, **CPU only** (no GPU).
- Image size: 64×64 (trade-off for CPU feasibility; 224×224 is standard but impractical).
- Normalization: ImageNet mean/std for Exp. B & C; same for consistency in Exp. A.
- Framework: PyTorch 2.9.1, torchvision 0.24.0.

### Actual Results

> **STATUS: EXPERIMENT EXECUTION PENDING** — CIFAR-10 download in progress (~170MB, slow connection). Smoke tests with synthetic data pass (36/36). Results below will be populated after the full experiment run completes.

| Method | Backbone | Trainable Params | Test Accuracy | Macro F1 | Training Time | Status |
|--------|----------|-----------------|--------------|---------|---------------|--------|
| A — Scratch | Custom CNN | ~550K | NOT RUN | NOT RUN | NOT RUN | PENDING |
| B — Feature Extr. | ResNet-18 frozen | ~5K | NOT RUN | NOT RUN | NOT RUN | PENDING |
| C — Fine-tuning | ResNet-18 unfrozen | ~2.5M+ | NOT RUN | NOT RUN | NOT RUN | PENDING |

*Note: image_size=64 (not 224) — results will deviate from what a GPU server would produce.*

### Main Conclusion (Preliminary)
Pipeline is fully implemented, tested (36/36 unit tests pass), and ready to execute as soon as the dataset is available. Based on theory: (C) fine-tuning should achieve highest accuracy, (B) feature extraction should converge fastest with fewest parameters, (A) scratch training should be slowest to converge on CPU with limited data augmentation.

### Important Limitations
1. **CPU-only:** Training ResNet-18 at 64×64 will take significantly longer than GPU. 224×224 is impractical.
2. **Results not yet measured:** Data download pending. All accuracy numbers above are UNAVAILABLE.
3. **Dataset assumption:** CIFAR-10 confirmed from project docs, not official assignment text.
4. **image_size=64 deviation:** Results not directly comparable to published CIFAR-10 benchmarks.

---

## How to Run

```bash
# From repository root (/home/ntbankey/uth/Lab-DL):

# 1. Install dependencies:
pip install -r labs/lab2/requirements.txt

# 2. Run unit tests (no real data needed):
pytest labs/lab2/tests -v

# 3. Smoke pipeline (downloads dataset if needed, uses 1000 samples):
python -m labs.lab2.main --mode smoke

# 4. EDA (visualize dataset):
python -m labs.lab2.main --eda

# 5. Full training:
python -m labs.lab2.main --mode full --experiments scratch,feature_extraction,fine_tuning

# 6. Compare completed runs:
python -m labs.lab2.main --compare
```

---

## Folder Structure

```text
labs/lab2/
├── README.md               ← This file (brief report EN + VI)
├── project_description.md  ← Requirements, design, assumptions
├── progress_status.md      ← Progress log with statuses
├── ORAL_EXAM_VI.md         ← Vietnamese oral examination guide
├── main.py                 ← Pipeline entry point (CLI)
├── requirements.txt        ← Python dependencies
├── eda/eda.py              ← Exploratory Data Analysis
├── preprocessing/preprocessing.py  ← Dataset loading, splitting
├── features/features.py    ← Transforms, DataLoaders
├── model/model.py          ← Model building, freeze/unfreeze
├── train/train.py          ← Training/validation loop
├── evaluate/evaluate.py    ← Metrics, confusion matrix
├── experiment/experiment.py ← Experiment orchestration
├── raw_data/               ← Dataset (NOT committed)
├── artifacts/              ← Figures, checkpoints, logs (runs NOT committed)
├── logbook/                ← Bug reports, experiment notes
└── tests/test_smoke.py     ← 36 unit + integration tests
```

---

## Tiếng Việt — Báo cáo ngắn

### Bài toán
So sánh ba chiến lược huấn luyện CNN trên CIFAR-10: (A) huấn luyện từ đầu, (B) feature extraction với backbone đóng băng, (C) fine-tuning một phần backbone. Câu hỏi trung tâm: ImageNet pretraining ảnh hưởng như thế nào đến độ chính xác, tốc độ hội tụ và chi phí tính toán?

### Dữ liệu
CIFAR-10 — 50,000 ảnh train và 10,000 ảnh test, 10 lớp cân bằng (32×32 RGB). Chia train/val: 90%/10%, seed=42. Tập test tách biệt hoàn toàn.

**GIẢ ĐỊNH:** Dataset là CIFAR-10 (xác nhận từ tài liệu dự án; chưa hỏi lại giảng viên).

### Phương pháp

| Thí nghiệm | Backbone | Pre-trained | Tham số huấn luyện | image_size |
|-----------|----------|-----------|-------------------|------------|
| A — Scratch | CNN tự thiết kế | Không | ~550K | 64×64 |
| B — Feature Extraction | ResNet-18 (đóng băng) | Có (ImageNet) | ~5K (chỉ head) | 64×64 |
| C — Fine-tuning | ResNet-18 (mở khoá cuối) | Có (ImageNet) | ~2.5M+ | 64×64 |

### Kết quả

> **TRẠNG THÁI: CHƯA CHẠY** — Đang tải CIFAR-10 (~170MB, kết nối chậm). Smoke test tổng hợp: 36/36 pass. Bảng kết quả sẽ được cập nhật sau khi chạy xong.

| Phương pháp | Backbone | Test Accuracy | Macro F1 | Thời gian | Trạng thái |
|------------|---------|--------------|---------|-----------|-----------|
| A — Scratch | CNN | N/A | N/A | N/A | CHƯA CHẠY |
| B — Feature Extr. | ResNet-18 | N/A | N/A | N/A | CHƯA CHẠY |
| C — Fine-tuning | ResNet-18 | N/A | N/A | N/A | CHƯA CHẠY |

### Kết luận sơ bộ
Pipeline đã triển khai đầy đủ, 36/36 test pass. Kỳ vọng: C > B > A về accuracy, B > C > A về tốc độ hội tụ và chi phí tính toán.

### Hạn chế quan trọng
1. Chỉ dùng CPU → image_size=64 (thay vì 224), kết quả chậm và có thể khác benchmark.
2. Kết quả thực chưa có (đang tải dataset).
3. GIẢ ĐỊNH về dataset (CIFAR-10) chưa được giảng viên xác nhận.
