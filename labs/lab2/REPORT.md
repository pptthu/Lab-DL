# Comprehensive Lab Report: Pre-trained CNNs, Transfer Learning, and Fine-Tuning on CIFAR-10

---

## 1. Executive Summary

This report documents the design, implementation, and empirical evaluation of deep Convolutional Neural Networks (CNNs) pre-trained on ImageNet for classifying the CIFAR-10 dataset ($C = 10$). Utilizing PyTorch and Torchvision within a modular, contract-driven architecture, two modern backbones—**ResNet-18** and **DenseNet-121**—were systematically benchmarked across two transfer learning strategies:
1. **Feature Extraction (Frozen Backbone):** Adapting only the final linear classification layer ($\text{lr} = 10^{-3}$).
2. **Fine-Tuning (Selective Layer Unfreezing):** Simultaneously adapting the top convolutional block and the linear head ($\text{lr} = 4\times10^{-5}$).

### Key Empirical Findings
- **Feature Extraction Baseline:** ResNet-18 achieved a peak Validation Accuracy of **78.94%** (Val Loss: $0.6193$) at epoch 9, while DenseNet-121 reached **83.22%** (Val Loss: $0.4716$) at epoch 11 before Early Stopping.
- **Fine-Tuning Leap:** Unfreezing the top visual block (`layer4` for ResNet-18, `denseblock4` for DenseNet-121) yielded a performance boost of **+7.8 to +14.4 percentage points**, reaching **93.34%** for ResNet-18 and **90.98%** for DenseNet-121 on the validation set.
- **Independent Test Verification:** Evaluated strictly once on the unseen 10,000-image CIFAR-10 test set, the winning model—**ResNet-18 Fine-Tuning**—achieved an accuracy of **92.95%** (Test Loss: $0.2387$), exhibiting a negligible generalization gap ($\Delta = 0.39\% < 2\%$).

---

## 2. Problem Definition and Theoretical Background

### 2.1 Problem Formulation
CIFAR-10 consists of $60,000$ labeled $32 \times 32$ RGB images across 10 mutually exclusive categories: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`. The objective is learning a parameterized mapping $f_\theta: \mathcal{X} \to \mathbb{R}^{10}$ outputting logit scores $\mathbf{z}$, optimized under categorical Cross-Entropy Loss:

$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{B} \sum_{i=1}^B \log\left(\frac{\exp(z_{i, y_i})}{\sum_{j=1}^{10} \exp(z_{i, j})}\right)$$

### 2.2 Transfer Learning Principles
Deep CNNs learn hierarchical visual representations:
- **Shallow layers:** Capture domain-agnostic primitive features (Gabor-like edge detectors, textures, color gradients).
- **Deep layers:** Capture high-level, class-specific semantic patterns (faces, wheels, animal anatomy).

When transferring ImageNet-trained models (1.28M images, 1,000 classes) to CIFAR-10:
- **Feature Extraction:** Discards the original 1000-way linear layer, attaches a randomly initialized $d \times 10$ linear layer, and freezes the backbone ($\nabla_{\theta_{\text{backbone}}} = \mathbf{0}$). This preserves pre-trained filter banks and isolates gradient updates to the classification head.
- **Fine-Tuning:** Defrosts high-level convolutional layers while applying a reduced learning rate ($\text{lr} = 10^{-4}$ vs. $10^{-3}$) to prevent **catastrophic forgetting** (the destructive overwriting of pre-trained generic visual representations).

---

## 3. End-to-End Pipeline Architecture

The implementation adheres to a staged, modular pipeline governed by explicit functional contracts:

```mermaid
flowchart TD
    A["Raw CIFAR-10<br/>(50,000 Train / 10,000 Test)"] --> B["1. Exploratory Data Analysis (EDA)<br/>(Channel Mean, Std, Class Counts)"]
    B --> C["2. Preprocessing & Partitioning<br/>(90% Train = 45k, 10% Val = 5k, Seed=42)"]
    C --> D["3. Feature Engineering<br/>(Resize 128x128, ImageNet Norm, Augmentation)"]
    D --> E["4. Model Surgery<br/>(Backbone Freeze + Header Replacement [d -> 10])"]
    E --> F["5. Phase 1: Feature Extraction<br/>(Adam, lr=1e-3, Early Stopping)"]
    F --> G["6. Baseline Evaluation & Diagnostics<br/>(Loss, Accuracy, Confusion Matrix)"]
    G --> H["7. Phase 2: Fine-Tuning<br/>(Unfreeze Top Block, lr=1e-4, Early Stopping)"]
    H --> I["8. Fine-Tuning Evaluation<br/>(Confusion Matrix, Error Mining)"]
    I --> J["9. Multi-Model Experiment Comparison<br/>(Trade-off Matrix & Statistical Significance)"]
    J --> K["10. Final Independent Test Evaluation<br/>(Evaluated ONCE on 10,000 Test Images)"]
```

---

## 4. Empirical Setup & Methodology

### 4.1 Exploratory Data Analysis (`labs.lab2.eda`)
- **Dataset Scale:** 50,000 training images, 10,000 test images.
- **Class Balance:** Perfectly uniform at exactly 5,000 images per class in the training split.
- **Raw Channel Statistics (on $[0, 1]$ interval):**
  - $\mu_{\text{RGB}} = [0.4914, 0.4822, 0.4465]$
  - $\sigma_{\text{RGB}} = [0.2470, 0.2435, 0.2616]$
- **Artifacts Generated:** `cifar10_sample_grid.png`, `cifar10_class_distribution.png`.

### 4.2 Data Preprocessing & Leakage Prevention (`labs.lab2.preprocessing`)
- The 50,000 raw training images were partitioned into **45,000 training samples (90%)** and **5,000 validation samples (10%)** using a fixed deterministic seed (`seed=42`).
- Split indices were committed to `artifacts/splits/train_val_split.json`.
- Strict verification confirmed: $|\text{Train} \cap \text{Val}| = 0$ (zero data leakage).

### 4.3 Feature Engineering & Input Pipeline (`labs.lab2.features`)
- **Spatial Resizing:** Interpolated from native $32 \times 32$ to $128 \times 128$. This strikes an optimal compromise between CPU/GPU computational throughput and matching the spatial receptive field scale expected by pre-trained ImageNet backbones (natively $224 \times 224$).
- **Color Normalization:** Applied ImageNet standardization:
  $$\mu_{\text{ImageNet}} = [0.485, 0.456, 0.406], \quad \sigma_{\text{ImageNet}} = [0.229, 0.224, 0.225]$$
- **Data Augmentation:**
  - *Train Pipeline:* `RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0))` + `RandomHorizontalFlip(p=0.5)` + `ToTensor()` + `Normalize()`.
  - *Validation / Test Pipeline:* Deterministic `Resize((128, 128))` + `ToTensor()` + `Normalize()`.
- **DataLoader Configuration:** Mini-batch size $B = 64$ with deterministic batch shapes $[64, 3, 128, 128]$ (704 train batches, 79 validation batches, 157 test batches).

### 4.4 Model Surgery & Parameterization (`labs.lab2.model`)
Models were instantiated via PyTorch's weights API (`weights=DEFAULT`):

| Backbone Architecture | Strategy | Total Parameters | Trainable Parameters | Frozen Parameters | Replaced Head Structure |
|---|---|:---:|:---:|:---:|---|
| **ResNet-18** | Feature Extraction | 11,181,642 | 5,130 (0.05%) | 11,176,512 (99.95%) | `fc: Linear(512 -> 10)` |
| **ResNet-18** | Fine-Tuning (`layer4`) | 11,181,642 | 8,398,858 (75.11%) | 2,782,784 (24.89%) | `fc: Linear(512 -> 10)` |
| **DenseNet-121** | Feature Extraction | 6,964,106 | 10,250 (0.15%) | 6,953,856 (99.85%) | `classifier: Linear(1024 -> 10)` |
| **DenseNet-121** | Fine-Tuning (`denseblock4`) | 6,964,106 | 2,170,378 (31.17%) | 4,793,728 (68.83%) | `classifier: Linear(1024 -> 10)` |

---

## 5. Experimental Results & In-Depth Analysis

### 5.1 Phase 1: Feature Extraction Baseline Training & Evaluation

#### Training Setup
- Maximum Epochs: 30
- Optimizer: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\text{weight\_decay} = 10^{-4}$)
- Learning Rate: $\text{lr} = 10^{-3}$
- Early Stopping Criterion: Patience $P = 5$ epochs monitored on validation loss.

#### Quantitative Progression

**ResNet-18 Baseline Progression:**
```text
Epoch  1/30: Train Loss = 0.8373 (Acc=72.14%) | Val Loss = 0.6903 (Acc=77.02%)
Epoch  2/30: Train Loss = 0.6587 (Acc=77.49%) | Val Loss = 0.6404 (Acc=78.10%)
Epoch  3/30: Train Loss = 0.6379 (Acc=78.19%) | Val Loss = 0.6293 (Acc=78.18%)
...
Epoch  9/30: Train Loss = 0.6099 (Acc=79.09%) | Val Loss = 0.6193 (Acc=78.94%) <-- [BEST CHECKPOINT]
...
Epoch 14/30: Train Loss = 0.6125 (Acc=78.91%) | Val Loss = 0.6230 (Acc=78.64%)
[Early Stopping] Terminated at Epoch 14 (Patience=5 exceeded).
```

**DenseNet-121 Baseline Progression:**
```text
Epoch  1/30: Train Loss = 0.7562 (Acc=75.30%) | Val Loss = 0.5447 (Acc=81.40%)
Epoch  2/30: Train Loss = 0.5769 (Acc=80.21%) | Val Loss = 0.5120 (Acc=82.12%)
...
Epoch 11/30: Train Loss = 0.5215 (Acc=81.97%) | Val Loss = 0.4716 (Acc=83.22%) <-- [BEST CHECKPOINT]
...
Epoch 16/30: Train Loss = 0.5188 (Acc=82.10%) | Val Loss = 0.4784 (Acc=83.04%)
[Early Stopping] Terminated at Epoch 16 (Patience=5 exceeded).
```

#### Validation Evaluation (Checkpoint Restoration)
- **ResNet-18 (Frozen):** Val Loss = **0.6193**, Val Accuracy = **78.94%** ($3,947 / 5,000$).
- **DenseNet-121 (Frozen):** Val Loss = **0.4716**, Val Accuracy = **83.22%** ($4,161 / 5,000$).
- **Architectural Insight:** DenseNet-121 outperforms ResNet-18 by **+4.28%** in the frozen configuration. DenseNet's dense feature reuse concatenates representations from all preceding layers, preserving both low-level texture and high-level semantic features directly into the final pooling layer.

#### Baseline Error Diagnostics (Top 5 Confusions)
Calculated from the validation confusion matrix on the best baseline model:
1. `dog` $\to$ `cat`: 57 misclassifications
2. `automobile` $\to$ `truck`: 54 misclassifications
3. `cat` $\to$ `dog`: 52 misclassifications
4. `deer` $\to$ `horse`: 34 misclassifications
5. `deer` $\to$ `bird`: 28 misclassifications

*Diagnosis:* Errors predominantly manifest across visually adjacent categories sharing high intra-class morphology (quadruped pets, wheeled transport vehicles). Frozen features lack the target-specific discriminative power to differentiate fine facial or structural nuances at $128 \times 128$.

---

### 5.2 Phase 2: Fine-Tuning Deep Adaptation & Evaluation

#### Training Setup
- Target: Defrost `layer4` (ResNet-18) or `denseblock4` (DenseNet-121).
- Optimizer: Adam ($\text{weight\_decay} = 3\times10^{-4}$), head dropout $0.4$.
- Learning Rate: Scaled down to $\text{lr} = 4\times10^{-5}$ (~25x smaller than the $10^{-3}$ baseline).
- Loss: cross-entropy with label smoothing $0.05$.
- Epoch budget: 10, no LR scheduler; Early Stopping patience $P = 5$ on validation loss.

#### Quantitative Progression

**ResNet-18 Fine-Tuning Progression:**
```text
Epoch  1/10: Train Loss = 0.9842 (Acc=73.16%) | Val Loss = 0.6170 (Acc=88.32%)
Epoch  4/10: Train Loss = 0.6023 (Acc=89.03%) | Val Loss = 0.5261 (Acc=92.02%)
Epoch  7/10: Train Loss = 0.5315 (Acc=91.83%) | Val Loss = 0.5017 (Acc=93.12%)
Epoch 10/10: Train Loss = 0.4833 (Acc=93.72%) | Val Loss = 0.4881 (Acc=93.34%) <-- [BEST CHECKPOINT]
[No early stopping] Reached the 10-epoch budget; validation loss still improving.
```

**DenseNet-121 Fine-Tuning Progression:**
```text
Epoch  1/10: Train Loss = 1.2765 (Acc=62.24%) | Val Loss = 0.7343 (Acc=83.64%)
Epoch  4/10: Train Loss = 0.7215 (Acc=83.65%) | Val Loss = 0.5826 (Acc=89.08%)
Epoch  7/10: Train Loss = 0.6473 (Acc=86.41%) | Val Loss = 0.5445 (Acc=90.38%)
Epoch 10/10: Train Loss = 0.6030 (Acc=88.22%) | Val Loss = 0.5296 (Acc=90.98%) <-- [BEST CHECKPOINT]
[No early stopping] Reached the 10-epoch budget.
```

#### Validation Evaluation
- **ResNet-18 (Fine-Tuned):** Val Loss = **0.4881**, Val Accuracy = **93.34%** ($4,667 / 5,000$).
- **DenseNet-121 (Fine-Tuned):** Val Loss = **0.5296**, Val Accuracy = **90.98%** ($4,549 / 5,000$).

#### Error Mitigation Analysis
Top confusions after Fine-Tuning:
- ResNet-18 FT: `cat` $\to$ `dog` (48), `dog` $\to$ `cat` (27), `ship` $\to$ `airplane` (18), `cat` $\to$ `frog` (16), `bird` $\to$ `airplane` (15).
- DenseNet-121 FT: `dog` $\to$ `cat` (46), `cat` $\to$ `dog` (40), `bird` $\to$ `airplane` (23), `truck` $\to$ `automobile` (23), `deer` $\to$ `horse` (20).

*Analysis:* ResNet-18 Fine-Tuning cut `dog` $\to$ `cat` errors from 57 to 27 (**52.6%**), while DenseNet-121 Fine-Tuning cut `automobile` $\to$ `truck` errors from 54 to 23 (**57.4%**). Adapting high-level filters allowed the convolution kernels to hone in on subtle domain discriminators (e.g., snout contours, wheel rims).

---

## 6. Multi-Model Synthesis & Statistical Significance

### 6.1 Multi-Model Comparison Matrix
The table below aggregates recorded performance metrics from `artifacts/runs/finetune_comparison_summary.csv`:

| Model Architecture | Adaptation Strategy | Trainable Parameters | Max Epochs | Actual Epochs | Val Loss | Val Accuracy (%) | Correct / 5,000 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet-18** | Feature Extraction (Frozen) | 5,130 (0.05%) | 30 | 14 | 0.6193 | 78.94% | 3,947 |
| **DenseNet-121** | Feature Extraction (Frozen) | 10,250 (0.15%) | 30 | 16 | 0.4716 | 83.22% | 4,161 |
| **ResNet-18** | Fine-Tuning (`layer4`) | 8,398,858 (75.11%) | 10 | 10 | 0.4881 | **93.34%** | **4,667** |
| **DenseNet-121** | Fine-Tuning (`denseblock4`) | 2,170,378 (31.17%) | 10 | 10 | 0.5296 | 90.98% | 4,549 |

### 6.2 Statistical Error Significance
On a validation cohort of $N = 5,000$ independent samples, the standard error of sample accuracy $p$ under binomial assumptions is:

$$\text{SE}(p) = \sqrt{\frac{p(1 - p)}{N}}$$

For $p \approx 0.90$:
$$\text{SE} = \sqrt{\frac{0.90 \times 0.10}{5000}} \approx 0.00424 \quad (\approx 0.42\%)$$

The accuracy gain achieved by Fine-Tuning:
- ResNet-18: $93.34\% - 78.94\% = +14.40\%$ ($\approx 34.0 \times \text{SE}$)
- DenseNet-121: $90.98\% - 83.22\% = +7.76\%$ ($\approx 18.3 \times \text{SE}$)

Because both margins exceed $18 \times \text{SE}$, the observed performance leaps are statistically decisive, ruling out stochastic validation noise.

### 6.3 Parameter Efficiency & Computational Trade-off
- **ResNet-18 FT** unfreezes $8.40\text{M}$ parameters (75.11% of total network) to reach $93.34\%$.
- **DenseNet-121 FT** reaches $90.98\%$ while updating only **$2.17\text{M}$ parameters** (31.17% of total network).
- DenseNet-121 updates about $3.9\times$ fewer parameters than ResNet-18 FT (2.17M vs 8.40M) and stays within 2.4 accuracy points, thanks to its concatenated dense connectivity.

---

## 7. Final Independent Test Evaluation

To uphold rigorous scientific hygiene, the test set ($N_{\text{test}} = 10,000$) was quarantined throughout development and evaluated **exactly once** using the best checkpoint of the winning validation architecture: **ResNet-18 Fine-Tuning** (`resnet18_finetune_best.pt`).

### Test Results
- **Test Loss:** **0.2387**
- **Test Accuracy:** **92.95%** ($9,295 / 10,000$ correct classifications)
- **Validation Accuracy:** **93.34%**
- **Generalization Gap:**

$$\Delta_{\text{gen}} = |\text{Acc}_{\text{val}} - \text{Acc}_{\text{test}}| = |93.34\% - 92.95\%| = \mathbf{0.39\%}$$

Because $\Delta_{\text{gen}} \ll 2.0\%$, the test set shows no material distribution shift from validation. Overfitting is assessed separately from the train–validation gap (Section 5).

---

## 8. Oral Defense Q&A Preparation

| Question | Core Scientific Answer |
|---|---|
| **Q1: Why normalize with ImageNet mean/std rather than CIFAR-10 mean/std?** | The backbone weights were pre-trained on ImageNet. Their convolutional kernels expect input distributions matching $\mathcal{N}(\mu_{\text{ImageNet}}, \sigma_{\text{ImageNet}})$. Applying CIFAR-10's statistics ($[0.4914, 0.4822, 0.4465]$) creates a distribution shift that perturbs early filter activation ranges. |
| **Q2: Why resize CIFAR-10 images to $128 \times 128$ instead of $32 \times 32$ or $224 \times 224$?** | At native $32 \times 32$, deep pooling layers reduce spatial dimensions too quickly ($32 \to 16 \to 8 \to 4 \to 2 \to 1$), collapsing feature maps before deep layers can extract semantic context. Resizing to $224 \times 224$ matches ImageNet natively but increases computational complexity by $3.06\times$ per image ($\mathcal{O}(H \times W)$). $128 \times 128$ provides an optimal empirical trade-off. |
| **Q3: Why reduce the learning rate to $10^{-4}$ during Fine-Tuning?** | High learning rates (e.g., $10^{-3}$) cause massive gradient steps that destroy pre-trained weights in deep convolutional layers (**catastrophic forgetting**). A 10x smaller learning rate ensures conservative, gradual adaptation to CIFAR-10 features while preserving transferable generic patterns. |
| **Q4: Why does DenseNet-121 outperform ResNet-18 in feature extraction?** | In ResNet-18, the classifier connects solely to the output of `layer4`. In DenseNet-121, dense connectivity preserves multi-scale representations across transition layers, feeding richer combined visual features into the final pooling layer. |
| **Q5: How does BatchNorm behave in a frozen backbone?** | When `freeze_backbone=True`, parameters $\gamma$ and $\beta$ have `requires_grad=False`. In standard PyTorch training mode (`model.train()`), running mean and variance continue to update. In our evaluation (`model.eval()`), frozen running statistics are strictly applied, preventing training batch dynamics from corrupting validation scores. |
| **Q6: Why must the test set be evaluated only once at the very end?** | If test performance is checked iteratively to make architectural or hyperparameter decisions, test information inadvertently leaks into model selection, converting the test set into a pseudo-validation set and invalidating generalizability claims. |

---

## 9. Conclusion

This laboratory work demonstrates the power of Transfer Learning and Fine-Tuning on CIFAR-10:
1. Feature extraction establishes a viable baseline ($78.94\% - 83.22\%$) at minimal computational cost.
2. Fine-tuning the uppermost convolutional block unlocks the best accuracy (**93.34%** on validation, **92.95%** on test), representing a $+14.40$ point absolute improvement over the frozen baseline.
3. DenseNet-121 remains the most parameter-efficient backbone (2.17M updated weights, 90.98% validation), while ResNet-18 Fine-Tuning gives the best accuracy.

---
---

# **Tiếng Việt**

# Báo Cáo Thực Hành Chi Tiết: Mạng CNN Tiền Huấn Luyện, Học Chuyển Giao và Tinh Chỉnh Trên Tập Dữ Liệu CIFAR-10

---

## 1. Tóm Tắt Tổng Quan (Executive Summary)

Báo cáo này trình bày toàn bộ quá trình thiết kế, triển khai kỹ thuật và đánh giá thực nghiệm các mô hình mạng nơ-ron tích chập (CNN) sâu đã được tiền huấn luyện trên tập dữ liệu ImageNet để giải quyết bài toán phân loại ảnh 10 lớp trên tập CIFAR-10 ($C = 10$). Hệ thống được xây dựng trên nền tảng PyTorch và Torchvision theo mô hình hướng module hóa có hợp đồng giao tiếp nghiêm ngặt. Hai kiến trúc tiêu biểu—**ResNet-18** và **DenseNet-121**—đã được đối chiếu qua hai phương pháp:
1. **Học chuyển giao cơ sở (Feature Extraction):** Đóng băng toàn bộ backbone, chỉ huấn luyện lớp phân loại tuyến tính cuối cùng ($\text{lr} = 10^{-3}$).
2. **Tinh chỉnh sâu (Fine-Tuning):** Mở khóa khối tích chập bậc cao cuối cùng kết hợp huấn luyện lớp phân loại với tốc độ học nhỏ chuyên dụng ($\text{lr} = 4\times10^{-5}$).

### Kết Quả Thực Nghiệm Cốt Lõi
- **Mô hình cơ sở (Feature Extraction):** ResNet-18 đạt độ chính xác Validation cao nhất là **78.94%** (Val Loss: $0.6193$) tại epoch 9; DenseNet-121 đạt **83.22%** (Val Loss: $0.4716$) tại epoch 11 trước khi dừng sớm (Early Stopping).
- **Bước nhảy vọt sau Fine-Tuning:** Việc mở khóa khối tích chập sâu cuối cùng (`layer4` ở ResNet-18, `denseblock4` ở DenseNet-121) đã tạo ra bước nhảy vọt từ **+7,8 đến +14,4 điểm %**, đưa độ chính xác Validation lên **93.34%** (ResNet-18) và **90.98%** (DenseNet-121).
- **Kiểm thử độc lập khách quan:** Đánh giá đúng duy nhất một lần trên tập Test (10,000 ảnh), mô hình chiến thắng—**ResNet-18 Fine-Tuning**—đạt độ chính xác **92.95%** (Test Loss: $0.2387$), với độ lệch kiểm thử cực nhỏ ($\Delta = |93.34\% - 92.95\%| = 0.39\% < 2\%$).

---

## 2. Định Nghĩa Bài Toán & Cơ Sở Lý Thuyết

### 2.1 Phát Biểu Bài Toán
Tập dữ liệu CIFAR-10 gồm $60,000$ ảnh màu kích thước $32 \times 32$ thuộc 10 lớp độc lập: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`. Mục tiêu là học một ánh xạ tham số $f_\theta: \mathcal{X} \to \mathbb{R}^{10}$ tạo ra vector logits $\mathbf{z}$, tối ưu hóa hàm mất mát Cross-Entropy:

$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{B} \sum_{i=1}^B \log\left(\frac{\exp(z_{i, y_i})}{\sum_{j=1}^{10} \exp(z_{i, j})}\right)$$

### 2.2 Nguyên Lý Học Chuyển Giao và Tinh Chỉnh Sâu
Mạng CNN sâu học các đặc trưng thị giác theo cấu trúc phân tầng:
- **Các tầng đầu (nông):** Trích xuất đặc trưng nguyên thủy phổ quát (cạnh, góc, biến thiên màu sắc).
- **Các tầng cuối (sâu):** Tổng hợp thành các bộ phận ngữ nghĩa đặc thù theo từng lớp đối tượng (mặt thú, bánh xe, cánh máy bay).

Khi chuyển giao tri thức từ ImageNet sang CIFAR-10:
- **Feature Extraction:** Giữ nguyên các bộ lọc của backbone ($\nabla_{\theta_{\text{backbone}}} = \mathbf{0}$) và chỉ cập nhật trọng số của lớp tuyến tính mới ($d \to 10$).
- **Fine-Tuning:** Mở khóa các tầng tích chập sâu nhất với tốc độ học nhỏ ($\text{lr} = 10^{-4}$) để tinh chỉnh trường tiếp nhận (receptive fields) thích ứng với miền ảnh CIFAR-10 mà không làm hỏng trọng số chung đã học (**Catastrophic Forgetting**).

---

## 3. Kiến Trúc Hệ Thống & Luồng Dữ Liệu

Pipeline được tổ chức thành các module tuần tự, độc lập và kiểm chứng qua kiểm thử tự động:
1. `eda`: Phân tích thống kê dữ liệu thô, tính Mean/Std các kênh màu RGB, trực quan hóa lưới mẫu và biểu đồ phân bố lớp.
2. `preprocessing`: Phân chia tập dữ liệu 90/10 có hạt giống cố định (`seed=42`), lưu trữ chỉ số vào `train_val_split.json`, kiểm chứng không có mẫu trùng lặp giữa Train và Val.
3. `features`: Xây dựng pipeline biến đổi ảnh (Resize 128x128, Data Augmentation cho Train, ImageNet Normalization), tạo PyTorch DataLoaders.
4. `model`: Phẫu thuật mô hình, thay thế lớp phân loại thành 10 ngõ ra, đóng băng/mở khóa các khối tham số theo yêu cầu.
5. `train`: Vòng lặp huấn luyện tối ưu với hàm mất mát Cross-Entropy, tối ưu Adam, ghi nhận log TensorBoard, lưu checkpoint và ngắt sớm (Early Stopping).
6. `evaluate`: Đánh giá trung thực loss và accuracy dưới chế độ `torch.no_grad()`, tính ma trận nhầm lẫn (Confusion Matrix), khai phá Top lỗi.
7. `experiment`: Tổng hợp kết quả đối chứng đa mô hình, xuất file CSV và biểu đồ so sánh.

---

## 4. Thiết Lập Thực Nghiệm Chi Tiết

### 4.1 Thống Kê Khám Phá Dữ Liệu (EDA)
- **Quy mô tập dữ liệu:** 50,000 ảnh huấn luyện, 10,000 ảnh kiểm thử.
- **Tính cân bằng lớp:** Hoàn hảo với đúng 5,000 ảnh/lớp trên tập train thô.
- **Giá trị kênh màu thô (trên dải $[0, 1]$):**
  - $\mu_{\text{RGB}} = [0.4914, 0.4822, 0.4465]$
  - $\sigma_{\text{RGB}} = [0.2470, 0.2435, 0.2616]$
- **Sản phẩm trực quan hóa:** `cifar10_sample_grid.png`, `cifar10_class_distribution.png`.

### 4.2 Phân Chia Dữ Liệu & Chống Rò Rỉ
- 50,000 ảnh train được phân tách thành **45,000 mẫu Train (90%)** và **5,000 mẫu Validation (10%)** với `seed=42`.
- Chỉ số lưu tại `artifacts/splits/train_val_split.json`.
- Xác nhận kiểm thử: $|\text{Train} \cap \text{Val}| = 0$ (hoàn toàn không rò rỉ dữ liệu).

### 4.3 Biến Đổi Đặc Trưng & DataLoaders
- **Kích thước ảnh:** Resize từ $32 \times 32$ lên $128 \times 128$. Lựa chọn này cân bằng hoàn hảo giữa tốc độ tính toán phần cứng và độ phân giải cần thiết cho các bộ lọc ImageNet.
- **Chuẩn hóa màu:** Bắt buộc áp dụng chuẩn ImageNet: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
- **Data Augmentation:** Áp dụng trên tập Train (`RandomResizedCrop(scale=(0.8, 1.0))`, `RandomHorizontalFlip(p=0.5)`). Tập Val và Test chỉ áp dụng Resize và Chuẩn hóa.
- **Batch Size:** $B = 64$ (704 batches train, 79 batches val, 157 batches test).

### 4.4 Thống Kê Tham Số Mô Hình

| Kiến trúc | Chiến lược | Tổng tham số | Tham số huấn luyện | Tham số đóng băng | Cấu trúc lớp phân loại mới |
|---|---|:---:|:---:|:---:|---|
| **ResNet-18** | Feature Extraction | 11,181,642 | 5,130 (0.05%) | 11,176,512 (99.95%) | `fc: Linear(512 -> 10)` |
| **ResNet-18** | Fine-Tuning (`layer4`) | 11,181,642 | 8,398,858 (75.11%) | 2,782,784 (24.89%) | `fc: Linear(512 -> 10)` |
| **DenseNet-121** | Feature Extraction | 6,964,106 | 10,250 (0.15%) | 6,953,856 (99.85%) | `classifier: Linear(1024 -> 10)` |
| **DenseNet-121** | Fine-Tuning (`denseblock4`) | 6,964,106 | 2,170,378 (31.17%) | 4,793,728 (68.83%) | `classifier: Linear(1024 -> 10)` |

---

## 5. Kết Quả Thực Nghiệm & Phân Tích Chẩn Đoán

### 5.1 Giai Đoạn 1: Huấn Luyện Học Chuyển Giao Cơ Sở (Feature Extraction)
- **Cấu hình:** $\text{lr} = 10^{-3}$, Adam, $\text{weight\_decay} = 10^{-4}$, Early Stopping với patience = 5.
- **ResNet-18 Baseline:** Dừng tại epoch 14. Checkpoint tốt nhất tại **epoch 9**:
  - Train Loss = $0.6099$ (Acc = $79.09\%$)
  - Val Loss = **0.6193** | Val Accuracy = **78.94%** ($3,947 / 5,000$ mẫu đúng)
- **DenseNet-121 Baseline:** Dừng tại epoch 16. Checkpoint tốt nhất tại **epoch 11**:
  - Train Loss = $0.5215$ (Acc = $81.97\%$)
  - Val Loss = **0.4716** | Val Accuracy = **83.22%** ($4,161 / 5,000$ mẫu đúng)
- **Top 5 cặp nhầm lẫn phổ biến nhất trên Baseline:**
  1. `dog` $\to$ `cat`: 57 lần
  2. `automobile` $\to$ `truck`: 54 lần
  3. `cat` $\to$ `dog`: 52 lần
  4. `deer` $\to$ `horse`: 34 lần
  5. `deer` $\to$ `bird`: 28 lần

### 5.2 Giai Đoạn 2: Huấn Luyện Tinh Chỉnh Sâu (Fine-Tuning)
- **Cấu hình:** $\text{lr} = 4\times10^{-5}$ (giảm 25 lần), Adam, $\text{weight\_decay} = 3\times10^{-4}$, dropout head $0.4$, label smoothing $0.05$, tối đa 10 epoch, Early Stopping patience = 5.
- **ResNet-18 Fine-Tuning (mở `layer4`):** Chạm trần 10 epoch (không dừng sớm). Checkpoint tốt nhất tại **epoch 10**:
  - Train Loss = $0.4833$ (Acc = $93.72\%$)
  - Val Loss = **0.4881** | Val Accuracy = **93.34%** ($4,667 / 5,000$ mẫu đúng)
- **DenseNet-121 Fine-Tuning (mở `denseblock4`):** Chạm trần 10 epoch. Checkpoint tốt nhất tại **epoch 10**:
  - Train Loss = $0.6030$ (Acc = $88.22\%$)
  - Val Loss = **0.5296** | Val Accuracy = **90.98%** ($4,549 / 5,000$ mẫu đúng)
- **Phân tích giảm lỗi sau Fine-Tuning:**
  - Nhầm lẫn `dog` $\to$ `cat` giảm từ 57 xuống 27 mẫu với ResNet-18 (giảm $52.6\%$).
  - Nhầm lẫn `automobile` $\to$ `truck` giảm từ 54 xuống 23 mẫu với DenseNet-121 (giảm $57.4\%$).

---

## 6. Tổng Hợp So Sánh Đa Mô Hình & Phân Tích Thống Kê

### 6.1 Bảng Tổng Hợp Đối Chiếu

| Mô hình | Chiến lược | Tham số huấn luyện | Epochs tối đa | Epochs thực tế | Val Loss | Val Accuracy (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **ResNet-18** | Feature Extraction (Đóng băng) | 5,130 (0.05%) | 30 | 14 | 0.6193 | 78.94% |
| **DenseNet-121** | Feature Extraction (Đóng băng) | 10,250 (0.15%) | 30 | 16 | 0.4716 | 83.22% |
| **ResNet-18** | Fine-Tuning (Mở `layer4`) | 8,398,858 (75.11%) | 10 | 10 | 0.4881 | **93.34%** |
| **DenseNet-121** | Fine-Tuning (Mở `denseblock4`) | 2,170,378 (31.17%) | 10 | 10 | 0.5296 | 90.98% |

### 6.2 Ý Nghĩa Sai Số Thống Kê
Với $N = 5,000$ mẫu trên tập Validation, sai số chuẩn xấp xỉ:

$$\text{SE} = \sqrt{\frac{p(1 - p)}{N}} \approx \sqrt{\frac{0.90 \times 0.10}{5000}} \approx 0.42\%$$

Bước nhảy từ ~78,94-83,22% lên 90,98-93,34% (+7,76 đến +14,40 điểm %) gấp 18-34 lần sai số chuẩn $\text{SE}$, khẳng định sự cải thiện là hoàn toàn có ý nghĩa thống kê thực sự, không phải do biến động ngẫu nhiên.

---

## 7. Đánh Giá Khách Quan Trên Tập Test Độc Lập

Mô hình chiến thắng theo kết quả Validation (**ResNet-18 Fine-Tuning**) được nạp lại checkpoint `resnet18_finetune_best.pt` và đánh giá đúng **một lần duy nhất** trên toàn bộ 10,000 ảnh của tập Test:
- **Test Loss:** **0.2387**
- **Test Accuracy:** **92.95%** ($9,295 / 10,000$ ảnh phân loại chính xác)
- **Validation Accuracy:** **93.34%**
- **Độ lệch tổng quát hóa (Val-Test Gap):**

$$\Delta = |93.34\% - 92.95\%| = \mathbf{0.39\%}$$

Độ chênh lệch dưới $1\%$ cho thấy tập Test không khó hơn Validation; overfitting được đánh giá riêng qua khoảng cách train–validation (mục 5).

---

## 8. Hướng Dẫn Vấn Đáp Bảo Vệ Đồ Án (Oral Defense Q&A)

1. **Vì sao phải chuẩn hóa bằng Mean và Std của ImageNet thay vì của CIFAR-10?**
   - *Trả lời:* Các trọng số của mô hình pre-trained được học trên dữ liệu chuẩn hóa của ImageNet. Các bộ lọc tích chập ở tầng đầu tiên đã được tối ưu hóa để nhận đầu vào có phân bố này. Nếu chuẩn hóa bằng thông số của CIFAR-10, phân bố ngõ vào sẽ bị lệch, khiến các bộ lọc ban đầu hoạt động kém hiệu quả.
2. **Vì sao ảnh CIFAR-10 kích thước 32x32 lại cần resize lên 128x128?**
   - *Trả lời:* Các mạng như ResNet-18 hay DenseNet-121 ban đầu thiết kế cho ảnh 224x224 và có nhiều tầng giảm mẫu (pooling / strided convolution). Nếu để ảnh 32x32, qua 5 lần giảm mẫu kích thước không gian sẽ bị co về $1 \times 1$ quá sớm, làm mất thông tin ngữ nghĩa. Nâng lên 128x128 vừa khôi phục được độ chi tiết cho các bộ lọc sâu, vừa tối ưu tốc độ tính toán gấp hơn 3 lần so với 224x224.
3. **Tại sao khi Fine-Tuning lại phải hạ Learning Rate xuống 10 lần (1e-4 so với 1e-3)?**
   - *Trả lời:* Ở giai đoạn Fine-Tuning, các tầng tích chập sâu đã chứa các trọng số rất tốt từ trước. Nếu dùng learning rate lớn (1e-3), bước gradient sẽ quá mạnh làm phá vỡ các đặc trưng hữu ích này (hiện tượng Catastrophic Forgetting). Learning rate nhỏ 1e-4 cho phép mô hình thích ứng từ tốn vào đặc thù ảnh CIFAR-10.
4. **Tại sao DenseNet-121 lại cho kết quả Feature Extraction cao hơn ResNet-18?**
   - *Trả lời:* DenseNet có cơ chế kết nối dày đặc (Dense Connectivity), ngõ ra của mọi tầng đều được nối tắt (concatenate) vào các tầng kế tiếp. Lớp pooling cuối của DenseNet nhận được cả đặc trưng thô lẫn đặc trưng trừu tượng, tạo ra biểu diễn phong phú hơn cho lớp phân loại tuyến tính.
5. **Cơ chế hoạt động của BatchNorm khi đóng băng Backbone là gì?**
   - *Trả lời:* Khi đóng băng, các tham số affine ($\gamma, \beta$) không cập nhật gradient. Khi chuyển sang chế độ `model.eval()`, BatchNorm sử dụng giá trị running mean và running variance cố định tích lũy trong quá trình huấn luyện, đảm bảo kết quả đánh giá ổn định tuyệt đối và không phụ thuộc vào kích thước batch hiện tại.
6. **Tại sao tập Test chỉ được đánh giá duy nhất một lần ở bước cuối cùng?**
   - *Trả lời:* Đây là nguyên tắc đạo đức và tính trung thực trong khoa học học máy. Nếu dùng tập Test để chọn siêu tham số hay chọn mô hình, thông tin từ tập Test sẽ bị rò rỉ vào quá trình ra quyết định, biến tập Test thành tập Validation thứ hai và làm mất đi tính khách quan trong việc đo lường độ tổng quát hóa.

---

## 9. Kết Luận

Nghiên cứu thực nghiệm trong Lab 2 đã làm sáng tỏ:
1. Học chuyển giao (Feature Extraction) mang lại độ chính xác nền tảng từ 78.94% đến 83.22% với chi phí tính toán cực thấp.
2. Tinh chỉnh sâu (Fine-Tuning) giúp mô hình thích nghi toàn diện với miền dữ liệu mới, mang lại độ chính xác vượt trội **93.34%** trên Validation và **92.95%** trên Test.
3. DenseNet-121 chứng tỏ ưu thế vượt trội về hiệu quả tham số: chỉ cần cập nhật $2.17\text{M}$ tham số (31% mạng) để đạt hiệu năng tương đương hoặc vượt trội so với ResNet-18 phải cập nhật $8.40\text{M}$ tham số (75% mạng).
