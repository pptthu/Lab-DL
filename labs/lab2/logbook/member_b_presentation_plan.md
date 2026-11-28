# Lab 2 — Lộ trình trình bày của Member B (Bảo)

> Phần A: **lộ trình mở file + phân bổ thời gian** cho buổi nói không slide.
> Người trình bày: Bảo (Member B). Phần của Đại (Member A) và Thư do hai bạn tự nói.
> Hình thức: mở thẳng dự án trên máy (VS Code + terminal + notebook + TensorBoard), vừa mở vừa nói.
> **Không chạy train.** Chỉ mở output đã lưu, mở TensorBoard từ log có sẵn, và chạy `pytest` nhanh.

## Giả định (sửa lại nếu khác)

| Mục | Giả định | Nếu khác thì sửa ở đâu |
|---|---|---|
| Thời lượng phần Bảo | **10 phút nói** + Q&A riêng sau đó | Có bảng co giãn 7 phút / 12 phút ở cuối |
| Người nghe | Cô chấm vấn đáp trực tiếp + cả lớp | Giọng nói vẫn giữ mức kỹ thuật, nhưng giải thích thuật ngữ ở lần đầu xuất hiện |
| Đại và Thư | Tự nói phần của mình | Bảo chỉ giữ **1 câu chuyển ý vào** và **1 câu chuyển ý ra** |
| Slide | Không có | Mọi thứ nằm trong file/thư mục dưới đây |

---

## Bảng tổng: 8 điểm dừng, 10 phút

| # | Thời gian | Mở gì | Vị trí chính xác | Nhấn mạnh điều gì |
|---|---|---|---|---|
| 1 | 0:00–1:00 | `labs/lab2/project_description.md` | dòng 20 (`- **Data flow:**`) + `labs/lab2/main.py` dòng 55–64 (`STAGES`) | Định vị: mình ở giữa pipeline; `experiment` bọc `model` + `train` |
| 2 | 1:00–3:30 | `labs/lab2/train/train.py` | `train_model` 18–160; `save_checkpoint` 163–176; `load_checkpoint` 179–197 | Vòng lặp train/val, loss có trọng số batch, optimizer chỉ trên `requires_grad`, chọn checkpoint theo **val loss nhỏ nhất** |
| 3 | 3:30–5:00 | `labs/lab2/evaluate/evaluate.py` | `evaluate` 18–56; `confusion_matrix` 59–95; `top_confusions` 98–125 | `no_grad` + `eval()`; hàng = thật, cột = dự đoán; bỏ đường chéo |
| 4 | 5:00–6:30 | `labs/lab2/experiment/experiment.py` | `make_summary_writer` 18–39; `_resolve_best_metrics` 42–73; `run_comparison` 76–197; `save_results` 200–253 | 1 cấu hình = 1 run TensorBoard; import TensorBoard trễ; báo đúng epoch đã lưu; CSV cột `hp_*` |
| 5 | 6:30–8:00 | `labs/lab2/notebook/lab2_presentation.ipynb` | mục 6 (cell 13) → mục 7 (cell 16) → mục 8 (cell 19) → mục 9 (cell 22) → mục 10 (cell 24) → mục 11 (cell 25) | **Số thật**: baseline val 78,18% / test 79,00%; top-5 nhầm lẫn; bảng so sánh 4 cấu hình |
| 6 | 8:00–9:00 | Terminal: `tensorboard --logdir labs/lab2/artifacts/logs` rồi `.../artifacts/runs` | tab Scalars | 3 scalar `Loss/train`, `Loss/val`, `Accuracy/val`; mỗi run 1 màu |
| 7 | 9:00–9:30 | Terminal: lệnh `pytest` (đã gõ sẵn) | 5 file test của mình | 24 passed — chỉ là test giả, **không phải** bằng chứng accuracy |
| 8 | 9:30–10:00 | Quay lại `labs/lab2/notebook/...` mục 11 (cell 25) hoặc mở `README.md` §18 (dòng 348–360) | — | Nói thẳng 4 giới hạn thật + cách khắc phục |

---

## Chi tiết từng điểm dừng

### Điểm 1 (0:00–1:00) — Định vị phần của mình trong pipeline

**Mở:** `labs/lab2/project_description.md`, cuộn tới **dòng 20**.

```text
- **Data flow:** eda -> preprocessing -> features -> model (backbone + header) -> train (+TensorBoard) -> evaluate ; experiment wraps model + train
```

**Mở tiếp:** `labs/lab2/main.py`, dòng **55–64** (`STAGES = ["eda", ..., "experiment", "results"]`).

**Nhấn mạnh**
- 8 stage; mình phụ trách 3 stage giữa: `train`, `evaluate`, `experiment`.
- `experiment` **không phải** stage song song — nó gọi lại `build_model`/`unfreeze_last_blocks` của Đại và `train_model` của mình, nên nó là lớp bọc (wrapper).
- Ai làm gì: Đại → `eda`, `preprocessing`, `features`, `model`; Thư → `main.py`, notebook, README, PR.

**Tuỳ chọn nếu muốn có "sơ đồ" trực quan:** `labs/lab2/README.md` §7 (dòng 53–91) có sơ đồ Mermaid. Chỉ mở nếu VS Code của bạn render Mermaid; nếu không thì đừng mở (sẽ thấy toàn chữ `flowchart TD`).

**Câu chuyển ý vào (dùng khi Thư/Đại vừa nói xong):**
> "Phần trên là dữ liệu và model đã sẵn sàng. Em nhận đầu vào từ `make_loaders` của Đại và `build_model` của Đại, nên em xin nói theo đúng thứ tự chạy: `train` trước, rồi `evaluate`, cuối cùng là `experiment` bọc cả hai."

**Câu chuyển ý ra (khi hết 10 phút):**
> "Đó là ba module của em; phần còn lại là cách Thư nối các stage trong `main.py` và phần đánh giá chung của nhóm."

---

### Điểm 2 (1:00–3:30) — `train.py`, 2 phút 30

**Mở:** `labs/lab2/train/train.py` (197 dòng). Không cuộn lướt toàn file — đi theo 6 mốc dòng:

| Dòng | Chỉ vào | Nói gì (ý chính, chi tiết lời thoại ở phần B) |
|---|---|---|
| 18–29 | Chữ ký `train_model` | 10 tham số; `writer=None` nghĩa là TensorBoard là tuỳ chọn |
| 32–35 | Docstring "Checkpoint selection criterion" | Vì sao chọn theo val **loss** và trả thêm `best_*` |
| 49–55 | 2 guard + `device` | Fail fast: `epochs<=0`, `lr<=0`; chuyển model sang device |
| 58–61 | `criterion` + `trainable_params` | CrossEntropyLoss; **chỉ** tham số `requires_grad=True`; nếu rỗng thì báo lỗi |
| 63–69 | Optimizer | Adam hoặc SGD momentum 0.9; tên khác thì raise |
| 86–109 | Nửa trên của vòng lặp | `model.train()`, 5 dòng cốt lõi (95–99), loss cộng dồn **có trọng số batch size** (101–103) |
| 112–138 | Nửa dưới của vòng lặp | `model.eval()` + `torch.no_grad()`; val loss và accuracy |
| 140–153 | TensorBoard + chọn checkpoint | 3 scalar; điều kiện `best_epoch == 0 or val < best` |
| 155–160 | Trả `history` | 6 khoá: 3 list + 3 `best_*` |
| 163–197 | 2 hàm checkpoint | `save_checkpoint` tạo thư mục cha; `load_checkpoint` dùng `weights_only=True` và đặt `eval()` |

**Nhấn mạnh 3 câu "ăn điểm"**
1. Loss cộng dồn phải nhân `batch_size` vì batch cuối có thể lẻ → trung bình mới đúng.
2. `model.eval()` để tắt Dropout và **dùng running stats của BatchNorm**; `no_grad()` để không dựng đồ thị và không tốn bộ nhớ.
3. Checkpoint chọn theo **val loss nhỏ nhất**, nên accuracy phải lấy **của đúng epoch đó** — không lấy `max(val_acc)`.

**Nếu cô hỏi ngay tại đây:** "Sao không chọn theo val acc?" → vì loss phản ánh độ tự tin (xác suất đúng), hai epoch cùng acc vẫn khác loss; chọn theo loss ổn định hơn khi số epoch ít.

---

### Điểm 3 (3:30–5:00) — `evaluate.py`, 1 phút 30

**Mở:** `labs/lab2/evaluate/evaluate.py` (125 dòng).

| Dòng | Chỉ vào | Nói gì |
|---|---|---|
| 28–30 | `device` + `model.eval()` | Đánh giá không cập nhật gì |
| 32–49 | Body | `no_grad`, loss có trọng số, `argmax(dim=1)`, đếm đúng/tổng |
| 51–56 | Chia trung bình | `total_samples == 0` → raise; trả 2 số float trong [0,1] |
| 76 | `np.zeros((C,C))` | Ma trận đếm, không phải xác suất |
| 89–93 | `cm[t, p] += 1` | **hàng = nhãn thật, cột = nhãn dự đoán**; ngoài phạm vi thì raise |
| 115–124 | Vòng lặp + `if i != j` | Bỏ đường chéo (dự đoán đúng) — chỉ giữ lỗi |
| 124 | `sort(..., reverse=True)` | Giảm dần theo số lần nhầm, cắt `k` |

**Nhấn mạnh**
- Cùng một công thức loss/accuracy với trong `train_model` → số val lúc train và số evaluate lại phải khớp (notebook: 0,6293 / 78,18% ở cả hai chỗ).
- Chỉ số **test** dùng một lần, sau khi đã chọn xong cấu hình bằng validation.

---

### Điểm 4 (5:00–6:30) — `experiment.py`, 1 phút 30

**Mở:** `labs/lab2/experiment/experiment.py` (253 dòng).

| Dòng | Chỉ vào | Nói gì |
|---|---|---|
| 35–39 | `try: from torch.utils.tensorboard import SummaryWriter` | Import **trễ**: máy không cài TensorBoard vẫn import module được; chỉ bắt `ImportError` |
| 42–73 | `_resolve_best_metrics` | Thứ tự ưu tiên: `best_epoch` → min `val_loss` → max `val_acc` → rỗng |
| 132–146 | 2 vòng lặp + `run_name` | Model × lưới; mỗi cấu hình tạo `run_name` và **1 thư mục log riêng** |
| 148–160 | `ckpt_path`, build, unfreeze | `freeze_backbone` từ lưới; `unfreeze_blocks > 0` thì gọi `unfreeze_fn` |
| 162–180 | `perf_counter` + `try/finally` | Đo `train_seconds`; `writer.close()` kể cả khi train lỗi |
| 183–195 | `_resolve_best_metrics` + dict kết quả | `best_val_acc`, `best_val_loss`, `best_epoch`, `train_seconds`, `checkpoint`, `tensorboard_logged` |
| 232–234 | `hp_*` columns | Mọi khoá trong `hparams` thành cột `hp_<tên>` — nên lưới có thêm biến là CSV tự có cột |

**Nhấn mạnh (và nói luôn giới hạn ở đây, đừng để cô bắt)**
- `loaders` được tạo **một lần bên ngoài** (dòng 127) → `batch_size` trong lưới **không có tác dụng**.
- Seed chỉ đặt một lần trước vòng lặp (ở notebook) → mỗi cấu hình chỉ 1 seed, không có ý nghĩa thống kê.
- **Lưới trong notebook khác lưới mặc định của `main.py`:** notebook dùng `lr=1e-3` cho frozen và `lr=1e-4` cho mở khối cuối; `main.py` `default_comparison_grid` (dòng 82–109) dùng **cùng một `lr`** cho cả hai. Vì vậy 4 con số trong báo cáo là của **lưới notebook**, không phải của lệnh `python -m labs.lab2.main`.

---

### Điểm 5 (6:30–8:00) — Notebook, output đã lưu, 1 phút 30

**Mở:** `labs/lab2/notebook/lab2_presentation.ipynb`. Chỉ mở 6 mục, không chạy cell nào.

| Mục (số cell trong bản hiện tại) | Output phải thấy | Câu chốt |
|---|---|---|
| 6 — Huấn luyện (cell 13) | Epoch 1/3 `0.8373 / 0.6903 / 77.02%`; epoch 3 `0.6379 / 0.6293 / 78.18%` | Loss train và val cùng giảm, val < train ở epoch 3 → chưa overfit ở 3 epoch |
| 7 — Val + confusion matrix (cell 16) | Val loss 0,6293; **78,18% (3909/5000)**; `(10, 10)`; heatmap | Nạp lại checkpoint rồi mới đánh giá — đúng quy trình |
| 8 — Top 5 nhầm lẫn (cell 19) | cat→dog 92; automobile→truck 64; horse→deer 54; ship→airplane 43; bird→frog 38 | Lỗi có cấu trúc: cùng hình dáng/bối cảnh, không phải lỗi ngẫu nhiên |
| 9 — Bảng so sánh (cell 22) | 4 dòng: 0,7760 / 0,9116 / 0,8246 / 0,9022 + `train_seconds` | Fine-tuning khối cuối tăng mạnh; DenseNet chậm hơn ResNet trên CPU |
| 10 — Test (cell 24) | Test loss 0,6251; **79,00% (7900/10000)**; lệch val–test 0,82% | Đây là test của **baseline ResNet-18 frozen**, không phải của cấu hình 91,16% |
| 11 — Tổng kết (cell 25) | 3 ý tóm tắt + 5 câu hỏi | Dùng để chuyển sang phần giới hạn |

**3 con số phải nói kèm nguồn, và 2 con số phải nói kèm nhãn**
- 78,18% = **validation** của baseline (ResNet-18 frozen, 3 epoch, ảnh 128).
- 79,00% = **test** của baseline — **không phải** của mô hình tốt nhất.
- 91,16% = **validation** của cấu hình ResNet-18 **mở khối cuối, lr=1e-4, 2 epoch** — **chưa** đánh giá trên test.
- Nguồn: output đã lưu trong notebook (cell 13/16/22/24) + `labs/lab2/artifacts/runs/model_comparison.csv`.

**Cảnh báo kỹ thuật (đã kiểm tra):** ở commit HEAD, notebook **chỉ có output ở cell 2**; toàn bộ output số liệu của các cell 4→24 là **thay đổi chưa commit** trong working tree. Nếu ai `git checkout`/reset hoặc nộp từ HEAD thì sẽ mất bảng số. Trước giờ trình bày phải kiểm tra lại (xem phần D) và nhắn Thư commit notebook.

---

### Điểm 6 (8:00–9:00) — TensorBoard, 1 phút

**Mở:** terminal đã chạy sẵn (xem phần D), trình duyệt ở `http://localhost:6006`.

**Đường dẫn log có thật trên máy:**
- Baseline: `labs/lab2/artifacts/logs/resnet18_baseline` (16 KB).
- So sánh: `labs/lab2/artifacts/runs/` (~140 MB) — 4 thư mục:
  `resnet18_run0_lr0.001_freezeTrue`, `resnet18_run1_lr0.0001_freezeTrue`,
  `densenet121_run0_lr0.001_freezeTrue`, `densenet121_run1_lr0.0001_freezeTrue`.

**Nói khi chỉ vào tab Scalars**
- 3 scalar do `train_model` ghi: `Loss/train`, `Loss/val`, `Accuracy/val`; trục x là **epoch**, không phải step.
- Mỗi thư mục run = 1 cấu hình; bật/tắt từng run để so 2 đường.
- 4 run trong `runs/` có **event file thật** (đuôi `events.out.tfevents.*`), mở được kể cả khi không có mạng.
- Baseline chỉ có 3 điểm (3 epoch); 4 run so sánh chỉ có 2 điểm (2 epoch) → đường ngắn là bình thường, không phải lỗi.

---

### Điểm 7 (9:00–9:30) — pytest, 30 giây

**Lệnh đã gõ sẵn trong terminal thứ 2:**
```bash
python -m pytest labs/lab2/tests/test_train.py labs/lab2/tests/test_evaluate.py \
  labs/lab2/tests/test_experiment.py labs/lab2/tests/test_checkpoint_consistency.py \
  labs/lab2/tests/test_tensorboard_optional.py -q
```
Kết quả đã kiểm tra ngày 2026-11-06: **24 passed in ~5s**.

**Nói ngay sau khi thấy kết quả (rất quan trọng):**
> "Đây là test trên dữ liệu giả, chạy vài giây, mục đích là kiểm tra hợp đồng hàm và các nhánh lỗi. Nó **không** chứng minh độ chính xác 78% — con số đó đến từ lần chạy thật trên CIFAR-10, ghi trong notebook."

---

### Điểm 8 (9:30–10:00) — Giới hạn thật, 30 giây (nói nhanh, dứt khoát)

Mở lại cell 25 (mục 11) hoặc `README.md` §18 dòng 348–360 (bảng so sánh) rồi nói 4 ý:

1. **1 seed / cấu hình** — chênh vài điểm chưa kiểm định ý nghĩa. (Notebook cell 20 đã ghi SE ≈ 0,5% với n=5.000.)
2. **`batch_size` trong lưới không có tác dụng** — loader tạo một lần bên ngoài.
3. **BatchNorm vẫn cập nhật thống kê khi backbone đóng băng** — vì `model.train()` bật train mode cho cả BN.
4. **Chỉ 2–3 epoch, ảnh 128 thay vì 224 vì CPU**; CIFAR-10 gốc 32×32 nên phóng to gây nhiễu nội suy.

Khắc phục nếu có thêm thời gian: GPU, 3–5 seed, scheduler (`CosineAnnealingLR`), augmentation mạnh hơn.

---

## Câu chuyển ý (chỉ 2 câu, dùng đúng 2 lần)

**Vào (sau phần Đại/Thư trước):**
> "Đầu vào của em là DataLoader đã có sẵn và model đã thay lớp cuối, nên em xin đi theo thứ tự chạy: train, evaluate, rồi experiment bọc cả hai."

**Ra (trước phần Đại/Thư sau):**
> "Ba module của em dừng ở việc tạo ra checkpoint, bảng số và log; phần nối các stage và phần kết luận chung của nhóm do Thư trình bày."

---

## Bảng số được phép nói + nguồn (chống bịa số)

| Con số | Nghĩa | Nguồn |
|---|---|---|
| 78,18% (3909/5000) | val acc baseline ResNet-18 frozen, 3 epoch, ảnh 128 | notebook cell 16 output (chưa commit) + README §18 |
| 79,00% (7900/10000), test loss 0,6251 | test acc **của baseline** | notebook cell 24 output + README §18 |
| 77,02 / 78,10 / 78,18% | val acc 3 epoch; val loss 0,6903 / 0,6404 / 0,6293 | notebook cell 13 output |
| 0,7760 / 0,9116 / 0,8246 / 0,9022 | best val acc 4 cấu hình, 2 epoch | notebook cell 22 + `artifacts/runs/model_comparison.csv` |
| 1097,18 / 1166,07 / 5132,97 / 3521,94 s | `train_seconds` 4 cấu hình | `artifacts/runs/model_comparison.csv` |
| 24 passed | 5 file test của Member B, dữ liệu giả | mình chạy 2026-11-06 |
| 5.130 / 11.181.642 (0,05%) | tham số trainable/tổng của ResNet-18 frozen | notebook cell 11 + README §18 |
| 8.398.858 (75,11%) | tham số trainable khi mở `layer4` | notebook cell 11 |
| 92 / 64 / 54 / 43 / 38 | số lần nhầm của top 5: cat→dog, automobile→truck, horse→deer, ship→airplane, bird→frog (validation) | notebook cell 19 + README §18 |

**Không được nói:** "test 91,16%", "fine-tune giúp tăng nhẹ" (thực tế +13,56 điểm val), "52 test" (đó là cả thư mục, cần CIFAR-10 đã tải; README còn ghi 23 — hai số này lệch nhau, đừng đọc số nào khi chưa kiểm tra lại).

---

## Co giãn thời lượng

**Nếu chỉ có 7 phút:** bỏ điểm 7 (pytest), rút điểm 2 còn 2 phút (chỉ nói dòng 58–61, 91–103, 112–138, 146–153), gộp điểm 6 vào điểm 5 (nói TensorBoard trong lúc bảng so sánh còn mở).

**Nếu có 12 phút:** thêm 2 phút ở điểm 2 để giải thích `best_epoch == 0 or val < best` (vì sao luôn có checkpoint, và cạm bẫy NaN: nếu epoch 1 ra NaN thì `NaN < NaN` luôn Sai nên checkpoint bị giữ ở epoch 1), thêm 1 phút ở điểm 5 để chỉ cell 19 (8 ảnh mẫu đúng/sai).

---

## Việc phải báo Thư (không tự sửa)

1. Notebook có output mới nhưng **chưa commit** (HEAD chỉ có output ở cell 2) — cần commit trước khi nộp.
2. `README.md` còn ghi "23 test / 8 file", "DenseNet-121 mở khối cuối: Pending", "`main.py` còn stub `NotImplementedError`" — cả ba đã lỗi thời so với code + notebook hiện tại.
3. `main.py` `default_comparison_grid` dùng **cùng lr** cho frozen và fine-tune, khác lưới notebook đã dùng để ra 4 con số trong báo cáo.
4. `train_model`: nhánh `best_epoch == 0 or val < best` không xử lý NaN (epoch sau có loss tốt hơn cũng không lưu được).

---

# Phần B — Kịch bản nói (làm dần từng điểm dừng)

> Quy ước: chữ **in đậm** là chỗ cần nhấn giọng. Dòng `[Màn hình: ...]` **không đọc** — đó là hành động.
> Tốc độ: ~130 từ/phút.

## B1. Điểm dừng 1 (0:00–1:00) — Định vị phần của mình (~140 từ)

[Màn hình: mở `labs/lab2/project_description.md`, cuộn tới dòng 20]

"Em xin bắt đầu bằng việc định vị phần của em. Đây là luồng dữ liệu của nhóm: eda, preprocessing, features, model, train, evaluate. Còn **experiment** thì bọc lấy model và train.

[Màn hình: chuyển sang `labs/lab2/main.py`, dòng 55–64]

Đây là danh sách stage trong `main.py`. Tám stage. Em phụ trách **ba stage giữa**: train, evaluate, experiment. Đại phụ trách bốn phần đầu. Thư phụ trách `main.py`, notebook và README.

Đầu vào của em là DataLoader đã có sẵn từ `features`, và model đã thay lớp cuối từ `model`. Nên em **không đụng vào dữ liệu, không đụng vào kiến trúc**. Em chỉ lo ba việc: huấn luyện ra checkpoint, đánh giá checkpoint đó, và chạy nhiều cấu hình để so sánh.

Em xin đi đúng thứ tự chạy: train trước, evaluate sau, experiment cuối."

**Câu cô có thể hỏi ngay tại đây:** "Ba module của em phụ thuộc vào phần của Đại ở chỗ nào?"
**Đáp án mẫu:** "Hai chỗ, đều qua hợp đồng hàm. Một, `make_loaders` trả batch `[B,3,H,W]` với nhãn 0–9. Hai, `build_model` trả logits `[B,10]`, và `unfreeze_last_blocks` để em mở khối cuối. Em chỉ phụ thuộc vào chữ ký hàm, nhưng em đã đọc code của Đại để hiểu đầu vào và đầu ra là gì."

## B2. Điểm dừng 2 (1:00–3:30) — `train.py` (~300 từ)

[Màn hình: mở `labs/lab2/train/train.py`, dòng 18–29 — chữ ký `train_model`]

"Đây là `train.py`, 197 dòng, ba hàm. Em nói hàm chính trước. `train_model` nhận model, hai loader, epochs, lr, device, đường dẫn checkpoint. `writer` mặc định là `None` — nghĩa là **TensorBoard là tuỳ chọn**, máy không cài vẫn chạy được.

[Màn hình: dòng 32–35 — docstring "Checkpoint selection criterion"]

Đây là quyết định quan trọng nhất của em: checkpoint chọn theo **validation loss nhỏ nhất**, và history trả thêm `best_epoch`, `best_val_loss`, `best_val_acc`. Nghĩa là **con số em báo cáo và file `.pt` em lưu là cùng một epoch**. Vì `max(val_acc)` có thể rơi vào epoch khác với epoch đã lưu — báo cáo sẽ lệch với file.

[Màn hình: dòng 49–61]

Hai dòng guard đầu hàm: epochs và lr phải dương, sai thì báo lỗi ngay. Rồi `CrossEntropyLoss`, và optimizer chỉ nhận tham số có `requires_grad=True`. Với ResNet-18 đóng băng, đó là **5.130 trên 11.181.642** tham số, khoảng **0,05%**.

[Màn hình: dòng 63–69]

Adam, hoặc SGD với momentum 0,9. Tên khác thì báo lỗi.

[Màn hình: dòng 86–109]

Mỗi epoch có hai nửa. Nửa train: `model.train()`, rồi năm dòng — zero_grad, forward, loss, backward, step. Loss cộng dồn **nhân với batch size**, vì batch cuối có thể lẻ, chia trung bình mới đúng.

[Màn hình: dòng 112–138]

Nửa validation: `model.eval()` và `torch.no_grad()`. `eval` để BatchNorm dùng running stats, và tắt dropout nếu model có — ví dụ VGG, còn ResNet-18 thì không có dropout. `no_grad` để không dựng đồ thị, không tốn bộ nhớ.

[Màn hình: dòng 140–153]

Nếu có writer, ghi ba scalar: `Loss/train`, `Loss/val`, `Accuracy/val`, trục x là epoch. Và đây là chỗ lưu checkpoint: chỉ lưu khi val loss nhỏ hơn best. Điều kiện `best_epoch == 0` để **chắc chắn epoch đầu luôn có file**.

[Màn hình: dòng 155–160 rồi 163–197]

Cuối cùng history trả ba list theo epoch: `train_loss`, `val_loss`, `val_acc`, cộng ba khoá `best_*`. Hai hàm còn lại: `save_checkpoint` tạo thư mục cha rồi `torch.save` state_dict; `load_checkpoint` dùng `weights_only=True`, chuyển sang device và đặt `eval()`."

**Câu cô có thể hỏi ngay tại đây:** "Vì sao chọn checkpoint theo validation loss mà không theo validation accuracy?"
**Đáp án mẫu:** "Đây là lựa chọn của nhóm: dùng **một tiêu chí duy nhất** cho mọi thí nghiệm, để checkpoint và con số báo cáo luôn cùng một epoch. Loss thấp khi mô hình đặt xác suất cao vào đúng lớp, nên loss **ổn định hơn** khi so giữa các epoch. Ở baseline thì hai tiêu chí trùng nhau: val loss giảm và val acc tăng suốt 3 epoch (val loss 0,6903 → 0,6404 → 0,6293; val acc 77,02% → 78,10% → 78,18%), nên chọn theo loss hay theo acc đều ra epoch 3 — nguồn: notebook cell 13."

**Câu luyện nói (30 giây):** Vì sao checkpoint chọn theo val loss, và vì sao history phải trả `best_val_acc` thay vì để người gọi tự lấy `max(val_acc)`?

## B3. Điểm dừng 3 (3:30–5:00) — `evaluate.py` (~200 từ)

[Màn hình: mở `labs/lab2/evaluate/evaluate.py`, dòng 18–30]

"Đây là `evaluate.py`, 125 dòng, ba hàm. `evaluate` nhận model, loader, device. Hai dòng đầu: chuyển model sang device rồi `model.eval()` — cùng lý do như lúc validation trong `train_model`.

[Màn hình: dòng 37–49]

Toàn bộ vòng lặp nằm trong `torch.no_grad()`. Mỗi batch: forward, `CrossEntropyLoss`, `argmax` lấy nhãn dự đoán, cộng dồn loss **nhân batch size** và đếm số đúng.

[Màn hình: dòng 51–56]

Chia cho tổng số mẫu. Loader rỗng thì báo lỗi. Trả hai float: `avg_loss` và accuracy trong khoảng 0 đến 1.

[Màn hình: dòng 59–76]

`confusion_matrix` cấp phát ma trận `C×C` bằng numpy, toàn số 0.

[Màn hình: dòng 89–93]

Với từng ảnh, tăng ô `[nhãn thật, nhãn dự đoán]` lên 1. **Hàng là nhãn thật, cột là nhãn dự đoán**. Nhãn ra ngoài phạm vi thì báo lỗi ngay, không âm thầm bỏ qua.

[Màn hình: dòng 115–124]

`top_confusions` bỏ đường chéo — vì đường chéo là đoán đúng — rồi sắp giảm dần theo số lần nhầm và cắt `k`. Nhờ vậy trên validation em đọc được ngay 5 cặp hay nhầm nhất, dùng ở mục 8 của notebook."

**Câu cô có thể hỏi ngay tại đây:** "Đã có accuracy rồi, sao còn cần confusion matrix?"
**Đáp án mẫu:** "Accuracy chỉ là một số tổng, không cho biết lỗi nằm ở đâu. Confusion matrix cho biết lớp nào bị nhầm thành lớp nào — ví dụ cat bị nhầm thành dog, automobile thành truck, toàn là cặp có hình dáng và bối cảnh giống nhau. Có bảng đó mới biết nên thêm dữ liệu hay augmentation cho nhóm lớp nào, thay vì đoán."

**Câu luyện nói (20 giây):** Hàng và cột của confusion matrix là gì, và vì sao `top_confusions` phải bỏ đường chéo?

---

# Phần C — Giải thích code (cheat sheet để ôn trước giờ nói)

## C1 — `train.py` (197 dòng, 3 hàm)

| Dòng | Làm gì | Vì sao |
|---|---|---|
| 18–29 | Chữ ký `train_model`, `writer=None` | TensorBoard là tuỳ chọn, không cài vẫn chạy |
| 32–35 | Docstring nêu tiêu chí chọn checkpoint | Hợp đồng với người gọi: báo cáo đúng epoch đã lưu |
| 49–52 | Guard `epochs > 0`, `lr > 0` | Fail fast, tránh chạy 0 epoch rồi trả history rỗng |
| 54–55 | `torch.device(...)`, `model.to(device)` | Chuẩn hoá device; ảnh và tham số phải cùng device |
| 58 | `nn.CrossEntropyLoss()` | Nhận logits thô + nhãn `int64`, không cần softmax |
| 59 | Lọc `p.requires_grad` | Chỉ đưa phần được mở khóa vào optimizer |
| 60–61 | Rỗng thì raise | Tránh optimizer không có gì để học |
| 63–69 | Adam / SGD momentum 0,9 / raise | Một chỗ duy nhất quyết định thuật toán |
| 71–83 | `history` 3 list + `best_*`; `inf`, 0, 0.0 | Epoch 1 luôn thắng khởi tạo |
| 86–87 | `range(1, epochs+1)`, `model.train()` | Epoch 1-based để làm trục x TensorBoard; bật BN/Dropout train mode |
| 95–99 | zero_grad → forward → loss → backward → step | Zero trước vì PyTorch cộng dồn `.grad` |
| 101–103 | `loss.item() * bs`, cộng `train_samples` | `.item()` tách khỏi đồ thị; nhân bs vì batch cuối có thể lẻ |
| 105–109 | Guard `train_samples == 0`, chia trung bình | Tránh chia 0, ra loss đúng trọng số |
| 112–117 | `model.eval()` + `torch.no_grad()` | BN dùng running stats, tắt dropout, không dựng đồ thị |
| 122–135 | Forward, loss, `argmax(dim=1)`, đếm đúng | Cùng công thức với `evaluate.py` |
| 140–144 | 3 `add_scalar(tag, value, epoch)` | `Loss/train`, `Loss/val`, `Accuracy/val`; x = epoch |
| 146–153 | Lưu khi `best_epoch == 0 or val < best` | Luôn có file; kèm `best_val_acc` của đúng epoch |
| 155–160 | Ghi 3 khoá `best_*` vào history | Người gọi không phải tự `max(val_acc)` |
| 163–176 | `save_checkpoint`: tạo thư mục cha + `torch.save(state_dict)` | `torch.save` không tự tạo thư mục |
| 179–197 | `load_checkpoint`: `map_location`, `weights_only=True`, `eval()` | Nạp an toàn, đúng device; mode không nằm trong file |

**Ba câu chốt:** (1) train 5 bước chuẩn, loss có trọng số theo số ảnh; (2) validate bằng `eval()` + `no_grad`; (3) checkpoint theo val loss nhỏ nhất và history trả `best_val_acc` cùng epoch.

**Cạm bẫy đã biết:** nếu epoch 1 ra NaN thì `NaN` không bao giờ bị thay → file giữ ở epoch 1.

## C2 — `evaluate.py` (125 dòng, 3 hàm)

| Dòng | Làm gì | Vì sao |
|---|---|---|
| 18–30 | Chữ ký `evaluate`; `device`, `model.to`, `model.eval()` | Cùng device; tắt dropout, BN dùng running stats |
| 32–35 | `CrossEntropyLoss` + 3 biến đếm | Cùng loss với lúc train để số liệu khớp |
| 37–49 | `no_grad`, forward, loss, `argmax`, cộng dồn theo bs | Kết quả vẫn đúng dù không có grad; không dựng đồ thị thì nhanh và nhẹ |
| 45–46 | `total_loss += loss.item() * bs` | Trung bình đúng trọng số với batch lẻ |
| 51–56 | Guard rỗng, chia, trả `float` | Tránh chia 0; `float` để dễ ghi JSON/CSV |
| 69–76 | Guard `num_classes > 0`; `np.zeros((C,C), int64)` | Ma trận **đếm**, không phải xác suất |
| 78–93 | `no_grad`, forward, `argmax`, `.cpu().numpy()`, `cm[t,p] += 1` | Tensor trên GPU không `.numpy()` được; kiểm tra khoảng nhãn thay vì bỏ im lặng |
| 89–93 | `cm[nhãn thật, nhãn dự đoán] += 1` | **Hàng = thật, cột = dự đoán**: tổng hàng = số mẫu thật của lớp đó |
| 108–113 | Nhận cả list, kiểm tra shape vs `len(class_names)` | Lệch shape thì chỉ số lớp sẽ sai mà không báo |
| 115–124 | Bỏ `i != j`, chỉ lấy `count > 0`, sort giảm dần, cắt `k` | Đường chéo là đoán đúng; cặp 0 lần không đáng liệt kê |

**Ba câu chốt:** (1) loss + accuracy trong `no_grad`, cùng công thức với train; (2) confusion matrix đếm theo cặp (thật, dự đoán); (3) top_confusions bỏ đường chéo, sắp giảm dần.

**Chi phí:** `confusion_matrix` lặp từng ảnh bằng Python — 5.000–10.000 ảnh vẫn ổn; nếu dữ liệu lớn hơn thì vector hoá (`np.add.at`).

## C3 — `experiment.py` (253 dòng, 3 hàm + 1 helper)

| Dòng | Làm gì | Vì sao |
|---|---|---|
| 18–39 | `make_summary_writer`: import TensorBoard **trong hàm**, chỉ bắt `ImportError` | Module import được khi máy không cài TensorBoard; lỗi khác vẫn nổi lên |
| 42–73 | `_resolve_best_metrics`: `best_epoch` → argmin `val_loss` → argmax `val_acc` → rỗng | Báo cáo đúng epoch đã lưu; vẫn đọc được history cũ không có `best_epoch` |
| 76–86 | Chữ ký `run_comparison` + 3 tham số tiêm hàm để test | Test chạy được không cần model thật, không cần CIFAR-10 |
| 104–117 | Nếu không truyền thì import `build_model` / `unfreeze_last_blocks` của Đại | Giữ hợp đồng module, nhưng cho phép thay bằng mock |
| 119–125 | `train_model_fn` import trong `try` rồi `except` import lại y hệt | Dư thừa (code smell), không sai |
| 127 | `train_loader, val_loader, _ = loaders` | **Cố tình bỏ test loader**: so sánh chỉ trên validation |
| 132–133 | Hai vòng lặp model × lưới | Mỗi cặp = 1 cấu hình = 1 run |
| 135–146 | Bóc siêu tham số có mặc định; `run_name`; writer riêng | Mỗi cấu hình một thư mục log, không ghi đè nhau |
| 151–160 | Build model → (nếu có) unfreeze | **Unfreeze trước khi tạo optimizer** (optimizer tạo bên trong `train_model` từ `requires_grad`) |
| 162–180 | `perf_counter`, `try/finally` đóng writer | Đo thời gian train; đóng file log kể cả khi train lỗi |
| 183–195 | `_resolve_best_metrics` + dict kết quả | `best_val_acc`, `best_val_loss`, `best_epoch`, `train_seconds`, `checkpoint`, `tensorboard_logged` |
| 200–253 | `save_results`: gom hợp các khoá `hparams`, cột `hp_*`, format 4 chữ số | Nhiều cấu hình có khoá khác nhau vẫn ra cùng một bảng; thứ tự cột cố định |

**Ba câu chốt:** (1) mỗi cấu hình một run TensorBoard, đo thời gian train; (2) báo `best_val_acc` của đúng epoch đã lưu; (3) CSV phẳng hoá siêu tham số thành cột `hp_*`.

**Bốn bẫy phải nhớ:** `batch_size` trong lưới **bị bỏ qua** (loader tạo ngoài); seed chỉ đặt một lần; `unfreeze_fn` import lỗi thì **im lặng thành `None`** → không mở khóa gì; `train_seconds` **không tính** thời gian build model/tải trọng số.
