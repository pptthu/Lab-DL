# Lab 1: Fashion-MNIST classification with an MLP (PyTorch): progress status
_Last updated: 2026-10-15 by Thư (Member B)_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `eda` | Member A (Toàn) | done | pass (synthetic CSV, 3 tests) | pass (60,000 rows) | 10 classes x 6,000; pixels 0-255; 0 missing; 43 duplicate rows (kept, EDA is read-only) |
| `preprocessing` | Member A (Toàn) | done | pass (6 tests) | pass (train 60,000 / test 10,000) | split 54,000 / 6,000, stratified (600 per class in val), seed 42 |
| `features` | Member A (Toàn) | done | pass (6 tests) | pass (real batch [64,1,28,28]) | train-only mean 0.2860, std 0.3527 (0-1 scale) |
| `model` | Member A (Toàn) | done | pass (6 tests, overfits 1 batch) | pass (real batch -> logits [64,10]) | default MLP 784-256-128-10 = 235,146 parameters |
| `train` | Member B (Thư) | done | pass (fake data, 4 tests; smoke 1 batch 1 epoch) | pass (5 epochs, val_loss 0.3031, val_acc 0.8905) | Adam lr=1e-3, CrossEntropyLoss, best ckpt saved to artifacts/models/mlp_baseline.pt |
| `evaluate` | Member B (Thư) | done | pass (fake data, 4 tests) | pass (VAL 6,000: loss 0.3031, acc 89.05%) | Trouser highest 98.0%, Shirt lowest 68.33%, CM sum 6000 |
| `visualize` | Member B (Thư) | done | pass (fake data, 3 tests) | pass (saved 3 figures in artifacts/figures/) | loss_curves.png, predictions_grid.png, confusion_matrix.png checked visually |
| `experiment` | Member B (Thư) | todo | - | - | |
| `main.py`, README, integration | Team leader | todo | - | - | |

## Handoff A -> B (sequential, no interleaving)
Member B starts coding only after ALL of the following are true:
- [x] split JSON committed in artifacts/splits/ (train/val indices): `train_val_split.json`
- [x] make_loaders returns a batch of shape [B,1,28,28] with labels 0-9
- [x] MLP forward on a dummy batch returns logits [B,10]
- [x] all Member A modules are `done` in this file and `pytest labs/lab1/tests` passes
- [x] Member A's PRs are merged into `main`; Member B branches from the updated `main`
- [x] Member A posts the handoff message (what was done, exact run commands, known problems)
After the handoff Member A only answers questions and reviews PRs; Member A does not edit Member B's modules.

## Audit: what actually runs
- `pytest labs/lab1/tests tests` -> pass, 41 passed (Python 3.13.7, torch 2.x, 2026-10-15)
- `run_eda(raw_data/fashion-mnist_train.csv, artifacts/figures)` -> pass: n_rows 60000, 6000 per class, pixel 0-255, n_missing 0, n_duplicates 43; wrote `class_counts.png`, `sample_grid.png`
- `python -m labs.lab1.preprocessing.preprocessing` -> pass: images (60000, 28, 28) uint8 0-255, labels int64 0-9, train 54000 / val 6000, wrote `artifacts/splits/train_val_split.json`
- load_csv on `fashion-mnist_test.csv` -> pass: (10000, 28, 28) uint8, 1000 per class
- loaders on real data -> pass: train 54000 / val 6000 / test 10000; batch x [64,1,28,28] float32, y int64 in 0-9
- `MLP()` forward on a real train batch -> pass: logits [64,10], untrained CrossEntropyLoss 2.2931 (ln 10 = 2.3026)
- `pytest labs/lab1/tests/test_train.py` -> pass: 4 passed in 3.64s
- `train_model` smoke run on 1 batch / 1 epoch -> pass: train_loss 2.3132, val_loss 2.2116, val_acc 0.3281 in 2.80s
- `train_model` normal run (5 epochs, Adam lr=1e-3, 54k train / 6k val) -> pass in 21.26s on CPU:
  - Epoch 1: train_loss=0.4752, val_loss=0.3826, val_acc=0.8632
  - Epoch 2: train_loss=0.3550, val_loss=0.3502, val_acc=0.8727
  - Epoch 3: train_loss=0.3178, val_loss=0.3240, val_acc=0.8823
  - Epoch 4: train_loss=0.2912, val_loss=0.3311, val_acc=0.8853
  - Epoch 5: train_loss=0.2714, val_loss=0.3031, val_acc=0.8905
  - Best model checkpoint saved to `artifacts/models/mlp_baseline.pt` (943,697 bytes)
- `pytest labs/lab1/tests/test_evaluate.py` -> pass: 4 passed in 2.14s
- `evaluate` on VAL (6,000 samples) with loaded `mlp_baseline.pt` -> pass:
  - Validation Loss: 0.3031, Validation Accuracy: 89.05% (5343/6000)
  - Confusion Matrix: shape (10, 10), sum 6000
  - Per-class accuracy: T-shirt/top 87.33%, Trouser 98.00%, Pullover 83.17%, Dress 89.00%, Coat 81.00%, Sandal 95.50%, Shirt 68.33%, Sneaker 95.67%, Bag 96.67%, Ankle boot 95.83%
- `pytest labs/lab1/tests/test_visualize.py` -> pass: 3 passed in 3.53s
- `visualize` real run -> pass: saved 3 figures to `artifacts/figures/`:
  - `loss_curves.png` (47,711 bytes): train vs val loss over 5 epochs
  - `predictions_grid.png` (64,104 bytes): 4x4 image grid with pred/true titles, wrong predictions in red
  - `confusion_matrix.png` (48,473 bytes): 10x10 annotated heatmap with class names

## Log (dated, one line per step)
- <date> <name>: scaffold created; interfaces agreed -> verified with `pytest labs/lab1/tests`
- 2026-10-13 Toàn: `eda`: implemented `run_eda` + `tests/test_eda.py` -> pytest pass; real run on 60,000 rows
- 2026-10-13 Toàn: `preprocessing`: implemented `load_csv`, `make_split`, `save_split`, `load_split` + tests; generated the shared split JSON -> pytest pass; real run on train/test CSV
- 2026-10-13 Toàn: `features`: implemented `compute_mean_std`, `build_transform`, `FashionDataset`, `make_loaders` + tests -> pytest pass; real batch checked
- 2026-10-13 Toàn: `model`: implemented `MLP`, `count_parameters` + tests -> pytest pass; forward on a real batch
- 2026-10-15 Thư: `train`: implemented `train_model`, `save_checkpoint`, `load_checkpoint` + `tests/test_train.py` -> 4 tests passed; smoke run (1 batch) and 5-epoch baseline run finished (val_acc 0.8905, saved `mlp_baseline.pt`)
- 2026-10-15 Thư: `evaluate`: implemented `evaluate`, `confusion_matrix`, `per_class_accuracy` + `tests/test_evaluate.py` -> 4 tests passed; evaluated saved checkpoint on VAL (6,000 samples: loss 0.3031, acc 89.05%)
- 2026-10-15 Thư: `visualize`: implemented `plot_loss_curves`, `show_predictions`, `plot_confusion_matrix` + `tests/test_visualize.py` -> 3 tests passed; generated & visually verified 3 figures in artifacts/figures/

## Open issues
- (issue): expected / actual / error / env / relevant code / recent changes (use docs/templates/bug_report.md)
- `save_split` signature: issue #2 says `save_split(split_dict, path)`, but `project_description.md` and the scaffold say `save_split(train_idx, val_idx, path)`. Implemented the repo contract; the leader should confirm.
- 43 duplicate rows in the Kaggle train CSV (identical label + pixels). Not removed, so indices in the split JSON match the raw CSV rows. Team decision needed if they should be dropped.
- `FashionDataset(images, labels, indices, transform)`: pass `indices=None` for the test set (uses every row).

---

**Tiếng Việt**

## Tiếng Việt (cách dùng)
- **Bàn giao A -> B (tuần tự, không xen kẽ):** Member B chỉ bắt đầu code khi Member A đã xong HẾT:
  - file JSON split đã commit trong artifacts/splits/ (chỉ số train/val)
  - make_loaders trả batch shape [B,1,28,28], nhãn 0-9
  - forward của MLP trên batch giả trả logits [B,10]
  - mọi module của A ở trạng thái `done`, `pytest` qua, PR đã merge vào `main`, A đã nhắn tin bàn giao. Sau đó A chỉ trả lời câu hỏi và duyệt PR, không sửa module của B.
- Cập nhật file này sau MỖI bước: đổi trạng thái, ghi kết quả chạy thật (không ghi "chạy được" nếu chưa chạy).
- Mỗi dòng log gồm: ngày, người làm, module, đã thay đổi gì, kiểm chứng bằng cách nào.
- Nếu một module không cần cho lab này, ghi `N/A` kèm lý do.
- Lỗi cần ghi đủ 6 mục: mong đợi, thực tế, thông báo lỗi, môi trường, code liên quan, thay đổi gần đây.
