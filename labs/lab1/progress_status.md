# Lab 1: Fashion-MNIST classification with an MLP (PyTorch): progress status
_Last updated: 2026-10-13 by Toàn (Member A)_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `eda` | Member A (Toàn) | done | pass (synthetic CSV, 3 tests) | pass (60,000 rows) | 10 classes x 6,000; pixels 0-255; 0 missing; 43 duplicate rows (kept, EDA is read-only) |
| `preprocessing` | Member A (Toàn) | done | pass (6 tests) | pass (train 60,000 / test 10,000) | split 54,000 / 6,000, stratified (600 per class in val), seed 42 |
| `features` | Member A (Toàn) | done | pass (6 tests) | pass (real batch [64,1,28,28]) | train-only mean 0.2860, std 0.3527 (0-1 scale) |
| `model` | Member A (Toàn) | done | pass (6 tests, overfits 1 batch) | pass (real batch -> logits [64,10]) | default MLP 784-256-128-10 = 235,146 parameters |
| `train` | Member B (name TBD) | todo | - | - | |
| `evaluate` | Member B (name TBD) | todo | - | - | |
| `visualize` | Member B (name TBD) | todo | - | - | |
| `experiment` | Member B (name TBD) | todo | - | - | |
| `main.py`, README, integration | Team leader | todo | - | - | |

## Handoff A -> B (sequential, no interleaving)
Member B starts coding only after ALL of the following are true:
- [x] split JSON committed in artifacts/splits/ (train/val indices): `train_val_split.json`
- [x] make_loaders returns a batch of shape [B,1,28,28] with labels 0-9
- [x] MLP forward on a dummy batch returns logits [B,10]
- [x] all Member A modules are `done` in this file and `pytest labs/lab1/tests` passes
- [ ] Member A's PRs are merged into `main`; Member B branches from the updated `main`
- [ ] Member A posts the handoff message (what was done, exact run commands, known problems)
After the handoff Member A only answers questions and reviews PRs; Member A does not edit Member B's modules.

## Audit: what actually runs
- `pytest labs/lab1/tests tests` -> pass, 34 passed (Python 3.11.9, torch 2.14.1+cpu, 2026-10-13)
- `run_eda(raw_data/fashion-mnist_train.csv, artifacts/figures)` -> pass: n_rows 60000, 6000 per class, pixel 0-255, n_missing 0, n_duplicates 43; wrote `class_counts.png`, `sample_grid.png`
- `python -m labs.lab1.preprocessing.preprocessing` -> pass: images (60000, 28, 28) uint8 0-255, labels int64 0-9, train 54000 / val 6000, wrote `artifacts/splits/train_val_split.json`
- load_csv on `fashion-mnist_test.csv` -> pass: (10000, 28, 28) uint8, 1000 per class
- loaders on real data -> pass: train 54000 / val 6000 / test 10000; batch x [64,1,28,28] float32, y int64 in 0-9
- `MLP()` forward on a real train batch -> pass: logits [64,10], untrained CrossEntropyLoss 2.2931 (ln 10 = 2.3026)

## Log (dated, one line per step)
- <date> <name>: scaffold created; interfaces agreed -> verified with `pytest labs/lab1/tests`
- 2026-10-13 Toàn: `eda`: implemented `run_eda` + `tests/test_eda.py` -> pytest pass; real run on 60,000 rows
- 2026-10-13 Toàn: `preprocessing`: implemented `load_csv`, `make_split`, `save_split`, `load_split` + tests; generated the shared split JSON -> pytest pass; real run on train/test CSV
- 2026-10-13 Toàn: `features`: implemented `compute_mean_std`, `build_transform`, `FashionDataset`, `make_loaders` + tests -> pytest pass; real batch checked
- 2026-10-13 Toàn: `model`: implemented `MLP`, `count_parameters` + tests -> pytest pass; forward on a real batch

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
