# Lab 1 progress status

Update this file after EVERY step.
The table below is the single source of truth for who is doing what.

| Module | Owner | Status (pending / in_progress / done) | Contract test (pass / fail / not_run) | Smoke run on real data (pass / fail / not_run) | Notes |
|---|---|---|---|---|---|
| `eda` | Member A (Toàn) | done | pass (3 tests) | pass | 60000 rows, 10 classes balanced, 43 duplicates kept, 0 missing |
| `preprocessing` | Member A (Toàn) | done | pass (6 tests) | pass | Stratified 90/10 split saved to artifacts/splits/train_val_split.json |
| `features` | Member A (Toàn) | done | pass (6 tests) | pass | train mean 0.2860, std 0.3527; loaders return [64,1,28,28] float32 |
| `model` | Member A (Toàn) | done | pass (6 tests) | pass | MLP 784->256->128->10: 235,146 trainable parameters |
| `train` | Member B (Thư) | done | pass (fake data, 4 tests) | pass (VAL 6,000: loss 0.3194, acc 88.77%) | 5 epochs, Adam lr=1e-3, best checkpoint saved to artifacts/models/mlp_baseline.pt |
| `evaluate` | Member B (Thư) | done | pass (fake data, 4 tests) | pass (VAL 6,000: loss 0.3194, acc 88.77%) | Trouser highest 98.5%, Shirt lowest 71.0%, CM sum 6000, ROC-AUC 0.9913 |
| `visualize` | Member B (Thư) | done | pass (fake data, 3 tests) | pass (saved 3 figures in artifacts/figures/) | loss_curves.png, predictions_grid.png, confusion_matrix.png checked visually |
| `experiment` | Member B (Thư) | done | pass (fake data, 1 test with 2 configs) | pass (4 configs, saved to artifacts/runs/experiment_results.csv) | Baseline best (val 89.05%, test 88.32%), larger network 567k params slower (0.43ms) |
| `main.py`, README, integration | Team leader (Thư) | done | pass (smoke run 1 epoch) | pass (full pipeline end-to-end) | README technical report (<200 words EN+VI), lab1_presentation.ipynb (25 cells), all 42 tests passing |

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
- `pytest labs/lab1/tests tests` -> pass, 42 passed (Python 3.13.7, torch 2.x)
- `run_eda(raw_data/fashion-mnist_train.csv, artifacts/figures)` -> pass: n_rows 60000, 6000 per class, pixel 0-255, n_missing 0, n_duplicates 43; wrote `class_counts.png`, `sample_grid.png`
- `python -m labs.lab1.preprocessing.preprocessing` -> pass: images (60000, 28, 28) uint8 0-255, labels int64 0-9, train 54000 / val 6000, wrote `artifacts/splits/train_val_split.json`
- load_csv on `fashion-mnist_test.csv` -> pass: (10000, 28, 28) uint8, 1000 per class
- loaders on real data -> pass: train 54000 / val 6000 / test 10000; batch x [64,1,28,28] float32, y int64 in 0-9
- `MLP()` forward on a real train batch -> pass: logits [64,10], untrained CrossEntropyLoss 2.2931 (ln 10 = 2.3026)
- `pytest labs/lab1/tests/test_train.py` -> pass: 4 passed
- `train_model` baseline run (5 epochs, Adam lr=1e-3, 54k train / 6k val) -> pass in ~21s on CPU:
  - Epoch 1: train_loss=0.4785, val_loss=0.3744, val_acc=0.8710
  - Epoch 2: train_loss=0.3532, val_loss=0.3481, val_acc=0.8772
  - Epoch 3: train_loss=0.3151, val_loss=0.3266, val_acc=0.8887
  - Epoch 4: train_loss=0.2898, val_loss=0.3194, val_acc=0.8877
  - Epoch 5: train_loss=0.2705, val_loss=0.3310, val_acc=0.8860
  - Best model checkpoint saved to `artifacts/models/mlp_baseline.pt`
- `pytest labs/lab1/tests/test_evaluate.py` -> pass: 4 passed
- `evaluate` on VAL (6,000 samples) with loaded `mlp_baseline.pt` -> pass:
  - Validation Loss: 0.3194, Validation Accuracy: 88.77% (5326/6000)
  - Confusion Matrix: shape (10, 10), sum 6000
  - Classification report: Trouser highest (98.50% recall, 98.75% F1), Shirt lowest (71.00% recall, 71.84% F1)
  - Macro ROC-AUC: 0.9913, Macro PR-AUC: 0.9428
- `pytest labs/lab1/tests/test_visualize.py` -> pass: 3 passed
- `visualize` real run -> pass: saved 3 figures to `artifacts/figures/`:
  - `loss_curves.png`: train vs val loss over 5 epochs
  - `predictions_grid.png`: 4x4 image grid with pred/true titles, wrong predictions in red
  - `confusion_matrix.png`: 10x10 annotated heatmap with class names
- `pytest labs/lab1/tests/test_experiment.py` -> pass: 1 passed
- `run_experiments` real run (4 configs, 5 epochs each) -> pass:
  - Baseline (256-128, Adam 1e-3, drop 0.0): val_acc 89.05%, 235k params, 0.18ms latency
  - SGD (256-128, SGD 0.01, mom 0.9): val_acc 88.57%, 235k params, 0.27ms latency
  - Larger network (512-256-128, Adam 1e-3): val_acc 88.10%, 567k params, 0.43ms latency
  - Dropout reg (256-128, Adam 1e-3, drop 0.2, wd 1e-4): val_acc 88.47%, 235k params, 0.22ms latency
  - Results saved to `artifacts/runs/experiment_results.csv`
  - Best config evaluated ONCE on TEST set (10,000 samples): Test Loss: 0.3212, Test Accuracy: 88.32% (8832/10000)
- `main.py` pipeline execution -> pass: all stages ran end-to-end seamlessly.
- `lab1_presentation.ipynb` -> 25 cells executed end-to-end with pre-rendered visuals, error analysis, and bilingual explanations.

## Log (dated, one line per step)
- <date> <name>: scaffold created; interfaces agreed -> verified with `pytest labs/lab1/tests`
- 2026-10-13 Toàn: `eda`: implemented `run_eda` + `tests/test_eda.py` -> pytest pass; real run on 60,000 rows
- 2026-10-13 Toàn: `preprocessing`: implemented `load_csv`, `make_split`, `save_split`, `load_split` + tests; generated the shared split JSON -> pytest pass; real run on train/test CSV
- 2026-10-13 Toàn: `features`: implemented `compute_mean_std`, `build_transform`, `FashionDataset`, `make_loaders` + tests -> pytest pass; real batch checked
- 2026-10-13 Toàn: `model`: implemented `MLP`, `count_parameters` + tests -> pytest pass; forward on a real batch
- 2026-10-15 Thư: `train`: implemented `train_model`, `save_checkpoint`, `load_checkpoint` + `tests/test_train.py` -> 4 tests passed; baseline run finished (saved `mlp_baseline.pt`)
- 2026-10-15 Thư: `evaluate`: implemented `evaluate`, `confusion_matrix`, `per_class_accuracy` + `tests/test_evaluate.py` -> 4 tests passed; evaluated saved checkpoint on VAL
- 2026-10-15 Thư: `visualize`: implemented `plot_loss_curves`, `show_predictions`, `plot_confusion_matrix` + `tests/test_visualize.py` -> 3 tests passed; generated & visually verified 3 figures in artifacts/figures/
- 2026-10-15 Thư: `experiment`: implemented `run_experiments`, `save_results` + `tests/test_experiment.py` -> test passed; ran 4 configs on real data; saved table to `artifacts/runs/experiment_results.csv`
- 2026-10-16 Thư: `integration`: aligned numbers to canonical run (val 88.77%, test 88.32%), refined presentation notebook with Shirt error analysis, ROC/PR curves, latency profiling, condensed README (<200 words), updated honest AI usage declaration.

## Open issues
- Resolved: `val_ratio` vs `val_fraction` parameter compatibility added to `make_split`.
- Resolved: All reported metrics aligned across notebook, README, and progress_status.
- 43 duplicate rows in the Kaggle train CSV kept intentionally to preserve index mapping.
