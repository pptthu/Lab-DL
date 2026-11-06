# Lab 1 / experiment: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab1/experiment/experiment.py` (used by `main.py` and `extended_study.py`) |
| Owner | Member B (Thư), assigned by the team leader |
| Status | done in `progress_status.md` (Run on the official data, 2026-11-04) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab1/experiment/experiment.py`, `labs/lab1/tests/test_experiment.py`, this guide |
| Files you must not edit | Member A's modules, committed split file, saved results of real runs, other owners' notebook cells |
| Agreed signatures | `run_experiments(configs, loaders, device, ckpt_prefix='exp_run')`; `save_results(results, csv_path)`; `save_history(results, csv_path)`; `analyze_seed_runs(runs, loaders, class_names, device, out_dir, top_k=10, prefix='', use_test=True)`; `build_model(cfg, num_classes=10)`; `compare_pairs(values, pairs)` (same as `project_description.md`) |
| Depends on / used by | `model`, `train`, `evaluate`, `features` / `main.py` stage 8, `extended_study.py`, notebook Stages 9-10 |
| Prohibited | comparing configurations on the test set; changing the decision rule after seeing results; adding configurations or a larger sweep without the team leader; starting the 35 runs (about 40 minutes or more on CPU) or the 20-epoch study without asking |
| Checks | Unit: `python -m pytest labs/lab1/tests/test_experiment.py -q` (fake data). Real run: `main.py` / `extended_study.py`, only when the team agrees |
| Done when | unit tests pass; real run recorded in `progress_status.md` and the README; this guide updated |

## What the brief requires

- Brief requirement: try different hyperparameters and compare.
- Team choices: 7 configurations x 5 seeds (42, 7, 123, 2024, 3407), each pair changes one factor; comparison on validation only; rule fixed before reading the results: a clear difference needs all 5 seeds in the same direction **and** |mean difference| >= 2 std; paired t-test with Holm correction and a 95% CI as a second check. Optional 20-epoch study (`extended_study.py`, baseline and dropout 0.2).

## How the module works (current code)

- `run_experiments`: trains each config with its seed; reports `best_val_acc` and `best_val_loss` from the epoch with the lowest validation loss (the saved checkpoint), plus `final_val_loss`, `n_params`, `train_seconds`, and the per-epoch `history`.
- `save_history`: one row per run and epoch (`name`, `seed`, `epoch`, `train_loss`, `val_loss`, `val_acc`); `main.py` writes `epoch_history.csv`, `extended_study.py` writes `extended_epoch_history.csv`.
- `analyze_seed_runs`: for one configuration across seeds, writes `seed_results.csv` (train/val/test per seed if `use_test`), `recall_by_class.csv`, `top_confusions.csv`.
- `compare_pairs`: paired differences, CI, `seeds_higher`, `rule_clear`, `p_value`, `p_holm`.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_experiment.py -q` | all pass |
| Real integration run | `python -m labs.lab1.main` | 35 runs; baseline test 88.00 +/- 0.39% (5 seeds); clear effects: no hidden layer -3.80, SGD vs Adam at lr 1e-3 -3.45, SGD lr 0.01 vs 1e-3 +3.14 points; not clear: larger network -0.52, dropout -0.46, weight decay -0.29 (Run, 2026-11-04, official data) |

## Open questions / limitations

- 5 seeds and one split: a non-significant result is not proof of no effect.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member B (Thư), do nhóm trưởng giao. Được sửa `experiment.py`, `tests/test_experiment.py`, file này. Chữ ký 6 hàm khớp `project_description.md` (thêm `save_history`: lưu loss và accuracy theo từng epoch ra CSV). Cấm so sánh cấu hình trên tập test, cấm đổi quy tắc sau khi xem kết quả, cấm thêm cấu hình hay sweep lớn khi chưa hỏi nhóm trưởng, cấm tự chạy 35 lần hoặc nghiên cứu 20 epoch.
- **Đề:** thử các siêu tham số khác nhau và so sánh. Nhóm chọn 7 cấu hình × 5 seed, mỗi cặp đổi một yếu tố, so trên validation, quy tắc chốt trước (cả 5 seed cùng chiều và |chênh lệch| >= 2 std), kiểm định t có hiệu chỉnh Holm.
- **Kiểm chứng:** baseline test 88,00 ± 0,39%; rõ: bỏ lớp ẩn −3,80, SGD so với Adam −3,45, SGD lr 0,01 +3,14 điểm; chưa rõ: mạng lớn hơn −0,52, dropout −0,46, weight decay −0,29 (đã chạy 04/11/2026 trên data chính thức).
