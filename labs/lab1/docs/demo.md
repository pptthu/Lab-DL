# Lab 1 / demo: how to run and present the lab

## For AI agents (read first)

| Field | Value |
|---|---|
| Scope | `labs/lab1/main.py`, `labs/lab1/extended_study.py`, `labs/lab1/notebook/lab1_presentation.ipynb` |
| Owner | team leader (Thư), assigned by the team leader |
| Status | Run end to end on the official data, 2026-11-04 |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `main.py`, `extended_study.py`, the notebook, the README, this guide (team leader's files) |
| Files you must not edit | the module files (owners in each module guide), raw data, committed split file |
| Prohibited | starting `main.py` (35 runs) or `extended_study.py` (10 runs x 20 epochs) without the team leader; adding a CNN or other models (team decision: MLP only); writing numbers into the README or notebook text that do not come from a recorded run |
| Checks | `python -m pytest labs/lab1/tests tests -q`; real runs only when the team agrees |
| Done when | the README, the notebook text and `progress_status.md` show the numbers of the same recorded run |

## Current (what runs today)

| Step | Command (repository root) | Time on the team laptop (CPU, 4 threads) |
|---|---|---|
| Install | `pip install -r labs/lab1/requirements.txt` | |
| Data | `python -m labs.lab1.preprocessing.download_official` (set `PYTHONIOENCODING=utf-8` on Windows with a Vietnamese path) | a few minutes (downloads about 30 MB) |
| Tests | `python -m pytest labs/lab1/tests tests -q` | under 1 minute |
| Full pipeline | `python -m labs.lab1.main` | 35 runs of 5 epochs; most runs take 1 to 2 minutes; the 2026-11-04 run took 2 h 3 min because one run stalled (cause not found) |
| 20-epoch study | `python -m labs.lab1.extended_study` (after `main.py`) | 24 minutes on 2026-11-04 |
| Notebook | Restart & Run All `labs/lab1/notebook/lab1_presentation.ipynb` (reads the results of the two scripts, retrains only the seed-42 baseline) | about 1 minute |

`main.py` stages: EDA -> load split -> train-only normalization and loaders -> MLP -> train (best checkpoint by validation loss) -> validation and one test evaluation -> figures -> 7 configurations x 5 seeds -> per-seed analysis.

Results of the recorded run (official data, 2026-11-04, Run):

- Saved model (seed 42): validation 89.05%, test 88.36%; notebook gives the same numbers.
- Baseline across 5 seeds: train 90.92%, validation 88.62 +/- 0.36%, test 88.00 +/- 0.39%.
- Clear effects (validation): no hidden layer -3.80, SGD vs Adam at lr 1e-3 -3.45, SGD lr 0.01 vs 1e-3 +3.14 points.
- 20 epochs: baseline overfits (val loss 0.2988 at best, 0.3972 at the end); test 88.62 +/- 0.19%; dropout keeps the final val loss lower (0.3095).

## Intended (what the brief needs)

- Deliverables (team leader, 2026-11-04): the full source code of the three labs and one written report (PDF, written later). Lab 1 is complete when `main.py`, the notebook and the README agree on one recorded run, which is the case for the 2026-11-04 run.

## What to show in the oral exam

- Data: why the official Zalando files and not the Kaggle CSVs; stratified split with seed 42; normalization from the training split only.
- Model: MLP 784-256-128-10, logits and CrossEntropyLoss, 235,146 parameters.
- Training: Adam, 5 epochs, checkpoint with the lowest validation loss; loss curves.
- Evaluation: validation for every choice, test once; confusion matrix and the Shirt error analysis.
- Experiments: one factor per pair, 5 seeds, rule fixed before reading the results, Holm correction; what "not significant" does and does not mean.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** phần này của nhóm trưởng (Thư): `main.py`, `extended_study.py`, notebook, README. Cấm sửa file module của người khác, data, file split đã commit. Cấm tự chạy `main.py` (35 lần) hay `extended_study.py` khi chưa hỏi nhóm trưởng, cấm thêm CNN (nhóm đã chốt chỉ làm MLP), cấm ghi số không lấy từ một lần chạy đã ghi lại.
- **Hiện tại:** tải data bằng `download_official` (vài phút, khoảng 30 MB), test dưới 1 phút, `main.py` gồm 35 lần chạy 5 epoch (lần 04/11 mất 2 giờ 3 phút vì có một lần chạy bị treo, chưa rõ nguyên nhân), `extended_study.py` 24 phút, notebook khoảng 1 phút.
- **Kết quả đã chạy (04/11/2026, data chính thức):** model seed 42: val 89,05%, test 88,36%, notebook ra cùng số; baseline 5 seed: test 88,00 ± 0,39%; tác động rõ: bỏ lớp ẩn −3,80, SGD so với Adam −3,45, SGD lr 0,01 +3,14 điểm; 20 epoch: baseline overfit, dropout giữ val loss cuối thấp hơn.
- **Bài nộp:** toàn bộ source của 3 lab và một bài báo cáo (PDF, viết sau).
- **Khi vấn đáp:** vì sao dùng data Zalando; chia phân tầng seed 42; chuẩn hóa chỉ trên train; MLP và logits; chọn checkpoint theo val loss; test chỉ dùng một lần; phân tích lỗi Shirt; mỗi cặp đổi một yếu tố, 5 seed, quy tắc chốt trước, hiệu chỉnh Holm.
