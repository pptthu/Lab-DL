# AI usage

Fill one section per lab. Be specific and honest: this file is part of the submission.

| Field | Lab 1 | Lab 2 | Lab 3 |
|---|---|---|---|
| Tools used (name, version) | Part A (Toàn, 2026-10-13): Claude Code (Claude Opus 5.5)<br>Part B (Thư, 2026-10-15): Antigravity IDE (Gemini 2.5 Flash / Pro) | | |
| Main purpose (planning / code draft / debugging / docs) | Part A: code draft, tests, smoke runs, progress notes<br>Part B: code draft, unit tests (fake data), training loop implementation, smoke & normal runs | | |
| Student contribution (what we designed, wrote, decided) | Part A: interfaces come from `project_description.md`; rule review before coding (data source, `save_split` signature)<br>Part B: verified optimizer choices (adam vs sgd), checked sample-weighted loss accumulation, verified checkpoint saving logic | | |
| Important AI-generated components (module / function) | Part A: `eda.run_eda`; `preprocessing` (4 functions); `features` (3 functions + `FashionDataset`); `model.MLP`, `count_parameters`; `tests/test_{eda,preprocessing,features,model}.py`<br>Part B: `train.train_model`, `train.save_checkpoint`, `train.load_checkpoint`, `tests/test_train.py` | | |
| Prompt purposes (one line per important prompt) | Part A: implement issue #2 (modules 1-4) following the repo rules, one module per branch, run and test each step<br>Part B: implement Step 1 (`train` module), run unit tests with fake data, execute smoke run and 5-epoch baseline run | | |
| Student modification (what we changed after AI output, and why) | Reviewed code structure, confirmed metric calculations and device placement | | |
| Validation method (smoke run, tests, manual inspection) | Part A: `pytest labs/lab1/tests tests` (34 passed); real runs on the Kaggle CSV<br>Part B: `pytest labs/lab1/tests/test_train.py` (4 passed, 34 total); real smoke run (1 batch) and normal run (5 epochs: val_acc 0.8905) | | |

---

**Tiếng Việt**

Mỗi lab điền một cột. Ghi cụ thể và trung thực vì file này là một phần bài nộp. Các dòng: công cụ dùng, mục đích chính, phần sinh viên đóng góp, thành phần do AI sinh ra quan trọng, mục đích các prompt, phần sinh viên chỉnh sửa, cách kiểm chứng.
