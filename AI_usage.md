# AI usage

Fill one section per lab. Be specific and honest: this file is part of the submission.

| Field | Lab 1 | Lab 2 | Lab 3 |
|---|---|---|---|
| Tools used (name, version) | Part A (Toàn, 2026-10-13): Claude Code (Claude Opus 5.5)<br>Part B (Thư, 2026-10-15): Antigravity IDE (Gemini 2.5 Flash / Pro) | | |
| Main purpose (planning / code draft / debugging / docs) | Part A: code draft, tests, smoke runs, progress notes<br>Part B: code draft, unit tests (fake data), training, evaluation, visualization, and experiment modules implementation, metric validation | | |
| Student contribution (what we designed, wrote, decided) | Part A: interfaces come from `project_description.md`; rule review before coding (data source, `save_split` signature)<br>Part B: designed 4 ablation experiment configurations (varying optimizer, hidden sizes, regularization), verified that test split was strictly untouched during experiment selection, visual inspection of figures | | |
| Important AI-generated components (module / function) | Part A: `eda.run_eda`; `preprocessing` (4 functions); `features` (3 functions + `FashionDataset`); `model.MLP`, `count_parameters`; `tests/test_{eda,preprocessing,features,model}.py`<br>Part B: `train.train_model`, `train.save_checkpoint`, `train.load_checkpoint`, `tests/test_train.py`, `evaluate.evaluate`, `evaluate.confusion_matrix`, `evaluate.per_class_accuracy`, `tests/test_evaluate.py`, `visualize.plot_loss_curves`, `visualize.show_predictions`, `visualize.plot_confusion_matrix`, `tests/test_visualize.py`, `experiment.run_experiments`, `experiment.save_results`, `tests/test_experiment.py` | | |
| Prompt purposes (one line per important prompt) | Part A: implement issue #2 (modules 1-4) following the repo rules, one module per branch, run and test each step<br>Part B: implement Step 1 (train), Step 2 (evaluate), Step 3 (visualize), and Step 4 (experiment), run unit tests, execute ablation experiments, evaluate best model once on test | | |
| Student modification (what we changed after AI output, and why) | Reviewed code structure, confirmed ablation plan, verified parameter counting and test evaluation logic | | |
| Validation method (smoke run, tests, manual inspection) | Part A: `pytest labs/lab1/tests tests` (34 passed); real runs on the Kaggle CSV<br>Part B: `pytest labs/lab1/tests` (42 passed); real 4-configuration experiment run; test set evaluated once (test_acc 88.99%) | | |

---

**Tiếng Việt**

Mỗi lab điền một cột. Ghi cụ thể và trung thực vì file này là một phần bài nộp. Các dòng: công cụ dùng, mục đích chính, phần sinh viên đóng góp, thành phần do AI sinh ra quan trọng, mục đích các prompt, phần sinh viên chỉnh sửa, cách kiểm chứng.
