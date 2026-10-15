# AI usage

Fill one section per lab. Be specific and honest: this file is part of the submission.

| Field | Lab 1 | Lab 2 | Lab 3 |
|---|---|---|---|
| Tools used (name, version) | Part A (Toàn, 2026-10-13): Claude Code (Claude Opus 5.5)<br>Part B & Integration (Thư, 2026-10-15): Antigravity IDE (Gemini 2.5 Flash / Pro) | | |
| Main purpose (planning / code draft / debugging / docs) | Part A: code draft, tests, smoke runs, progress notes<br>Part B & Integration: code draft, unit tests (fake data), training, evaluation, visualization, experiments, pipeline integration (`main.py`), report (`README.md`), and presentation notebook | | |
| Student contribution (what we designed, wrote, decided) | Part A: interfaces come from `project_description.md`; rule review before coding (data source, `save_split` signature)<br>Part B & Integration: designed 4 ablation experiment configurations (varying optimizer, hidden sizes, regularization), verified that test split was strictly untouched during experiment selection, reviewed loss weighting logic, verified all 42 automated tests, and conducted manual verification of all generated figures | | |
| Important AI-generated components (module / function) | Part A: `eda.run_eda`; `preprocessing` (4 functions); `features` (3 functions + `FashionDataset`); `model.MLP`, `count_parameters`; `tests/test_{eda,preprocessing,features,model}.py`<br>Part B & Integration: `train.train_model`, `train.save_checkpoint`, `train.load_checkpoint`, `tests/test_train.py`, `evaluate.evaluate`, `evaluate.confusion_matrix`, `evaluate.per_class_accuracy`, `tests/test_evaluate.py`, `visualize.plot_loss_curves`, `visualize.show_predictions`, `visualize.plot_confusion_matrix`, `tests/test_visualize.py`, `experiment.run_experiments`, `experiment.save_results`, `tests/test_experiment.py`, `main.main`, `lab1_presentation.ipynb` | | |
| Prompt purposes (one line per important prompt) | Part A: implement issue #2 (modules 1-4) following the repo rules, one module per branch, run and test each step<br>Part B & Integration: implement Step 1 (train), Step 2 (evaluate), Step 3 (visualize), Step 4 (experiment), and Step 5 (integration, README technical report, presentation notebook) | | |
| Student modification (what we changed after AI output, and why) | Confirmed metric calculations, verified parameter counting, ensured test set single-evaluation contract, validated notebook cells run cleanly without tqdm flood | | |
| Validation method (smoke run, tests, manual inspection) | Part A: `pytest labs/lab1/tests tests` (34 passed); real runs on the Kaggle CSV<br>Part B & Integration: `pytest labs/lab1/tests` (42 passed); real 4-configuration experiment run; test set evaluated once (test_acc 88.99%); end-to-end smoke run of `main.py` | | |

---

**Tiếng Việt**

Mỗi lab điền một cột. Ghi cụ thể và trung thực vì file này là một phần bài nộp. Các dòng: công cụ dùng, mục đích chính, phần sinh viên đóng góp, thành phần do AI sinh ra quan trọng, mục đích các prompt, phần sinh viên chỉnh sửa, cách kiểm chứng.
