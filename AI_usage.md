# AI usage

Fill one section per lab. Be specific and honest: this file is part of the submission.

| Field | Lab 1 | Lab 2 | Lab 3 |
|---|---|---|---|
| Tools used (name, version) | Part A (Toàn, 2026-10-13): Claude Code (Claude 3.5 Sonnet)<br>Part B & Integration (Thư, 2026-10-15): Google Antigravity IDE (Gemini 2.5 Pro / Flash) | | |
| Main purpose (planning / code draft / debugging / docs) | Part A: code draft, unit tests, smoke runs, progress notes<br>Part B & Integration: module code draft, unit tests, pipeline integration (`main.py`), report (`README.md`), and presentation notebook (`lab1_presentation.ipynb`) | | |
| Student contribution (what we designed, wrote, decided) | Part A: interfaces defined from `project_description.md`; rule review before coding.<br>Part B & Integration: Task assignment and technical direction adhering strictly to course requirements and Lecture Chapter 2 (slide 14); directed referencing presentation ideas (pipeline flowchart, ROC/PR curves, inference latency profiling) from peer M's repository (`deeplearning_LAB_submission`) and re-implementing them independently to fit our modular architecture; removed exaggerated claims, verified all 42 automated unit tests of Lab 1. | | |
| Important AI-generated components (module / function) | Part A: `eda.run_eda`, `preprocessing` (4 functions), `features` (3 functions + `FashionDataset`), `model.MLP`, `count_parameters`, unit tests.<br>Part B & Integration: `train.train_model`, `evaluate.evaluate`, `evaluate.confusion_matrix`, `evaluate.per_class_accuracy`, `visualize.plot_loss_curves`, `visualize.show_predictions`, `visualize.plot_confusion_matrix`, `experiment.run_experiments`, `experiment.save_results`, `main.main`, `lab1_presentation.ipynb`. | | |
| Prompt purposes (one line per important prompt) | Part A: implement modules 1-4 following repo contracts, one branch per module.<br>Part B: implement Step 1 (train), Step 2 (evaluate), Step 3 (visualize), Step 4 (experiment), Step 5 (integration & presentation notebook), and refine error analysis (Shirt class) and honest reporting. | | |
| Student modification (what we changed after AI output, and why) | Fixed `val_ratio` parameter call, added bilingual Vietnamese oral defense explanations, adjusted wording to acknowledge statistical margins of error, aligned numbers to a single canonical run, condensed README to <200 words. | | |
| Validation method (smoke run, tests, manual inspection) | Part A: `pytest labs/lab1/tests tests` (34 passed); verified on Kaggle CSV.<br>Part B & Integration: `pytest labs/lab1/tests` (42 passed); end-to-end execution of `main.py` and pre-rendered run of all 25 cells in `lab1_presentation.ipynb`. | | |

---

**Tiếng Việt**

Mỗi lab điền một cột. Ghi cụ thể và trung thực vì file này là một phần bài nộp. Các dòng: công cụ dùng, mục đích chính, phần sinh viên đóng góp, thành phần do AI sinh ra quan trọng, mục đích các prompt, phần sinh viên chỉnh sửa, cách kiểm chứng.
