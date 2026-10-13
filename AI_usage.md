# AI usage

Fill one section per lab. Be specific and honest: this file is part of the submission.

| Field | Lab 1 | Lab 2 | Lab 3 |
|---|---|---|---|
| Tools used (name, version) | Part A (Toàn, 2026-10-13): Claude Code (Claude Opus 5.5) | | |
| Main purpose (planning / code draft / debugging / docs) | Part A: code draft, tests, smoke runs, progress notes | | |
| Student contribution (what we designed, wrote, decided) | Part A: interfaces come from `project_description.md`; rule review before coding (data source, `save_split` signature); _student to complete after reviewing the code_ | | |
| Important AI-generated components (module / function) | Part A: `eda.run_eda`; `preprocessing` (4 functions); `features` (3 functions + `FashionDataset`); `model.MLP`, `count_parameters`; `tests/test_{eda,preprocessing,features,model}.py` | | |
| Prompt purposes (one line per important prompt) | Part A: implement issue #2 (modules 1-4) following the repo rules, one module per branch, run and test each step | | |
| Student modification (what we changed after AI output, and why) | Part A: _student to complete after reviewing the code_ | | |
| Validation method (smoke run, tests, manual inspection) | Part A: `pytest labs/lab1/tests tests` (34 passed); real runs on the Kaggle CSV (see `labs/lab1/progress_status.md`, Audit) | | |

---

**Tiếng Việt**

Mỗi lab điền một cột. Ghi cụ thể và trung thực vì file này là một phần bài nộp. Các dòng: công cụ dùng, mục đích chính, phần sinh viên đóng góp, thành phần do AI sinh ra quan trọng, mục đích các prompt, phần sinh viên chỉnh sửa, cách kiểm chứng.
