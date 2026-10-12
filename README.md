# Deep Learning Labs (Lab-DL)

Group coursework for the Deep Learning course: three labs, each done by a pair of members.

| Lab | Topic | Folder |
|---|---|---|
| 1 | Fashion-MNIST classification (MLP, PyTorch) | `labs/lab1/` |
| 2 | Pre-trained CNNs, transfer learning and fine-tuning (CIFAR-10 assumed) | `labs/lab2/` |
| 3 | Hugging Face: sentiment analysis and fine-tuning (IMDB assumed) | `labs/lab3/` |

## Repository structure
```text
Lab-DL/
├── README.md, AI_usage.md, CONTRIBUTING.md, AGENTS.md
├── requirements.txt, pytest.ini, .gitignore
├── .github/                  # issue + pull request templates
├── docs/
│   ├── workflow.md           # the lecturer's vibe-coding workflow
│   ├── templates/            # bug report, experiment note
│   └── agent/dl-lab-workflow/  # SKILL.md + templates for the AI agent
├── shared/                   # paths.py, seed.py (repo-relative paths, seeds)
├── tests/                    # tests of shared helpers
└── labs/
│   ├── lab1/  (eda, preprocessing, features, model, train, evaluate, visualize, experiment)
│   ├── lab2/  (eda, preprocessing, features, model, train, evaluate, experiment)
│   ├── lab3/  (sentiment_demo, eda, preprocessing, features, model, train, evaluate)
```
Inside each lab, every module is one sub-folder with one script (`eda/eda.py`, `model/model.py`, ...).
Each lab keeps its own `project_description.md` and `progress_status.md`.

## Setup
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r labs/<lab_folder>/requirements.txt
pytest                      # contract + smoke tests of every lab
```
Datasets are not stored in git. Each lab README explains where to put them.

## Working method
Define -> Design -> Break down -> Build MVP -> Implement -> Test -> Refactor -> Commit -> Document (see `docs/workflow.md`).
Work is assigned as GitHub Issues in the project board. Git rules are in `CONTRIBUTING.md`; AI use is logged in `AI_usage.md`.

---

**Tiếng Việt**

## Tổng quan
Bài tập nhóm môn Deep Learning: ba lab, mỗi lab do một cặp thành viên thực hiện. Bảng lab và cấu trúc thư mục ở phần tiếng Anh phía trên.

- Trong mỗi lab, mỗi module là một sub-folder chứa một script. Mỗi lab có `project_description.md` và `progress_status.md` riêng.
- Dataset không đưa lên git; README của từng lab hướng dẫn nơi đặt dữ liệu.
- Quy trình: Define -> Design -> Break down -> Build MVP -> Implement -> Test -> Refactor -> Commit -> Document (xem `docs/workflow.md`).
- Việc được giao bằng GitHub Issue trong project board. Quy tắc Git nằm trong `CONTRIBUTING.md`; việc dùng AI được ghi trong `AI_usage.md`.
