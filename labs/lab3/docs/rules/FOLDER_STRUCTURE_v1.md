# Cấu trúc thư mục Lab 3 (v1)

- Tác giả: Thư | Tạo: 2026-11-19T16:54:00+07:00 | Cập nhật: Thư, 2026-11-19T16:54:00+07:00
- Trạng thái: cây đích của Lab 3 làm lại (plan v1, L15). Theo `labs/lab1/docs/rules/FOLDER_STRUCTURE_v2.md`; không có `artifacts/`, `raw_data/`, `logbook/`.

## 1. Cây thư mục `labs/lab3/`

    labs/lab3/
    ├── sentiment_demo/ eda/ preprocessing/ features/ model/ train/ evaluate/ experiment/ visualize/   # 9 module, mỗi folder 1 script
    ├── data/
    │   ├── raw/        # aclImdb (bản gốc Stanford, không commit)
    │   └── splits/     # train_val_split.json: ID review + ID phim của val (commit)
    ├── outputs/
    │   ├── models/     # checkpoint tốt nhất (không commit, nằm trên Kaggle)
    │   ├── logs/       # <ts>_<run>.log, _history.csv, _config.json
    │   ├── figures/    # EDA, đường huấn luyện, confusion matrix, max_length
    │   ├── results/    # bảng zero-shot / fine-tuned, 3 seed, max_length
    │   ├── demo/       # Bài 1: token + nhãn câu mẫu; ví dụ dự đoán sai
    │   └── _smoke/     # đầu ra smoke run, không dùng để báo cáo
    ├── kaggle/         # notebook chạy main.py trên Kaggle
    ├── docs/           # xem §2
    ├── notebook/  tests/
    └── main.py  README.md  requirements.txt  progress_status.md  project_description.md

## 2. Cây thư mục `labs/lab3/docs/`

    docs/
    ├── 00_README_v1.md          # mục lục, đọc đầu tiên
    ├── lab3_description_v1.md   # desc hiện hành
    ├── rules/                   # RULES_v1.md, FOLDER_STRUCTURE_v1.md
    ├── templates/               # khuôn desc, log, model card
    ├── plans/                   # detail plan có version
    ├── phases/                  # desc từng module: <module>_v1.md (9 file)
    ├── models/                  # model card
    ├── experiments/             # experiment_log.md: pilot, 3 seed, max_length
    └── bugs/                    # chỉ tạo khi có lỗi đầu tiên

## 3. Loại kết quả → đường dẫn

| Loại | Đường dẫn | Commit? |
|---|---|---|
| Dữ liệu gốc | `data/raw/aclImdb/` | Không |
| Split train/val | `data/splits/train_val_split.json` | Có |
| Checkpoint | `outputs/models/<ts>_<run>_best/` | Không |
| Log mỗi lần chạy | `outputs/logs/<ts>_<run>.log`, `_history.csv`, `_config.json` | Có |
| Hình, bảng kết quả, demo | `outputs/{figures,results,demo}/` | Có |
| Smoke run | `outputs/_smoke/` | Không |

`<ts>` = `YYYYMMDD_HHMMSS` lúc bắt đầu chạy.

Lý do commit log và kết quả (Thư, 2026-11-19): normal run chạy trên Kaggle GPU, máy khác không chạy lại được; không commit thì cô và nhóm không xem được bằng chứng, notebook không đọc được kết quả. Khác Lab 1 (`.gitignore`: `labs/lab1/outputs/*/*`).

## 4. Đường dẫn trong code
Dùng `shared/paths.py`: `data_dir("lab3", ...)`, `output_dir("lab3", ...)` (đã có từ Lab 1). `.gitignore` có khối riêng cho Lab 3: bỏ qua `data/raw/`, `outputs/models/`, `outputs/_smoke/`; cho phép file `.csv` trong `data/splits/`, `outputs/logs/`, `outputs/results/`, `outputs/demo/` (vì repo bỏ qua `*.csv` ở mọi nơi).
