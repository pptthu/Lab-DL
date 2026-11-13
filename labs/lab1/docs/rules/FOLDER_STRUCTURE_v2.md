# Cấu trúc thư mục Lab 1 (v2)

- Tác giả: Thư | Tạo: 2026-11-14T00:29:40+07:00 | Cập nhật: Thư, 2026-11-14T01:44:12+07:00
- Trạng thái: đã chuyển sang cây này ngày 2026-11-14 (O5 — đổi thư mục `artifacts/`, `raw_data/` sang `data/`, `outputs/`), bằng `git mv`, kèm sửa `.gitignore`. `artifacts/`, `raw_data/` của Lab 1 đã bỏ; Lab 2, 3 vẫn dùng `artifacts/` (`shared/paths.py` giữ `artifact_dir`).

## 1. Cây thư mục `labs/lab1/`

    labs/lab1/
    ├── eda/ preprocessing/ features/ model/ train/ evaluate/ experiment/ visualize/   # 8 module, mỗi folder 1 script
    ├── data/
    │   ├── raw/        # fashion-mnist_train.csv, fashion-mnist_test.csv (không commit)
    │   └── splits/     # train_val_split.json (54000 train / 6000 val)
    ├── outputs/
    │   ├── models/     # checkpoint .pt
    │   ├── logs/       # .log, _history.csv, _config.json
    │   ├── figures/    # loss curve, accuracy curve, confusion matrix
    │   ├── results/    # CSV đánh giá, bảng so sánh
    │   ├── demo/       # ảnh dự đoán của best checkpoint trên test
    │   └── _smoke/     # mọi đầu ra của smoke run, không dùng để báo cáo
    ├── docs/           # xem §2
    ├── notebook/  tests/
    └── main.py  README.md  requirements.txt  progress_status.md  project_description.md

Đã bỏ khỏi gốc lab: `artifacts/`, `raw_data/`. `extended_study.py` đã xoá khi sửa `experiment` (O10, 2026-11-14). `logbook/` đã chuyển vào `docs/experiments/` (PR #65).

## 2. Cây thư mục `labs/lab1/docs/`

    docs/
    ├── 00_README_v2.md          # mục lục, đọc đầu tiên
    ├── lab1_description_v1.md   # desc hiện hành của Lab 1
    ├── rules/                   # luật luôn bật
    ├── templates/               # khuôn để copy
    ├── plans/                   # detail plan có version
    ├── phases/                  # desc từng module: <module>_v1.md (9 file)
    ├── models/                  # model card đã điền
    ├── experiments/             # sổ ghi thí nghiệm experiment_log.md (thay logbook/)
    └── bugs/                    # ghi chép lỗi (thay logbook/)

`phases/` có 9 file module (chuyển từ gốc `docs/` bằng `git mv`, giữ lịch sử); `experiments/` có `experiment_log.md`. `models/`, `bugs/` chỉ tạo khi có file đầu tiên.

## 3. Loại kết quả → đường dẫn

| Loại | Đường dẫn đích | Code hiện tại (sau O5) |
|---|---|---|
| Dữ liệu gốc | `data/raw/` | `data/raw/` |
| Split train/val | `data/splits/train_val_split.json` | `data/splits/train_val_split.json` |
| Checkpoint | `outputs/models/<ts>_<run>_best.pt`, `_last.pt` | như đích (2026-11-14) |
| Log mỗi lần chạy | `outputs/logs/<ts>_<run>.log`, `_history.csv`, `_config.json` | như đích; thêm `outputs/results/epoch_history.csv` gộp history của 5 run |
| Biểu đồ | `outputs/figures/` | `outputs/figures/` |
| Bảng đánh giá, so sánh | `outputs/results/` | `outputs/results/` |
| Ảnh demo | `outputs/demo/` | `outputs/demo/<ts>_predictions.png`, `.csv` |
| Đầu ra smoke run (log, checkpoint, hình) | `outputs/_smoke/` | `outputs/_smoke/{models,logs,figures,results,demo}/` |
| Model card (có sơ đồ kiến trúc mermaid) | `docs/models/<tên>_v<N>.md` | `mlp_baseline_v1.md`, `mlp_larger_network_v1.md` |
| Ghi chép thí nghiệm, lỗi | `docs/experiments/`, `docs/bugs/` | `docs/experiments/experiment_log.md` (đã chuyển); `docs/bugs/` chưa có |

`<ts>` = `YYYYMMDD_HHMMSS` lúc bắt đầu chạy.

## 4. Trả lời 3 nhận xét của cô

- **"Lưu model trong artifacts là sai":** trọng số vào `outputs/models/`. Bỏ `artifacts/` vì tên quá chung, không cho biết bên trong là gì.
- **"Thư mục lab phân rã quá":** gom các thư mục lẻ ở gốc lab còn 3 nhóm: `data/` (vào), `outputs/` (ra), `docs/` (tài liệu). Giữ 8 folder module vì 3 team làm song song, mỗi người chỉ sửa folder của mình.
- **"Docs sai, đạt khoảng 70%":** docs chia theo loại (luật, khuôn, plan, phase, model card). Desc do người viết theo khuôn khoảng 20 dòng; md từng module đã viết lại vào `phases/` theo khuôn phase của M.
