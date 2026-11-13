# Detail plan v1 — Cập nhật tài liệu yêu cầu Lab 1

- **Tác giả:** Thư (leader) duyệt; AI soạn theo thảo luận
- **Tạo:** 2026-11-13T00:00:00+07:00 | **Cập nhật:** 2026-11-13 (v1)
- **Trạng thái:** Leader yêu cầu sửa từ v0 ngày 2026-11-13; bản đang dùng
- **Nguồn quyết định:** handoff_2026-11-10.md (C1–C13) + các mục chốt trong thảo luận 2026-11-12/13 (O3–O12)
- **Phạm vi:** chỉ tài liệu md trong `Lab-DL/labs/lab1/docs/`. Không sửa code, không đổi thư mục code/dữ liệu/kết quả của Lab-DL trong plan này.

### Thay đổi so với v0

- Nơi lưu: mọi file md mới nằm trong `labs/lab1/docs/` (repo Lab-DL), **không** lưu vào `Tài liệu/Lab 1 MD/` nữa. Lý do (leader): các file này thuộc về Lab 1.
- `labs/lab1/docs/` được chia thư mục con (§2).
- Vì các file nằm trong repo, chúng đi chung PR cuối cùng.

---

## 0. Plan này sửa 3 nhận xét của cô thế nào

| Nhận xét của cô | Cách sửa | Chốt ở |
|---|---|---|
| Lưu model trong `artifacts/` là sai | Trọng số vào `outputs/models/`; bỏ thư mục chung `artifacts/` | C3, O3 |
| Thư mục lab phân rã quá | Giữ 8 folder module; gom phần rải rác vào `data/`, `outputs/`, `docs/` (docs chia theo loại tài liệu) | C2, C4, O4, O10 |
| Phần docs sai (đạt khoảng 70%) | Template kiểu M + desc ngắn do người viết; md từng module viết lại ở giai đoạn sau | C7, C13, O8 |

## 1. Quyết định đã chốt (O3–O12)

| # | Quyết định |
|---|---|
| O3 | Kết quả để trong `labs/lab1/outputs/{models,logs,figures,results,demo}` (giống M Lab 1: `outputs/practice_1/model/`; Cookiecutter: `models/`) |
| O4 | `labs/lab1/data/raw/` (thay `raw_data/`) và `labs/lab1/data/splits/` (thay `artifacts/splits/`), như M `data/lab1/splits/` |
| O5 | Lab 1 làm một PR, dùng `git mv`; thêm hàm đường dẫn riêng cho lab1, không sửa `artifact_dir` cũ (Lab 2, 3 đang dùng). Chỉ làm khi leader cho phép, sau khi nhánh `docs/lab1-part-a-docs` đã merge |
| O6 | Phân vai người/AI: khoảng 5 dòng trong `docs/rules/RULES_v1.md` |
| O7 | Mỗi template một file, có version `_vN`. 4 file cũ trong `Tài liệu/Lab 1 MD/` (ngoài repo) giữ nguyên, không dùng làm nơi lưu nữa |
| O8 | md từng module viết theo khuôn C13. `demo` = phase "nạp best checkpoint → dự đoán ảnh test → `outputs/demo/`", code đặt trong `visualize/` |
| O9 | Chỉ Lab 1. Lab 2, 3 làm sau khi Lab 1 chạy ổn |
| O10 | MLP only; 5 cấu hình × seed 42; mỗi cấu hình tối đa 30 epoch + early stopping. Bỏ `extended_study.py`, bỏ đa seed và p-value/CI95 |
| O11 | Baseline đạt test ≥ 88% trên tập test chính thức (baseline hiện có: 88.00 ± 0.39%, `artifacts/runs/seed_results.csv`) |
| O12 | Model card tách Initial model / Trained model theo note N7 |
| Thêm | History mỗi epoch lưu CSV. Chênh train/val acc chỉ báo cáo, không làm ngưỡng đạt (chọn B). Chưa di chuyển `progress_status.md` và `project_description.md` |
| Thêm (v1) | Tài liệu md mới lưu trong `labs/lab1/docs/`, chia thư mục như §2 |

## 2. Cấu trúc `labs/lab1/docs/` và danh sách file

    labs/lab1/docs/
    ├── 00_README_v1.md            # mục lục docs + quy tắc version (đọc đầu tiên)
    ├── lab1_description_v1.md     # desc chính của Lab 1 = file md yêu cầu để AI sinh báo cáo/slide (C5)
    ├── rules/                     # luật luôn bật (như agents/rules/ của M)
    │   ├── RULES_v1.md
    │   └── FOLDER_STRUCTURE_v1.md
    ├── templates/                 # khuôn để copy
    │   ├── DESC_TEMPLATE_v1.md
    │   ├── LOG_TEMPLATE_v1.md
    │   └── MODEL_CARD_TEMPLATE_v1.md
    ├── plans/                     # detail plan có version
    │   ├── detail_plan_lab1_md_v0.md
    │   └── detail_plan_lab1_md_v1.md
    ├── phases/                    # md từng module theo khuôn C13 (giai đoạn sau, O8)
    ├── models/                    # model card đã điền (sau khi train lại)
    ├── experiments/               # thay logbook/ (C4, giai đoạn sau)
    └── bugs/                      # thay logbook/ (C4, giai đoạn sau)

**Tạo trong plan này (7 file):**

| File | Nội dung | Căn cứ |
|---|---|---|
| `rules/RULES_v1.md` | Quy trình: desc → discussion → detail plan → duyệt → lưu `_vN` → small prompt → smoke → normal → audit. AI nói lại luật trước khi lưu README. Không để kết quả trong notebook. Mỗi thay đổi chỉ sửa một file nguồn. Phân vai người/AI | cô #9, #10, C6, C7, O6, N1 |
| `rules/FOLDER_STRUCTURE_v1.md` | Cây thư mục đích của `labs/lab1/` và của `docs/`; bảng "loại kết quả → đường dẫn"; trả lời 3 nhận xét của cô | C2–C4, O3, O4, O8, cô #1 |
| `templates/DESC_TEMPLATE_v1.md` | Khuôn desc khoảng 20 dòng (§3) | C13, cô #8 |
| `templates/LOG_TEMPLATE_v1.md` | Template log (§4) + quy tắc dừng (§5) | C9, C11, N2, N6 |
| `templates/MODEL_CARD_TEMPLATE_v1.md` | Template model card (§6) | O12, N7, cô #7 |
| `lab1_description_v1.md` | Desc thật của Lab 1 theo khuôn §3 (§7) | C5, C12, N3, N5, O10, O11 |
| `00_README_v1.md` | Mục lục docs; quy tắc version: dùng bản số lớn nhất, bản cũ giữ nguyên; chỉ rõ `lab1_description_v1.md` là desc hiện hành | O7, cô #9 |

Thứ tự tạo: RULES → FOLDER_STRUCTURE → DESC_TEMPLATE → LOG_TEMPLATE → MODEL_CARD_TEMPLATE → lab1_description → 00_README. Mỗi lượt một file; AI in nháp ra chat, chỉ ghi khi leader nói "lưu".

**Không đụng trong plan này:**
- 9 file module hiện có ở gốc `docs/` (`demo.md`, `eda.md`, …): giữ nguyên chỗ cũ; giai đoạn sau viết lại theo khuôn C13 và chuyển vào `phases/` bằng `git mv`.
- `project_description.md`, `progress_status.md` ở gốc lab: giữ nguyên. `00_README_v1.md` ghi rõ desc hiện hành là `docs/lab1_description_v1.md`.
- `Lab-DL/docs/templates/` (template chung của repo: `bug_report.md`, `experiment_note.md`, `module_guide.md`): giữ nguyên.
- Các thư mục `phases/`, `models/`, `experiments/`, `bugs/`: chỉ tạo khi có file đầu tiên (git không lưu thư mục rỗng).

**Cây thư mục đích của `labs/lab1/` (ghi trong `FOLDER_STRUCTURE_v1.md`):**

    labs/lab1/
    ├── eda/ preprocessing/ features/ model/ train/ evaluate/ experiment/ visualize/   # 8 module (C2)
    ├── data/raw/  data/splits/                          # O4
    ├── outputs/models/ logs/ figures/ results/ demo/    # O3
    ├── docs/  (cấu trúc như trên)                       # C4, O8
    ├── notebook/  tests/
    └── main.py  README.md  requirements.txt  progress_status.md  project_description.md

Bỏ khỏi gốc lab (giai đoạn sau, O5): `artifacts/`, `raw_data/`, `logbook/`, `extended_study.py`.

## 3. Khuôn desc (khoảng 20 dòng, C13)

    # <Lab/Module> — <tên ngắn>  (v<N>)
    - Tác giả: <tên> | Tạo: YYYY-MM-DDTHH:MM:SS+07:00 | Cập nhật: <tên>, <thời điểm>
    - Trạng thái: To Do | In Progress | Done
    ## Background
    <1–3 câu: vì sao cần bước này>
    ## Purpose
    - Làm: <kết quả cụ thể>
    - Không làm: <ngoài phạm vi>
    ## Input / Output
    - Input: <đường dẫn> (<định dạng, shape>)
    - Output: <đường dẫn> (<định dạng>)
    - Code: labs/lab1/<module>/<file>.py
    ## Success criteria
    - <chỉ số> <ngưỡng> trên <tập dữ liệu>
    - <file phải tồn tại / nạp lại được>
    ## How-to-do
    1. <bước 1 = 1 small prompt>
    2. ...
    ## Quyết định (người ghi)
    - <ngày>: <quyết định> — <lý do>

## 4. Template log (C11)

Mỗi lần chạy sinh 3 file trong `outputs/logs/`: `<YYYYMMDD_HHMMSS>_<run>.log`, `<…>_history.csv` (mỗi epoch một dòng; biểu đồ vẽ từ file này), `<…>_config.json`.

    === RUN START 2026-11-13T10:15:00+07:00 ===
    run: baseline | seed: 42 | device: cpu
    config: hidden_sizes=[256,128] lr=0.001 optimizer=adam dropout=0.0 weight_decay=0.0 batch_size=64 max_epochs=30
    early_stopping: monitor=val_loss patience=5 min_delta=1e-4
    architecture: MLP 784-256-128-10 (ReLU) | params: 235,146
    data: train=54000 val=6000 | split: labs/lab1/data/splits/train_val_split.json
    epoch,train_loss,train_acc,val_loss,val_acc,lr,time_s
    1,...
    === RUN END 2026-11-13T10:17:30+07:00 ===
    best_epoch: <n> | stopped_epoch: <n> | stop_reason: early_stopping | max_epochs
    best_ckpt: outputs/models/<ts>_baseline_best.pt | last_ckpt: outputs/models/<ts>_baseline_last.pt

(Số param và số mẫu train/val lấy từ code hiện có; thời gian chỉ là ví dụ.)

## 5. Quy tắc dừng (C9)

1. Tối đa 30 epoch, giống nhau cho mọi cấu hình.
2. Sau mỗi epoch tính `val_loss`. Nếu giảm ít nhất `1e-4` so với best: lưu `_best.pt`, reset bộ đếm về 0. Nếu không: tăng bộ đếm thêm 1.
3. Dừng khi bộ đếm = 5 (`early_stopping`) hoặc khi hết 30 epoch (`max_epochs`).
4. Luôn lưu `_last.pt`. Đánh giá và demo dùng `_best.pt`.
5. Val tách từ train. Test chạy một lần, sau khi đã chọn cấu hình trên val (C10).
6. Theo dõi `val_loss` thay vì chữ "acc" trong note của cô vì `val_loss` nhạy hơn; `val_acc` vẫn được log và báo cáo.

## 6. Template model card (O12)

    # Model card — <tên>  (v<N>)
    - Tác giả, ngày tạo | Người cập nhật, ngày cập nhật
    ## Purpose
    ## Initial model
    - Architecture: <vd 784→256→128→10, ReLU, dropout p> | Sơ đồ: outputs/figures/<model>_architecture.png
    - Xây từ đâu / backbone: <tự viết, không dùng pretrained>
    - Số param: <…> | Input: [B,1,28,28] (chuẩn hóa theo mean/std train) | Output: logits [B,10]
    - Source: labs/lab1/model/model.py
    ## Trained model
    - Người train | Ngày train xong | Log/config: outputs/logs/<ts>_<run>.*
    - Checkpoint: outputs/models/<ts>_<run>_best.pt (best_epoch, stopped_epoch)
    - Kết quả: val_acc, test_acc
    - Ghi chú

Model card đã điền lưu ở `docs/models/<tên>_v<N>.md`.

## 7. Nội dung chính của `lab1_description_v1.md`

- **Số model (N5):** 5 cấu hình MLP, seed 42, tối đa 30 epoch + early stopping:
  1. baseline (256,128), Adam, lr 1e-3
  2. softmax_regression (không hidden layer)
  3. larger_network (512,256,128)
  4. sgd_lr_0.01
  5. dropout_0.2
- **Success criteria:**
  - Mọi run có log và `_best.pt` nạp lại dự đoán được.
  - Baseline đạt test ≥ 88% (tập test chính thức 10k).
  - Baseline có val_acc cao hơn softmax_regression.
  - Có đủ: loss curve train+val vẽ từ log, ảnh dự đoán trên test, confusion matrix, bảng so sánh 5 cấu hình (có cột chênh train_acc − val_acc, chỉ để báo cáo).
- **Không làm:** CNN, Optuna, đa seed, p-value/CI95, test trên ảnh ngoài dataset, đo tốc độ inference.

## 8. Chỗ handoff lệch so với file (phát hiện khi kiểm)

- `draft_lab1_train.md`: không có trong thư mục DL.
- `MODELS.md` của M: phần thân là hướng dẫn tải model, không có bảng số param như phần đầu file ghi.
- `agents/rules/` của M: không có trong bản nộp.
- O5: file nhắc `logbook/` là `CONTRIBUTING.md`, không phải `AGENTS.md`.
- §7 handoff: M Lab 1 **có** validation + early stopping (`practice_1.ipynb` cell 11, `patience = 5`).
- O10: số lần chạy thật là 46 (1 baseline + 35 + 10), không phải 35 hay 45.

## 9. Bước tiếp theo

1. AI tạo 7 file ở §2, mỗi lượt một file; leader đọc và nói "lưu".
2. Cập nhật handoff khi leader nói "ghi handoff".
3. Git: làm trên nhánh `docs/lab1-progress-kaggle-note` (đã có 1 commit `progress_status.md`). Xong 7 file thì mở **một PR** gồm: commit progress, `docs/plans/` (v0, v1) và 7 file mới. Chỉ push/PR khi leader bảo.
4. Giai đoạn Lab-DL (thư mục, code, viết lại md module vào `phases/`): chỉ làm khi leader cho phép (O5).
