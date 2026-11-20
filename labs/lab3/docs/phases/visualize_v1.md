# visualize — đường huấn luyện, confusion matrix, `max_length`, ví dụ sai (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T12:37:50+07:00 (Phúc, AI soạn)
- **Người phụ trách**: Phúc (Member B)
- **Status**: In Progress (chạy xong, chờ Thư audit)

---

## Name
Hình và phân tích lỗi — vẽ từ file trong `outputs/`, không train, không chạy model

## Background
Phần mở rộng (plan §5). Mỗi hình có lý do riêng của Lab 3: đường huấn luyện là bằng chứng chọn điểm dừng; confusion matrix so zero-shot (học trên SST-2, câu ngắn) với fine-tuned trên review dài; hình `max_length` nối với EDA; ví dụ sai chia theo review bị cắt / không bị cắt và theo rating gốc.

## Goals / Purpose
- `plot_training_curves(history_csv)` → `outputs/figures/<run>_curves.png`: train loss, val loss, val acc theo bước (trục phụ: epoch); đánh dấu checkpoint tốt nhất
- `plot_confusion(pred_csvs)` → `outputs/figures/confusion_zero_shot_vs_finetuned.png` (2 ma trận cạnh nhau, cùng 25k test)
- `plot_maxlen(maxlen_runs_csv)` → `outputs/figures/maxlen_val_acc.png` (thanh lỗi = std nếu có nhiều seed)
- `error_analysis(pred_csv, test, max_length)` → `outputs/demo/error_examples.csv` + `outputs/results/error_by_group.csv`: tỉ lệ sai theo nhóm (bị cắt / không bị cắt; rating 1, 2, 3, 4, 7, 8, 9, 10) và vài ví dụ sai tự tin nhất mỗi nhóm
- Không làm:
  - ROC/PR — bỏ theo plan L17
  - Hình EDA — thuộc `eda`

## Input / Output
- **Input:** `outputs/logs/*_history.csv` (`train`); `outputs/results/*_test_predictions.csv` (`evaluate`); `outputs/results/maxlen_runs.csv` (`experiment`); test thô + số token (`preprocessing`, `eda`)
- **Output:** `outputs/figures/*.png`, `outputs/demo/error_examples.csv`, `outputs/results/error_by_group.csv` (commit)

## How to do it (general plan)
1. Đường huấn luyện cho pilot và từng seed của lần chạy chính
2. Confusion matrix 2×2 (số lượng + tỉ lệ) cho zero-shot và fine-tuned (seed có val acc cao nhất, ghi rõ seed)
3. Hình `max_length`
4. Phân tích lỗi theo nhóm
5. Test với CSV giả nhỏ
6. Chạy trên máy Thư sau khi tải `outputs/` từ Kaggle về (chỉ đọc file)

## Pipeline
```
_history.csv → plot_training_curves
*_test_predictions.csv → plot_confusion, error_analysis (+ test thô, số token)
maxlen_runs.csv → plot_maxlen
  → outputs/figures/, outputs/demo/, outputs/results/ → notebook, README
```

## Detailed plan
- **Đường huấn luyện vẽ cả val loss lẫn val acc:** cho thấy chỗ val loss tăng trong khi acc còn tăng (HF course ch.3.5) — giải thích vì sao chọn theo acc (L8)
- **Nhóm theo rating (Thư duyệt 2026-11-20):** rating gốc có sẵn trong tên file (`<id>_<rating>.txt`; nhãn âm ≤ 4, dương ≥ 7 theo README). Giả thuyết: lỗi tập trung ở rating 4 và 7 (sát ranh giới) — bảng này kiểm, không khẳng định trước
- **Bỏ nhóm "có từ phủ định":** review IMDB nhiều câu gần như chắc chắn có phủ định (Konstantinova et al. 2012; `eda` đo tỉ lệ thật) → nhóm không tách được gì. Phủ định được đo bằng 6 câu kiểm soát ở `sentiment_demo` / `evaluate`
- **Ví dụ sai:** chọn theo xác suất sai cao nhất, mỗi nhóm vài ví dụ; lý do sai ghi là giả thuyết, không khẳng định
- **Gotchas:**
  - Notebook chỉ hiển thị hình đã lưu, không vẽ lại từ model (RULES §2)
  - Review dài: cắt hiển thị trong CSV ví dụ (vd 300 ký tự đầu) nhưng giữ `id` để tra bản đầy đủ

## Triển khai (2026-11-20)
- **Hàm** (`labs/lab3/visualize/visualize.py`): `plot_training_curves(history_csv, out_path=None, title=None)` (`:34`); `confusion_counts(df)` (`:76`); `plot_confusion(pred_csvs, out_path=None)` (`:84`); `maxlen_table(maxlen_runs, main_runs=None)` (`:117`); `plot_maxlen(maxlen_runs, main_runs=None, out_path=None)` (`:131`); `error_analysis(pred_csv, test_df, token_lengths, max_length, results_dir=None, demo_dir=None, per_group=3)` (`:156`); `run_all(...)` (`:207`). Chạy: `python -m labs.lab3.visualize.visualize` (chỉ đọc file trong `outputs/`)
- **Chỗ code khác plan:**
  - `maxlen_table` lấy giá trị 512 từ `main_runs.csv` (thí nghiệm `maxlen` không chạy lại 512)
  - Độ dài token lấy từ `outputs/results/eda_token_lengths.csv` của EDA (đếm trên văn bản đã làm sạch, gồm `[CLS]`/`[SEP]`); bị cắt ⇔ `n_tokens > max_length`
  - Confusion matrix và phân tích lỗi dùng seed có **val acc** cao nhất (seed 44), chọn trên val
- **Kết quả** (chi tiết `experiment_log.md` S5):
  - Hình: `outputs/figures/pilot_seed42_curves.png`, `main_seed{42,43,44}_curves.png`, `maxlen{128,256}_seed42_curves.png`, `maxlen_val_acc.png`, `confusion_zero_shot_vs_finetuned.png` (tổng mỗi ma trận = 25.000)
  - Zero-shot bỏ sót 13,5% review tích cực (FN 1.690); fine-tuned FN 685, FP 971
  - Tỉ lệ lỗi (seed 44): bị cắt 10,05% so với không bị cắt 6,14%; rating 4: 17,23%, rating 7: 11,70% (cao nhất) → giả thuyết "lỗi tập trung ở 4 và 7" đúng trên số liệu này
  - File: `outputs/results/error_by_group.csv`, `outputs/demo/error_examples.csv`
- **Test / smoke:** `pytest labs/lab3/tests/test_visualize.py` 4 passed (CSV giả, tổng confusion = số mẫu); smoke trên máy: hình và CSV trong `outputs/_smoke/`

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_visualize.py` đạt: tạo đủ file từ CSV giả, tổng confusion matrix = số mẫu
- **Smoke run:** hình và CSV có trong `outputs/_smoke/`
- **Normal run:** đủ hình ở Goals; tổng mỗi confusion matrix = 25.000; `error_by_group.csv` có đủ nhóm

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md), [experiment_v1.md](experiment_v1.md), [eda_v1.md](eda_v1.md)
- Nguồn: HF LLM course ch.3.5 https://huggingface.co/learn/llm-course/chapter3/5
- Code: `labs/lab3/visualize/visualize.py` | Test: `labs/lab3/tests/test_visualize.py`
