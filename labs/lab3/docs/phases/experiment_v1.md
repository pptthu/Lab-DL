# experiment — pilot, 3 seed, thí nghiệm `max_length` (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T14:37:54+07:00 (Thư)
- **Người phụ trách**: Phúc (Member B)
- **Status**: Done — audit 2026-11-20

---

## Name
Thí nghiệm — sinh cấu hình cho pilot, lần chạy chính 3 seed, thí nghiệm `max_length`; gộp bảng kết quả

## Background
Phần mở rộng (plan §4 S2–S4, §5). Không có trong 7 bước của đề. Mục đích: chốt số epoch bằng thực nghiệm (pilot), đo độ lệch theo seed (Dodge et al. 2020: std SST 0,0028, tập nhỏ 0,06–0,10), và trả lời việc cắt review ảnh hưởng accuracy bao nhiêu (Sun et al. 2019).

## Goals / Purpose
- `make_configs(stage)` trả danh sách `cfg` cho `pilot` / `main` / `maxlen`; mọi tham số đọc từ một chỗ
- `run_stage(stage)` gọi `train.train_model` cho từng `cfg`, ghi `outputs/results/<stage>_runs.csv` (run, seed, `max_length`, best_step, best_val_acc, stopped_step, stop_reason, train_time)
- `seed_spread(main_runs)` trả mean, std của val acc 3 seed → quyết định số seed cho `maxlen` (S4)
- Không làm:
  - Chạy test — thuộc `evaluate`, chỉ cho lần chạy chính (thí nghiệm `max_length` so trên val)
  - Tìm siêu tham số tự động (Optuna) — ngoài phạm vi (plan §7)

## Input / Output
- **Input:** cấu hình gốc (plan L9); `max_length` chốt sau S1; trần epoch, patience chốt sau S2
- **Output:** `outputs/results/pilot_runs.csv`, `main_runs.csv`, `maxlen_runs.csv`; log từng run (qua `train`); ghi tóm tắt vào `docs/experiments/experiment_log.md`

## How to do it (general plan)
1. `pilot`: 1 seed (42), trần epoch trên mức 2–4 của BERT gốc (số cụ thể theo thời gian/epoch đo ở smoke trên Kaggle), không dừng sớm
2. Thư xem đường cong pilot → chốt trần, patience, threshold
3. `main`: 3 seed, dừng sớm theo bước 2
4. `seed_spread` → std nhỏ (≈ ≤ 0,3 điểm): `maxlen` 1 seed / giá trị; std lớn (≈ ≥ 1 điểm): 3 seed / giá trị `[cần chốt sau S3]`
5. `maxlen`: 128 / 256 / 512 (bỏ giá trị đã chạy ở `main`), so trên val
6. Test: `make_configs` sinh đúng số run, tham số đúng; `run_stage` với `train_model` giả
7. Smoke: `pilot` vài step; normal run trên Kaggle

## Pipeline
```
make_configs(pilot) → run_stage → pilot_runs.csv + đường cong → Thư chốt trần/patience
make_configs(main)  → run_stage → main_runs.csv → seed_spread → số seed cho maxlen
make_configs(maxlen)→ run_stage → maxlen_runs.csv → visualize
main (3 checkpoint) → evaluate (test một lần)
```

## Detailed plan
- **Seed lần chạy chính (Thư duyệt 2026-11-20):** 42, 43, 44 — **cố định trước khi chạy**, báo cáo cả 3, không chọn seed cho kết quả đẹp. Seed của HF Trainer đổi cả khởi tạo đầu phân loại lẫn thứ tự dữ liệu
- **Giới hạn:** Bouthillier et al. 2021: seed khởi tạo chỉ là *"a small part of the variance"*, phần lớn đến từ cách chia dữ liệu. Lab 3 giữ test chính thức cố định (để so với bài báo) → std đo được **chưa gồm** độ lệch do cách chia; ghi rõ trong báo cáo
- **Tiêu chí "khác biệt có ý nghĩa" ở `maxlen`:** chênh lệch val acc lớn hơn khoảng 2 lần std theo seed đo ở `main`; nhỏ hơn thì kết luận "không phân biệt được" (plan L12)
- **Chi phí ước tính:** 1 pilot + 3 main + 2 maxlen = 6 lần chạy trên Kaggle T4 `[đoán: 30–60 phút / lần, kiểm ở smoke trên Kaggle]`
- **Gotchas:**
  - `max_length` 512 tốn bộ nhớ và thời gian nhất → chạy cuối, giảm batch nếu thiếu VRAM và ghi lại trong log
  - Kaggle ngắt phiên sau một thời gian chạy liên tục → mỗi run ghi kết quả ngay khi xong, chạy lại được từng run

## Triển khai (2026-11-20)
- **Hàm** (`labs/lab3/experiment/experiment.py`): `make_configs(stage, smoke=False) -> list[dict]` (`:37`); `append_run_row(row, path)` (`:74`); `run_row(cfg, result, n_train) -> dict` (`:86`); `run_stage(stage, parts=None, smoke=False, runs=None, train_fn=None, model_fn=None, results_dir=None, **train_kw)` (`:98`); `seed_spread(main_runs) -> dict` (`:144`); `merge_run_tables(src, dst) -> int` (`:163`). Chạy: `python -m labs.lab3.main --stage timing|pilot|main|maxlen [--runs …]`
- **Chỗ code khác plan:**
  - Thêm giai đoạn `timing` (300 step) để đo thời gian/epoch trên Kaggle trước khi chọn trần pilot (theo yêu cầu bổ sung của Thư 2026-11-20)
  - `run_stage` nhận `runs` để mỗi kernel Kaggle chạy đúng một run; bảng `<stage>_runs.csv` ghi sau mỗi run và được gộp khi tải về (`merge_run_tables`)
  - `make_configs` báo lỗi nếu giá trị `[cần chốt]` trong `config.py` còn `None` (không đoán)
- **Kết quả normal run** (chi tiết `experiment_log.md` S2–S4):
  - `timing_runs.csv`: 0,356 s/step → chọn trần pilot 6 epoch
  - `pilot_runs.csv`: best val acc 0,9308 (epoch 4,0) → chốt trần 6, patience 8, threshold 0 [AI chốt theo uỷ quyền của Thư 2026-11-20 — Thư đã duyệt 2026-11-20]
  - `main_runs.csv`: 3 seed, val acc 0,9303 ± 0,0013 → `seed_spread`: std ≤ 0,3 điểm → 1 seed / `max_length` (ngưỡng của plan)
  - `maxlen_runs.csv` (seed 42): val acc 128: 0,8729; 256: 0,9152; 512 (`main_seed42`): 0,9288 → cùng seed: 256 hơn 128 4,24 điểm, 512 hơn 256 1,36 điểm, đều lớn hơn 2 × std (0,26 điểm); trung bình 3 seed ở 512 (0,9303) chỉ để tham khảo
  - Chạy trên tài khoản Kaggle của Thư (`pptt1103`); Kaggle cho tối đa 2 phiên GPU cùng lúc → các run chạy theo cặp
- **Test / smoke:** `pytest labs/lab3/tests/test_experiment.py` 6 passed (`train_model` giả); smoke trên máy: `pilot_runs.csv` 1 dòng, `main_runs.csv` 3 dòng

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_experiment.py` đạt
- **Smoke run:** `outputs/_smoke/results/pilot_runs.csv` có 1 dòng
- **Normal run:** `pilot_runs.csv` (1 dòng), `main_runs.csv` (3 dòng, std ghi lại), `maxlen_runs.csv` (đủ giá trị); `experiment_log.md` ghi quyết định sau mỗi giai đoạn

## Kết quả audit (2026-11-20)
- **Test:** `pytest labs/lab3/tests/test_experiment.py` → 6 passed ✔
- **Smoke run** (`python -m labs.lab3.main --smoke`, máy Thư, CPU, 2026-11-20): `outputs/_smoke/results/pilot_runs.csv` 1 dòng ✔
- **Normal run:** `pilot_runs.csv` 1 dòng, `main_runs.csv` 3 dòng (mean 0,9303, std 0,0013 tính lại khớp), `maxlen_runs.csv` 2 dòng (128, 256; 512 lấy từ `main_seed42`); `experiment_log.md` ghi quyết định sau S2, S3, S4 ✔
- Số dòng code ghi ở mục Triển khai khớp code hiện tại ✔

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md), [visualize_v1.md](visualize_v1.md), [eda_v1.md](eda_v1.md)
- Nguồn: Dodge et al. 2020 https://arxiv.org/abs/2002.06305 ; Sun et al. 2019 https://arxiv.org/abs/1905.05583 ; Bouthillier et al. 2021 https://arxiv.org/abs/2103.03098
- Code: `labs/lab3/experiment/experiment.py` | Test: `labs/lab3/tests/test_experiment.py`
