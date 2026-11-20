# evaluate — metric bằng thư viện `evaluate`, test một lần, zero-shot (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T14:37:54+07:00 (Thư)
- **Người phụ trách**: Phúc (Member B)
- **Status**: Done — audit 2026-11-20

---

## Name
Đánh giá model đã fine-tune (Bài 2.7) — accuracy/F1 bằng `evaluate`, test 25k một lần mỗi model, baseline zero-shot

## Background
Bài 2.1 yêu cầu thư viện `evaluate`; Bài 2.7 yêu cầu đánh giá model đã fine-tune. Plan L2, L6, L7, L12, L13: test 25k chỉ dùng một lần; so với pipeline zero-shot của Bài 1 trên cùng 25k test; báo trung bình ± std 3 seed; ngưỡng ≥ 91%.

## Goals / Purpose
- `compute_metrics(eval_pred)` trả `accuracy`, `f1` (dùng trong Trainer khi đánh giá val)
- `evaluate_on_test(model_dir, tokenized_test)` trả accuracy, F1 và ghi `outputs/results/<run>_test_predictions.csv` (`id`, `label`, `pred`, `prob_pos`)
- `evaluate_zero_shot(test)` chạy pipeline SST-2 của `sentiment_demo` trên 25k test, ghi dự đoán cùng định dạng
- `summarize(...)` ghi `outputs/results/test_summary.csv`: từng seed, trung bình ± std, zero-shot; cột `pass` so với ngưỡng 91%; nếu `eda` thấy review trùng train–test: thêm cột acc trên test **bỏ các review trùng**
- `evaluate_control_sentences(model_dir)` chạy 6 câu kiểm soát của `sentiment_demo` (`CONTROL_SENTENCES`) qua model sau fine-tune → `outputs/demo/control_sentences.csv`: so zero-shot và fine-tuned trên phủ định, tương phản, trung tính
- Không làm:
  - Vẽ confusion matrix, phân tích lỗi — thuộc `visualize` (đọc file dự đoán)
  - Chọn checkpoint — thuộc `train` (theo val)

## Input / Output
- **Input:** checkpoint tốt nhất của từng seed (`train`); test đã tokenize (`features`); test thô (`preprocessing`) cho zero-shot
- **Output:** `outputs/results/*_test_predictions.csv`, `outputs/results/test_summary.csv` (commit)

## How to do it (general plan)
1. `compute_metrics` với `evaluate.load("accuracy")`, `evaluate.load("f1")`
2. `evaluate_on_test`: `trainer.predict` trên test → softmax → nhãn; ghi CSV dự đoán
3. `evaluate_zero_shot`: đổi `POSITIVE`→1, `NEGATIVE`→0; `truncation=True`
4. `summarize`: gộp các seed, tính mean ± std (std mẫu, n−1), so ngưỡng
5. Kiểm "test một lần": mỗi run có cờ trong `_config.json`; gọi lần hai thì báo lỗi
6. Test: `compute_metrics` so với tính tay trên ví dụ nhỏ; đổi nhãn zero-shot đúng
7. Smoke run vài chục review; normal run trên Kaggle (S5)

## Pipeline
```
checkpoint tốt nhất (mỗi seed) + test tokenized → evaluate_on_test → *_test_predictions.csv
pipeline SST-2 (sentiment_demo) + test thô → evaluate_zero_shot → zero_shot_test_predictions.csv
  → summarize → test_summary.csv (mean ± std, pass ≥ 91%) → visualize, README, model card
```

## Detailed plan
- **Thư viện `evaluate`:** đề Bài 2.1; tutorial chính thức HF dùng `evaluate.load("accuracy")`
- **F1 nhị phân, lớp dương = 1 (POSITIVE):** IMDB cân bằng nên F1 gần bằng accuracy; báo cả hai
- **Accuracy là chỉ số chính:** DistilBERT paper và các model card báo accuracy; ngưỡng 91% so trên trung bình 3 seed (L13)
- **Zero-shot trên cùng 25k test:** so sánh công bằng (L6). Pipeline cắt ở 512 token
- **Review trùng train–test (Thư duyệt 2026-11-20):** giữ test chính thức nguyên vẹn để so được với DistilBERT paper (92,82); acc trên test bỏ review trùng chỉ báo thêm, không thay chỉ số chính
- **Câu kiểm soát:** cùng 6 câu CheckList (Ribeiro et al. 2020) cho cả model trước và sau fine-tune — đo phủ định bằng câu có định nghĩa rõ thay vì nhóm review IMDB theo từ phủ định
- **Gotchas:**
  - `evaluate.load` tải script metric từ Hub lần đầu → Kaggle phải bật Internet
  - Test chỉ chạy sau khi đã chốt mọi thứ trên val; thí nghiệm `max_length` so trên **val**, không chạy test (plan S4)

## Triển khai (2026-11-20)
- **Hàm** (`labs/lab3/evaluate/evaluate.py`): `accuracy_f1(labels, preds) -> dict` (`:33`); `compute_metrics(eval_pred) -> dict` (`:42`); `softmax_pos(logits)` (`:53`); `predict_logits(model, tokenized_split, collator, batch_size)` (`:85`); `evaluate_on_test(model_dir, tokenized_test, run, collator=None, run_config_path=None, results_dir=None, model=None) -> dict` (`:99`); `evaluate_zero_shot(test_ds, pipe=None, results_dir=None, batch_size=64) -> dict` (`:129`); `duplicate_test_uids(train_df, test_df) -> set` (`:151`); `summarize(pred_files, zero_shot_file=None, seeds=None, duplicate_uids=None, out_path=None, pass_accuracy=0.91)` (`:162`); `evaluate_control_sentences(model_dir, label, pipe_zero_shot=None, pipe_finetuned=None, out_path=None)` (`:204`). Gọi từ `main.run_test` (`labs/lab3/main.py:103`)
- **Chỗ code khác plan:**
  - Cột CSV dự đoán là `uid` (không phải `id`) để khớp `features.KEEP_COLUMNS` của phần A
  - `evaluate_on_test` cần thêm `run` và `run_config_path` để đặt cờ "test một lần"; ngoài cờ, file dự đoán đã có cũng bị coi là đã test
  - Review trùng train–test tính lại bằng `preprocessing.text_hash` (cùng cách EDA) → 123 review, khớp `eda_duplicates.csv`
- **Kết quả normal run** (Kaggle T4, test 25.000; chi tiết `experiment_log.md` S5):
  - Fine-tuned test acc **0,9315 ± 0,0020** (seed 42 / 43 / 44: 0,9300 / 0,9309 / 0,9338), F1 0,9323 ± 0,0021 → **đạt ngưỡng 91%**
  - Zero-shot SST-2: acc 0,8922, F1 0,8892 (fine-tuned hơn 3,93 điểm)
  - Bỏ 123 review trùng: acc trung bình 0,9312 (gần như không đổi)
  - Câu kiểm soát: fine-tuned (seed 44) sai cả 2 câu phủ định, zero-shot đúng cả 2 → ghi đúng như đo
  - 3 file `_config.json` của lần chạy chính có `test_evaluated: true`; mỗi file dự đoán 25.000 dòng
  - File: `outputs/results/test_summary.csv`, `*_test_predictions.csv`; `outputs/demo/control_sentences.csv`
- **Test / smoke:** `pytest labs/lab3/tests/test_evaluate.py` 7 passed (metric so với tính tay, đổi nhãn zero-shot, gọi test lần hai báo lỗi); smoke trên máy: đủ file trong `outputs/_smoke/results/`

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_evaluate.py` đạt: metric đúng trên ví dụ tính tay, đổi nhãn zero-shot đúng, gọi test lần hai thì báo lỗi
- **Smoke run:** file dự đoán và `test_summary.csv` có trong `outputs/_smoke/results/`
- **Normal run:** mỗi file dự đoán 25.000 dòng; `test_summary.csv` có 3 seed + mean ± std + zero-shot; accuracy trung bình ≥ 0,91; `control_sentences.csv` có 6 câu × 2 model

## Kết quả audit (2026-11-20)
- **Test:** `pytest labs/lab3/tests/test_evaluate.py` → 7 passed ✔
- **Normal run:** 4 file dự đoán, mỗi file 25.000 dòng, 25.000 `uid` khác nhau; accuracy tính lại từ file khớp `test_summary.csv` (0,9300 / 0,9309 / 0,9338; zero-shot 0,8922); `test_summary.csv` có 3 seed + mean + std + zero-shot, mean 0,9315 ≥ 0,91 ✔; `control_sentences.csv` 6 câu × 2 model ✔
- **Test một lần:** chỉ 3 file `_config.json` của lần chạy chính có `test_evaluated: true`; timing, pilot, `max_length` đều `false` ✔
- **Smoke run:** lần smoke trên máy Thư (2026-11-20) bị dừng ở bước train vì quá giờ trên CPU, chưa tới bước test; bước test được kiểm bằng test đơn vị và lần chạy thật trên Kaggle
- Số dòng code ghi ở mục Triển khai khớp code hiện tại ✔

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [sentiment_demo_v1.md](sentiment_demo_v1.md), [train_v1.md](train_v1.md), [visualize_v1.md](visualize_v1.md)
- Nguồn: HF Evaluate https://huggingface.co/docs/evaluate/transformers_integrations ; Sanh et al. 2019 https://arxiv.org/abs/1910.01108
- Code: `labs/lab3/evaluate/evaluate.py` | Test: `labs/lab3/tests/test_evaluate.py`
