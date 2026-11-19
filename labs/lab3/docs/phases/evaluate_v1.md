# evaluate — metric bằng thư viện `evaluate`, test một lần, zero-shot (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T01:16:36+07:00 (Thư)
- **Người phụ trách**: Thư
- **Status**: To Do

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

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_evaluate.py` đạt: metric đúng trên ví dụ tính tay, đổi nhãn zero-shot đúng, gọi test lần hai thì báo lỗi
- **Smoke run:** file dự đoán và `test_summary.csv` có trong `outputs/_smoke/results/`
- **Normal run:** mỗi file dự đoán 25.000 dòng; `test_summary.csv` có 3 seed + mean ± std + zero-shot; accuracy trung bình ≥ 0,91; `control_sentences.csv` có 6 câu × 2 model

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [sentiment_demo_v1.md](sentiment_demo_v1.md), [train_v1.md](train_v1.md), [visualize_v1.md](visualize_v1.md)
- Nguồn: HF Evaluate https://huggingface.co/docs/evaluate/transformers_integrations ; Sanh et al. 2019 https://arxiv.org/abs/1910.01108
- Code: `labs/lab3/evaluate/evaluate.py` | Test: `labs/lab3/tests/test_evaluate.py`
