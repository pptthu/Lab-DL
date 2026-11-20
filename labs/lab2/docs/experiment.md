# experiment — So sánh model và siêu tham số (v2)

- **Created**: 2026-11-04 · **Last Updated**: 2026-11-20 (Bảo) — viết lại theo code hiện tại
- **Người phụ trách**: Member B (Bảo)
- **Status**: Done
- **Code**: [`labs/lab2/experiment/experiment.py`](../experiment/experiment.py)

---

## Name
Experiments — chạy `model × cấu hình`, mỗi run một checkpoint + một thư mục TensorBoard, xuất bảng CSV

## Background
Cần trả lời "cấu hình nào tốt nhất?" giữa feature extraction (đóng băng backbone) và fine-tuning
(mở khoá khối cuối) trên nhiều backbone. `experiment` là lớp điều phối: nó **không** tự viết vòng lặp train,
mà ghép `build_model` + `unfreeze_last_blocks` (Member A) với `train_model` (Member B) cho từng ô của lưới.

## Goals / Purpose
- `run_comparison` duyệt `model_names × hparam_grid`, mỗi cặp là một run độc lập và trả list kết quả.
- Báo cáo đúng **epoch đã lưu checkpoint** (`best_epoch`, `best_val_loss`, `best_val_acc`), không lấy `max(val_acc)`.
- `save_results` làm phẳng `hparams` thành cột `hp_*` và ghi CSV để notebook/README dùng lại.
- `make_summary_writer` biến TensorBoard thành tuỳ chọn (máy không cài vẫn import và chạy được).
- Không làm:
  - Không đánh giá trên test — chỉ so sánh trên validation; test do `evaluate` chạy một lần cho cấu hình thắng.
  - Không lưu lịch sử từng epoch vào CSV (chỉ metric của epoch tốt nhất); history chi tiết do `main.py` ghi riêng.

## Input / Output
- **Input:** `model_names: List[str]` (vd `['resnet18','densenet121']`); `hparam_grid: List[Dict]`;
  `loaders: (train, val, test)`; `device`; `log_root`; tuỳ chọn `build_model_fn`, `unfreeze_fn`, `train_model_fn`
  (dùng để test/inject), `enable_tensorboard: bool = True`.
- **Key đọc từ mỗi ô lưới (kèm mặc định):** `lr` 1e-3, `epochs` 1, `optimizer_name` `'adam'`,
  `weight_decay` 0.0, `freeze_backbone` True, `pretrained` True, `unfreeze_blocks` 0.
- **Output:** list dict, mỗi phần tử gồm `model`, `hparams`, `best_val_acc`, `best_val_loss`, `best_epoch`,
  `train_seconds`, `checkpoint`, `tensorboard_logged`.
  File: `{log_root}/{run_name}_best.pt`, log `{log_root}/{run_name}/`, CSV do `save_results` ghi.
  `run_name = f"{model}_run{idx}_lr{lr}_freeze{freeze_backbone}"`.

## How to do it (general plan)
1. Resolve hàm phụ thuộc: import động `build_model` (bắt buộc), `unfreeze_last_blocks` (thiếu thì bỏ qua), `train_model`.
2. Chỉ lấy `train_loader, val_loader` từ `loaders`; phần tử thứ ba (test) bị bỏ qua.
3. Với mỗi `(model, hparams)`: tạo `run_name`, `writer` cho run (nếu bật) và đường dẫn checkpoint.
4. Dựng model theo `freeze_backbone`/`pretrained`, mở khoá nếu `unfreeze_blocks > 0`.
5. Đo thời gian bằng `time.perf_counter()` và gọi `train_model(...)`; `writer.close()` trong `finally`
   để không rò rỉ file handle khi train lỗi.
6. Đọc metric của epoch đã lưu bằng `_resolve_best_metrics`, thêm kết quả vào list; cuối cùng `save_results` xuất CSV.

## Pipeline
```
main.py: model_names = [model_name, 'densenet121']
         grid = default_comparison_grid(lr, epochs, weight_decay)
              # ô 1: freeze=True, unfreeze_blocks=0 | ô 2: freeze=True, unfreeze_blocks=1
run_comparison(model_names, grid, (train, val, test), device, log_root='artifacts/runs/experiment')
   └─ mỗi run: build_model → (unfreeze) → train_model → _resolve_best_metrics → results.append
save_results(results, 'artifacts/runs/experiment_results.csv')
```

## Detailed plan
- **TensorBoard tuỳ chọn:** `make_summary_writer` import `SummaryWriter` **bên trong hàm** và chỉ bắt `ImportError`
  → lỗi khác (quyền ghi, ...) không bị che; không có TensorBoard thì `writer=None`, `tensorboard_logged=False`.
- **Chọn epoch báo cáo:** `_resolve_best_metrics` ưu tiên `history["best_epoch"]`; nếu không có thì `argmin(val_loss)`,
  rồi `argmax(val_acc)`, cuối cùng `(0.0, None, None)` khi history rỗng. `best_epoch` là 1-based.
- **CSV:** cột `model`, `hp_<key>` (sắp alphabet), `best_val_acc`/`best_val_loss` 4 chữ số thập phân,
  `train_seconds` 2 chữ số, `checkpoint`; giá trị `None` ghi ô rỗng; `results` rỗng vẫn ghi header.
- **Hàm hiện có:** `make_summary_writer` (`experiment.py:18`), `_resolve_best_metrics` (`:42`),
  `run_comparison` (`:76`), `save_results` (`:200`).
- **Chỗ code chưa khớp plan:** không có.
- **Gotchas:**
  - `batch_size` trong `hparam_grid` **không có tác dụng**: loader được tạo một lần ở `main.py` rồi truyền vào.
  - `save_results` và checkpoint ghi đè file cùng đường dẫn (`open(...,'w')`, `torch.save`): chạy lại cùng tên run
    sẽ thay kết quả cũ — đổi `log_root` hoặc `run_name` nếu muốn giữ cả hai.
  - Seed chỉ đặt một lần trước `run_comparison`; chênh lệch vài điểm % giữa các run chưa kiểm định ý nghĩa thống kê.
  - Mỗi cell hiện vẫn chạy tuần tự, cùng DataLoader — rất chậm trên CPU với `epochs` lớn hoặc ảnh 128.

## Success criteria
- **Test:** `python -m pytest labs/lab2/tests/test_experiment.py labs/lab2/tests/test_tensorboard_optional.py labs/lab2/tests/test_checkpoint_consistency.py -q`
  (ghi kết quả + CSV; chạy được khi không có TensorBoard; writer đóng cả khi train raise; báo cáo đúng epoch checkpoint).
- **Smoke run:** lưới 1 model × 1 ô, `epochs=1`, loader giả → list kết quả dài đúng `len(model_names) * len(grid)`,
  file `.pt` tồn tại, CSV đọc được bằng `csv.DictReader`.
- **Normal run:** `main.py` in `Comparison results saved to: ...` và sinh `artifacts/runs/experiment_results.csv`.

## Links
- Desc lab: [../project_description.md](../project_description.md) | Model: [model.md](model.md)
- Module liên quan: [train.md](train.md), [evaluate.md](evaluate.md), [features.md](features.md)
- Tiến độ và bằng chứng chạy: [../progress_status.md](../progress_status.md)
- Test: `labs/lab2/tests/test_experiment.py`, `test_tensorboard_optional.py`, `test_checkpoint_consistency.py`
