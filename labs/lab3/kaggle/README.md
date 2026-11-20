# Chạy Lab 3 trên Kaggle GPU (Member B)

- **Created**: 2026-11-20T04:40:00+07:00
- **Last Updated**: 2026-11-20T04:40:00+07:00 (Phúc)

Repo `pptthu/Lab-DL` là private nên Kaggle không clone được. Code được đưa lên bằng một **Kaggle Dataset private** `lab3-redo-code`, tạo từ những file đã `git add` (cây index: `git write-tree` → `git archive <tree>`). Mã `<HEAD>+tree:<cây>` được ghi vào `BUILD_INFO.json` và vào log của mỗi lần chạy (`code: …`), nên biết chính xác code nào đã chạy. Thay đổi chưa stage không được đưa lên. Mỗi bước (đo thời gian, pilot, từng seed, từng `max_length`, test) chạy trong **một kernel riêng**, nên bước sau lỗi không làm mất bước trước.

Các lần chạy của Lab 3 (2026-11-20) dùng tài khoản Kaggle của Thư (`pptt1103`); tên tài khoản được đọc từ token (`kaggle config view`), không ghi cố định trong code.

## Chuẩn bị (một lần)
- Kaggle CLI ≥ 1.8 (`pip install kaggle`), token để ở `~/.kaggle/access_token`, **không bao giờ để trong repo**.
- Kiểm quota trước mỗi bước: `kaggle quota`.

## Các lệnh (chạy từ gốc repo)
| Việc | Lệnh |
|---|---|
| Đưa code đã stage lên dataset | `python -m labs.lab3.kaggle.kaggle_run publish-code` |
| Chạy một bước | `python -m labs.lab3.kaggle.kaggle_run push --stage pilot` (thêm `--runs main_seed43`, `--cpu`, `--smoke` khi cần) |
| Bước test (S5) | `push --stage test --model-kernels lab3-main-main-seed42 lab3-main-main-seed43 lab3-main-main-seed44` |
| Xem trạng thái / chờ (hỏi mỗi 10 phút) | `status --stage pilot` / `wait --stage pilot` |
| Tải kết quả về `labs/lab3/outputs/` | `fetch --stage pilot` |

## Kernel làm gì (`kernel_template.py`)
1. Tìm code trong dataset, chép sang `/tmp/lab3`, đặt `LAB3_CODE_COMMIT`.
2. Cài đúng phiên bản `transformers`, `datasets`, `evaluate`, `accelerate` (như `labs/lab3/requirements.txt`); `torch` giữ bản của Kaggle (phiên bản ghi trong log).
3. Chỉ dùng **1 GPU** (`CUDA_VISIBLE_DEVICES=0`). Với T4 x2, Trainer sẽ tự chạy DataParallel, khiến batch thực thành 32 thay vì 16 (plan L9).
4. Chạy `python -m labs.lab3.main --stage …` (tải aclImdb qua Internet).
5. Chép `outputs/{logs,results,figures,demo}` và checkpoint tốt nhất sang `/kaggle/working/outputs/`. Nếu bước chạy lỗi thì vẫn chép log rồi mới báo lỗi.

`fetch` gộp `<stage>_runs.csv` vào bảng trong repo (không ghi trùng run) và chép đè các file còn lại. Ví dụ `_config.json` sau bước test có cờ `test_evaluated: true`. Checkpoint không vào repo, nằm trong output của kernel (bước test đọc qua `kernel_sources`).
