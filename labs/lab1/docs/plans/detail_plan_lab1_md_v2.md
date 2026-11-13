# Detail plan v2 — Cập nhật tài liệu yêu cầu Lab 1

- **Tác giả:** Thư duyệt; AI soạn theo thảo luận
- **Tạo:** 2026-11-14T00:34:29+07:00 | **Cập nhật:** 2026-11-14T00:55:48+07:00 (v2)
- **Trạng thái:** bản đang dùng. v0, v1 giữ nguyên.
- **Phạm vi:** chỉ tài liệu md của Lab 1 (`labs/lab1/`). Không sửa code, không đổi thư mục code/dữ liệu/kết quả.

## 1. Quyết định thêm (sau O3–O12)

| # | Ngày | Quyết định |
|---|---|---|
| D1 | 13/11 | Smoke run = chạy cả pipeline 1 epoch, 2 batch, seed 42; đầu ra vào `outputs/_smoke/`, không dùng để báo cáo (`rules/RULES_v1.md` bước 7) |
| D2 | 13/11 | Desc module dùng `templates/DESC_TEMPLATE_v3.md` (khuôn phase của M + Success criteria). Desc cả lab vẫn dùng v1. v2 giữ theo luật version, không dùng |
| D3 | 13/11 | Tên file module: `docs/phases/<module>_vN.md`; `git mv` trong commit riêng để giữ lịch sử |
| D4 | 13/11 | Goals / Không làm / Success criteria: AI đề xuất theo cách viết trong DESC_TEMPLATE_v3, đánh dấu `[đề xuất]`, Thư duyệt |
| D5 | 13/11 | Status module: "Done (theo yêu cầu cũ) — chờ sửa theo yêu cầu mới" cho tới khi code sửa xong |
| D6 | 13/11 | `logbook/` của Lab 1 chuyển vào `docs/experiments/` (PR #65); Lab 2, 3 giữ `logbook/` |
| D7 | 13/11 | Lab 1: mọi file md và ô markdown notebook chỉ tiếng Việt, dịch từ phần tiếng Anh (đầy đủ hơn). Code `.py` giữ nguyên; chú thích mới viết tiếng Việt. Lab 2, 3 chưa đổi |
| D8 | 14/11 | Mọi file md có ngày tạo và ngày cập nhật đủ giờ phút giây, ghi thời điểm thật. Ngày tạo: commit đầu trong git, hoặc mốc sớm nhất có bằng chứng. Tác giả file cũ: `pptthu`. Notebook không cần |
| D9 | 14/11 | Đổi nội dung file có version thì tạo `_vN+1` (`00_README_v2`, `FOLDER_STRUCTURE_v2`) |
| D10 | 13/11 | `project_description.md` (tổng quan để trình bày với cô) viết lại sau khi sửa code xong |
| D11 | 13/11 | Git theo `CONTRIBUTING.md` và template PR: chạy pytest, cập nhật `progress_status.md`, `AI_usage.md` trước khi mở PR |
| D12 | 14/11 | Không dùng chữ "leader" trong tài liệu Lab 1; ghi tên người (Thư) |
| D13 | 14/11 | `evaluate`: test chạy cho cả baseline và cấu hình được chọn trên val, mỗi model đúng một lần |
| D14 | 14/11 | Sơ đồ kiến trúc vẽ bằng mermaid trong model card `docs/models/`; không viết code vẽ, `model.py` giữ nguyên |
| D15 | 14/11 | Ngưỡng test của baseline ≥ 87%, theo MLP tốt nhất trong bài báo gốc Fashion-MNIST (Xiao et al. 2017, Bảng 3: 0,871). Giữ tiêu chí baseline > softmax_regression (bài báo gốc: MLP 0,871 so với LogisticRegression 0,842) |
| D16 | 14/11 | Duyệt Goals / Không làm / Success criteria của 9 file `phases/`; bỏ dấu `[đề xuất]` |
| D17 | 14/11 | Đợt sửa này ghi đè lên file hiện hành, không tạo version mới |

## 2. Các bước đã làm

| Bước | Việc | Kết quả |
|---|---|---|
| a | Khuôn module | `DESC_TEMPLATE_v2` rồi `v3` |
| b | Sửa `DESC_TEMPLATE_v1` dòng 4 | Bỏ `[cần chốt]`, ghi `phases/<module>_vN.md` |
| c | `git mv` 9 file module vào `phases/` | Commit riêng, chỉ đổi tên |
| d | Viết lại 9 file `phases/` | Theo v3; mỗi file có mục "Chỗ code chưa khớp plan" |
| e | Mục lục và cây thư mục | `00_README_v2.md`, `rules/FOLDER_STRUCTURE_v2.md` |
| — | Tiếng Việt | `README.md`, `docs/experiments/experiment_log.md`, 16 ô markdown notebook |
| — | Ngày giờ | Header thật cho 22 file |

## 3. Còn mở

- Không còn mục `[cần chốt]` (đã chốt ở D13–D16)

## 4. Bước tiếp theo

1. Sửa code theo `lab1_description_v1.md` và từng file `phases/`, mỗi module một nhánh `lab1/<module>` (chỉ làm khi Thư cho phép)
2. Đổi thư mục `artifacts/`, `raw_data/` sang `outputs/`, `data/` (O5)
3. Viết model card `docs/models/mlp_baseline_v1.md` (phần Initial model và sơ đồ mermaid)
4. Chạy smoke run → normal run; cập nhật README, phần Trained model của model card, Status module
5. Viết lại `project_description.md` (D10)
