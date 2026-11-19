# Luật làm việc Lab 3 (v1)

- Tác giả: Thư | Tạo: 2026-11-19T16:37:42+07:00 | Cập nhật: Thư, 2026-11-19T16:37:42+07:00
- Áp dụng cho: Thư và AI agent khi làm Lab 3 (nhánh làm lại Lab 3, `labs/lab3/docs/plans/detail_plan_lab3_md_v1.md`). Luật luôn bật, đọc trước mọi việc.

## 1. Quy trình
Mỗi bước (lab hoặc module) đi theo thứ tự, không nhảy cóc:
1. **Desc:** dàn ý theo `templates/DESC_TEMPLATE_v3.md` (module) hoặc `DESC_TEMPLATE_v1.md` (cả lab).
2. **Discussion + research:** AI hỏi lại chỗ chưa rõ (tối đa 5 câu/lượt). Mỗi đề xuất phải có nguồn (bài báo, tài liệu HF, model card) và nói rõ đâu là biết chắc, đâu là suy luận.
3. **Detail plan:** lưu vào `plans/`.
4. **Duyệt:** Thư duyệt hoặc yêu cầu sửa.
5. **Lưu có version:** `_v0`, `_v1`… Dùng bản số lớn nhất; sửa nội dung thì hỏi Thư sửa đè hay tạo `_vN+1`.
6. **Small prompt:** mỗi prompt làm một bước.
7. **Smoke run:** cả pipeline với vài chục review, vài step, seed 42, chạy trên máy Thư. Đầu ra vào `outputs/_smoke/`, không dùng để báo cáo.
8. **Normal run:** chạy trên Kaggle GPU bằng `main.py`; tải `outputs/` (log, csv, json, hình) về repo. Log theo `templates/LOG_TEMPLATE_v1.md`, ghi thêm GPU, phiên bản thư viện, thời gian chạy.
9. **Audit:** đối chiếu code và kết quả với desc. Chỉ Done khi audit xong.

## 2. Luật chung
- Số chưa đo được thì không chốt trước: làm bước đo rồi mới chốt (`max_length` sau EDA, số epoch sau pilot, cách làm thí nghiệm `max_length` sau 3 seed).
- Tập test 25k chỉ dùng một lần cho mỗi model, sau khi đã chọn xong trên validation.
- Không để kết quả trong notebook. Notebook chỉ đọc `outputs/`, không train.
- Số liệu trong tài liệu phải lấy từ file kết quả của một lần chạy có log, ghi đường dẫn file.
- Mỗi thay đổi chỉ sửa một file nguồn: tham số đặt ở một chỗ, nơi khác đọc từ đó.
- Tài liệu tiếng Việt, giữ thuật ngữ tiếng Anh, không song ngữ; có Created/Last Updated đủ giờ phút giây (ngoại lệ so với `Lab-DL/AGENTS.md` dòng 6, như Lab 1).
- Không dùng lại code, số liệu hay tài liệu của Lab 3 cũ.

## 3. Phân vai người và AI
- Thư viết/duyệt desc, mục tiêu, success criteria, phần "không làm". AI chi tiết hoá và research.
- Thư quyết định, ghi kèm ngày và lý do. AI đánh dấu `[đề xuất]` / `[cần chốt]` cho tới khi được duyệt.
- Thư duyệt ở 4 cổng: desc, detail plan, smoke run, audit.
- AI không tự thêm thí nghiệm, không đổi cấu trúc ngoài plan đã duyệt. Chưa chắc thì hỏi.
