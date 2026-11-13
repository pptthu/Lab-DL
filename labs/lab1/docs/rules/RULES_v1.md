# Luật làm việc Lab 1 (v1)

- Tác giả: Thư | Tạo: 2026-11-13T11:22:06+07:00 | Cập nhật: Thư, 2026-11-13T11:22:06+07:00
- Áp dụng cho: cả nhóm và AI agent khi làm Lab 1. Luật này luôn bật, đọc trước mọi việc.

## 1. Quy trình
Mỗi bước (lab hoặc module) đi theo thứ tự sau, không nhảy cóc:
1. **Desc:** người viết dàn ý theo `templates/DESC_TEMPLATE_v1.md`.
2. **Discussion:** AI hỏi lại chỗ chưa rõ, mỗi lượt tối đa 5 câu.
3. **Detail plan:** AI chi tiết hoá từ desc, lưu vào `plans/`.
4. **Duyệt:** leader đọc plan và nói duyệt hay sửa.
5. **Lưu có version:** tên file `_v0`, `_v1`… Luôn dùng bản số lớn nhất, bản cũ giữ nguyên.
6. **Small prompt:** mỗi prompt chỉ làm một bước (ví dụ: load data, rồi build model).
7. **Smoke run (= dry run):** chạy cả pipeline với 1 epoch, 2 batch, seed 42 (giống `--smoke` của M). Mọi đầu ra ghi vào `outputs/_smoke/`, không ghi vào `outputs/models|logs|results`. Đạt khi: chạy hết không lỗi, log có đủ 3 phần (đầu / mỗi epoch / cuối), checkpoint nạp lại dự đoán được. Số liệu của smoke run không dùng để báo cáo.
8. **Normal run:** chạy thật, sinh log theo `templates/LOG_TEMPLATE_v1.md`.
9. **Audit:** đối chiếu code và kết quả với desc. Chỉ đánh dấu Done khi audit xong; lỗi nào không sửa thì ghi lý do vào desc.

## 2. Luật chung
- Trước khi lưu README, AI phải nói lại nó hiểu luật thế nào. Leader đọc thấy đúng mới cho lưu.
- Không để kết quả trong notebook. Code phải lưu model, log, hình, số liệu vào `outputs/` (xem `FOLDER_STRUCTURE_v1.md`) để làm báo cáo mà không phải chạy lại.
- Mỗi thay đổi chỉ sửa một file nguồn. Đổi tham số thì sửa ở một chỗ, các file khác đọc từ đó.
- Số liệu ghi trong tài liệu phải lấy từ file kết quả của một lần chạy có log và ghi đường dẫn file đó.
- Tài liệu Lab 1 viết tiếng Việt, giữ thuật ngữ tiếng Anh, không song ngữ. Đây là ngoại lệ so với `Lab-DL/AGENTS.md` (dòng 6: "Documents and docstrings: English first, Vietnamese below"), chỉ áp dụng cho tài liệu trong `labs/lab1/docs/`.

## 3. Phân vai người và AI
- Người viết dàn ý desc, mục tiêu, success criteria và phần "không làm". AI chỉ chi tiết hoá.
- Người quyết định, ghi kèm ngày và lý do. AI đề xuất và đánh dấu `[đề xuất]` hoặc `[cần chốt]` cho tới khi được duyệt.
- Người duyệt ở 4 cổng: desc, detail plan, smoke run, audit.
- AI không tự thêm thí nghiệm, không đổi cấu trúc thư mục ngoài plan đã duyệt.
- Chưa chắc thì AI hỏi, không đoán.
