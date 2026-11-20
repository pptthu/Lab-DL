# Khuôn desc cả lab (v1)

- Tác giả: Thư | Tạo: 2026-11-19T17:00:38+07:00 | Cập nhật: Thư, 2026-11-19T17:00:38+07:00
- Dùng cho: desc của cả Lab 3 (`docs/lab3_description_vN.md`). Desc từng module dùng `DESC_TEMPLATE_v3.md`.
- Nguồn: chép từ `labs/lab1/docs/templates/DESC_TEMPLATE_v1.md`, giữ tên file để khớp `rules/RULES_v1.md`; chỉ đổi đường dẫn sang Lab 3 và thêm dòng "Căn cứ".

## Cách dùng
- Copy khối bên dưới, thay phần `<...>`. Giữ khoảng 20–30 dòng.
- Thư viết dàn ý (Purpose, Success criteria, phần "Không làm"). AI chỉ chi tiết hoá rồi đưa Thư duyệt.
- Success criteria phải có con số và tập dữ liệu cụ thể, không ghi chung chung kiểu "chạy tốt".
- Mỗi quyết định có căn cứ (bài báo, tài liệu HF, kết quả đo), ghi ở cột "Căn cứ" hoặc sau dấu —.
- How-to-do: mỗi bước là một small prompt.
- Đổi nội dung thì hỏi Thư: sửa đè hay lưu `_vN+1`.

## Khuôn

    # <Lab/Module> — <tên ngắn>  (v<N>)
    - Tác giả: <tên> | Tạo: YYYY-MM-DDTHH:MM:SS+07:00 | Cập nhật: <tên>, <thời điểm>
    - Trạng thái: To Do | In Progress | Done
    ## Background
    <1–3 câu: vì sao cần bước này, ứng với phần nào của đề>
    ## Purpose
    - Làm: <kết quả cụ thể>
    - Không làm: <ngoài phạm vi>
    ## Input / Output
    - Input: <đường dẫn> (<định dạng, kích thước>)
    - Output: <đường dẫn> (<định dạng>)
    - Code: labs/lab3/<module>/<file>.py
    ## Success criteria
    - <chỉ số> <ngưỡng> trên <tập dữ liệu> — <căn cứ của ngưỡng>
    - <file phải tồn tại / nạp lại được>
    ## How-to-do
    1. <bước 1 = 1 small prompt>
    2. ...
    ## Quyết định (người ghi)
    - <ngày>: <quyết định> — <căn cứ>
