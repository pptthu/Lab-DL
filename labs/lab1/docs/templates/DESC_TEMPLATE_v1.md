# Khuôn desc (v1)

- Tác giả: Thư | Tạo: 2026-11-13T11:22:06+07:00 | Cập nhật: Thư, 2026-11-13T17:21:44+07:00
- Dùng cho: desc của cả lab (`docs/lab1_description_vN.md`) và desc từng module (`docs/phases/<module>_vN.md`).

## Cách dùng
- Copy khối bên dưới, thay phần `<...>`. Giữ khoảng 20 dòng.
- Người viết dàn ý (Purpose, Success criteria, phần "Không làm"). AI chỉ chi tiết hoá rồi đưa người duyệt.
- Success criteria phải có con số và tập dữ liệu cụ thể, không ghi chung chung kiểu "chạy tốt".
- How-to-do: mỗi bước là một small prompt.
- Đổi nội dung thì lưu bản mới `_vN+1`, không sửa đè bản cũ.

## Khuôn

    # <Lab/Module> — <tên ngắn>  (v<N>)
    - Tác giả: <tên> | Tạo: YYYY-MM-DDTHH:MM:SS+07:00 | Cập nhật: <tên>, <thời điểm>
    - Trạng thái: To Do | In Progress | Done
    ## Background
    <1–3 câu: vì sao cần bước này>
    ## Purpose
    - Làm: <kết quả cụ thể>
    - Không làm: <ngoài phạm vi>
    ## Input / Output
    - Input: <đường dẫn> (<định dạng, shape>)
    - Output: <đường dẫn> (<định dạng>)
    - Code: labs/lab1/<module>/<file>.py
    ## Success criteria
    - <chỉ số> <ngưỡng> trên <tập dữ liệu>
    - <file phải tồn tại / nạp lại được>
    ## How-to-do
    1. <bước 1 = 1 small prompt>
    2. ...
    ## Quyết định (người ghi)
    - <ngày>: <quyết định> — <lý do>
