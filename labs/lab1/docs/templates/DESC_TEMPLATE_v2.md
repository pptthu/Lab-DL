# Khuôn desc (v2)

- Tác giả: Thư | Tạo: 2026-11-13T17:19:00+07:00 | Cập nhật: Thư, 2026-11-13T17:19:00+07:00
- Dùng cho: desc từng module `docs/phases/<module>_vN.md`. Desc cả lab (`lab1_description_vN.md`) vẫn dùng `DESC_TEMPLATE_v1.md`.
- Thay đổi so với v1: thêm 4 mục Interface, Pipeline, Run & Test, Gotchas và mục Links; thêm dòng Người phụ trách; sắp lại thứ tự mục. v1 giữ nguyên.

## Cách dùng
- Copy khối bên dưới, thay phần `<...>`. Desc module khoảng 40–80 dòng.
- Người viết dàn ý Purpose, Không làm, Success criteria. Nếu AI viết nháp thì đánh dấu `[đề xuất]` cho tới khi người duyệt.
- Người phụ trách: lấy theo phân công của leader (với 9 module cũ: trường Owner của file cũ).
- Input / Output: ghi đường dẫn code đang dùng thật. Nếu đường dẫn sẽ đổi, thêm dòng "Sẽ chuyển sang <đường dẫn mới> ở giai đoạn O5." Không ghi đường dẫn mới như thể đã đổi.
- Interface: chép đúng chữ ký hàm trong code, kèm `file.py:dòng`.
- Run & Test: chỉ ghi lệnh và kết quả có bằng chứng (log, csv, test). Chưa có thì ghi "chưa chạy".
- Gotchas: lấy từ code và ghi chép cũ, ghi nguồn. Không bịa thêm.
- How-to-do: mỗi bước là một small prompt.
- Đổi nội dung thì lưu bản `_vN+1`, không sửa đè bản cũ.

## Khuôn

    # <Module> — <tên ngắn>  (v<N>)
    - Tác giả: <tên> | Tạo: YYYY-MM-DDTHH:MM:SS+07:00 | Cập nhật: <tên>, <thời điểm>
    - Người phụ trách: <Member X (tên)>
    - Trạng thái: To Do | In Progress | Done
    ## Background
    <1–3 câu: vì sao cần module này, đề yêu cầu gì>
    ## Purpose
    - Làm: <kết quả cụ thể>
    - Không làm: <ngoài phạm vi>
    ## Input / Output
    - Input: <đường dẫn đang dùng> (<định dạng, shape>)
    - Output: <đường dẫn đang dùng> (<định dạng>)
    - Sẽ chuyển sang <đường dẫn theo rules/FOLDER_STRUCTURE_v1.md> ở giai đoạn O5.
    - Code: labs/lab1/<module>/<file>.py
    ## Interface
    | Hàm | Chữ ký | Dòng |
    |---|---|---|
    | <tên> | `<chữ ký đúng như code>` | `<file>.py:<dòng>` |
    ## Pipeline
        <module trước> → <hàm 1> → <hàm 2> → <output> → <module sau>
    ## How-to-do
    1. <bước 1 = 1 small prompt>
    2. ...
    ## Success criteria
    - <chỉ số> <ngưỡng> trên <tập dữ liệu>
    - <file phải tồn tại / nạp lại được>
    ## Run & Test
    | Bước | Lệnh | Kết quả | Bằng chứng |
    |---|---|---|---|
    | Smoke run | <lệnh> | <kết quả> hoặc "chưa chạy" | <log/csv> |
    | Normal run | <lệnh> | <kết quả> hoặc "chưa chạy" | <log/csv> |
    | Test | python -m pytest labs/lab1/tests/test_<module>.py -q | <kết quả> hoặc "chưa chạy" | <ngày chạy> |
    ## Gotchas
    - <lỗi dễ gặp / giới hạn> (nguồn: <file:dòng hoặc ghi chép cũ>)
    ## Quyết định (người ghi)
    - <ngày>: <quyết định> — <lý do>
    ## Links
    - Desc lab: docs/lab1_description_vN.md | Module liên quan: phases/<module>_vN.md
    - Test: labs/lab1/tests/test_<module>.py
