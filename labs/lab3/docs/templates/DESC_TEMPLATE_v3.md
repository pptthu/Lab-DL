# Khuôn phase doc module (v3)

- **Created**: 2026-11-19T17:00:38+07:00
- **Last Updated**: 2026-11-19T17:00:38+07:00 (Thư)
- Dùng cho: desc từng module `docs/phases/<module>_vN.md`. Desc cả lab dùng `DESC_TEMPLATE_v1.md`.
- Nguồn: chép từ `labs/lab1/docs/templates/DESC_TEMPLATE_v3.md` (khuôn phase của M + Success criteria), giữ tên file để khớp `rules/RULES_v1.md`. Đổi so với Lab 1: đường dẫn sang Lab 3; bỏ ghi chú "giai đoạn O5" (Lab 3 dùng `data/`, `outputs/` ngay từ đầu); thêm yêu cầu ghi nguồn cho tham số; Normal run chạy trên Kaggle.

## Cách dùng
- Copy khối bên dưới, thay phần `<...>`. Khoảng 40–80 dòng.
- **Goals / Purpose** (`[đề xuất]` tới khi duyệt): 2–4 dòng. Mỗi dòng là một **kết quả kiểm được** (file nào, hàm trả gì, con số nào), không phải một hoạt động. Viết "`train_model` trả `best_step`, `stop_reason`", không viết "huấn luyện model".
- **Không làm:** chỉ ghi những việc người đọc hoặc AI **dễ tưởng** thuộc module này; mỗi dòng nói rõ việc đó thuộc module nào hoặc vì sao ngoài đề. Không ghi câu phủ định của goal.
- **Success criteria** (`[đề xuất]` tới khi duyệt): mỗi tiêu chí kiểm được bằng **một lệnh hoặc một file**, có số, tập dữ liệu, nơi kiểm. Chia 3 tầng: Test → Smoke run (máy Thư) → Normal run (Kaggle). Ngưỡng accuracy chỉ đặt ở desc lab và kiểm ở `evaluate`; module không tự đặt ngưỡng mới.
- Input / Output ghi đường dẫn code dùng thật (`shared/paths.py`: `data_dir`, `output_dir`).
- How to do it: mỗi bước một small prompt, ghi rõ tham số; luôn có smoke run trước normal run.
- Detailed plan: tham số cụ thể kèm **căn cứ** (bài báo, tài liệu HF, kết quả đo ở bước trước). Tham số chưa đo được thì ghi `[cần chốt sau <bước>]`, không đoán. Hàm lấy đúng chữ ký kèm `file.py:dòng`; gotcha ghi nguồn, không bịa.
- Số liệu chạy chỉ ghi khi có bằng chứng (log, csv, test), kèm đường dẫn; chưa có thì ghi "chưa chạy".

## Khuôn

    # <module> — <tên ngắn> (v<N>)

    - **Created**: YYYY-MM-DDTHH:MM:SS+07:00
    - **Last Updated**: YYYY-MM-DDTHH:MM:SS+07:00 (<người cập nhật>)
    - **Người phụ trách**: <tên>
    - **Status**: To Do | In Progress | Done

    ---

    ## Name
    <Tên phase — các việc chính>

    ## Background
    <Ứng với bước nào của đề (Bài 1.x / Bài 2.x); module này dùng cho module nào>

    ## Goals / Purpose  [đề xuất]
    - <kết quả kiểm được>
    - Không làm:
      - <việc dễ tưởng thuộc module này> — <thuộc module nào / vì sao ngoài đề>

    ## Input / Output
    - **Input:** <từ module nào, định dạng, kích thước>
    - **Output:** <file / đối tượng, đường dẫn trong `data/` hoặc `outputs/`>

    ## How to do it (general plan)
    1. <bước = 1 small prompt, có tham số>
    2. ...
    n. Smoke run (máy Thư) → kiểm <…>
    n+1. Normal run (Kaggle)

    ## Pipeline
    ```
    <module trước> → <hàm> → <bước> → <output> → <module sau>
    ```

    ## Detailed plan
    - **<Tham số / quyết định>:** <giá trị> — <căn cứ>
    - **Hàm:** `<chữ ký đúng như code>` (`<file>.py:<dòng>`)
    - **Chỗ code chưa khớp plan:** <…> (`<file>.py:<dòng>`)
    - **Gotchas:**
      - <lỗi dễ gặp / giới hạn> (<nguồn>)

    ## Success criteria  [đề xuất]
    - **Test:** <lệnh pytest> đạt, gồm <test cần có>
    - **Smoke run:** <file nào phải có trong `outputs/_smoke/`, kiểm gì>
    - **Normal run:** <chỉ số / file phải có trong `outputs/`>

    ## Links
    - Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
    - Module liên quan: [<module>_v1.md](<module>_v1.md)
    - Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md)
    - Code: `labs/lab3/<module>/<module>.py` | Test: `labs/lab3/tests/test_<module>.py`
