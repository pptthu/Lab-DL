# Khuôn phase doc module (v3)

- **Created**: 2026-11-13T23:32:00+07:00
- **Last Updated**: 2026-11-13T23:32:00+07:00 (Thư)
- Dùng cho: desc từng module `docs/phases/<module>_vN.md`. Desc cả lab vẫn dùng `DESC_TEMPLATE_v1.md`.
- Thay đổi so với v2: theo khuôn phase của M (`M/deeplearning_LAB_submission/docs/lab2/phases/PHASE_TEMPLATE.md`, `MODEL.md`, `TRAINING_INFO.md`): thêm Name, Detailed plan; gộp Interface, Gotchas vào Detailed plan; bằng chứng chạy trỏ sang `progress_status.md`; thêm cách viết Goals / Không làm / Success criteria. v1, v2 giữ nguyên.

## Cách dùng
- Copy khối bên dưới, thay phần `<...>`. Khoảng 40–80 dòng.
- **Goals / Purpose** (`[đề xuất]` tới khi duyệt): 2–4 dòng. Mỗi dòng là một **kết quả kiểm được** (file nào, hàm trả gì, con số nào), không phải một hoạt động. Viết "train_model trả best_epoch, stop_reason", không viết "huấn luyện model".
- **Không làm:** chỉ ghi những việc người đọc hoặc AI **dễ tưởng** thuộc module này; mỗi dòng nói rõ việc đó thuộc module nào hoặc vì sao ngoài đề. Không ghi câu phủ định của goal (kiểu "không bị lỗi").
- **Success criteria** (`[đề xuất]` tới khi duyệt): mỗi tiêu chí kiểm được bằng **một lệnh hoặc một file**, có số, tập dữ liệu, nơi kiểm. Chia 3 tầng: Test → Smoke run → Normal run. Ngưỡng accuracy chỉ đặt ở desc lab và kiểm ở `evaluate`; module không tự đặt ngưỡng mới.
- Input / Output ghi đường dẫn code đang dùng thật, kèm "Sẽ chuyển sang … ở giai đoạn O5". Không ghi đường dẫn mới như thể đã đổi.
- How to do it: mỗi bước một small prompt, ghi rõ tham số; luôn có bước smoke run trước normal run.
- Detailed plan: tham số cụ thể kèm lý do; hàm hiện có lấy đúng chữ ký kèm `file.py:dòng`; gotcha lấy từ code và ghi chép cũ, ghi nguồn, không bịa.
- Số liệu chạy chỉ ghi khi có bằng chứng (log, csv, test), kèm đường dẫn; chưa có thì ghi "chưa chạy".

## Khuôn

    # <module> — <tên ngắn> (v<N>)

    - **Created**: YYYY-MM-DDTHH:MM:SS+07:00
    - **Last Updated**: YYYY-MM-DDTHH:MM:SS+07:00 (<người cập nhật>)
    - **Người phụ trách**: <Member X (tên)>
    - **Status**: To Do | In Progress | Done

    ---

    ## Name
    <Tên phase — các việc chính>

    ## Background
    <Ứng với phần nào của đề; module này dùng cho ai; bản cũ thiếu gì>

    ## Goals / Purpose  [đề xuất]
    - <kết quả kiểm được>
    - Không làm:
      - <việc dễ tưởng thuộc module này> — <thuộc module nào / vì sao ngoài đề>

    ## Input / Output
    - **Input:** <từ module nào, định dạng, shape>
    - **Output:** <file / đối tượng, tên file>
      - Hiện code lưu ở <đường dẫn> (`<file>.py:<dòng>`). Sẽ chuyển sang <đường dẫn mới> ở giai đoạn O5.

    ## How to do it (general plan)
    1. <bước = 1 small prompt, có tham số>
    2. ...
    n. Smoke run → kiểm <…>
    n+1. Normal run

    ## Pipeline
    ```
    <module trước> → <hàm> → <bước> → <output> → <module sau>
    ```

    ## Detailed plan
    - **<Tham số / quyết định>:** <giá trị> — <lý do>
    - **Hàm hiện có:** `<chữ ký đúng như code>` (`<file>.py:<dòng>`)
    - **Chỗ code chưa khớp plan:** <…> (`<file>.py:<dòng>`)
    - **Gotchas:**
      - <lỗi dễ gặp / giới hạn> (<nguồn>)

    ## Success criteria  [đề xuất]
    - **Test:** <lệnh pytest> đạt, gồm <test cần có>
    - **Smoke run:** <file nào phải có, kiểm gì>
    - **Normal run:** <chỉ số / file phải có>

    ## Links
    - Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md) | Khuôn liên quan: <…>
    - Module liên quan: [<module>_v1.md](<module>_v1.md)
    - Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (<số liệu, dòng>)
    - Code: `labs/lab1/<module>/<file>.py` | Test: `labs/lab1/tests/test_<module>.py`
