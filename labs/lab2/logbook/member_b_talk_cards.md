# Thẻ nói lại — Member B (Bảo), Lab 2

> Mục đích: mỗi thẻ là một câu trả lời **nói ra miệng** được, không đọc code.
> Ba mức: **10 giây** (một câu) → **30 giây** (3–4 câu, có ví dụ) → **60 giây** (tóm tắt cả phần).
> Chỉ dùng số đã có trong "Bảng số được phép nói" của file `member_b_presentation_plan.md`.

---

## A. Mức 10 giây — một hàm, một câu

| Hàm | Câu nói |
|---|---|
| `train_model` | "Vòng lặp train rồi validation mỗi epoch, cuối cùng lưu phiên bản có validation loss thấp nhất." |
| `save_checkpoint` | "Tạo thư mục cha rồi lưu trọng số ra file." |
| `load_checkpoint` | "Nạp trọng số lên đúng thiết bị và chuyển model sang chế độ đánh giá." |
| `evaluate` | "Chạy model trên tập đánh giá và trả về loss trung bình cùng tỉ lệ đoán đúng." |
| `confusion_matrix` | "Đếm xem mỗi lớp thật bị đoán thành lớp nào." |
| `top_confusions` | "Lấy ra những cặp lớp bị nhầm nhiều nhất, bỏ qua các dự đoán đúng." |
| `make_summary_writer` | "Tạo kênh ghi TensorBoard; máy không cài thì trả về `None` để chương trình vẫn chạy." |
| `_resolve_best_metrics` | "Lấy đúng con số của epoch đã được lưu thành file." |
| `run_comparison` | "Chạy nhiều cấu hình, mỗi cấu hình một thư mục log và một checkpoint riêng." |
| `save_results` | "Đổ bảng so sánh ra CSV, mỗi siêu tham số thành một cột." |

---

## B. Mức 30 giây — nói cho người chưa đọc code hiểu

**`train_model`**
> "Mỗi epoch, em cho model học hết tập train, rồi chấm điểm trên tập validation. Điểm chấm là loss. Em giữ lại phiên bản có loss thấp nhất — giống giữ bài làm tốt nhất trong các lần thi thử. Cuối cùng em trả về lịch sử từng epoch, kèm con số của phiên bản đã lưu, để báo cáo và file không bị lệch nhau."

**`save_checkpoint` / `load_checkpoint`**
> "File checkpoint chỉ chứa trọng số, không chứa kiến trúc mạng và không chứa chế độ train hay eval. Nên khi lưu em tạo thư mục trước, còn khi nạp em phải tự đặt lại chế độ đánh giá. Em nạp bằng `weights_only=True` để không chạy mã lạ từ file."

**`evaluate`**
> "Em cho model chạy trên tập không dùng để học, tắt mọi cập nhật, rồi tính hai số: loss trung bình và tỉ lệ đoán đúng. Hai số này tính theo số ảnh, nên batch cuối lẻ vẫn không làm sai kết quả."

**`confusion_matrix`**
> "Đây là bảng 10 nhân 10. Hàng là nhãn thật, cột là nhãn model đoán. Ô giao của cat và dog đếm số ảnh cat bị đoán thành dog. Đường chéo là những ảnh đoán đúng."

**`top_confusions`**
> "Em bỏ đường chéo đi vì đó là đoán đúng, chỉ giữ phần sai. Rồi sắp các ô sai từ nhiều xuống ít, lấy ra vài cặp đầu. Nhìn bảng đó là biết model yếu ở nhóm lớp nào."

**`make_summary_writer`**
> "TensorBoard là công cụ theo dõi, không bắt buộc. Em import nó ngay trong hàm, nên máy nào chưa cài thì hàm trả `None` và mọi thứ vẫn chạy, chỉ là không có biểu đồ."

**`_resolve_best_metrics`**
> "Có một cái bẫy: epoch có accuracy cao nhất có thể không phải epoch đã lưu file. Nên em không lấy accuracy cao nhất, mà lấy accuracy của đúng epoch đã lưu. Hàm này làm việc đó."

**`run_comparison`**
> "Nó là một vòng lặp hai lớp: mỗi model, mỗi bộ siêu tham số là một lần train riêng. Mỗi lần có thư mục log riêng, checkpoint riêng, và được bấm giờ riêng. Chạy xong thì gom thành một danh sách kết quả."

**`save_results`**
> "Mỗi siêu tham số thành một cột có tiền tố `hp_`, để cột nhập và cột kết quả không lẫn nhau. Thứ tự cột cố định, nên hai lần chạy so sánh được với nhau."

---

## C. Mức 30 giây — khái niệm (không phải hàm)

**Transfer learning**
> "Em lấy mạng đã học trên ImageNet, giữ nguyên phần trích đặc trưng, chỉ thay lớp cuối cho 10 lớp của CIFAR-10. Như vậy chỉ khoảng 5.130 trên 11.181.642 tham số là phải học, tức 0,05%."

**Fine-tuning**
> "Thay vì đóng băng tất cả, em mở khóa thêm khối cuối để nó chỉnh theo dữ liệu mới. Vì sửa cả trọng số đã học nên phải dùng learning rate nhỏ để không phá đặc trưng cũ."

**Vì sao chuẩn hóa theo ImageNet**
> "Mạng được học với ảnh đã trừ mean và chia std của ImageNet. Đưa ảnh cùng chuẩn đó vào thì các bộ lọc mới 'nhìn' đúng; lệch chuẩn là đặc trưng sai ngay từ lớp đầu."

**Vì sao ảnh 128 thay vì 224**
> "Máy chỉ có CPU, 224 rất chậm. 128 nhanh hơn đáng kể mà vẫn đủ để backbone nhận ra đặc trưng. CIFAR-10 gốc chỉ 32 nhân 32, nên phóng to lên là nội suy — có nhiễu, em nói thật điểm này."

**Checkpoint**
> "Là file trọng số tại một thời điểm. Em chọn thời điểm có validation loss thấp nhất, và ghi lại cả accuracy của chính thời điểm đó."

**TensorBoard**
> "Ba đường: loss train, loss validation, accuracy validation. Trục hoành là epoch. Mỗi cấu hình là một màu, nên so sánh bằng cách bật tắt từng run."

**Vì sao validation khác test**
> "Validation dùng để chọn: chọn model, chọn siêu tham số. Test chỉ dùng một lần, cho model đã chọn, để báo cáo. Nếu chọn trên test thì con số không còn khách quan nữa."

**Vì sao 1 seed là hạn chế**
> "Mỗi cấu hình em chỉ chạy một lần. Chênh nhau khoảng một điểm có thể chỉ do may mắn. Muốn kết luận cấu hình này hơn cấu hình kia thì phải chạy nhiều seed rồi so trung bình."

---

## D. Mức 60 giây — tóm tắt cả ba module (khi bị hỏi "phần em làm gì")

> "Em làm ba module nối tiếp nhau. `train` nhận DataLoader và model, chạy train và validation mỗi epoch, giữ checkpoint có validation loss thấp nhất, và ghi ba chỉ số lên TensorBoard. `evaluate` nạp lại checkpoint đó, tính loss, accuracy, ma trận nhầm lẫn và các cặp lớp hay nhầm nhất. `experiment` bọc cả hai: mỗi model và mỗi bộ siêu tham số là một lần train riêng, có log và checkpoint riêng, cuối cùng xuất ra CSV để so sánh.
> Điểm em muốn nhấn là mọi lựa chọn đều dựa trên validation; test chỉ dùng một lần. Và con số em báo cáo luôn là của đúng epoch đã lưu file."

---

## E. Khi bị hỏi ngoài phạm vi — câu nói an toàn

- "Phần này em chưa kiểm tra trực tiếp, em chỉ biết qua tài liệu. Em xin phép mở code kiểm tra lại."
- "Cái đó thuộc module của bạn Đại / phần của Thư. Em không sửa module bạn ấy, nhưng em đọc để hiểu đầu vào đầu ra."
- "Em chưa có số liệu cho câu này. Em chỉ có số của lần chạy đã ghi trong notebook."

## F. Ba câu **không được nói**

1. "Fine-tuning giúp tăng nhẹ" — thực tế chênh lớn; nói "tăng rõ rệt trên validation".
2. "Test của mô hình tốt nhất là 91%" — 91,16% là **validation** của cấu hình fine-tune; test 79,00% là của **baseline**.
3. "Chạy `python -m labs.lab2.main` ra bảng này" — bảng đó từ **lưới trong notebook**, khác lưới mặc định của `main.py`.
