# Sổ ghi thí nghiệm Lab 3

- **Created**: 2026-11-20T02:27:41+07:00 (pptthu)
- **Last Updated**: 2026-11-20T02:27:41+07:00 (pptthu)
- Mỗi giai đoạn của plan §4 (S1–S5) một mục: chạy gì, số liệu (kèm file), quyết định, người chốt. Số liệu lấy từ file trong `outputs/`, không chép tay từ màn hình.

---

## S1 — EDA trên dữ liệu gốc (2026-11-20, máy Thư, CPU)

**Lệnh:** `python -m labs.lab3.eda.eda` (51 giây; dữ liệu đọc từ bộ đệm parquet). Tokenizer `distilbert-base-uncased` (commit `12040acc…`), đếm token gồm `[CLS]`/`[SEP]`, không cắt.
**File:** `outputs/results/eda_summary.csv`, `eda_duplicates.csv`, `eda_ratings.csv`, `eda_token_lengths.csv`; hình `outputs/figures/eda_*.png`.

### Số liệu chính

| | train | test |
|---|---|---|
| Số review (neg / pos) | 25.000 (12.500 / 12.500) | 25.000 (12.500 / 12.500) |
| Số phim / tối đa review mỗi phim | 3.456 / 30 | 3.581 / 30 |
| Độ dài token: median / trung bình | 222 / 297,6 | 219 / 290,8 |
| P90 / P95 / max | 584 / 763 / 3.047 | 565 / 743 / 2.875 |
| % review > 128 / > 256 / > 512 token | 87,9 / 41,1 / 13,3 | 87,4 / 40,1 / 12,3 |
| % review có thẻ `<br />` / số token `<br />` trung bình mỗi review | 58,7 / 16,3 | 58,1 / 16,0 |
| % review có ít nhất một từ phủ định | 89,4 | 89,1 |

Đối chiếu: Sun et al. 2019 (Table 1) báo 12,69% review IMDb dài hơn 512 token — khớp với 12,3–13,3% đo ở đây (khác tokenizer cased/uncased và bước làm sạch nên lệch nhẹ).

**Trùng lặp** (`eda_duplicates.csv`, so khớp văn bản đã làm sạch): train 98 bản thừa (192 review nằm trong nhóm trùng), test 201 bản thừa (394 review); **123 review test có nội dung trùng một review train** (0,49% test); 0 nhóm trùng trong train có nhãn mâu thuẫn.

**Phim ở cả train và test:** 1 phim (`tt0406816`, *The Guardian*, 2006): 2 review train (`train_neg_12498`, `train_neg_12499`) và 23 review test. README ghi hai tập *"contain a disjoint set of movies"* → gần đúng, có 2/25.000 review ở ranh giới. Nội dung review khớp phim (Coast Guard, Costner, Kutcher) → xác nhận cách đọc ID phim (dòng i của `urls_*.txt` = review id i) là đúng.

### Hệ quả cho các module
- **`preprocessing` (làm sạch):** 58% review có `<br />`, mỗi review tốn trung bình 16 token cho thẻ HTML → bỏ thẻ là có căn cứ (ở `max_length` 256, 16 token ≈ 6% chỗ).
- **`preprocessing` (chia):** review trùng tồn tại thật → gộp nhóm khi chia train/val là cần thiết (đã làm: 18 phim được gộp nhóm vì có review trùng).
- **`evaluate`:** 123 review test trùng train → báo thêm acc trên test bỏ 123 review này (quyết định đã duyệt 2026-11-20).
- **`visualize`:** 89% review có từ phủ định → xác nhận bỏ nhóm "có phủ định" trong phân tích lỗi là đúng (nhóm gần như là cả tập).

### Đề xuất `max_length` cho lần chạy chính: **512** `[đề xuất — chờ Thư chốt]`

Chi phí đo trên độ dài thật của train (batch 16, pad theo batch, `outputs/results/eda_token_lengths.csv`):

| `max_length` | Review giữ nguyên (không bị cắt) | % token của train được giữ | Độ dài batch trung bình | Token xử lý mỗi epoch (so với 256) |
|---|---|---|---|---|
| 128 | 12,1% | 41,5% | 128 | 0,5× |
| 256 | 58,9% | 68,3% | 256 | 1× |
| 512 | 86,7% | 89,0% | 505 | ≈ 2× |

Lý do chọn 512:
1. **Ít mất thông tin nhất:** cắt ở 256 làm 41% review mất phần sau (giữ 68% token); 512 chỉ cắt 13% (giữ 89%). Sun et al. 2019 cho thấy trên IMDb phần văn bản giữ lại ảnh hưởng accuracy (Table 2: head-only 510 token lỗi 5,63%, head+tail 5,42%).
2. **So sánh công bằng với Bài 1:** pipeline zero-shot cắt ở 512 (giới hạn của DistilBERT). Fine-tune ở 256 mà thua/thắng zero-shot thì không tách được ảnh hưởng của độ dài.
3. **Chi phí chấp nhận được:** ≈ 2 lần số token so với 256 (DistilBERT ở 512 token chi phí chủ yếu ở lớp tuyến tính nên thời gian tăng gần tuyến tính — suy luận); chạy trên Kaggle GPU. Thời gian mỗi epoch đo ở smoke run trên Kaggle (Member B) trước pilot.
4. Thí nghiệm `max_length` (S4) so 128 và 256 với 512 → trả lời trực tiếp "cắt review mất bao nhiêu accuracy".

**Phương án dự phòng:** nếu smoke trên Kaggle cho thấy 512 không vừa VRAM với batch 16 hoặc quá chậm cho 1 pilot + 3 seed trong quota GPU, dùng 256 (mức phổ biến trong các model card DistilBERT IMDB, vd Niophy 91,2%) và ghi lý do vào mục S2.

`labs/lab3/config.py`: `MAX_LENGTH = 512` (ghi chú "đề xuất, chờ Thư chốt").

---

## S2 — Pilot (chưa chạy, Member B)
## S3 — Lần chạy chính 3 seed (chưa chạy, Member B)
## S4 — Thí nghiệm `max_length` (chưa chạy, Member B)
## S5 — Test một lần (chưa chạy, Member B)
