# Sổ ghi thí nghiệm Lab 3

- **Created**: 2026-11-20T02:27:41+07:00 (pptthu)
- **Last Updated**: 2026-11-20T12:33:55+07:00 (Phúc, AI soạn)
- Mọi lần chạy Kaggle (S2–S5) chạy trên tài khoản Kaggle của Thư (`pptt1103`).
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

### `max_length` cho lần chạy chính: **512** (Thư chốt 2026-11-20)

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

`labs/lab3/config.py`: `MAX_LENGTH = 512`. Cũng chốt: giữ split train 22.499 / val 2.501 (không ép tròn 2.500 vì chia theo phim).

---

## S2 — Pilot (Member B: Phúc, AI chạy)

### S2a — Đo thời gian trên Kaggle GPU (2026-11-20)

**Lệnh:** `python -m labs.lab3.main --stage timing` trong kernel Kaggle `pptt1103/lab3-timing` (tài khoản Kaggle của Thư, `pptt1103`) (Tesla T4, 1 GPU, fp16), 300 step (`config.TIMING_MAX_STEPS`), seed 42, `max_length` 512, batch 16, đánh giá val mỗi 75 step.
**File:** `outputs/logs/20261120_095745_timing_seed42.log` (+ `_history.csv`, `_config.json`), `outputs/results/timing_runs.csv`.
**Code:** log ghi `code: 5cd7132…`. Đây là bản code trước khi gỡ các commit theo yêu cầu "không commit" (2026-11-20). Commit này chỉ còn ở nhánh local `backup/member-b-phuc-commits`. Các file Python chạy trên Kaggle giống bản đang stage; chỉ công cụ `labs/lab3/kaggle/kaggle_run.py` (chạy trên máy) khác. Từ pilot trở đi, log ghi `b2c21f7+tree:<mã cây index>`.

| Đo | Giá trị |
|---|---|
| Thời gian train 300 step (gồm 4 lần đánh giá val) | 106,7 s → **0,356 s/step** |
| Số step mỗi epoch (22.499 / 16) | 1.407 |
| Ước tính mỗi epoch (`est_epoch_min`) | **8,3 phút** (hơi cao hơn thực tế vì 300 step có 4 lần đánh giá, một epoch cũng chỉ 4 lần) |
| Val acc ở step 300 (0,21 epoch) | 0,906 (`best_is_last_eval: yes` — bình thường với lần đo ngắn, không dùng để chọn gì) |
| Phiên bản | python 3.13.16, torch 2.11.0+cu130, transformers 5.22.1, datasets 5.1.0, evaluate 0.4.6 |

**Quyết định — trần epoch của pilot: 6** (`config.PILOT_MAX_EPOCHS`), ≈ 50 phút GPU.
- Plan §4 S2: trần phải *trên* mức 2–4 epoch của BERT gốc để thấy đường cong sau điểm tốt nhất.
- Mosbach et al. 2021 (§3.2 dùng 3 epoch; §5.2: *"it is even desirable to train for a larger number of iterations"*) → trần gấp đôi 3 epoch.
- Chi phí: 6 × 8,3 ≈ 50 phút, nằm trong quota 30 giờ/tuần (đo `kaggle quota`: còn 30,00 h trước khi chạy).
- Quy tắc kiểm tra giữ nguyên (LOG_TEMPLATE §3.5): nếu checkpoint tốt nhất rơi vào lần đánh giá cuối → tăng trần, chạy lại pilot.
- Người chốt: AI, theo uỷ quyền của Thư 2026-11-20 — **[AI chốt theo uỷ quyền của Thư 2026-11-20 — Thư đã duyệt 2026-11-20]**.
- Nguồn đã đọc trực tiếp: Mosbach et al. 2021, bản HTML https://arxiv.org/html/2006.04884v3 (§3.2, §5.2).

### S2b — Pilot (2026-11-20)

**Lệnh:** `python -m labs.lab3.main --stage pilot` trong kernel `pptt1103/lab3-pilot` (Tesla T4, 1 GPU, fp16). Seed 42, `max_length` 512, trần 6 epoch, **không dừng sớm**, đánh giá val mỗi 352 step (≈ 1/4 epoch). Code `b2c21f7+tree:19a3b6018058`.
**File:** `outputs/logs/20261120_100723_pilot_seed42.log` (+ `_history.csv`, `_config.json`), `outputs/results/pilot_runs.csv`, hình `outputs/figures/pilot_seed42_curves.png`.

| Số liệu (chỉ VAL, không nhìn test) | Giá trị |
|---|---|
| Thời gian | 1.872 s cho 6 epoch → **5,2 phút/epoch** (S2a ước 8,3 vì lần đo ngắn có tỉ lệ đánh giá cao hơn) |
| Checkpoint tốt nhất | step 5.632 (**epoch 4,0**), **val acc 0,9308**, val F1 0,9292 |
| `best_is_last_eval` | **no** (cuối pilot, epoch 6,0: 0,9260) → trần 6 đủ, không phải chạy lại |
| Val acc theo epoch (lần đánh giá cuối mỗi epoch) | 1,0: 0,9212 · 2,0: 0,9204 · 3,0: 0,9224 · 4,0: 0,9308 · 5,0: 0,9260 · 6,0: 0,9260 |
| Val loss | thấp nhất ở epoch 1,0 (0,2110) rồi tăng tới 0,42–0,44 ở epoch 5,5–6 |
| Dao động val acc giữa hai lần đánh giá liền nhau | tới ±0,005–0,015 (vd 0,9212 → 0,9064 ở epoch 1,25); sai số chuẩn của acc ≈ 0,93 với 2.501 mẫu ≈ 0,005 |

**Nhận xét:**
- Val loss tăng từ epoch 1 trong khi val acc vẫn tăng tới epoch 4. Nếu chọn theo val loss thì sẽ dừng ở epoch 1 (acc 0,9212). Kết quả này xác nhận quyết định L8 (chọn theo val acc).
- Đường val acc có **một đoạn bằng dài** (epoch 1,0 → 2,25: 5 lần đánh giá liền không vượt 0,9212) trước khi tăng tiếp.

**Cách chọn quy tắc dừng:** mô phỏng `EarlyStoppingCallback` trên chính đường val của pilot, dùng đúng điều kiện trong mã transformers 5.22.1 (`check_metric_value`: cải thiện khi `metric > best` và `|metric − best| > threshold`; không cải thiện thì bộ đếm +1, đủ `patience` thì dừng):

| threshold | patience 1–4 | patience 5 | patience 6–8 | patience ≥ 9 |
|---|---|---|---|---|
| 0 | dừng ở epoch 1,25–2,0, best 0,9212 (**bỏ lỡ**) | dừng 5,25, best 0,9308 | dừng 5,5–6,0, best 0,9308 | chạy hết 6 epoch |
| 0,001 | như trên | như trên | như trên | như trên |
| 0,002 | bỏ lỡ | dừng 2,25, best 0,9232 (**bỏ lỡ**) | dừng 2,5–3,0 (**bỏ lỡ**) | p 9: dừng 3,25 (bỏ lỡ); p ≥ 10: hết 6 epoch |

**Quyết định — lần chạy chính (S3) và thí nghiệm `max_length` (S4):** **[AI chốt theo uỷ quyền của Thư 2026-11-20 — Thư đã duyệt 2026-11-20]**
- **Trần 6 epoch** (`config.MAIN_MAX_EPOCHS`), bằng trần pilot. Lịch lr linear giảm theo trần (gotcha trong `train_v1.md`), nên giữ trần thì lr của lần chạy chính giống hệt pilot; đường cong pilot là dự báo hợp lệ. Điểm tốt nhất ở epoch 4,0 < 6 (không rơi vào lần đánh giá cuối).
- **patience = 8 lần đánh giá (= 2 epoch)** (`config.PATIENCE`). Trên pilot, patience 5 là mức nhỏ nhất không bỏ lỡ điểm tốt nhất, nhưng sát ranh giới (patience 4 dừng ở epoch 2). Seed khác có thể có đoạn bằng dài hơn, nên chọn 8 để có biên +3. Chi phí: tối đa 6 epoch ≈ 31 phút mỗi lần chạy.
- **threshold = 0** (`config.THRESHOLD`, mặc định của HF). Dao động giữa hai lần đánh giá (≈ sai số chuẩn 0,005) lớn hơn mọi threshold nhỏ, nên threshold không lọc được nhiễu. Trên pilot, threshold 0,002 làm dừng sớm ở epoch 2,25–3,25. Nhiễu được xử lý bằng patience.
- Checkpoint vẫn chọn theo val acc cao nhất (L8). Early stopping chỉ quyết định khi nào ngừng train, không thay đổi cách chọn checkpoint.

**Nguồn (đã đọc trực tiếp):**
- Hugging Face, `EarlyStoppingCallback` (https://huggingface.co/docs/transformers/main_classes/callback): *"stop training when the specified metric worsens for `early_stopping_patience` evaluation calls"*; threshold: *"how much the specified metric must improve"*; cần `load_best_model_at_end`, và *"if … save_steps differs from eval_steps, the early stopping will not occur until the next save step"* (code đặt `save_steps = eval_steps`). Điều kiện cụ thể đọc từ mã `check_metric_value` của transformers 5.22.1 đã cài.
- Mosbach et al. 2021 (https://arxiv.org/html/2006.04884v3) §5.2: *"it is even desirable to train for a larger number of iterations"* → không dừng sớm ngay khi val acc đi ngang.
- Dodge et al. 2020 (https://arxiv.org/html/2002.06305v1) §3: đánh giá *"ten times per epoch"* trong lúc train → đánh giá dày (Lab 3: 4 lần/epoch) đủ để thấy dao động. Lưu ý: "early stopping" của Dodge là dừng sớm *các lần chạy kém trong nhiều lần chạy*, không phải patience trên một lần chạy; ở đây chỉ dùng phần về tần suất đánh giá.
## S3 — Lần chạy chính 3 seed (2026-11-20, Member B: Phúc, AI chạy)

**Lệnh:** mỗi seed một kernel Kaggle (`pptt1103/lab3-main-main-seed42|43|44`; Kaggle cho tối đa 2 phiên GPU cùng lúc → seed 44 chạy sau): `python -m labs.lab3.main --stage main --runs main_seed<s>`. Tesla T4, fp16, `max_length` 512, trần 6 epoch, patience 8, threshold 0 (S2b). Code `b2c21f7+tree:821781a67490`.
**File:** `outputs/results/main_runs.csv`; log `outputs/logs/20261120_105134_main_seed42.*`, `…_105200_main_seed43.*`, `…_112347_main_seed44.*` (tên đúng trong `main_runs.csv`, cột `log`); hình `outputs/figures/main_seed{42,43,44}_curves.png`.

| Seed | Checkpoint tốt nhất | Val acc | Val F1 | Dừng ở | `stop_reason` | `best_is_last_eval` | Thời gian train |
|---|---|---|---|---|---|---|---|
| 42 | step 4.224 (epoch 3,0) | 0,9288 | 0,9290 | epoch 5,0 | early_stopping | no | 1.611 s |
| 43 | step 7.744 (epoch 5,5) | 0,9308 | 0,9316 | epoch 6,0 | max_epochs | no | 2.043 s |
| 44 | step 3.872 (epoch 2,75) | 0,9312 | 0,9321 | epoch 4,75 | early_stopping | no | 1.499 s |
| **Trung bình ± std (n−1)** | | **0,9303 ± 0,0013** | | | | | |

Nguồn số: `seed_spread(main_runs.csv)` → mean 0,930295, std 0,001286, min 0,928828, max 0,931228.

**Nhận xét:**
- Cả 3 seed đều `best_is_last_eval: no`, nên trần 6 epoch là đủ (LOG_TEMPLATE §3.5). Seed 43 chạy hết trần vì vẫn còn cải thiện ở epoch 5,5; vẫn còn 2 lần đánh giá sau điểm tốt nhất.
- Epoch tốt nhất khác nhau nhiều giữa các seed (2,75 / 3,0 / 5,5), nhưng val acc chỉ chênh 0,24 điểm. Đường val gần như phẳng trong khoảng epoch 3–6, nên không có một "số epoch đúng" duy nhất. Đây là lý do dùng early stopping thay vì đặt cứng số epoch (L10).
- **Không tất định trên GPU:** pilot và `main_seed42` cùng seed 42, cùng lịch lr (trần 6) nhưng khác kết quả (pilot 0,9308 ở epoch 4,0; main 0,9288 ở epoch 3,0). Nguyên nhân: phép tính fp16 và các kernel CUDA không tất định, nên chạy lại cùng seed vẫn không ra y hệt. Nhắc lại giới hạn đã ghi trong `experiment_v1.md`: std đo được chỉ gồm seed (khởi tạo đầu phân loại, thứ tự dữ liệu) và tính không tất định của GPU, **chưa gồm** độ lệch do cách chia dữ liệu.
- Val acc ở đây là giá trị **cao nhất** trong các lần đánh giá của mỗi lần chạy (đã dùng để chọn checkpoint), nên hơi lạc quan. Số báo cáo cuối cùng là test (S5).

**Ghi chú seed 43: điểm tốt nhất sát trần 6 epoch (Thư yêu cầu ghi 2026-11-20; KHÔNG chạy lại).** Trích `outputs/logs/20261120_105200_main_seed43_history.csv`:

| Epoch | 3,75 | 4,0 | 5,0 | 5,5 | 6,0 |
|---|---|---|---|---|---|
| Val acc | 0,9304 | 0,9296 | 0,9304 | **0,9308** | 0,9308 |
| Val loss | 0,337 | 0,324 | 0,383 | 0,400 | 0,395 |

- Ghi trung thực: lần đánh giá **cuối** (epoch 6,0) có val acc **bằng đúng** giá trị tốt nhất (0,930828). `best_is_last_eval = False` chỉ vì Trainer giữ điểm tốt nhất xuất hiện **trước** (epoch 5,5; điều kiện "lớn hơn hẳn"), không phải vì đường cong đã đi xuống.
- Lý do giữ trần 6, không tăng trần và chạy lại:
  - (a) Val acc chững từ epoch 3,75 (0,9304). Phần tăng sau đó chỉ 0,0004, tức 1 review / 2.501, nhỏ hơn nhiều so với sai số chuẩn (≈ 0,005).
  - (b) Val loss có xu hướng tăng từ epoch 2 (0,206 ở epoch 2,0 → 0,400 ở epoch 5,5; có dao động, vd 0,257 ở epoch 3,0). Pilot chạy đủ 6 epoch, không dừng sớm, cũng không cải thiện sau epoch 4.
  - (c) Test đã dùng ở S5. Chạy lại sau khi đã thấy test là vi phạm L2.
- Người quyết: Thư, 2026-11-20.

**Quyết định — số seed cho thí nghiệm `max_length` (S4): 1 seed/giá trị (seed 42)** (`config.MAXLEN_SEEDS = (42,)`).
- Std = 0,13 điểm ≤ 0,3 điểm, đúng ngưỡng "std nhỏ → 1 seed / giá trị" đã ghi sẵn trong plan §4 S4 và `experiment_v1.md` bước 4. Đây là áp dụng quy tắc có sẵn, **không phải AI tự quyết** (không rơi vào vùng 0,3–1 điểm cần uỷ quyền).
- Tiêu chí "khác biệt có ý nghĩa" ở S4 (`experiment_v1.md`): chênh val acc > 2 × std ≈ **0,26 điểm**; nhỏ hơn thì kết luận "không phân biệt được".
## S4 — Thí nghiệm `max_length` (2026-11-20, Member B: Phúc, AI chạy)

**Lệnh:** 2 kernel chạy song song: `pptt1103/lab3-maxlen-maxlen128-seed42`, `…-maxlen256-seed42` (`python -m labs.lab3.main --stage maxlen --runs maxlen<L>_seed42`). Seed 42, cùng quy tắc với S3 (trần 6, patience 8, threshold 0), chỉ đổi `max_length`. Giá trị 512 lấy từ S3: so sánh chính dùng `main_seed42` (cùng seed), trung bình 3 seed chỉ để tham khảo. **So trên val, không chạy test** (plan S4). Code `b2c21f7+tree:56668a2ce8ed`. Chạy trên tài khoản Kaggle của Thư (`pptt1103`).
**File:** `outputs/results/maxlen_runs.csv`; log `outputs/logs/20261120_115617_maxlen128_seed42.*`, `…_115638_maxlen256_seed42.*`; hình `outputs/figures/maxlen_val_acc.png`, `maxlen128_seed42_curves.png`, `maxlen256_seed42_curves.png`.

| `max_length` | Review không bị cắt (S1) | % token train được giữ (S1) | Val acc (checkpoint tốt nhất) | Val F1 | Epoch tốt nhất | Số seed | Thời gian train |
|---|---|---|---|---|---|---|---|
| 128 | 12,1% | 41,5% | **0,8729** | 0,8739 | 0,75 | 1 | 267 s |
| 256 | 58,9% | 68,3% | **0,9152** | 0,9153 | 2,75 | 1 | 822 s |
| 512 (seed 42, `main_seed42`) | 86,7% | 89,0% | **0,9288** | 0,9290 | 3,0 | 1 | 1.611 s |
| 512 (trung bình 3 seed, tham khảo) | 86,7% | 89,0% | 0,9303 ± 0,0013 | — | 2,75–5,5 | 3 | 1.499–2.043 s |

Cả hai lần chạy đều `best_is_last_eval: no`, `stop_reason: early_stopping`.

**Kết luận** (tiêu chí S3: chênh > 2 × std ≈ 0,26 điểm thì có ý nghĩa):
- So **cùng điều kiện (seed 42)**: 512 so với 256 **+1,36 điểm** (0,9288 − 0,9152), gấp khoảng 5 lần ngưỡng → khác biệt có ý nghĩa. Tham khảo: trung bình 3 seed ở 512 (0,9303) hơn 256 1,51 điểm. Thư sửa 2026-11-20: 128 và 256 chỉ có seed 42, nên so với seed 42 ở 512.
- 256 so với 128 (cùng seed 42): **+4,24 điểm** (0,9152 − 0,8729) → khác biệt có ý nghĩa.
- Cắt review càng ngắn thì val acc càng giảm, và giảm nhanh khi phần giữ lại nhỏ: ở 128 chỉ còn 41,5% token. Hướng này khớp với Sun et al. 2019 (phần văn bản giữ lại ảnh hưởng accuracy trên IMDb) và xác nhận lựa chọn `max_length` 512 ở S1.
- Giới hạn: 128 và 256 chỉ có 1 seed (theo quy tắc S3). Std giả định bằng std đo được ở 512, chưa đo trực tiếp cho 128 và 256.
- Ở 128, checkpoint tốt nhất rất sớm (epoch 0,75) rồi val acc không vượt lại trong 8 lần đánh giá. Đây là quan sát, chưa kiểm nguyên nhân.
## S5 — Test một lần (2026-11-20, Member B: Phúc, AI chạy)

**Lệnh:** kernel `pptt1103/lab3-test` (Tesla T4): `python -m labs.lab3.main --stage test --model-dirs <3 checkpoint>`. Checkpoint đọc từ output của 3 kernel S3 (`kernel_sources`). Code `b2c21f7+tree:f727a9a97cc1`. Chạy sau khi đã chốt xong mọi thứ trên val (S2–S4). **Mỗi model dùng test đúng một lần:** sau bước này cả 3 file `outputs/logs/*_main_seed*_config.json` có `"test_evaluated": true`; gọi lần hai sẽ báo lỗi (`evaluate._check_and_mark_test`).
**File:** `outputs/results/test_summary.csv`, `main_seed{42,43,44}_test_predictions.csv`, `zero_shot_test_predictions.csv` (mỗi file 25.000 dòng); `outputs/demo/control_sentences.csv`; hình `outputs/figures/confusion_zero_shot_vs_finetuned.png`; phân tích lỗi `outputs/results/error_by_group.csv`, `outputs/demo/error_examples.csv` (chạy trên máy: `python -m labs.lab3.visualize.visualize`).

### Kết quả trên 25.000 review test chính thức

| Model | Test acc | Test F1 | Acc bỏ 123 review trùng train (n = 24.877) |
|---|---|---|---|
| Fine-tuned seed 42 | 0,9300 | 0,9304 | 0,9297 |
| Fine-tuned seed 43 | 0,9309 | 0,9319 | 0,9305 |
| Fine-tuned seed 44 | 0,9338 | 0,9345 | 0,9335 |
| **Fine-tuned: trung bình ± std (n−1)** | **0,9315 ± 0,0020** | **0,9323 ± 0,0021** | 0,9312 ± 0,0020 |
| Zero-shot (pipeline SST-2, Bài 1) | 0,8922 | 0,8892 | 0,8921 |

- **Ngưỡng L13 (≥ 91%, trung bình 3 seed): đạt** (`pass = True` trong `test_summary.csv`). Kết quả nằm trong khoảng kỳ vọng 92–93% của plan; DistilBERT paper (Sanh et al. 2019, Table 2) báo 92,82.
- Fine-tuned hơn zero-shot **+3,93 điểm** acc, lớn hơn rất nhiều so với std theo seed (0,20 điểm).
- Bỏ 123 review trùng train chỉ đổi acc khoảng 0,03 điểm, nên trùng lặp không làm kết quả đẹp lên.
- Std trên test (0,0020) cùng cỡ với std trên val (0,0013, S3). Thứ tự 3 seed trên val và trên test trùng nhau (42 < 43 < 44), nhưng chỉ có 3 seed và chênh lệch nhỏ nên không dùng để kết luận gì.

### Confusion matrix (seed 44: val acc cao nhất ở S3, chọn trên val)

| | Zero-shot: FP (neg → pos) | Zero-shot: FN (pos → neg) | Fine-tuned: FP | Fine-tuned: FN |
|---|---|---|---|---|
| Số lượng | 1.004 (8,0% neg) | 1.690 (13,5% pos) | 971 (7,8% neg) | 685 (5,5% pos) |

Zero-shot nghiêng về NEGATIVE: bỏ sót 13,5% review tích cực. Fine-tune làm giảm FN từ 1.690 xuống 685; FP gần như không đổi (1.004 → 971). Giả thuyết, chưa kiểm: pipeline SST-2 học trên câu ngắn, còn review IMDB tích cực thường có đoạn chê xen vào.

### Phân tích lỗi (seed 44, `error_by_group.csv`)

| Nhóm | n | Lỗi | Tỉ lệ lỗi |
|---|---|---|---|
| Tất cả | 25.000 | 1.656 | 6,62% |
| Bị cắt (> 512 token) | 3.086 | 310 | **10,05%** |
| Không bị cắt (≤ 512 token) | 21.914 | 1.346 | 6,14% |
| Rating 1 / 2 / 3 / 4 | 5.022 / 2.302 / 2.541 / 2.635 | | 3,52% / 5,08% / 8,78% / **17,23%** |
| Rating 7 / 8 / 9 / 10 | 2.307 / 2.850 / 2.344 / 4.999 | | **11,70%** / 5,61% / 3,16% / 3,62% |

- Giả thuyết trong `visualize_v1.md` ("lỗi tập trung ở rating 4 và 7, sát ranh giới"): **đúng** trên số liệu này. Rating 4 và 7 có tỉ lệ lỗi cao gấp 3–5 lần rating 1 và 10.
- Review bị cắt sai nhiều hơn (10,05% so với 6,14%). Cùng hướng với S4 (cắt nhiều hơn thì acc thấp hơn). Lưu ý: review dài cũng có thể khó hơn vì lý do khác (nhiều ý trái chiều), nên đây chỉ là tương quan, chưa phải nguyên nhân.
- Ví dụ sai tự tin nhất mỗi nhóm: `outputs/demo/error_examples.csv` (300 ký tự đầu, giữ `uid` để tra bản đầy đủ). Lý do sai chỉ là giả thuyết.

### 6 câu kiểm soát (`control_sentences.csv`)

| Câu | Zero-shot | Fine-tuned (seed 44) |
|---|---|---|
| (1) rõ tích cực | POSITIVE 0,9999 ✓ | POSITIVE 0,9968 ✓ |
| (2) rõ tiêu cực | NEGATIVE 0,9998 ✓ | NEGATIVE 0,9988 ✓ |
| (3) phủ định câu tiêu cực "not bad at all" | POSITIVE 0,9980 ✓ | **NEGATIVE 0,8339 ✗** |
| (4) phủ định câu tích cực "would not say … good" | NEGATIVE 0,9997 ✓ | **POSITIVE 0,7411 ✗** |
| (5) tương phản (but) | NEGATIVE 0,9997 ✓ | NEGATIVE 0,9987 ✓ |
| (6) trung tính | NEGATIVE 0,9955 | NEGATIVE 0,6695 |

- Model fine-tune **sai cả 2 câu phủ định**, trong khi zero-shot đúng cả 2. Đây là kết quả đo thật; câu không được sửa cho đẹp (`sentiment_demo_v1.md`).
- Như vậy, acc cao trên IMDB (93%) không có nghĩa là model xử lý phủ định trong câu ngắn tốt hơn. Giả thuyết, chưa kiểm: review IMDB dài, nhãn quyết định bởi tổng thể nhiều câu, nên model ít học được phủ định trong một câu đơn. SST-2 có nhãn cho từng cụm từ nên học phủ định tốt hơn. Chỉ có 2 câu, nên không kết luận chung về năng lực phủ định.
- Ở câu trung tính, fine-tuned ít tự tin hơn (0,67 so với 0,9955), hợp lý hơn với một câu không mang cảm xúc.
