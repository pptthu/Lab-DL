# eda — khám phá dữ liệu IMDB, đo độ dài token (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T14:29:08+07:00 (Thư)
- **Người phụ trách**: Thư
- **Status**: Done — audit 2026-11-20

---

## Name
EDA — phân bố nhãn, độ dài review (từ và token), trùng lặp; số liệu để chốt `max_length`

## Background
Không có trong 7 bước của đề, nhưng cần cho quyết định `max_length` (plan §4 bước S1): review IMDB dài, DistilBERT nhận tối đa 512 token; Sun et al. 2019 cho thấy cách cắt văn bản ảnh hưởng accuracy trên IMDb. Đầu vào cho thí nghiệm `max_length` và phân tích lỗi.

## Goals / Purpose
- `outputs/results/eda_summary.csv`: số review và tỉ lệ nhãn mỗi split (train, test chính thức); độ dài token: trung bình, median, P90, P95, max; % review dài hơn 128 / 256 / 512 token
- `outputs/figures/`: phân bố nhãn, histogram độ dài token (vạch 128/256/512), độ dài theo nhãn
- Số review trùng nội dung trong train, trong test và giữa train–test (băm văn bản đã làm sạch)
- Số token `<br />` chiếm trước khi làm sạch (trung bình / review) — bằng chứng cho bước làm sạch của `preprocessing`
- Tỉ lệ review chứa ít nhất một từ phủ định (`not`, `no`, `n't`, `never`, `nothing`, `none`, `nor`, `neither`, `nobody`, `without`) — kiểm giả định "gần như review nào cũng có phủ định" (lý do bỏ nhóm phủ định ở `visualize`)
- Phân bố rating (1–4, 7–10) mỗi split — dùng cho phân tích lỗi theo rating
- Không làm:
  - Chọn `max_length` — Thư chốt từ bảng này: **512** (2026-11-20, `docs/experiments/experiment_log.md` S1)
  - Làm sạch, chia split — thuộc `preprocessing`

## Input / Output
- **Input:** dữ liệu thô từ `preprocessing.load_split("train" | "test")` (25k + 25k, chưa chia val); tokenizer `distilbert-base-uncased`. Độ dài token đo trên văn bản đã qua `preprocessing.clean_text()` — đó là thứ model nhận; số token `<br />` đo trên văn bản gốc. Chạy **trước** phần chia train/val của `preprocessing` (Thư chốt 2026-11-20: làm EDA trước) để số liệu EDA làm căn cứ cho bước làm sạch và gộp review trùng
- **Output:** `outputs/results/eda_summary.csv`, `outputs/figures/eda_*.png`; smoke run: `outputs/_smoke/`

## How to do it (general plan)
1. Đếm nhãn mỗi split
2. Độ dài theo từ và theo token (tokenize **không cắt**, chỉ đếm)
3. Bảng phân vị và % vượt 128 / 256 / 512
4. Đếm trùng lặp (so khớp nội dung đã làm sạch)
5. Vẽ 3 hình
6. Test với dataset giả nhỏ
7. Smoke run vài chục review; normal run full 50k trên máy Thư (chỉ tokenize, không train)

## Pipeline
```
preprocessing.load_split (thô) → clean_text → đếm nhãn, tokenize không cắt → bảng độ dài + trùng lặp
  → outputs/results/eda_summary.csv, outputs/figures/eda_*.png → Thư chốt max_length
```

## Detailed plan
- **Đo theo token, không chỉ theo từ:** `max_length` tính bằng token WordPiece; một từ có thể thành nhiều token
- **512 thực tế còn 510 token nội dung:** 2 vị trí dành cho `[CLS]` và `[SEP]`
- **Trùng lặp (Thư duyệt 2026-11-20):** chỉ đếm và báo cáo ở module này; cách xử lý: train–val gộp nhóm (`preprocessing`), train–test giữ test nguyên và báo thêm acc trên test bỏ review trùng (`evaluate`), trong cùng train giữ nguyên
- **Từ phủ định:** Konstantinova et al. 2012 (SFU Review corpus): 23,7% câu trong review phim có phủ định; "not" và "no" chiếm hơn 55% dấu hiệu phủ định → review IMDB nhiều câu gần như chắc chắn có phủ định (suy luận, bảng này kiểm)
- **Gotchas:**
  - Tokenize 50k review không cắt có thể báo cảnh báo "sequence longer than 512" — chỉ đếm, không đưa vào model nên bỏ qua được
  - Tokenize full data trên CPU mất vài phút — dùng tokenizer fast, `batched=True`

## Triển khai (2026-11-20)
- **Hàm** (`labs/lab3/eda/eda.py`): `NEGATION_RE` (`:35`); `token_lengths(texts, tokenizer, batch_size=1000)` (`:39`); `length_summary(lengths, limits)` (`:51`); `duplicate_counts(train, test)` (`:69`); `negation_share(texts)` (`:88`); `split_summary(df, tokenizer)` (`:94`); 3 hàm vẽ (`:120`–`:162`); `run_eda(train, test, tokenizer, results_dir, figures_dir) -> dict` (`:178`). Chạy: `python -m labs.lab3.eda.eda [--smoke]`
- **Chỗ code khác plan:** ngoài `eda_summary.csv`, ghi thêm `eda_duplicates.csv`, `eda_ratings.csv` và `eda_token_lengths.csv` (số token từng review — `visualize` dùng để biết review nào bị cắt). Chạy trên 2 split chính thức (val tách sau)
- **Kết quả normal run** (51 s, máy Thư): median 222 / 219 token (train / test); > 128: 87,9 / 87,4%; > 256: 41,1 / 40,1%; > 512: 13,3 / 12,3% (Sun et al. 2019: 12,69%); 58,7% review có `<br />`, tốn trung bình 16,3 token; 89,4% có từ phủ định; 123 review test trùng train; 0 nhóm trùng có nhãn mâu thuẫn. `max_length` = 512, Thư chốt 2026-11-20 (phân tích: `docs/experiments/experiment_log.md` mục S1)
- **Test / smoke:** `pytest labs/lab3/tests/test_eda.py` 5 passed; smoke ghi `outputs/_smoke/{results,figures}/` không lỗi

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_eda.py` đạt: phân vị và % vượt ngưỡng tính đúng trên dữ liệu giả
- **Smoke run:** CSV và 3 hình có trong `outputs/_smoke/`
- **Normal run:** `eda_summary.csv` có đủ 2 split chính thức (train, test), tổng 50.000 review; 3 hình trong `outputs/figures/`

## Kết quả audit (2026-11-20)
- **Test:** `pytest labs/lab3/tests/test_eda.py` → 5 passed ✔
- **Smoke run** (`run_member_a(smoke=True)`, 38 giây, máy Thư, 40 review mỗi split): 4 CSV và 3 hình trong `outputs/_smoke/` ✔
- **Normal run:** `outputs/results/eda_summary.csv` có train và test, tổng 50.000 review; số liệu ở mục Triển khai khớp file (median 222 / 219; > 512: 13,28 / 12,34%; `<br />` 58,66%; phủ định 89,44%; 123 review test trùng train); 3 hình trong `outputs/figures/` ✔
- Số dòng code ghi ở mục Triển khai khớp code hiện tại ✔

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [preprocessing_v1.md](preprocessing_v1.md), [experiment_v1.md](experiment_v1.md), [visualize_v1.md](visualize_v1.md)
- Nguồn: Sun et al. 2019 https://arxiv.org/abs/1905.05583 ; Konstantinova et al. 2012 https://www.uhu.es/noa.cruz/lrec.pdf
- Code: `labs/lab3/eda/eda.py` | Test: `labs/lab3/tests/test_eda.py`
