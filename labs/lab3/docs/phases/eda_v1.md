# eda — khám phá dữ liệu IMDB, đo độ dài token (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T01:16:36+07:00 (Thư)
- **Người phụ trách**: Thư
- **Status**: To Do

---

## Name
EDA — phân bố nhãn, độ dài review (từ và token), trùng lặp; số liệu để chốt `max_length`

## Background
Không có trong 7 bước của đề, nhưng cần cho quyết định `max_length` (plan §4 bước S1): review IMDB dài, DistilBERT nhận tối đa 512 token; Sun et al. 2019 cho thấy cách cắt văn bản ảnh hưởng accuracy trên IMDb. Đầu vào cho thí nghiệm `max_length` và phân tích lỗi.

## Goals / Purpose
- `outputs/results/eda_summary.csv`: số review và tỉ lệ nhãn mỗi split; độ dài token: trung bình, median, P90, P95, max; % review dài hơn 128 / 256 / 512 token
- `outputs/figures/`: phân bố nhãn, histogram độ dài token (vạch 128/256/512), độ dài theo nhãn
- Số review trùng nội dung trong train, trong test và giữa train–test (băm văn bản đã làm sạch)
- Số token `<br />` chiếm trước khi làm sạch (trung bình / review) — bằng chứng cho bước làm sạch của `preprocessing`
- Tỉ lệ review chứa ít nhất một từ phủ định (`not`, `no`, `n't`, `never`, `nothing`, `none`, `nor`, `neither`, `nobody`, `without`) — kiểm giả định "gần như review nào cũng có phủ định" (lý do bỏ nhóm phủ định ở `visualize`)
- Phân bố rating (1–4, 7–10) mỗi split — dùng cho phân tích lỗi theo rating
- Không làm:
  - Chọn `max_length` — Thư chốt từ bảng này (`[cần chốt sau S1]`)
  - Làm sạch, chia split — thuộc `preprocessing`

## Input / Output
- **Input:** `DatasetDict` từ `preprocessing.load_datasets()`; tokenizer `distilbert-base-uncased`
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
preprocessing.load_datasets → đếm nhãn, tokenize không cắt → bảng độ dài + trùng lặp
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

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_eda.py` đạt: phân vị và % vượt ngưỡng tính đúng trên dữ liệu giả
- **Smoke run:** CSV và 3 hình có trong `outputs/_smoke/`
- **Normal run:** `eda_summary.csv` có đủ 3 split, tổng 50.000 review; 3 hình trong `outputs/figures/`

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [preprocessing_v1.md](preprocessing_v1.md), [experiment_v1.md](experiment_v1.md), [visualize_v1.md](visualize_v1.md)
- Nguồn: Sun et al. 2019 https://arxiv.org/abs/1905.05583 ; Konstantinova et al. 2012 https://www.uhu.es/noa.cruz/lrec.pdf
- Code: `labs/lab3/eda/eda.py` | Test: `labs/lab3/tests/test_eda.py`
