# Overfit, chọn checkpoint và độ tự tin của model — Lab 3 (v1)

- **Created**: 2026-11-24T16:51:29+07:00 (pptthu, AI soạn)
- **Last Updated**: 2026-11-24T16:51:29+07:00 (pptthu, AI soạn)
- Không train lại, không chạy lại test. Mọi số liệu tính từ file đã có trong `outputs/`: đường huấn luyện lấy từ `outputs/logs/*_history.csv`, độ tự tin lấy từ `outputs/results/main_seed<s>_test_predictions.csv` (cột `prob_pos`). Phần tính trên test chỉ để báo cáo, không dùng để chọn model hay cấu hình (plan L2).

## 1. Hiện tượng
Ở mọi lần chạy, train loss giảm gần về 0, val loss thấp nhất ở epoch 1–2 rồi tăng lên, nhưng val accuracy vẫn tăng thêm vài epoch nữa.

| Lần chạy | Train loss cuối | Val loss thấp nhất (epoch) | Val loss ở lần đánh giá cuối (epoch) |
|---|---:|---|---|
| Pilot seed 42 (6 epoch, không dừng sớm) | 0,018 | 0,211 (1,0) | 0,424 (6,0) |
| Seed 42 | 0,042 | 0,211 (1,0) | 0,379 (5,0) |
| Seed 43 | 0,023 | 0,206 (2,0) | 0,395 (6,0) |
| Seed 44 | 0,039 | 0,199 (1,0) | 0,384 (4,75) |

Theo định nghĩa trong slide Chapter 3 (trang 10, "Validation"), overfit là khi model "performs well on the training data but poorly on unseen data". Nhìn theo loss thì đây là overfit: model làm tốt trên train, còn loss trên val xấu đi. Mục 3 cho thấy phần bị overfit là **độ tự tin**, còn **khả năng đoán đúng** không giảm.

## 2. Chọn checkpoint theo val loss hay val accuracy

### 2.1 Bằng chứng (chỉ dùng val)
Mỗi lần đánh giá (≈ 1/4 epoch), log ghi cả val loss và val acc. Vì vậy, trong cùng một lần train, đọc được checkpoint nào sẽ được chọn nếu dùng val loss, và checkpoint nào nếu dùng val acc. Hai checkpoint cùng seed, cùng dữ liệu, chỉ khác quy tắc chọn.

| Lần chạy | Chọn theo val loss: epoch, val acc | Chọn theo val acc: epoch, val acc | Chênh val acc |
|---|---|---|---:|
| Pilot seed 42 | 1,0: 0,9212 | 4,0: 0,9308 | +0,96 điểm |
| Seed 42 | 1,0: 0,9208 | 3,0: 0,9288 | +0,80 điểm |
| Seed 43 | 2,0: 0,9276 | 5,5: 0,9308 | +0,32 điểm |
| Seed 44 | 1,0: 0,9256 | 2,75: 0,9312 | +0,56 điểm |

- Cả 4 lần chạy cùng một hướng: chọn theo val loss thì dừng ở epoch 1–2 và val acc thấp hơn.
- Test của các checkpoint chọn theo val acc: 0,9315 ± 0,0020, không thấp hơn val (0,9303 ± 0,0013). Cách chọn này không làm model bị overfit vào val.

### 2.2 Vì sao Lab 3 khác Lab 1, Lab 2
- Đề chỉ cần nhãn khen/chê, nên metric chính là accuracy (`project_description.md` §1). Slide Chapter 3, trang 10 ghi: "Choose the best model during training (e.g., based on validation accuracy)".
- Val loss chỉ là chỉ báo thay thế, dùng được khi nó đi cùng chiều với accuracy. Ở Lab 1 (MLP train từ đầu) và Lab 2, hai chỉ báo gần trùng nhau: epoch có val loss thấp nhất lệch 0–0,6 điểm val acc so với epoch có val acc cao nhất. Ở Lab 3 (fine-tune Transformer pre-trained), hai chỉ báo tách nhau như bảng 2.1.
- Quyết định chọn theo val acc có trong plan trước khi chạy (`plans/detail_plan_lab3_md_v1.md`, L8). Pilot (`experiment_log.md`, S2b) và 3 seed ở bảng trên là số đo xác nhận quyết định đó.

### 2.3 Giới hạn của bằng chứng
- Chênh lệch trung bình khoảng 0,5–0,7 điểm, cỡ 1–2 lần sai số chuẩn của val acc (√(0,93 × 0,07 / 2.501) ≈ 0,005). Hướng nhất quán qua 4 lần chạy, nhưng độ lớn chưa chắc chắn.
- Val acc ở checkpoint được chọn là giá trị lớn nhất trong khoảng 20 lần đánh giá, nên hơi lạc quan.
- Không có số test cho checkpoint chọn theo val loss: checkpoint đó không được lưu (`SAVE_TOTAL_LIMIT = 2`), và test 25k chỉ chạy một lần cho model cuối (plan L2). Nhóm M có số test cho cả hai cách (EX4: 90,34% so với 91,12%), nhưng đó là dùng test để so hai cách chọn.

## 3. Overfit theo loss là vấn đề độ tự tin

### 3.1 Vì sao loss tăng mà accuracy không giảm
- Accuracy chỉ xét model đoán đúng hay sai.
- Cross-entropy loss xét thêm model chắc chắn đến đâu. Đoán sai với xác suất 0,99 bị phạt −ln(0,01) ≈ 4,6; đoán sai với xác suất 0,6 chỉ bị phạt −ln(0,4) ≈ 0,9.
- Càng train, model càng đẩy xác suất về gần 0 hoặc 1. Câu đúng thì loss giảm rất ít (đã gần 0), câu sai thì loss tăng rất mạnh. Vì vậy val loss tăng trong khi số câu đúng không đổi hoặc còn tăng.

### 3.2 Số đo trên test (chỉ báo cáo)
ECE (Expected Calibration Error) đo model nói chắc có đúng mức không: gom các câu theo độ tự tin (10 nhóm từ 0,5 đến 1), trong mỗi nhóm lấy |độ tự tin trung bình − tỉ lệ đúng|, rồi lấy trung bình có trọng số theo số câu. ECE = 0 là nói chắc đúng mức. Độ tự tin của một câu = max(`prob_pos`, 1 − `prob_pos`).

| Seed | Test acc | Độ tự tin trung bình | ECE | Câu sai có độ tự tin > 0,95 | Câu sai có độ tự tin > 0,99 | Test log loss |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0,9300 | 0,988 | 0,058 | 71,9% (của 1.750 câu sai) | 49,8% | 0,301 |
| 43 | 0,9309 | 0,994 | 0,064 | 84,9% (của 1.728) | 74,2% | 0,403 |
| 44 | 0,9338 | 0,987 | 0,053 | 70,3% (của 1.656) | 45,7% | 0,276 |

Model nói chắc bao nhiêu và thật ra đúng bao nhiêu (seed 44):

| Độ tự tin | Số review | Độ tự tin trung bình | Tỉ lệ đúng thật |
|---|---:|---:|---:|
| 0,5–0,7 | 302 | 0,607 | 0,583 |
| 0,7–0,9 | 535 | 0,816 | 0,593 |
| 0,9–0,99 | 1.844 | 0,968 | 0,699 |
| ≥ 0,99 | 22.319 | 0,998 | 0,966 |

**Nhận xét.**
- Model đúng khoảng 93% nhưng tự tin trung bình khoảng 99%: tự tin quá mức (overconfident). ECE 0,05–0,06 nghĩa là model nói chắc hơn thực tế khoảng 5–6 điểm phần trăm.
- Seed 43 có checkpoint muộn nhất (epoch 5,5) và cũng tự tin quá mức nhất (ECE 0,064, log loss 0,403). Điều này khớp với giải thích ở 3.1: train càng lâu, model càng chắc chắn.
- Hiện tượng này đã được nghiên cứu: mạng nơ-ron hiện đại thường tự tin quá mức dù accuracy cao (Guo et al. 2017); Transformer pre-trained cũng vậy (Desai & Durrett 2020).

### 3.3 Hệ quả
- Nhãn dự đoán dùng được; test 0,9315 ± 0,0020.
- Xác suất `prob_pos` **không dùng làm độ tin cậy được**. Ví dụ: các "lỗi tự tin nhất" trong `outputs/demo/error_examples.csv` có `prob_pos` khoảng 0,001. Nếu cần dùng xác suất (đặt ngưỡng, báo mức tin cậy) thì phải calibrate trước, ví dụ temperature scaling: chia logit cho một hệ số T học trên val; accuracy không đổi (Guo et al. 2017, §4.2). Bài này chỉ dùng nhãn nên không làm.

## 4. Các cách chặn overfit trong bài

| Cách | Có | Giá trị, vị trí |
|---|:---:|---|
| Chọn checkpoint tốt nhất thay vì checkpoint cuối | ✔ | `load_best_model_at_end=True`, `metric_for_best_model="accuracy"` (`train/train.py`) |
| Early stopping | ✔ | patience 8 lần đánh giá ≈ 2 epoch (`config.PATIENCE`); seed 42 dừng ở epoch 5,0, seed 44 ở 4,75 |
| Trần epoch chốt từ pilot | ✔ | 6 epoch (`experiment_log.md` S2b) |
| Learning rate nhỏ, warmup rồi giảm tuyến tính | ✔ | 2e-5, warmup 10% (`config.py`) |
| Weight decay | ✔ | 0,01 |
| Dropout | ✔ | mặc định DistilBERT: 0,1 trong các lớp, 0,2 trước đầu phân loại (`outputs/results/model_summary.json`); nhóm không chỉnh |
| Đo đúng: val tách theo phim, test chạy một lần, 3 seed | ✔ | `preprocessing`, `evaluate` (cờ `test_evaluated`) |
| Tăng dropout, tăng weight decay, label smoothing, LoRA | ✘ | ngoài phạm vi (plan §7); xem mục 5 |
| Calibration (temperature scaling) | ✘ | không cần cho nhãn; xem 3.3 |

## 5. Vì sao không thêm regularization
Nhóm M dùng cùng model (`distilbert-base-uncased`), cùng IMDb, cùng test 25k, và đã thử (báo cáo trong repo của M, `docs/lab3/experiments/`):

| Báo cáo M | Cấu hình | Test acc | So với baseline 91,19% |
|---|---|---:|---:|
| EX3 | Cosine scheduler | 91,24% | +0,05 |
| EX3 | Weight decay 0,10 | 91,35% | +0,16 |
| EX3 | Dropout đầu phân loại 0,30 | 91,48% | +0,29 |
| EX4 | Gộp: weight decay 0,1, dropout 0,3, label smoothing 0,1, cosine | 91,12% | −0,07 |
| EX14 so với EX6 | LoRA + Cleanlab + head+tail so với full fine-tune 512 | 93,14% so với 93,23% | −0,09 |

- Mỗi cấu hình của M chạy 1 seed và được so trên test, nên chỉ dùng làm tham khảo. Các chênh lệch (0,05–0,29 điểm) không lớn hơn độ lệch giữa các seed của nhóm (std 0,20 điểm).
- Kết luận: với bài này, regularization thêm gần như không đổi được accuracy. Vấn đề còn lại là độ tự tin (mục 3), và sửa độ tự tin thì dùng calibration, không phải regularization.

## 6. Tóm tắt khi vấn đáp
1. Model có overfit theo loss: train loss về khoảng 0,02–0,04, val loss tăng từ khoảng 0,2 lên khoảng 0,4.
2. Phần bị overfit là độ tự tin: model đúng 93% nhưng tự tin trung bình 99%, ECE 0,05–0,06.
3. Khả năng đoán đúng không giảm: val acc tăng tới epoch 3–5; test 93,15% không thấp hơn val 93,03%.
4. Chọn checkpoint theo val acc vì đề cần nhãn và slide Chapter 3 trang 10 ghi chọn theo validation accuracy. Bằng chứng: pilot và 3 seed, chọn theo val loss thì dừng ở epoch 1–2 và val acc thấp hơn 0,3–1 điểm, chỉ so trên val.
5. Đã chặn bằng early stopping, trần epoch, weight decay, dropout mặc định. Không thêm regularization vì số của nhóm M cho thấy gần như không đổi được accuracy; xác suất đầu ra cần calibrate nếu muốn dùng.

## Tài liệu tham khảo
1. Slide Chapter 3 — Training Deep Neural Networks, trang 10 "Validation" (tài liệu môn học).
2. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599
3. S. Desai, G. Durrett. Calibration of Pre-trained Transformers. EMNLP 2020. https://arxiv.org/abs/2003.07892
4. Hugging Face LLM Course, chương 3 (fine-tuning): https://huggingface.co/learn/llm-course/chapter3
5. Báo cáo của nhóm M: `docs/lab3/experiments/EX3_ANTI_OVERFITTING_REPORT.md`, `EX4_RESULTS_AND_DIAGNOSTICS_REPORT.md`, `EX6_512_TOKENS_BREAKTHROUGH_REPORT.md`, `EX14_LORA_DENOISED_HEAD_TAIL_BREAKTHROUGH_REPORT.md`.
