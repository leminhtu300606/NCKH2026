# Hệ thống bảo vệ riêng tư video UAV trên thiết bị biên

## Mục đích và phạm vi

Hệ thống nhận video UAV, phát hiện người, theo dõi đối tượng, ước lượng rủi ro
nhận dạng lại, chọn mức bảo vệ, che vùng nhạy cảm và chỉ phát hành frame qua
`OutputGate` sau khi xác minh. Đây là pipeline nghiên cứu trên **người (person)** trong
VisDrone; detector hiện tại là **model person**, **không phải** detector khuôn mặt hoặc
biển số. Toàn bộ dữ liệu train/val/test và toàn bộ metric đánh giá đều được tính trên
hộp bounding box **person**, không dùng nhãn mặt hoặc biển số.

Luồng xử lý chính:

```text
frame -> YOLO person -> ByteTrack -> 21 features -> XGBoost risk
      -> RiskScheduler (P0-P3/A1-A4) -> blur/pixelate -> verify -> OutputGate
```

Tracker runtime dùng `ultralytics.trackers.byte_tracker.BYTETracker` với cấu
hình trong archive: high/low threshold `0.5/0.1`, new-track threshold `0.6`,
buffer `30`, match threshold `0.8` và score fusion bật.

`B1` đến `B8` là các operating point của cùng pipeline:

- `B1`: phát hiện toàn frame mỗi frame.
- `B2`/`B3`/`B4`: giảm tần suất phát hiện, lần lượt kết hợp tracking.
- `B5`/`B6`: dùng XGBoost risk; `B6` thêm output gate fail-safe.
- `B7`: thêm adaptive mask.
- `B8`: thêm motion hold, scheduler bốn action, runtime risk evidence và
  adaptive region protection. `recall_first` full-frame fallback chỉ dành cho
  thí nghiệm upper-bound và tắt mặc định để cấu hình có thể triển khai thực tế.

## Thành phần và dữ liệu

- [edge_models/yolo26s.pt](edge_models/yolo26s.pt): detector **person** (không phải face/license plate).
- [edge_models/risk_model.json](edge_models/risk_model.json): XGBoost artifact
  từ archive, với 21 feature, train trên dữ liệu **person**.
- [data/canonical_zip](data/canonical_zip): risk dataset, OOF predictions,
  feature schema, CV summary và metric gốc trong ZIP; toàn bộ sample là
  **person track**, không có face/license plate.
- [edge_models/risk_model.json.metadata.json](edge_models/risk_model.json.metadata.json):
  thứ tự feature bắt buộc khi phục vụ model.
- [edge_models/feature_schema.json](edge_models/feature_schema.json): định nghĩa
  feature, tất cả mô tả đặc trưng **person track**.
- [edge_models/tau_val.json](edge_models/tau_val.json): ngưỡng được chọn trên
  validation sequence-disjoint.
- [VisDrone2019-VID-test-dev](VisDrone2019-VID-test-dev): dataset đánh giá chuẩn,
  gồm 17 sequence, có `sequences/` và `annotations/`; annotation chỉ chứa
  **person box** (category 1), không có face/license plate.
- [visdrone-train](visdrone-train): dữ liệu dùng để xây detector export, không
  phải bảng feature huấn luyện XGBoost; chỉ dùng cho detector **person**.

`risk_dataset.csv` trong [data/canonical_zip](data/canonical_zip) là bảng dữ
liệu huấn luyện/đánh giá được đóng gói trong ZIP. Archive ghi nhận tracking
chưa hoàn tất (`completed=false`, 52/80 sequence), nên metric risk archive
được báo cáo đúng trạng thái và không được trộn với metric video end-to-end.

## Đánh giá toàn hệ thống

Lệnh chuẩn đánh giá tất cả sequence và tất cả policy:

```powershell
python test_step6.py
python run_step6.py `
  --out results/system_evaluation `
  --data-root VisDrone2019-VID-test-dev `
  --max-frames 200 --reps 1 `
  --policies B8 --imgsz 1024 --backend pytorch
python make_step6_report.py --results results/system_evaluation
```

Khi không truyền `--sequences`, chương trình tự lấy mọi thư mục sequence trong
`--data-root`. Vì vậy kết quả chuẩn không thể vô tình chỉ chạy một vài sequence.
Các policy dùng chung detection cache trong cùng lần chạy để so sánh công bằng.
Kết quả được lưu trong [results/system_evaluation](results/system_evaluation):

- `step6_metrics.json`: protocol và metric tổng hợp theo policy.
- `step6_policies.csv`: một dòng cho mỗi policy.
- `step6_sequences.csv`: phân rã theo sequence và repetition.
- `step6_report.md`: báo cáo diễn giải.
- `step6_faults.json`: kiểm tra fail-safe và output gate.
- `step6_privacy.json`: đánh giá Re-ID oracle nếu đã bật.

Trong báo cáo, **B8 là cấu hình chính của toàn bộ hệ thống**. B1-B7 chỉ được
dùng làm baseline/ablation để đo đánh đổi giữa coverage, độ trễ và chi phí
detector; không trình bày chúng như các kết quả tổng thể độc lập.

Các tham số mặc định của protocol là CPU, `imgsz=1024`, confidence `0.25`,
IoU `0.5`, tối đa 200 frame/sequence, một repetition, warm-up 30 frame và
loại annotation có `occlusion == 2`. Đây là benchmark kiến trúc end-to-end,
không cộng hoặc trộn kết quả từ các thư mục bước riêng lẻ.

Có thể benchmark detector ONNX sau khi export:

```powershell
python run_step6.py --backend onnx --weights edge_models\best.onnx
```

Hệ thống từ chối cấu hình ONNX nếu checkpoint không có đuôi `.onnx`; TensorRT
và công suất/energy chỉ được báo cáo khi môi trường có runtime hoặc cảm biến
tương ứng.

## Chỉ số

- **Detection recall/precision/F1**: đúng/sai theo ground-truth **person box** (không phải face/license plate).
- **AP/AP50/AP75/APsmall/APmedium/APlarge/AR và HOTA/DetA/AssA/MOTA/IDF1**:
  chưa có trong benchmark hiện tại vì evaluator chưa xuất các metric này.
- **Region recall**: tỷ lệ ground-truth box được hợp các vùng bảo vệ phủ ít nhất
  50% diện tích.
- Khi cần đo upper-bound, có thể bật B8 `recall_first` để dùng vùng bảo vệ toàn
  khung. Đây không phải cấu hình triển khai vì làm mất tính hữu dụng của video.
- **ULFR**: tỷ lệ frame còn ít nhất một ground-truth box chưa được phủ; thấp hơn
  là tốt hơn. Đây là coverage-only, chưa bao gồm mọi khả năng Re-ID/OCR.
- **Unsafe release**: frame được phát hành nhưng còn box chưa phủ; đây là
  coverage-proxy, khác với gate violation.
- **TTP p50/p95**: số frame từ lúc identity xuất hiện đến lần đầu được bảo vệ;
  identity không được bảo vệ là censored.
- **Calls/100 frames, FPS, p50/p95/p99 latency**: chi phí và thời gian end-to-end
  sau warm-up.
- **Blocked/correct block/false block/gate violation**: tính đúng của cổng phát
  hành, tách biệt với chất lượng detector.
- **Re-ID attack/protection success, PSNR, SSIM**: hiệu quả và chất lượng vùng
  che khi oracle Re-ID được chạy.
- **Face Recognition Accuracy, Rank-1, TAR, FAR, EER, OCR Exact Match Accuracy,
  CER, MS-SSIM, LPIPS, FID**: chưa đánh giá vì hệ thống chỉ sử dụng dữ liệu
  **person**, không có face identity hoặc license-plate/OCR ground truth và
  chưa có evaluator tương ứng. Không đánh giá face/license plate vì không nằm
  trong phạm vi dữ liệu.
- **XGBoost AUC, PR-AUC, Brier, F1 và ECE**: chất lượng risk model trên split
  sequence-disjoint; đây là metric của model, không thay thế metric end-to-end.
- **Energy/frame** chỉ có ý nghĩa khi đo bằng cảm biến công suất đã hiệu chuẩn.

## Giới hạn và cách diễn giải

1. Detector là detector **person** nên toàn bộ bảo vệ nhạy cảm là proxy; **chỉ dùng dữ liệu person**, không dùng face hoặc license plate; chưa chứng minh cho mặt hoặc biển số.
2. Kết quả detector/tracking trên `VisDrone2019-VID-test-dev` chỉ có ý nghĩa
   trong split này; không được gọi là kết quả production hoặc edge GPU.
3. Latency hiện đo trên CPU. `overload-ms=250` là ngưỡng đo cho máy này, không
   đại diện cho deadline 40 ms của thiết bị edge GPU.
4. XGBoost có ECE cao, nên risk score chưa phải xác suất được calibration.
   Logistic Regression hiện có thể đạt AUC/PR-AUC tốt hơn XGBoost trên test.
5. Pipeline phục vụ dùng cửa sổ quá khứ, trong khi dữ liệu archive huấn luyện
   được mô tả là có cửa sổ nhìn tới tương lai. Đây là train/serve skew cần sửa
   khi có lại dữ liệu gốc.
6. `ROI_DETECTION` trong benchmark hiện dùng detection toàn frame từ cache; số
   lần gọi ROI vì vậy không chứng minh mức tiết kiệm GPU thực tế.
7. ULFR coverage-only không phát hiện trường hợp đã phủ nhưng Re-ID/OCR vẫn nhận
   đúng. Đánh giá riêng tư mạnh hơn cần nhãn hoặc oracle cross-sequence hợp lệ.
8. Các sequence/frame local có thể trùng với dữ liệu detector dùng để xây model;
   không dùng chúng để tuyên bố holdout độc lập nếu chưa kiểm tra hash và nguồn
   phân chia.
9. Kết quả chỉ hợp lệ cùng `data_root`, danh sách sequence, frame cap, model,
   threshold và protocol được ghi trong `step6_metrics.json`.

## Kiểm tra nhanh

```powershell
python test_step6.py
python -c "import xgboost as xgb; m=xgb.XGBClassifier(); m.load_model('edge_models/risk_model.json'); print(m.n_features_in_)"
```
