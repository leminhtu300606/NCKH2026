# Hệ thống bảo vệ riêng tư video UAV trên thiết bị biên

## Mục đích và phạm vi

Hệ thống nhận video UAV, phát hiện người, theo dõi đối tượng, ước lượng rủi ro
nhận dạng lại, chọn mức bảo vệ, che vùng nhạy cảm và chỉ phát hành frame qua
OutputGate sau khi xác minh. Đây là pipeline nghiên cứu trên *người (person)* trong
VisDrone; detector hiện tại là *model person*. Toàn bộ dữ liệu train/val/test và toàn bộ metric đánh giá đều được tính trên
hộp bounding box *person*.

Luồng xử lý chính:

text
frame -> YOLO person -> ByteTrack -> 21 features -> XGBoost risk
      -> RiskScheduler (P0-P3/A1-A4) -> blur/pixelate -> verify -> OutputGate

Tracker runtime dùng ultralytics.trackers.byte_tracker.BYTETracker với cấu
hình trong archive: high/low threshold 0.5/0.1, new-track threshold 0.6,
buffer 30, match threshold 0.8 và score fusion bật.

B1 đến B8 là các operating point của cùng pipeline:

- B1: phát hiện toàn frame mỗi frame.
- B2`/B3`/`B4`: giảm tần suất phát hiện, lần lượt kết hợp tracking.
- B5`/B6`: dùng XGBoost risk; B6 thêm output gate fail-safe.
- B7: thêm adaptive mask.
- B8: thêm motion hold, scheduler bốn action, runtime risk evidence và
  adaptive region protection. recall_first full-frame fallback chỉ dành cho
  thí nghiệm upper-bound và tắt mặc định để cấu hình có thể triển khai thực tế.

## Thành phần và dữ liệu

- [models/best.pt](models/best.pt): detector *person*.
- [models/xgboost_risk.json](models/xgboost_risk.json): XGBoost artifact
  từ archive, với 21 feature, train trên dữ liệu *person*.
- [zip_extract_full/artifacts](zip_extract_full/artifacts): risk dataset, OOF predictions,
  feature schema, CV summary và metric gốc trong ZIP; toàn bộ sample là
  *person track*.
- [zip_extract_full/artifacts/risk_model.json.metadata.json](zip_extract_full/artifacts/risk_model.json.metadata.json):
  thứ tự feature bắt buộc khi phục vụ model.
- [zip_extract_full/artifacts/feature_schema.json](zip_extract_full/artifacts/feature_schema.json): định nghĩa
  feature, tất cả mô tả đặc trưng *person track*.
- [config/tau_val.json](config/tau_val.json): ngưỡng được chọn trên
  validation sequence-disjoint.
- [VisDrone2019-VID-test-dev](VisDrone2019-VID-test-dev): dataset đánh giá chuẩn,
  gồm 17 sequence, có sequences/ và annotations/; annotation chỉ chứa
  *person box* (category 1).
- [visdrone-train](visdrone-train): dữ liệu dùng để xây detector export, không
  phải bảng feature huấn luyện XGBoost; chỉ dùng cho detector *person*.

risk_dataset.csv trong [zip_extract_full/artifacts](zip_extract_full/artifacts) là bảng dữ
liệu huấn luyện/đánh giá được đóng gói trong ZIP. Archive ghi nhận tracking
chưa hoàn tất (completed=false, 52/80 sequence), nên metric risk archive
được báo cáo đúng trạng thái và không được trộn với metric video end-to-end.

## Đánh giá toàn hệ thống

Lệnh chuẩn chạy Bước 5 (tích hợp hệ thống) và Bước 6 (benchmark phần cứng), sau đó xuất báo cáo:

python step5_integrated_system.py
python step6_hardware_benchmark.py
python report_system_metrics.py

Các tuỳ chọn chạy nhanh cho Bước 5 (giới hạn số sequence/frame, không sửa code):

   NCKH_BACKEND=onnx NCKH_MAX_SEQS=3 NCKH_MAX_FRAMES=40 python step5_integrated_system.py
   NCKH_BACKEND=openvino python step5_integrated_system.py

Kết quả được lưu trong [results/](results/):

- step5: results/system_metrics_unified.csv, results/output_gate_log.json, outputs/protected_video.mp4
- step6: results/hardware_benchmark.csv
- report: results/system_report.md

Các tham số mặc định của protocol là CPU, imgsz=1024, confidence 0.25,
IoU 0.5, tối đa 200 frame/sequence, một repetition, warm-up 30 frame và
loại annotation có occlusion == 2. Đây là benchmark kiến trúc end-to-end,
không cộng hoặc trộn kết quả từ các thư mục bước riêng lẻ.

Hai script step5_integrated_system.py và step6_hardware_benchmark.py dùng
IMG_SIZE=1024, confidence 0.25, NMS IoU 0.70 để cân bằng precision/recall
trên detector người. Bước 5 dùng ByteTrack với high/low/new-track thresholds
0.10/0.01/0.08, match threshold 0.80 và score fusion bật. Detection và
tracking precision/recall/F1 được tính độc lập, ghép một-một trong cùng frame
ở IoU 0.5; chỉ category pedestrian (1) được chấm và annotation occlusion 2
bị loại. Đây là operating point recall-first: precision có thể giảm, nên cần
đọc cả hai metric. Bước 5 dùng box ứng viên detector (kèm track khi bỏ qua
detect) làm vùng blur; IoU coverage được đo trên box trước padding và chỉ trên
GT ở frame được phát hành. Confidence 0.25 giúp giảm false positive và diện
tích video bị blur so với ngưỡng rất thấp; không diễn giải coverage conditional
này như recall chuẩn của detector. PSNR/SSIM chỉ được tính trên frame được
phát hành (không tính frame bị gate chắn đen) để phản ánh chất lượng vùng che
thực tế.
Với cấu hình hiện tại, coverage trên lần chạy 17 sequence là
0.7073; precision YOLO là 0.0892, và 54.14% frame bị output gate chặn.
Coverage có điều kiện trên frame phát hành; nếu tính các GT nằm trong frame
bị chặn đen là được bảo vệ hoàn toàn, effective privacy coverage là 0.9619.
Phần hiệu quả toàn cục này đến từ việc loại bỏ nhiều frame, không đồng nghĩa
với video hữu dụng hoặc coverage IoU trên mọi frame.
Đây là kết quả exploratory vì threshold được chọn từ cùng 17 sequence, không
phải đánh giá holdout độc lập. Độ phân giải cao hơn làm tăng latency và VRAM.
Hai script tự chọn cuda:0 khi PyTorch CUDA khả dụng, nếu không sẽ chạy CPU.
Môi trường Windows/Python 3.15 đã kiểm tra với RTX 3050 và PyTorch CUDA 13.0:

python -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 `
  --index-url https://download.pytorch.org/whl/cu130

### Backend detector (Bước 5)

Hiện Bước 5 chỉ triển khai inference PyTorch. Biến `NCKH_BACKEND` được giữ để
chọn backend trong tương lai; nếu đặt `onnx` hoặc `openvino` mà chưa triển khai,
script sẽ in cảnh báo và fallback về PyTorch:

```powershell
python step5_integrated_system.py                                  # PyTorch (mặc định)
$env:NCKH_BACKEND="onnx";     python step5_integrated_system.py    # fallback PyTorch (cảnh báo)
$env:NCKH_BACKEND="openvino"; python step5_integrated_system.py    # fallback PyTorch (cảnh báo)
```

Các file ONNX (`models/best_raw_dyn.onnx`, `models/best_raw.onnx`) chỉ giữ
làm tham chiếu cho Bước 6 / xuất mô hình; chưa dùng trực tiếp trong Bước 5.
Chỉnh nhanh phạm vi đánh giá: `NCKH_MAX_SEQS`, `NCKH_MAX_FRAMES`,
`NCKH_OUT_METRICS`, `NCKH_OUT_GATE_LOG`.

Đo round-robin trên 1 frame thật (1360x765), 20 lần, FP32, cùng conf 0.0075 /
IoU 0.7: PyTorch 493.6 ms, ONNX + IOBinding 439.2 ms, OpenVINO 363.3 ms.
Parity PT vs ONNX trên 90 frame: 99.51% box khớp, IoU trung bình 0.9989,
chênh confidence tuyệt đối 0.00024, precision/recall/F1 gần như trùng.

### Chỉ số

- *YOLO detection recall/precision/F1* và *ByteTrack recall/precision/F1*:
  ghép một-một theo frame với ground-truth pedestrian category 1 ở IoU >= 0.5;
  loại annotation occlusion level 2. Không gộp detection và tracking thành một metric.
- *AP/AP50/AP75/APsmall/APmedium/APlarge/AR và HOTA/DetA/AssA/MOTA/IDF1*:
  chưa có trong benchmark hiện tại vì evaluator chưa xuất các metric này.
- *Region recall*: tỷ lệ ground-truth box được hợp các vùng bảo vệ phủ ít nhất
  50% diện tích.
- Khi cần đo upper-bound, có thể bật B8 recall_first để dùng vùng bảo vệ toàn
  khung. Đây không phải cấu hình triển khai vì làm mất tính hữu dụng của video.
- *ULFR*: tỷ lệ frame còn ít nhất một ground-truth box chưa được phủ; thấp hơn
  là tốt hơn. Đây là coverage-only.
- *Unsafe release*: frame được phát hành nhưng còn box chưa phủ; đây là
  coverage-proxy, khác với gate violation.
- *TTP p50/p95*: số frame từ lúc identity xuất hiện đến lần đầu được bảo vệ;
  identity không được bảo vệ là censored.
- *Calls/100 frames, FPS, p50/p95/p99 latency*: chi phí và thời gian end-to-end
  sau warm-up.
- *Blocked/correct block/false block/gate violation*: tính đúng của cổng phát
  hành, tách biệt với chất lượng detector.
- *Re-ID attack/protection success, PSNR, SSIM*: hiệu quả và chất lượng vùng
  che khi oracle Re-ID được chạy.
- Các chỉ số nhận dạng đối tượng khác ngoài người không nằm trong phạm vi đánh giá vì hệ thống chỉ sử dụng dữ liệu *person*.
- *XGBoost AUC, PR-AUC, Brier, F1 và ECE*: chất lượng risk model trên split
  sequence-disjoint; đây là metric của model, không thay thế metric end-to-end.
  Mô hình runtime được calibrate bằng Isotonic Regression trên OOF
  sequence-disjoint (lưu tại `models/xgboost_risk_calibrated.joblib`), giúp
  xác suất rủi ro gần xác suất thực hơn (ECE/Brier giảm rõ) và được đánh giá
  thêm trên temporal hold-out sequence-disjoint.
- *Energy/frame* chỉ có ý nghĩa khi đo bằng cảm biến công suất đã hiệu chuẩn.

## Giới hạn và cách diễn giải

1. Detector là detector *person* nên toàn bộ bảo vệ nhạy cảm là proxy; *chỉ dùng dữ liệu person*.
2. Kết quả detector/tracking trên VisDrone2019-VID-test-dev chỉ có ý nghĩa
   trong split này; không được gọi là kết quả production hoặc edge GPU.
3. Latency hiện đo trên CPU. overload-ms=250 là ngưỡng đo cho máy này, không
   đại diện cho deadline 40 ms của thiết bị edge GPU.
4. XGBoost có ECE cao, nên risk score chưa phải xác suất được calibration.
   Logistic Regression hiện có thể đạt AUC/PR-AUC tốt hơn XGBoost trên test.
5. Pipeline phục vụ dùng cửa sổ quá khứ, trong khi dữ liệu archive huấn luyện
   được mô tả là có cửa sổ nhìn tới tương lai. Đây là train/serve skew cần sửa
   khi có lại dữ liệu gốc.
6. ROI_DETECTION trong benchmark hiện dùng detection toàn frame từ cache; số
   lần gọi ROI vì vậy không chứng minh mức tiết kiệm GPU thực tế.
7. ULFR coverage-only không phát hiện trường hợp đã phủ nhưng Re-ID vẫn nhận
   đúng. Đánh giá riêng tư mạnh hơn cần nhãn hoặc oracle cross-sequence hợp lệ.
8. Các sequence/frame local có thể trùng với dữ liệu detector dùng để xây model;
   không dùng chúng để tuyên bố holdout độc lập nếu chưa kiểm tra hash và nguồn
   phân chia.
9. Kết quả chỉ hợp lệ cùng data_root, danh sách sequence, frame cap, model,
   threshold và protocol được ghi trong step6_metrics.json.

## Kiểm tra nhanh

python -c "import xgboost as xgb; m=xgb.XGBClassifier(); m.load_model('models/xgboost_risk.json'); print(m.n_features_in_)"