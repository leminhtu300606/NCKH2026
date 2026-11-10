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
hình: high/low threshold 0.30/0.05, new-track threshold 0.25, match threshold
0.95, buffer 30 và score fusion bật. Bộ ngưỡng này được chọn trên val
sequence-disjoint rồi giữ nguyên khi chạy test-dev (xem
[config/eval_config.json](config/eval_config.json)).

B1 đến B8 là các operating point của cùng pipeline:

- B1: phát hiện toàn frame mỗi frame.
- B2`/B3`/`B4`: giảm tần suất phát hiện, lần lượt kết hợp tracking.
- B5`/B6`: dùng XGBoost risk; B6 thêm output gate fail-safe.
- B7: thêm adaptive mask.
- B8: thêm motion hold, scheduler bốn action, runtime risk evidence và
  adaptive region protection. recall_first full-frame fallback chỉ dành cho
  thí nghiệm upper-bound và tắt mặc định để cấu hình có thể triển khai thực tế.

## Thành phần và dữ liệu

- [final_extracted/artifacts/yolo26s.pt](final_extracted/artifacts/yolo26s.pt): detector *person*.
- [final_extracted/artifacts/risk_model.json](final_extracted/artifacts/risk_model.json): XGBoost artifact
  từ archive, với 25 feature (schema v2), train trên dữ liệu *person*.
- [final_extracted/artifacts](final_extracted/artifacts): risk dataset, OOF predictions,
  feature schema, CV summary và metric gốc; toàn bộ sample là *person track*.
- [final_extracted/artifacts/risk_model.json.metadata.json](final_extracted/artifacts/risk_model.json.metadata.json):
  thứ tự feature bắt buộc khi phục vụ model.
- [final_extracted/artifacts/feature_schema.json](final_extracted/artifacts/feature_schema.json): định nghĩa
  feature, tất cả mô tả đặc trưng *person track*.
- [config/tau_val.json](config/tau_val.json): ngưỡng được chọn trên
  validation sequence-disjoint.
- [VisDrone2019-VID-test-dev](VisDrone2019-VID-test-dev): dataset đánh giá chuẩn,
  gồm 17 sequence, có sequences/ và annotations/; annotation chỉ chứa
  *person box* (category 1).
- [visdrone-train](visdrone-train): dữ liệu dùng để xây detector export, không
  phải bảng feature huấn luyện XGBoost; chỉ dùng cho detector *person*.

risk_dataset.csv trong [final_extracted/artifacts](final_extracted/artifacts) là bảng dữ
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

- step5 (deployable): results/system_metrics_system.csv, results/output_gate_log.json, outputs/protected_video.mp4
- step5 (eval, detect mọi frame): results/system_metrics_eval.csv, results/output_gate_log_eval.json
- step5 (bản tổng hợp dùng cho báo cáo): results/system_metrics_unified.csv
- step6: results/hardware_benchmark.csv
- report: results/system_report.md

Xem [results/README.md](results/README.md) để biết cấu hình, chính sách đánh
giá và ngày chạy của từng con số.

Các tham số mặc định của protocol (lấy từ notebook huấn luyện và artifact sinh
ra từ chính notebook đó) là CPU, imgsz=1024, confidence 0.15, NMS IoU 0.5,
max_det=1000, chạy toàn bộ frame của tập test (17 sequence, 6635 frame), một
repetition, warm-up 30 frame và loại annotation có occlusion == 2. Toàn bộ
tham số nằm trong [config/eval_config.json](config/eval_config.json). Đây là
benchmark kiến trúc end-to-end, không cộng hoặc trộn kết quả từ các thư mục
bước riêng lẻ.

Hai script step5_integrated_system.py và step6_hardware_benchmark.py dùng
IMG_SIZE=1024. Bước 5 dùng ByteTrack với ngưỡng
high/low/new-track 0.30/0.05/0.25, match threshold 0.95, buffer 30, score
fusion bật — đồng bộ với confidence detector 0.15 và được chọn trên val
sequence-disjoint. Detection và tracking precision/recall/F1 được tính độc lập, ghép
một-một trong cùng frame ở IoU 0.5; chỉ category pedestrian (1) được chấm và
annotation occlusion 2 bị loại. Đây là operating point recall-first: precision
có thể giảm, nên cần đọc cả hai metric. Bước 5 dùng box ứng viên detector (kèm
track khi bỏ qua detect) làm vùng blur; IoU coverage được đo trên box trước
padding và chỉ trên GT ở frame được phát hành. PSNR/SSIM chỉ được tính trên
frame được phát hành (không tính frame bị gate chắn đen) để phản ánh chất lượng
vùng che thực tế.

Các bộ số cũ (sinh với confidence 0.25, IoU 0.7, 200 frame/sequence,
ByteTrack 0.10/0.01/0.08/0.80) đã được dọn khỏi repo sau khi đồng bộ theo
notebook/artifact, nên không so sánh trực tiếp với kết quả hiện tại.

Với cấu hình hiện tại (test-dev 17 sequence, 6.635 frame), coverage IoU 0.5
trên frame phát hành là 0.4182; YOLO precision 0.5183 / recall 0.4887 trên
frame có detect (recall toàn cục 0.2661 vì chỉ gọi detect 53,8% frame), và
15,13% frame bị output gate chặn. Coverage có điều kiện trên frame phát hành;
nếu tính các GT nằm trong frame bị chặn đen là được bảo vệ hoàn toàn,
effective privacy coverage là 0.6982. Phần hiệu quả toàn cục này đến từ việc
loại bỏ một phần frame, không đồng nghĩa với video hữu dụng hoặc coverage IoU
trên mọi frame. Ngưỡng gate 0.35408 được hiệu chuẩn ở mức frame trên val
sequence-disjoint (7 sequence, 2.846 frame); do isotonic bão hoà ở 0.35408,
tỷ lệ chặn thực tế là 14,55% thay vì mục tiêu 5%, và không tune trên test-dev.
Độ phân giải cao hơn làm tăng latency và VRAM.
Hai script tự chọn cuda:0 khi PyTorch CUDA khả dụng, nếu không sẽ chạy CPU.
Môi trường Windows/Python 3.12 đã kiểm tra. Cài đặt:

# CPU (mặc định)
pip install -r requirements.txt

# Hoặc CUDA (GPU NVIDIA, đổi chỉ mục cu124 cho khớp CUDA của máy):
pip install -r requirements-cuda.txt

Hoặc dùng install.bat (chọn 1 = CPU, 2 = CUDA).

### Backend detector (Bước 5)

Bước 5 chạy được cả hai backend:

- `pytorch` (mặc định): dùng `final_extracted/artifacts/yolo26s.pt`.
- `onnx`: dùng `final_extracted/artifacts/yolo26s.onnx`. Lưu ý model này xuất
  head thô `(1, 5, 21504)` (nc=1 person, **không NMS trong graph**), nên
  postprocess trong script phải transpose → lọc conf → un-letterbox → NMS
  (`cv2.dnn.NMSBoxes`). Bản trước đây giả định output `(1, 300, 6)` nên trả về
  box rác (conf > 1, toạ độ sai); lỗi này đã được sửa.

```powershell
python step5_integrated_system.py                                  # PyTorch (mặc định)
$env:NCKH_BACKEND="onnx"; python step5_integrated_system.py        # ONNX Runtime
```

Video demo: mặc định bật, ghi ra `outputs/protected_video.mp4` (canvas
1280x720 vì 17 sequence test có 7 độ phân giải khác nhau). Video demo hiển thị
khung **đã che (blur)** cho cả frame bị gate chắn, kèm nhãn
`BLOCKED/RELEASED` và giá trị risk; metric vẫn tính trên frame thật của hệ
thống (frame bị chắn = đen) nên video không làm sai lệch số liệu.

Biến môi trường điều khiển nhanh: `NCKH_BACKEND`, `NCKH_MAX_SEQS`, `NCKH_MAX_FRAMES` (0 = tất cả), `NCKH_WRITE_VIDEO` (0 = tắt video),
`NCKH_DEMO_OVERLAY` (0 = ghi đúng output thô, frame bị chắn sẽ đen),
`NCKH_PROGRESS_EVERY` (số frame giữa 2 dòng log tiến độ, 0 = tắt),
`NCKH_OUT_METRICS`, `NCKH_OUT_GATE_LOG`, `NCKH_OUT_VIDEO`.

Biến môi trường cho protocol đánh giá:

- `NCKH_TRACK_EVAL=1`: bỏ qua scheduler, detect **mọi frame** để đo
  MOTA/IDF1/HOTA đúng protocol tracking (mặc định 0 = chế độ triển khai
  skip-frame + Kalman predict).
- `NCKH_ORACLE_GATE=1`: bật cổng containment dùng ground-truth; chỉ dùng để
  tham chiếu upper-bound, mặc định tắt và không dùng trong kết quả deployable.
- `NCKH_DATA_ROOT`: trỏ sang split khác (ví dụ
  `visdrone_val/VisDrone2019-VID-val`).
- `NCKH_SEQ`: lọc danh sách sequence, phân tách bằng dấu phẩy.
- `NCKH_TRACK_HIGH`, `NCKH_TRACK_LOW`, `NCKH_TRACK_NEW`, `NCKH_TRACK_MATCH`,
  `NCKH_TRACK_BUFFER`: override ngưỡng ByteTrack khi sweep trên val.

Bước 6 đo latency trên 1 frame thật (1360x765), 100 lần sau warm-up 10 lần,
cùng operating point với Bước 5 (đọc từ `config/eval_config.json`: imgsz=1024,
conf 0.15, IoU 0.5), FP32. Latency end-to-end bao gồm detect YOLO ->
tính 25 đặc trưng rủi ro -> XGBoost predict_proba -> GaussianBlur, phản ánh
chi phí thực tế. Báo cáo thêm tỉ lệ frame vượt deadline thiết bị biên (mặc
định 40 ms, tuỳ qua NCKH_EDGE_DEADLINE_MS). Kết quả chỉ đại diện cho máy
phát triển (CPU), không phải deadline 40 ms của thiết bị edge GPU.

### Chỉ số

- *YOLO detection recall/precision/F1* và *ByteTrack recall/precision/F1*:
  ghép một-một theo frame với ground-truth pedestrian category 1 ở IoU >= 0.5;
  loại annotation occlusion level 2. Không gộp detection và tracking thành một metric.
- *AP50/AP@0.5:0.95*: COCO-style 101-point interpolation trên detection của
  detector person.
- *MOTA/MOTP/IDF1/HOTA/DetA/AssA/IDSW/Frag/MT/ML*: tính bằng
  [TrackEval](https://github.com/JonathonLuiten/TrackEval) — evaluator chính
  thức của MOTChallenge — trên `target_id` của VisDrone VID. ID chỉ duy nhất
  trong từng sequence nên đánh giá theo sequence rồi gộp bằng
  `combine_sequences`. Nếu TrackEval không khả dụng, các metric này ghi `-`
  (N/A), không ghi 0 giả.
- *Detection toàn cục*: precision/recall/F1 trên **mọi frame có GT**, kể cả
  frame scheduler bỏ qua detect; đây là chỉ số trung thực cho chế độ tiết kiệm
  năng lượng, tách khỏi precision/recall trên frame thực sự gọi detect.
- *Region recall*: tỷ lệ ground-truth box được hợp các vùng bảo vệ phủ ít nhất
  50% diện tích.
- *ULFR (unsafe leak frame rate)*: tỷ lệ frame phát hành còn ít nhất một GT box
  chưa được che >= 90% trên tổng số frame có GT. Frame bị gate chặn (đen) không
  tính là hở, nên ULFR là coverage-only và phải đọc cùng block rate.
- *Unsafe release rate*: tỷ lệ frame released còn GT hở trên số frame released
  có GT; *unsafe GT boxes* / *leak exposure* là số GT-frame bị hở.
- *Identity leak rate*: tỷ lệ `target_id` không được bảo vệ ở bất kỳ frame nào;
  frame bị chặn được tính là bảo vệ hoàn toàn.
- *TTP p50/p95*: số frame từ lúc identity xuất hiện đến lần đầu được bảo vệ;
  identity không được bảo vệ là censored (báo kèm censored rate).
- *Blur-area ratio*: trung bình tỷ lệ diện tích frame bị blur (union mask,
  không double-count) trên frame released.
- *Over-protection ratio*: `1 - diện tích GT được phủ / diện tích đã blur`;
  cao hơn nghĩa là blur lan ra ngoài người nhiều, mất utility.
- *PSNR background / protected*: PSNR giới hạn theo mask nền và mask vùng che.
  PSNR vùng che thấp hơn là che mạnh hơn; PSNR nền cao là giữ được chi tiết
  cảnh. Giá trị 100 dB là quy ước khi hai ảnh giống hệt trên mask.
- *Gate precision/recall/F1, false block rate, leak rate*: coi positive =
  frame nguy hiểm (nếu phát hành thì hở GT), dự đoán positive = frame bị chặn.
  Ground-truth chỉ dùng để gán nhãn **sau khi** gate đã quyết định.
- *Frame-risk ROC-AUC/PR-AUC/Brier/ECE*: calibration của frame risk theo nhãn
  unsafe ở trên, không phải metric chất lượng model XGBoost trên
  `risk_dataset.csv`; hai nguồn nhãn không được trộn.
- Hỗ trợ đo upper-bound: có thể bật B8 recall_first để dùng vùng bảo vệ toàn
  khung. Đây không phải cấu hình triển khai vì làm mất tính hữu dụng của video.
- *Calls/100 frames, FPS, p50/p95/p99 latency*: chi phí và thời gian end-to-end
  sau warm-up.
- *Blocked/correct block/false block/gate violation*: tính đúng của cổng phát
  hành, tách biệt với chất lượng detector. Các metric gate dùng cùng nhãn
  unsafe như ULFR nên không phụ thuộc decision của gate.
- *Re-ID attack/protection success, PSNR, SSIM*: hiệu quả và chất lượng vùng
  che khi oracle Re-ID được chạy.
- Các chỉ số nhận dạng đối tượng khác ngoài người không nằm trong phạm vi đánh giá vì hệ thống chỉ sử dụng dữ liệu *person*.
- *XGBoost AUC, PR-AUC, Brier, F1 và ECE*: chất lượng risk model trên split
  sequence-disjoint; đây là metric của model, không thay thế metric end-to-end.
  Mô hình runtime được calibrate bằng Isotonic Regression trên OOF
  sequence-disjoint (lưu tại `final_extracted/artifacts/risk_model_calibrated.joblib`), giúp
  xác suất rủi ro gần xác suất thực hơn (ECE/Brier giảm rõ) và được đánh giá
  thêm trên temporal hold-out sequence-disjoint.
- *Energy/frame* chỉ có ý nghĩa khi đo bằng cảm biến công suất đã hiệu chuẩn.



## Kiểm tra nhanh

python -c "import xgboost as xgb; m=xgb.XGBClassifier(); m.load_model('final_extracted/artifacts/risk_model.json'); print(m.n_features_in_)"