# Hệ thống bảo vệ riêng tư video UAV trên thiết bị biên

## Mục đích và phạm vi

Hệ thống nhận video UAV, phát hiện người, theo dõi đối tượng, ước lượng rủi ro
nhận dạng lại, chọn mức bảo vệ, che vùng nhạy cảm và chỉ phát hành frame qua
OutputGate. Đây là pipeline nghiên cứu trên *người (person)* trong VisDrone;
detector hiện tại là *model person*. Toàn bộ dữ liệu và metric đánh giá đều
tính trên hộp bounding box *person*.

Luồng xử lý chính:

```text
frame -> YOLO person (OpenVINO) -> ByteTrack -> 25 features -> LogisticRegression risk
      -> RiskScheduler (P0-P3/A1-A4) -> blur/pixelate -> verify -> OutputGate
```

Tracker runtime dùng `ultralytics.trackers.byte_tracker.BYTETracker` với cấu
hình đồng bộ notebook chuẩn `notebook4234ae1bfe.ipynb` (CELL 2/13 CONFIG):
high/low 0.35/0.05, new-track 0.25, match 0.70, buffer 45.

> **Một cấu hình, một chế độ, một bộ output.**
> Mọi tham số nằm trong `CONFIG` ở
> [step5_integrated_system.py](step5_integrated_system.py).
> Hệ thống luôn chạy **scheduler thích ứng + output gate**; không còn chế độ
> eval/deployable, không còn `NCKH_TRACK_EVAL` / `NCKH_ORACLE_GATE`.

## Chạy nhanh

```powershell
python run_pipeline.py all          # evaluate -> report (val)
python run_pipeline.py evaluate     # Bước 5: chạy hệ thống trên val
python run_pipeline.py report       # tổng hợp results/system_report.md
python run_pipeline.py calibrate    # hiệu chuẩn ngưỡng gate trên val
python run_pipeline.py verify       # kiểm tra CONFIG ↔ gate log ↔ metrics
```

Báo cáo held-out trên test-dev (không dùng để chọn tham số):

```powershell
python run_pipeline.py evaluate --split test-dev
python run_pipeline.py report   --split test-dev
```

### Nguyên tắc split

- **val** (`visdrone_val/VisDrone2019-VID-val`, 7 sequence, 2.846 frame): split
  **chọn tham số** (detector, tracker, ngưỡng gate) và hiệu chuẩn.
- **test-dev** (`VisDrone2019-VID-test-dev`, 17 sequence, 6.635 frame): split
  **held-out** dùng để **báo cáo**; không tham gia chọn tham số.

`visdrone-train` trong repo là bản YOLO của **đúng 7 sequence val** (cùng tên
`uav..._frame....jpg`, cùng 2.846 frame), nên **không phải split rời rạc** và
không dùng để chọn tham số "rời rạc" được.

Tham số của `calibrate`:

```powershell
python run_pipeline.py calibrate --target-block-rate 0.40
python run_pipeline.py calibrate --target-coverage 0.80
python run_pipeline.py calibrate --apply   # ghi ngưỡng mới vào CONFIG
```

`run_pipeline.py` là entrypoint duy nhất; các module còn lại là thư viện
(`step5_integrated_system.py`, `report_system_metrics.py`, ...).

## Thành phần và dữ liệu

- [final_extracted/artifacts/yolo26s_1024_openvino_model](final_extracted/artifacts/yolo26s_1024_openvino_model):
  detector *person* duy nhất (OpenVINO, export ở imgsz 1024).
- [final_extracted/artifacts/risk_model_calibrated.joblib](final_extracted/artifacts/risk_model_calibrated.joblib):
  risk model 25 feature (schema v2) + calibrator, train trên *person track*.
- [final_extracted/artifacts/risk_dataset.csv](final_extracted/artifacts/risk_dataset.csv):
  bảng nhãn risk dùng khi chạy runtime.
- `CONFIG` trong [step5_integrated_system.py](step5_integrated_system.py):
  **nguồn cấu hình duy nhất** cho mọi lệnh của `run_pipeline.py`.
- [visdrone_val/VisDrone2019-VID-val](visdrone_val/VisDrone2019-VID-val):
  **split chọn tham số** — 7 sequence, 2.846 frame (sequence-disjoint). Dùng cho
  hiệu chuẩn ngưỡng gate và mọi quyết định tham số.
- `VisDrone2019-VID-test-dev`: **split held-out để báo cáo** — 17 sequence,
  6.635 frame, có `target_id`; **không** dùng để chọn tham số.
- `visdrone-train`: bản YOLO của đúng 7 sequence val (cùng 2.846 frame), **không
  phải split rời rạc**; không dùng để chọn tham số.

`risk_dataset.csv` trong `final_extracted/artifacts` là bảng huấn luyện/đánh giá
đóng gói trong ZIP. Archive ghi tracking chưa hoàn tất (52/80 sequence), nên
metric risk archive được báo đúng trạng thái và không trộn với metric video
end-to-end.

## Cấu hình duy nhất

Mọi giá trị dưới đây nằm trong `CONFIG` của
[step5_integrated_system.py](step5_integrated_system.py). Không có biến môi
trường nào ghi đè operating point; biến `NCKH_*` chỉ còn là công tắc
debug/tăng tốc (xem bảng cuối).

| Khóa | Giá trị | Ghi chú |
|---|---|---|
| `notebook_source` | `notebook4234ae1bfe.ipynb` | nguồn chuẩn |
| `data_root` | `visdrone_val/VisDrone2019-VID-val` | split chọn tham số |
| `person_categories` | `[1, 2]` | pedestrian + people |
| `exclude_occlusion_2` | `false` | notebook không lọc occlusion |
| `backend` | `openvino` | detector duy nhất |
| `openvino_model` | `final_extracted/artifacts/yolo26s_1024_openvino_model` | thư mục model |
| `img_size` | 1024 | notebook CONFIG |
| `conf_thres` / `iou_thres` | 0.05 / 0.5 | notebook CONFIG |
| `max_det` | 1000 | notebook CONFIG |
| `det_tiles` / `tile_overlap` | `1x1` / 0.15 | `1x1` = nguyên frame |
| `max_frames_per_seq` | 0 = toàn bộ | protocol |
| `warmup_frames` | 30 | protocol |
| `track_high/low/new/match` | 0.35 / 0.05 / 0.25 / 0.80 | notebook + ablation val (nhóm 3) |
| `track_buffer` | 60 | ablation val: IDSW 973 → 486, IDF1 +7,3pp |
| `gmc` | `false` | ablation val: GMC giảm IDF1/HOTA |
| `track_all` | `false` | ByteTrack chuẩn |
| `blur_mode` / `blur_kernel` | `roi` / 15 | notebook `ANON_KSIZE` |
| `coverage_iou_threshold` / `containment_threshold` | 0.5 / 0.9 | protocol coverage |
| `tau` | 0.188065… | track-level fallback |
| `gate_target_block_rate` | 0.40 | chặn tối đa 40% frame |
| `gate_threshold` | 0.215140… | hiệu chuẩn trên val (block 0.3995) |
| `gate_risk_scale` / `gate_crowd_ref` | 0.5 / 168.0 | trọng số gate score |
| `iou_match_threshold` | 0.5 | ghép detection/tracking với GT |

Gate score = `leak_score + gate_risk_scale * frame_risk`, với
`leak_score = 0.5*no_track + 0.0*missing + 0.0*low_conf + 0.1*(crowd/crowd_ref)`
(tuned offline trên val). Ngưỡng hiệu chuẩn lưu ở
`results/gate_calibration.json`; không còn khối `CONFIG["calibration"]`.

Scheduler (`risk_low/medium/high/critical`, `detect_interval_*`,
`roi_expand_*`) nằm trong khối `scheduler` của cùng `CONFIG`.

### Vì sao block rate 0,40

Hiệu chuẩn trên val với mục tiêu **chặn 40% frame** cho ngưỡng **0,215140…**
(hiệu chuẩn lại trên records của tracker `bt_v3`); run hiện tại đạt block rate
**0,3995** (1137/2846 frame). Với ngân sách chặn này, privacy đo trên **frame đã
phát hành** đạt:

- `coverage_released_containment90` = **0,7744** (headline)
- `coverage_released_iou50` = **0,7520**
- `coverage_incl_blocked_containment90` = 0,8045 (phụ, tính cả frame bị chặn)
- `ulfr` = 0,5623 (unsafe leak frame rate)
- identity leak rate = 0,0090 (3/335 danh tính)

Block rate thấp hơn (0,20) cho coverage thấp hơn rõ rệt; 0,40 là điểm cân bằng
được chọn vì vẫn giữ utility ở mức chấp nhận được (blur-area 0,0735,
over-protection 0,5865). **Mọi số privacy phải đọc kèm block rate.**

`python run_pipeline.py calibrate` in **đường cong privacy–utility** (block rate
→ coverage) và **leave-one-sequence-out** để chứng minh ngưỡng không bị một
sequence chi phối. Số liệu val dùng để chọn tham số; số trích dẫn là
`results/system_metrics_testdev.csv` (held-out).

## Kết quả held-out (test-dev)

`python run_pipeline.py evaluate --split test-dev` (17 sequence, 6.635 frame) chạy
đúng ngưỡng và trọng số đã chọn trên val, **không hiệu chỉnh lại**:

| Chỉ số | val (selection) | test-dev (held-out) |
|---|---|---|
| Block rate | 0,3995 | **0,2360** (1566/6635) |
| Coverage released @ containment90 | 0,7744 | **0,5405** |
| Coverage released @ IoU 0.5 | 0,7520 | 0,5126 |
| Coverage incl. blocked @ containment90 | 0,8045 | 0,6673 |
| Blocked-GT share | 0,1335 | 0,2759 |
| ULFR | 0,5623 | 0,7545 |
| Unsafe release rate | 0,9387 | 0,9285 |
| Identity leak rate | 0,0090 | 0,0833 |
| TTP p50 / p95 (frame) | 0 / 14 | 0 / 49,0 |
| PSNR / SSIM | 38,18 / 0,9558 | 40,26 / 0,9791 |
| Blur-area / over-protection | 0,0735 / 0,5865 | 0,0361 / 0,5018 |
| Gate precision / recall | 0,9551 / 0,4103 | 0,6405 / 0,1779 |
| Frame-risk ROC-AUC / ECE | 0,4589 / 0,4097 | 0,8087 / 0,5593 |
| Detection recall / AP50 | 0,7050 / 0,5936 | 0,4985 / 0,3988 |
| MOTA / IDF1 / HOTA / IDSW | 0,3417 / 0,5224 / 0,3955 / 486 | 0,2179 / 0,3593 / 0,2913 / 1215 |
| `fps_stream` / `fps_wall` | 7,14 / 7,25 | 14,03 / 14,35 |

Ba nguyên nhân của khoảng cách val → test-dev, tất cả đều đo được và **không**
do đổi định nghĩa metric:

1. **Detector yếu hơn trên test-dev**: recall 0,7050 → 0,4985, AP50 0,5936 →
   0,3988 (test-dev có 7 độ phân giải 680x382–2720x1530 và khác domain). Không
   phát hiện được người thì không thể che, nên coverage released bị chặn trên
   bởi recall.
2. **Ngân sách chặn 40% trên val tập trung vào một sequence**: `uav0000268_05773_v`
   bị chặn 963/978 frame (98,5%) và chiếm 963/1137 = 85% số frame bị chặn của
   val; 6 sequence còn lại chỉ chặn 0–20%. Nói cách khác "40% của val" thực chất
   là "chặn gần hết sequence đông nhất". test-dev không có sequence nào giống vậy,
   nên block rate thực tế chỉ còn 0,2360.
3. **Gate chuyển miền kém, chủ yếu vì hạng tử `no_track`**: trên test-dev
   1.196/6.635 frame (18,0%) không có track, nhưng chỉ 52,9% trong số đó là
   unsafe (base rate 91,8% trên frame có GT) — chúng chiếm 75,8% số frame bị
   chặn (1187/1566). Ngược lại 379 frame bị chặn *có* track thì 100% unsafe.
   Trên val, frame không track chiếm 36,3% và 96,1% trong số đó unsafe (base
   95,4%), tức chặn chúng gần như miễn phí về coverage, nên quét trọng số chọn
   `no_track = 0.5`; sang test-dev cùng trọng số đó tiêu phần lớn ngân sách chặn
   vào frame gần như trống (415/1566 = 26,5% frame bị chặn không có GT nào).
   Ngoài ra risk model đổi miền: 7 sequence val không có trong
   `risk_dataset.csv` (OOD, ROC-AUC 0,4589), còn 13/17 sequence test-dev trùng
   split `test` của archive (ROC-AUC 0,8087), nên `gate_score` có precision
   0,6405 < base rate 0,9178 trên test-dev. Đây là **hạn chế chuyển miền của
   gate**, không phải lỗi của ngưỡng (ngưỡng vẫn giữ đúng vai trò giới hạn block
   rate).

Vì vậy con số trích dẫn held-out là **0,5405 containment90 trên frame đã phát
hành ở block rate 0,2360**. Muốn cải thiện phải làm detector mạnh hơn trên
test-dev hoặc thiết kế lại gate theo hướng bất biến theo sequence, rồi chọn lại
trên val — không được tune trực tiếp trên test-dev.

## Thực nghiệm chọn tham số (nhóm 2–4)

Mọi tham số được chọn trên **val**; test-dev chỉ chạy **một lần** sau khi chốt.
Công cụ và quy trình nằm trong [experiments/README.md](experiments/README.md).
`experiments/protocol_report.py` sinh bảng val ↔ test-dev và đường cong
privacy–utility; `experiments/compare_ablations.py` so sánh các ablation.

### Nhóm 3 — tracking

Ablation trên val, cùng detector và **không ghi video**. Baseline được chạy lại ở
ngưỡng gate cuối 0,215140, còn `bt_v2`/`bt_v3`/GMC chạy ở ngưỡng 0,215735, nên
block/coverage/ULFR chỉ đọc như tham khảo; delta tracking/IDSW là so sánh chặt.
FPS dao động lớn giữa các lần chạy trên cùng máy nên không dùng để phân định
(trừ GMC giảm rõ rệt):

| Run | Block | Coverage | ULFR | ID leak | MOTA | IDF1 | HOTA | IDSW | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (match 0,70, buffer 45) | 0,4023 | 0,7729 | 0,5580 | 0,0090 | 0,3280 | 0,4492 | 0,3578 | 973 | 11,71 |
| `bt_v2` (+ high 0,25) | 0,3791 | 0,7713 | 0,5807 | 0,0090 | 0,3213 | 0,5100 | 0,3891 | 557 | 8,98 |
| **`bt_v3` (match 0,80, buffer 60)** | **0,3946** | **0,7747** | 0,5666 | 0,0090 | **0,3417** | **0,5224** | **0,3955** | **486** | 9,39 |
| GMC `sparseOptFlow` | 0,3858 | 0,7748 | 0,5731 | 0,0090 | 0,2482 | 0,4109 | 0,3298 | 1293 | 5,75 |

`bt_v3` được chọn: giảm một nửa IDSW, IDF1 +7,3pp, HOTA +3,8pp, MOTA +1,4pp;
coverage gần như không đổi (chênh 0,18pp, cùng bậc với khác biệt ngưỡng gate),
ULFR +0,86pp. GMC bị loại vì hại tracking và giảm FPS một nửa. **Re-ID
association không được bật**: pipeline chỉ có một detector, không có appearance
encoder hay trọng số Re-ID trong `final_extracted/artifacts`, và thêm encoder sẽ
tăng chi phí mỗi frame trên thiết bị biên — trái với ràng buộc một mô hình. Thay
vào đó, nới ngưỡng association và giữ track lâu hơn, đúng phần chi phối IDSW của
cảnh UAV.

### Nhóm 2 — OutputGate

Bốn hướng thiết kế lại đã thử và **bị loại bằng số liệu val**, giữ nguyên
`gate_score = leak_score + 0,5·frame_risk` với trọng số hiện tại:

| Hướng | Kết quả val | Kết luận |
|---|---|---|
| Quét 72 tổ hợp trọng số (`risk_scale = 0,5`) | trần coverage 0,7753 (baseline 0,7732); id-leak 0,0090 → 0,0299; blocked-GT 0,1274 → 0,0489 | không đánh đổi |
| Ngân sách chặn theo từng sequence | coverage 0,7732 → 0,7293; blocked-GT share 0,135 → 0,411 | dồn ngân sách vào frame nhiều GT |
| Chuẩn hoá hạng risk theo sequence (oracle/causal) | 0,7752 / 0,7704 | không tăng coverage |
| Thả 26 frame "không có bằng chứng" (risk = 0) | coverage 0,7721 (baseline 0,7732), block 0,3935; cả 26 frame đều có GT | giảm coverage, không đáng đổi |

Các số 0,7732/0,1274 trong bảng này là baseline tại ngưỡng 0,215735 trên records
baseline (`results/abl_baseline_frame_records.json`, 72 tổ hợp trong
`results/gate_sweep_val.json` và `results/gate_policies_val.json`); bảng ablation ở
trên dùng baseline chạy lại ở ngưỡng cuối 0,215140, nên không trộn hai bộ số khi
so sánh. Cấu hình "bằng chứng" bỏ risk (`A4`) đạt 0,7770 nhưng id-leak 0,0269 —
cũng bị loại.

Nguyên nhân gốc của lãng phí ngân sách là **không phải** trọng số mà là
**chuyển miền**: trên test-dev 26,5% frame bị chặn không có GT, và `no_track`
chiếm 75,8% số frame bị chặn dù chỉ 52,9% trong số đó unsafe. Sửa được điều này
cần đặc trưng bất biến theo sequence, không thể làm bằng tinh chỉnh ngưỡng trên
val.

### Nhóm 4 — giao thức val → test-dev

Đường cong privacy–utility (chặn top-K theo `gate_score`):

| Block rate | val coverage | test-dev coverage | val ULFR | test-dev ULFR |
|---:|---:|---:|---:|---:|
| 0,00 | 0,7465 | 0,5582 | 0,9535 | 0,9178 |
| 0,10 | 0,7533 | **0,5613** | 0,8573 | 0,8479 |
| 0,20 | 0,7626 | 0,5520 | 0,7576 | 0,7934 |
| 0,30 | 0,7713 | 0,4871 | 0,6592 | 0,6869 |
| **0,40 (val operating point)** | **0,7744** | 0,4501 | 0,5620 | 0,5812 |
| 0,50 | 0,7591 | 0,4171 | 0,4636 | 0,4790 |

Hai split có **hình dạng đường cong khác nhau**: val đạt đỉnh quanh block 0,35–0,40,
test-dev đạt đỉnh quanh 0,10 rồi giảm. Đây là bằng chứng định lượng cho domain
shift: operating point 0,40 chọn trên val là tối ưu cho val nhưng **không** tối
ưu cho test-dev. Theo giao thức, không được lấy phát hiện này để chỉnh ngưỡng
trên test-dev; nó được báo cáo như một hạn chế và là hướng việc tiếp theo (chọn
operating point theo ràng buộc utility thay vì block rate cố định, hoặc hiệu
chuẩn lại khi có val gần miền test-dev hơn).

## Output

| File | Nội dung |
|---|---|
| `results/system_metrics.csv` | metric hệ thống trên **val** (chọn tham số) |
| `results/system_metrics_testdev.csv` | metric **held-out** trên test-dev |
| `results/output_gate_log.json` | gate log + protocol + calibration (val) |
| `results/output_gate_log_testdev.json` | gate log của run test-dev |
| `results/val_frame_records.json` | per-frame records để hiệu chuẩn lại |
| `results/gate_calibration.json` | kết quả `run_pipeline.py calibrate` |
| `results/stage_profile.json` | phân rã thời gian (khi `NCKH_PROFILE=1`) |
| `results/system_report.md` | báo cáo tổng hợp val |
| `results/system_report_testdev.md` | báo cáo held-out test-dev |
| `results/protocol_report.md` | bảng val ↔ test-dev + đường cong privacy–utility |
| `outputs/protected_video.mp4` | video demo đã che (val, toàn bộ 2.846 frame) |
| `outputs/protected_video_testdev.mp4` | video demo test-dev (`uav0000073_04464_v`, 312 frame) |

`report_system_metrics.py` đọc `evaluation_config` trong gate log và cảnh báo
ngay đầu báo cáo nếu số liệu không khớp `CONFIG` hiện hành.

Kết quả protocol cũ đã được xóa để tránh trích dẫn nhầm. Chỉ dùng
`results/system_metrics.csv` (val) và `results/system_metrics_testdev.csv`
(held-out).

## Một chế độ chạy

- Detect mỗi frame: mọi `detect_interval_*` = 1 (cấu hình đang dùng), nên
  không có frame nào bị bỏ detect; cơ chế scheduler thích ứng vẫn tồn tại và có
  thể bật lại bằng `CONFIG["scheduler"]`. Nếu bật skip-frame, frame bị bỏ detect
  dùng Kalman predict của ByteTrack (không gọi `update` với box rỗng).
- Output gate luôn bật; quyết định chặn chỉ dựa trên **gate score** (leak proxy
  runtime + frame risk), không dùng ground-truth. Frame bị chặn được ghi đen và
  tính là bảo vệ hoàn toàn.
- Detection AP được chấm trên **frame thực sự gọi detect** (`detection_ap50`),
  để scheduler không bị tính là false negative. Thêm
  `detection_ap50_tracked` / `detection_ap50_95_tracked` chấm trên track output
  ở **mọi frame** để đối chứng phần bị bỏ detect được Kalman lấp ở mức nào.
- Tracking (MOTA/MOTP/IDF1/HOTA/IDSW/Frag) chấm trên toàn bộ frame bằng
  TrackEval.

## Chỉ số

- *YOLO detection precision/recall/F1* và *ByteTrack precision/recall/F1*: ghép
  một-một theo frame với GT person category 1+2 ở IoU >= 0.5; giữ occlusion
  level 2. Không gộp detection và tracking thành một metric.
- *AP50 / AP@0.5:0.95*: COCO-style 101-point interpolation trên detection của
  detector person, chỉ trên frame có gọi detect.
- *MOTA/MOTP/IDF1/HOTA/DetA/AssA/IDSW/Frag/MT/ML*: tính bằng
  [TrackEval](https://github.com/JonathonLuiten/TrackEval) trên `target_id`
  của VisDrone VID. ID chỉ duy nhất trong từng sequence nên đánh giá theo
  sequence rồi gộp bằng `combine_sequences`. Đây là **nguồn duy nhất** cho
  metric tracking; nếu TrackEval không khả dụng, toàn bộ metric tracking ghi
  `-` (N/A), không ghi 0 giả.
- *Detection toàn cục*: precision/recall/F1 trên **mọi frame có GT**, kể cả
  frame scheduler bỏ detect; tách khỏi precision/recall trên frame có detect.
- *Region recall*: tỷ lệ GT box được hợp vùng bảo vệ phủ ít nhất 50% diện tích.
- *ULFR (unsafe leak frame rate)*: tỷ lệ frame phát hành còn ít nhất một GT box
  chưa được che >= 90% trên tổng số frame có GT. Frame bị chặn (đen) không tính
  là hở, nên ULFR là coverage-only và phải đọc cùng block rate.
- *Unsafe release rate*: tỷ lệ frame released còn GT hở trên số frame released
  có GT.
- *Identity leak rate*: tỷ lệ `target_id` không được bảo vệ ở bất kỳ frame nào;
  frame bị chặn được tính là bảo vệ hoàn toàn.
- *TTP p50/p95*: số frame từ lúc identity xuất hiện đến lần đầu được bảo vệ;
  identity không bao giờ được bảo vệ là censored.
- *Blur-area ratio* / *over-protection ratio*: mức blur trên frame released và
  phần blur lan ra ngoài người.
- *PSNR background / protected, SSIM*: chất lượng vùng che và utility của cảnh.
  Giá trị 100 dB là quy ước khi hai ảnh giống hệt trên mask.
- *Gate precision/recall/F1, false block rate, leak rate*: positive = frame
  nguy hiểm (nếu phát hành thì hở GT), dự đoán positive = frame bị chặn. GT chỉ
  dùng để gán nhãn **sau khi** gate đã quyết định.
- *Frame-risk ROC-AUC/PR-AUC/Brier/ECE*: calibration của frame risk theo nhãn
  unsafe, không phải metric chất lượng risk model trên `risk_dataset.csv`.
- *Calls/100 frames, FPS, p50/p95/p99 latency*: chi phí end-to-end sau warm-up.
- *Risk-model AUC, PR-AUC, Brier, F1, ECE*: chất lượng risk model trên split
  sequence-disjoint; model runtime là **LogisticRegression** đã calibrate
  (artifact `risk_model_calibrated.joblib`, `model_kind=logreg`, 25 feature).

## Detector

Dự án dùng **một detector duy nhất**: OpenVINO model trong
`final_extracted/artifacts/yolo26s_1024_openvino_model` (khóa
`openvino_model` trong `CONFIG`). Không còn nhánh PyTorch/ONNX/TensorRT, nên
không cần cài `onnxruntime` hay xuất engine riêng.

Thiết bị suy luận mặc định `intel:GPU.0` (iGPU Iris Xe), tự fallback về CPU nếu
không mở được (`resolve_openvino_device`). Đường chạy mặc định là
`FastOpenVINODetector`; đặt `NCKH_FAST_DETECT=0` để quay lại predictor của
Ultralytics khi cần đối chứng.

Nếu cần export lại từ checkpoint gốc (không nằm trong repo):

```powershell
yolo export model=<checkpoint.pt> format=openvino imgsz=1024
# trỏ CONFIG["openvino_model"] tới thư mục *_openvino_model mới
```

## Cài đặt

```powershell
# CPU
pip install -r requirements.txt

# GPU NVIDIA: gỡ bản CPU trước, rồi cài torch CUDA + phần còn lại
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.15.0+cu124 torchvision==0.29.2+cu124
pip install -r requirements-cuda.txt
```

Runtime tự chọn `cuda:0` khi PyTorch CUDA khả dụng, nếu không chạy CPU và in
cảnh báo rõ ràng khi phát hiện `nvidia-smi` có nhưng `torch.cuda.is_available()`
là False. Hoặc dùng `install.bat` (1 = CPU, 2 = CUDA).

## Hiệu năng và tăng tốc

Các tối ưu đang bật (không đổi định nghĩa metric):

1. **Blur theo ROI** thay vì blur toàn frame: cắt từng cửa sổ `box + kernel/2`,
   ghép mosaic, gọi `cv2.GaussianBlur` một lần rồi dán lại. Kết quả giống hệt
   từng pixel; chỉ giảm chi phí CPU.
2. **PSNR một lần diff** bằng `cv2.norm(NORM_L2SQR)`; suy ra cả PSNR toàn
   frame, PSNR nền và PSNR vùng che.
3. **SSIM chỉ tính trong vùng che** (`dilate(mask) + 6 px`) rồi cộng phần ngoài
   bằng 1.0; trùng bản toàn frame tới ~1e-16.
4. **SSIM chạy nền** qua `ThreadPoolExecutor`, lấy kết quả FIFO nên giá trị
   trung bình không đổi.
5. **Đọc JPEG trước bằng thread** (`FramePrefetcher`) và **ghi video ở thread
   riêng** (`AsyncVideoWriter`) với backpressure.
6. **Đo tách stage** bằng `NCKH_PROFILE=1`; bảng stage được đưa vào báo cáo.
7. **Detect bất đồng bộ**: một `ThreadPoolExecutor(max_workers=1)` chạy
   detect frame kế tiếp trong lúc luồng chính blur/mask/PSNR frame hiện tại
   (chỉ khi mọi `detect_interval_*=1`, tức cấu hình đang dùng).
8. **Fast OpenVINO path** (`FastOpenVINODetector`): letterbox + chuẩn hoá bằng
   numpy, gọi thẳng `CompiledModel` (bỏ lớp predictor của Ultralytics), NMS 1
   thread. Đã đối chứng **bit-identical** với đường Ultralytics trên toàn bộ
   464 frame của một sequence val và 17/17 sequence test-dev (kể cả frame
   382x680 bị phóng to).
9. **Vector hoá thống kê lân cận** (`frame_neighbor_stats`): bỏ vòng lặp
   Python `iou()` O(T²) mỗi frame.
10. **`cv2.copyTo` thay boolean fancy indexing** khi dán vùng blur (bit-identical,
    nhanh hơn ~5–12 ms/frame ở mask lớn).
11. **`cv2.meanStdDev`** cho `crop_quality_stats` thay `np.mean/np.std`
    (lệch ≤ 1e-16, không đổi metric).
12. **Warmup 3 inference** trước khi đo để hấp thụ ~20 s compile kernel đầu tiên
    của OpenVINO trên iGPU.

### Biến môi trường (chỉ debug/tăng tốc)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `NCKH_DEVICE` | tự chọn | Ép thiết bị, ví dụ `cpu`, `cuda:0` |
| `NCKH_BLUR_MODE` | `roi` | `full` để chạy lại cách cũ khi đối chứng |
| `NCKH_BLUR_KERNEL` | `15` | Kernel Gaussian khi đối chứng |
| `NCKH_CV_THREADS` | `min(4, số CPU)` | Số thread OpenCV (0 = tuần tự) |
| `NCKH_TORCH_THREADS` | mặc định PyTorch | Số thread PyTorch |
| `NCKH_DECODE_THREADS` | `3` | Số thread đọc trước JPEG (0 = tuần tự) |
| `NCKH_DECODE_PENDING` | `16` | Số frame giữ trong RAM khi đọc trước |
| `NCKH_SSIM_THREADS` | `2` | Số thread tính SSIM nền (0 = tính ngay) |
| `NCKH_SSIM_PENDING` | `8` | Số phép SSIM tối đa đang chờ |
| `NCKH_ASYNC_DETECT` | `1` | `0` để tắt detect bất đồng bộ (đối chứng FPS) |
| `NCKH_FAST_DETECT` | `1` | `0` để quay lại đường Ultralytics (đối chứng) |
| `NCKH_ENCODE_QUEUE` | `8` | Số frame xếp hàng cho thread ghi video |
| `NCKH_PROFILE` | `0` | `1` để in bảng stage + ghi JSON |
| `NCKH_PROFILE_OUT` | `results/stage_profile.json` | Nơi ghi profile |
| `NCKH_STRICT_METRICS` | `0` | `1` để tính lại PSNR/SSIM và assert trùng |
| `NCKH_ULTRALYTICS_LOG` | `ERROR` | Mức log của ultralytics khi đo |
| `NCKH_MAX_SEQS` / `NCKH_MAX_FRAMES` | `0` = tất cả | Chạy nhanh trên subset |
| `NCKH_SEQ` | rỗng | Lọc sequence, cách nhau dấu phẩy |
| `NCKH_WRITE_VIDEO` | `1` | `0` để tắt video |
| `NCKH_DEMO_OVERLAY` | `1` | `0` để ghi output thô (frame chặn = đen) |
| `NCKH_PROGRESS_EVERY` | `25` | Số frame giữa 2 dòng log (0 = tắt) |
| `NCKH_OUT_METRICS` / `NCKH_OUT_GATE_LOG` / `NCKH_OUT_VIDEO` / `NCKH_OUT_FRAME_RECORDS` | theo CONFIG | Đổi đường dẫn output khi debug |

### Định nghĩa FPS và trạng thái tiêu chí FPS ≥ 20

- `fps_wall` = `1000 * N / Σ latency`, với `latency` là thời gian xử lý từng
  frame (không gồm phần SSIM còn trong hàng đợi).
- `fps_stream` = `frames_done / thời gian tường`; `fps_stream_with_encode` cộng
  thêm thời gian thread ghi video trả nốt hàng đợi.

**Số đo hiện tại (val đầy đủ, 2.846 frame, detect mỗi frame, có ghi video demo):**

| Chỉ số | Giá trị |
|---|---|
| `fps_stream` | **7,14** |
| `fps_stream_with_encode` | **7,14** |
| `fps_wall` | **7,25** |
| latency p50 / p95 / p99 | 127,2 / 250,1 / 348,0 ms |
| detection calls/100 frame | 100 |
| thời gian toàn run | ~6,6 phút |

Khi tắt ghi video (`NCKH_WRITE_VIDEO=0`) cùng cấu hình đo được `fps_stream`
9,05 trên máy này; chênh lệch còn lại là encode demo và tranh chấp CPU giữa các
lần chạy. Con số dùng để so sánh ablation là `fps_wall` đo trong cùng điều kiện
(không ghi video).

**Tiêu chí FPS ≥ 20 KHÔNG đạt trên máy đo.** Đây là hạn chế phần cứng, không
phải lỗi logic, và được báo cáo trung thực thay vì hạ chuẩn privacy:

- Model OpenVINO cố định **1024 FP32**; raw inference ~24 ms + preprocess ~14 ms
  trên Iris Xe, cộng tranh chấp CPU với 4 core laptop.
- Không thể INT8 PTQ: `nncf` không có trong môi trường (`openvino 2026.4.1`).
- Không thể reshape sang 640/768/896: anchor/shape đã bake vào model export.
- MX330 (dGPU) không dùng được (~6,5 s/inference).
- Các tối ưu đã bật (async, fast path, vector hoá, `cv2.copyTo`) đưa FPS từ
  ~6,5 lên ~9,6 trên val; muốn ≥ 20 phải **giảm tần suất detect** (skip-frame),
  việc này làm thay đổi privacy/tracking metric đã hiệu chuẩn — không được thực
  hiện để giữ nguyên kết quả đã kiểm chứng.

Nếu chạy trên máy có GPU rời/CUDA hoặc model INT8 640, con số FPS sẽ khác; khi
đó phải chạy lại `evaluate` và `calibrate` vì operating point phụ thuộc tốc độ
detector.

### Đối chứng chất lượng

```powershell
# Blur ROI vs blur toàn frame trên cùng subset
$env:NCKH_MAX_SEQS="2"; $env:NCKH_WRITE_VIDEO="0"
python run_pipeline.py evaluate
$env:NCKH_BLUR_MODE="full"; $env:NCKH_OUT_METRICS="results/system_metrics_full.csv"
python run_pipeline.py evaluate

# Đối chứng PSNR/SSIM tối ưu: mỗi frame tính cả bản cũ và assert trùng
$env:NCKH_STRICT_METRICS="1"; $env:NCKH_SEQ="uav0000009_03358_v"
$env:NCKH_OUT_METRICS="results/system_metrics_strict.csv"
python run_pipeline.py evaluate

# Tìm nút thắt
$env:NCKH_PROFILE="1"; $env:NCKH_SEQ="uav0000009_03358_v"
python run_pipeline.py evaluate > results\run_profile.log 2>&1
python run_pipeline.py report
```

## Kiểm tra nhanh

```powershell
python -c "import joblib; cal=joblib.load('final_extracted/artifacts/risk_model_calibrated.joblib'); print(cal['model'].n_features_in_)"
```
