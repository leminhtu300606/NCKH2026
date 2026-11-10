"""Bước 5: Tích hợp hệ thống hoàn chỉnh - phát hiện, theo dõi, ước lượng rủi ro,
điều phối thích ứng, che đối tượng và kiểm soát đầu ra.

Chỉ sử dụng các chỉ số được công nhận trong literature:
- Detection: Precision, Recall, F1 (ground-truth person box, IoU >= 0.5)
- Tracking: MOTA, MOTP, IDF1, IDSW, Frag
- Risk model: ROC AUC, PR AUC, Accuracy, Precision/Recall/F1, Brier Score, ECE
- Scheduler/Gate: Detection calls/100 frames, Average detection interval,
  Output gate block rate, Output gate release rate
- Privacy protection: PSNR (dB), SSIM, ULFR, unsafe release, TTP,
  identity leak, blur-area ratio, over-protection ratio, PSNR tách vùng
- Gate quality: confusion matrix, precision/recall/F1, false block rate,
  leak rate, ROC-AUC/PR-AUC/Brier/ECE của frame risk
- End-to-end: Detection Precision/Recall/F1, Latency p50/p95/p99, FPS,
  Detection calls/100 frames

Không tự tạo chỉ số mới.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Tắt cảnh báo spam từ ultralytics/YOLO (không ảnh hưởng kết quả)
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
warnings.filterwarnings("ignore", message=".*Use 'quantize' instead.*")

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import xgboost as xgb
from ultralytics.trackers.byte_tracker import BYTETracker

from ultralytics import YOLO

from metrics_unified import (
    iou,
    containment,
    iou_matrix,
    containment_matrix,
    average_precision_coco,
    precision_recall_f1_one_to_one,
    mota_motp,
    expected_calibration_error,
    privacy_utility_metrics,
    gate_quality_metrics,
)
from perf_utils import (
    configure_threads,
    resolve_device,
    resolve_half,
    roi_gaussian_blur,
    runtime_info,
    warn_if_cuda_missing,
)

try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

WEIGHTS_PATH = Path("final_extracted/artifacts/yolo26s.pt")
RISK_MODEL_PATH = Path("final_extracted/artifacts/risk_model.json")
RISK_CALIBRATED_PATH = Path("final_extracted/artifacts/risk_model_calibrated.joblib")
DATA_ROOT = Path(os.getenv("NCKH_DATA_ROOT", "VisDrone2019-VID-test-dev"))
SEQUENCES_DIR = DATA_ROOT / "sequences"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
OUT_VIDEO = Path(os.getenv("NCKH_OUT_VIDEO", "outputs/protected_video.mp4"))
OUT_GATE_LOG = Path(os.getenv("NCKH_OUT_GATE_LOG", "results/output_gate_log.json"))
OUT_METRICS = Path(os.getenv("NCKH_OUT_METRICS", "results/system_metrics_unified.csv"))
COORDINATOR_RULES = Path("config/coordinator_rules.json")
GATE_THRESHOLD_PATH = Path("config/gate_threshold_val.json")
TAU_PATH = Path("config/tau_val.json")
RISK_DATASET_PATH = Path("final_extracted/artifacts/risk_dataset.csv")

# ===== HỢP ĐỒNG ĐẶC TRƯNG (schema v2) =====
# 25 đặc trưng này phải khớp 1:1 với 2 notebook Kaggle
# (notebook4234ae1bfe.ipynb và notebook426604c70d.ipynb) ở cả thứ tự lẫn cách tính.
# 4 đặc trưng chẩn đoán (DROPPED_FEATURES) vẫn được tính để runtime còn nạp được
# model 21-feature cũ (LEGACY_FEATURES) và để đối chiếu với notebook.
FEATURES = [
    # chất lượng detection / appearance
    "conf", "conf_std5", "conf_delta", "conf_mean5", "conf_slope5",
    "lap_var_log", "brightness", "motion_blur_ratio", "crop_contrast",
    # độ ổn định track theo thời gian
    "track_age", "hit_ratio", "time_since_update",
    "speed_norm", "pred_uncert", "iou_prev", "scale_change",
    # hình học / phối cảnh
    "min_side_px", "area_norm", "height_norm", "center_y_norm", "dist_edge",
    # bối cảnh đám đông / detection
    "nearest_dist_norm", "n_neighbors_r2_norm", "overlap_max",
    "n_det_expected_ratio",
]
DROPPED_FEATURES = ["track_state", "aspect_dev", "is_trunc", "frame_det_delta"]
LEGACY_FEATURES = [
    "conf", "conf_std5", "conf_delta", "track_age", "hit_ratio",
    "time_since_update", "track_state", "min_side_px", "aspect_dev",
    "dist_edge", "is_trunc", "speed_norm", "pred_uncert", "lap_var_log",
    "brightness", "motion_blur_ratio", "nearest_dist_norm",
    "n_neighbors_r2_norm", "overlap_max", "frame_det_delta",
    "n_det_expected_ratio",
]
FEATURE_POOL = list(dict.fromkeys(FEATURES + DROPPED_FEATURES + LEGACY_FEATURES))
assert len(FEATURES) == 25, len(FEATURES)


def resolve_feature_names(model: Any) -> list[str]:
    """Chọn đúng danh sách đặc trưng theo model đang nạp.

    - Payload calibrated do notebook train ghi ra có khóa `features`.
    - Model XGBoost thô thì dựa vào `n_features_in_` (25 mới / 21 cũ).
    Nhờ vậy step5 chạy được cả model cũ lẫn model mới mà không cần sửa file.
    """
    if isinstance(model, dict) and model.get("features"):
        names = [str(f) for f in model["features"]]
    else:
        inner = model["model"] if isinstance(model, dict) and "model" in model else model
        n_in = getattr(inner, "n_features_in_", None)
        if n_in == len(FEATURES):
            names = list(FEATURES)
        elif n_in == len(LEGACY_FEATURES):
            names = list(LEGACY_FEATURES)
        else:
            names = list(FEATURES)
    missing = [f for f in names if f not in FEATURE_POOL]
    if missing:
        raise ValueError(f"Model yêu cầu đặc trưng runtime không tính được: {missing}")
    return names

# Giá trị mặc định; sẽ được override bởi config/eval_config.json trong main()
# để đảm bảo đồng nhất chỉ số giữa hệ thống và notebook huấn luyện Kaggle.
# Các giá trị dưới đây lấy đúng theo artifact sinh bởi notebook:
#   - IMGSZ = 1024 (notebook4234ae1bfe / notebook426604c70d, cell CONFIG)
#   - conf/iou track = 0.15 / 0.5 (final_extracted/artifacts/track_config.json)
#   - ByteTrack = tune trên val sequence-disjoint (0.30/0.05/0.25/0.95, buffer 30)
#   - MAX_DET = 1000 (cell CONFIG notebook)
IMG_SIZE = 1024
# Video demo: bật mặc định (outputs/protected_video.mp4). Tắt bằng
# NCKH_WRITE_VIDEO=0 hoặc "write_video": false trong config/eval_config.json
# nếu môi trường không có encoder tương thích.
WRITE_VIDEO = os.getenv("NCKH_WRITE_VIDEO", "1") == "1"
CONF_THRES = 0.15
IOU_THRES = 0.5
TRACK_HIGH_THRESH = 0.30
TRACK_LOW_THRESH = 0.05
TRACK_NEW_THRESH = 0.25
TRACK_MATCH_THRESH = 0.95
MAX_DET = 1000
# Video demo: ghi khung đã che + nhãn BLOCKED/RELEASED (không ảnh hưởng metric).
DEMO_OVERLAY = os.getenv("NCKH_DEMO_OVERLAY", "1") == "1"
PROTECT_PADDING = 0.15
# Cho phép chạy nhanh để so sánh backend mà không sửa code:
#   NCKH_BACKEND=pytorch NCKH_MAX_SEQS=3 NCKH_MAX_FRAMES=40 python step5_integrated_system.py
# pytorch mặc định vì nhanh hơn ONNX-CPU ở imgsz 1024 trên máy này; đặt
# NCKH_BACKEND=onnx nếu cần đối chứng backend ONNX.
DETECTOR_BACKEND = os.getenv("NCKH_BACKEND", "pytorch")
ONNX_MODEL_PATH = Path(os.getenv("NCKH_ONNX_MODEL", "final_extracted/artifacts/yolo26s.onnx"))
MAX_SEQS = int(os.getenv("NCKH_MAX_SEQS", "0"))  # 0 = tất cả sequence
MAX_FRAMES_PER_SEQ = int(os.getenv("NCKH_MAX_FRAMES", "0"))  # 0 = tất cả frame
WARMUP_FRAMES = 30
PROGRESS_EVERY = int(os.getenv("NCKH_PROGRESS_EVERY", "25"))  # 0 = tắt log tiến độ

# Chế độ đánh giá tracker: detect MỖI frame để đo MOTA/IDF1/HOTA đúng protocol.
# Mặc định tắt để giữ hành vi hệ thống thật (skip-frame + lan truyền track).
TRACK_EVAL = os.getenv("NCKH_TRACK_EVAL", "0") == "1"
# Cổng output dùng ground-truth (oracle) chỉ để tham chiếu upper-bound.
# Mặc định tắt: quyết định block không được dùng GT.
ORACLE_GATE = os.getenv("NCKH_ORACLE_GATE", "0") == "1"

CONF_WINDOW = 5
PRED_UNCERT_WINDOW = 5
EWMA_ALPHA = 0.3
TRACK_BUFFER_FRAMES = 30
TRUNC_MARGIN_PX = 2
NEIGHBOR_RADIUS_FRAC = 0.05
NEIGHBOR_K = 5
CROP_RESIZE = 224
SSIM_INTERVAL = 5

# Che khuon mat: blur theo ROI (mac dinh) thay vi blur toan frame.
#   NCKH_BLUR_MODE=roi   -> chi blur trong tung box (nhanh hon nhieu, ket qua
#                           trong vung che giong het blur toan frame)
#   NCKH_BLUR_MODE=full  -> cach cu, de doi chung/kiem tra PSNR
BLUR_MODE = os.getenv("NCKH_BLUR_MODE", "roi").strip().lower()
if BLUR_MODE not in {"roi", "full"}:
    print(
        f"[Bước 5] Cảnh báo: NCKH_BLUR_MODE='{BLUR_MODE}' không hợp lệ "
        "(chỉ nhận 'roi' hoặc 'full'), dùng mặc định 'roi'."
    )
    BLUR_MODE = "roi"
BLUR_KERNEL = int(os.getenv("NCKH_BLUR_KERNEL", "51"))

DEVICE = resolve_device()
configure_threads()
print(f"[Bước 5] Sử dụng thiết bị: {DEVICE} (CUDA: {torch.cuda.is_available()})")
if not torch.cuda.is_available():
    warn_if_cuda_missing("[Bước 5]")
print(
    f"[Bước 5] Blur mode: {BLUR_MODE} | kernel: {BLUR_KERNEL} | "
    f"cv_threads: {cv2.getNumThreads()} | torch_threads: {torch.get_num_threads()}"
)


def load_tau() -> float:
    """Ngưỡng tau chọn trên validation sequence-disjoint.

    `config/tau_val.json` được sinh bởi step_train_risk_model.py theo target FPR.
    Fallback 0.5 nếu chưa có, thay vì dùng artifact tau.json cũ (giá trị hỏng:
    AUC < 0.5, n_val = 0).
    """
    if TAU_PATH.exists():
        with TAU_PATH.open("r", encoding="utf-8") as f:
            return float(json.load(f).get("tau", 0.5))
    return 0.5


def load_eval_config() -> dict[str, Any]:
    """Đọc cấu hình đánh giá dùng chung từ config/eval_config.json.

    File này được thiết kế để dùng đồng nhất giữa step5_integrated_system.py
    và notebook huấn luyện trên Kaggle, giúp các chỉ số (AP, precision,
    recall, ...) tính toán giống hệt nhau ở cả hai nơi.
    """
    cfg_path = Path("config/eval_config.json")
    default_cfg = {
        # --- Detector (theo notebook CONFIG, cell 2/13) ---
        "conf_thres": 0.15,
        "iou_thres": 0.50,
        "img_size": 1024,
        "max_det": 1000,
        "backend": "pytorch",
        # device/half không đặt ở đây: perf_utils tự chọn cuda:0 khi có CUDA
        # (NCKH_DEVICE) và tự bật FP16 khi chạy CUDA (NCKH_HALF=0 để tắt).
        # --- Protocol đánh giá trên VisDrone (giữ như README/report) ---
        "iou_match_threshold": 0.5,
        "exclude_occlusion_2": True,
        "category_person": 1,
        # 0 = chạy toàn bộ frame của tập test (17 sequence / 6635 frame)
        "max_frames_per_seq": 0,
        "warmup_frames": 30,
        # --- ByteTrack: tune trên val sequence-disjoint (không phải yaml Kaggle) ---
        "track_high_thresh": 0.30,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.25,
        "track_match_thresh": 0.95,
        "track_buffer": 30,
        # --- Video demo ---
        "write_video": True,
        "video_fps": 30.0,
        "video_width": 1280,
        "video_height": 720,
    }
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Gộp với default để tránh thiếu khóa
        for k, v in default_cfg.items():
            cfg.setdefault(k, v)
        return cfg
    return default_cfg


def load_risk_model() -> Any:
    """Tải model calibration nếu hợp lệ, ngược lại fallback về XGBoost thô.

    File final_extracted/xgboost_risk_calibrated.joblib trong artifact có thể bị hỏng
    (joblib báo "input stream corrupted"). Trường hợp đó chỉ ghi cảnh báo ngắn
    gọn và dùng final_extracted/xgboost_risk.json.
    """
    if RISK_CALIBRATED_PATH.exists() and _HAS_JOBLIB:
        try:
            cal = joblib.load(str(RISK_CALIBRATED_PATH))
            if isinstance(cal, dict) and "model" in cal and "calibrator" in cal:
                print(
                    f"[Bước 5] Đã tải mô hình đã calibrate (ECE raw={cal.get('ece_raw', float('nan')):.4f} "
                    f"-> cal={cal.get('ece_calibrated', float('nan')):.4f}): {RISK_CALIBRATED_PATH}"
                )
            else:
                print(f"[Bước 5] Đã tải mô hình đã calibration: {RISK_CALIBRATED_PATH}")
            return cal
        except Exception:
            print(
                "[Bước 5] Cảnh báo: không đọc được model calibration "
                f"(file hỏng?), dùng model XGBoost thô thay thế."
            )
    model = xgb.XGBClassifier()
    model.load_model(str(RISK_MODEL_PATH))
    print(f"[Bước 5] Đã tải mô hình XGBoost thô: {RISK_MODEL_PATH}")
    return model


def run_onnx_detect(
    ort_session: Any,
    frame: np.ndarray,
    input_name: str,
    conf_thres: float,
    iou_thres: float,
    img_size: int,
) -> np.ndarray:
    """Chạy inference ONNX detector và trả về detections_np (x1,y1,x2,y2,conf,cls).

    Model ONNX (`final_extracted/artifacts/yolo26s.onnx`) xuất ra **head thô**
    của Ultralytics: (1, 4+nc, N) = (1, 5, 21504) với nc = 1 (person), KHÔNG có
    NMS nhúng trong graph. Vì vậy phải:
      1. transpose về (N, 4+nc) = (21504, 5);
      2. lọc theo conf, đổi cx,cy,w,h -> x1,y1,x2,y2;
      3. un-letterbox về toạ độ ảnh gốc (trừ padding top/left, chia scale);
      4. NMS bằng cv2.dnn.NMSBoxes (vector hoá) thay cho vòng lặp Python.
    """
    h, w = frame.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    top = (img_size - new_h) // 2
    left = (img_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)

    outputs = ort_session.run(None, {input_name: blob})
    raw = np.asarray(outputs[0])
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2 or raw.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    # (4+nc, N) -> (N, 4+nc); nếu model đã xuất (N, 4+nc) thì giữ nguyên.
    preds = raw.T if raw.shape[0] < raw.shape[1] else raw
    if preds.shape[1] < 5:
        return np.empty((0, 6), dtype=np.float32)

    if preds.shape[1] == 5:
        # nc = 1: cột thứ 5 chính là confidence của lớp person.
        confs = preds[:, 4]
        classes = np.zeros(len(preds), dtype=np.float32)
    else:
        # nc > 1: không có objectness (YOLOv8+), conf = max class score.
        scores = preds[:, 4:]
        classes = scores.argmax(axis=1).astype(np.float32)
        confs = scores.max(axis=1)

    keep = confs >= float(conf_thres)
    if not keep.any():
        return np.empty((0, 6), dtype=np.float32)

    boxes = preds[keep]
    confs = confs[keep]
    classes = classes[keep]

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = np.clip((cx - bw / 2.0 - left) / scale, 0.0, w)
    y1 = np.clip((cy - bh / 2.0 - top) / scale, 0.0, h)
    x2 = np.clip((cx + bw / 2.0 - left) / scale, 0.0, w)
    y2 = np.clip((cy + bh / 2.0 - top) / scale, 0.0, h)

    widths = x2 - x1
    heights = y2 - y1
    valid = (widths > 1.0) & (heights > 1.0)
    if not valid.any():
        return np.empty((0, 6), dtype=np.float32)
    x1, y1, widths, heights = x1[valid], y1[valid], widths[valid], heights[valid]
    confs, classes = confs[valid], classes[valid]

    # NMS vector hoá: cv2.dnn.NMSBoxes nhận box dạng [x, y, w, h].
    nms_boxes = np.stack([x1, y1, widths, heights], axis=1).tolist()
    idx = cv2.dnn.NMSBoxes(
        nms_boxes, confs.astype(np.float32).tolist(),
        float(conf_thres), float(iou_thres),
    )
    if idx is None or len(idx) == 0:
        return np.empty((0, 6), dtype=np.float32)
    idx = np.asarray(idx).reshape(-1)[:MAX_DET]

    det_out = np.column_stack(
        [x1[idx], y1[idx], x1[idx] + widths[idx], y1[idx] + heights[idx],
         confs[idx], classes[idx]]
    )
    return det_out.astype(np.float32)


def predict_risk_batch(risk_model: Any, X: np.ndarray) -> np.ndarray:
    """Dự đoán xác suất rủi ro cho cả ma trận đặc trưng (n_samples, n_features).

    Gọi `predict_proba` một lần cho toàn bộ track trong frame thay vì từng
    track, giảm overhead đáng kể trên frame đông người (VisDrone có thể có
    hàng chục track/frame).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.size == 0:
        return np.empty((0,), dtype=np.float64)

    if isinstance(risk_model, dict) and "model" in risk_model:
        raw = np.asarray(risk_model["model"].predict_proba(X), dtype=np.float64)[:, 1]
        cal = risk_model.get("calibrator")
        if cal is not None and "X_thresholds_" in cal and "y_values_" in cal:
            raw = np.interp(
                raw,
                np.asarray(cal["X_thresholds_"], dtype=np.float64),
                np.asarray(cal["y_values_"], dtype=np.float64),
            )
        return np.clip(raw, 0.0, 1.0)
    return np.asarray(risk_model.predict_proba(X), dtype=np.float64)[:, 1]


def predict_risk(risk_model: Any, X: np.ndarray) -> float:
    """Dự đoán xác suất rủi ro cho một vector đặc trưng.

    Hỗ trợ cả mô hình XGBoost thô và payload đã calibrate
    {'model': XGBClassifier, 'calibrator': _RiskCalibrator, ...}.
    """
    return float(predict_risk_batch(risk_model, X)[0])


def load_risk_labels_lookup() -> dict:
    """Tải nhãn rủi ro ground truth từ risk_dataset.csv để đánh giá model tại runtime.

    Lookup key: (sequence, frame_idx) -> list of label (0/1).
    Vì track_id runtime (ByteTrack gán mới) khác track_id trong artifact
    (rủi ro artifact được sinh từ một lần tracking cũ) và risk_dataset.csv
    không lưu box coordinates, chúng ta match theo (sequence, frame):
    nếu có bất kỳ artifact track nào trong cùng frame có label=1, frame
    đó được coi là rủi ro (conservative).
    """
    lookup: dict[tuple[str, int], list[int]] = {}
    if not RISK_DATASET_PATH.exists():
        print(
            f"[Bước 5] Cảnh báo: không tìm thấy {RISK_DATASET_PATH}, "
            "bỏ qua đánh giá risk model ground truth."
        )
        return lookup
    df = pd.read_csv(RISK_DATASET_PATH)
    n_skipped = 0
    for _, row in df.iterrows():
        if pd.isna(row.get("label")):
            n_skipped += 1
            continue
        seq = str(row["sequence"])
        ws = int(row["window_start"])
        we = int(row["window_end"])
        lbl = int(row["label"])
        for f in range(ws, we + 1):
            lookup.setdefault((seq, f), []).append(lbl)
    print(
        f"[Bước 5] Đã tải nhãn rủi ro cho {len(lookup)} (sequence, frame) từ {RISK_DATASET_PATH} "
        f"(bỏ qua {n_skipped} dòng không có nhãn)"
    )
    return lookup


NORM_MIN_SAMPLES = 50


class SequenceStandardizer:
    """Chuẩn hóa z-score online theo sequence, chỉ dùng dữ liệu quá khứ.

    Khớp với cách huấn luyện trong step_train_risk_model.py:
    - min_n vector đầu tiên giữ nguyên giá trị thô;
    - các vector sau chuẩn hóa bằng mean/std của các vector trước đó (Welford).
    Nhờ vậy model không bị lệch phân phối giữa các sequence (domain shift).
    """

    def __init__(self, n_features: int, min_n: int = NORM_MIN_SAMPLES):
        self.n = 0
        self.min_n = min_n
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.n == 0:
            self.n = 1
            self.mean = x.copy()
            return x.astype(np.float32)
        if self.n < self.min_n:
            out = x.copy()
        else:
            sd = np.sqrt(np.maximum(self.m2 / self.n, 1e-6))
            out = (x - self.mean) / sd
        self.n += 1
        d = x - self.mean
        self.mean = self.mean + d / self.n
        self.m2 = self.m2 + d * (x - self.mean)
        return out.astype(np.float32)


def limit_frames(images: list[Path], max_frames: int) -> list[Path]:
    """Giới hạn số frame/sequence; `max_frames <= 0` nghĩa là lấy toàn bộ."""
    if max_frames is None or int(max_frames) <= 0:
        return images
    return images[: int(max_frames)]


def load_annotations_with_ids(
    seq_name: str,
) -> tuple[dict[int, list[list[float]]], dict[int, list[int]]]:
    """Đọc annotation VisDrone, trả về box và `target_id` song song.

    Cột 2 của VisDrone VID là `target_id` (track ID do dataset cung cấp).
    Trước đây code bỏ qua cột này nên IDF1/HOTA không thể tính đúng. ID 0 là
    ID hợp lệ; ID chỉ duy nhất trong phạm vi một sequence.
    """
    ann_path = ANNOTATIONS_DIR / f"{seq_name}.txt"
    gt: dict[int, list[list[float]]] = {}
    gt_ids: dict[int, list[int]] = {}
    if not ann_path.exists():
        return gt, gt_ids
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            frame = int(parts[0])
            target_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            category = int(parts[7])
            occlusion = int(parts[9])
            if category != 1 or occlusion == 2:
                continue
            gt.setdefault(frame, []).append([x, y, x + w, y + h])
            gt_ids.setdefault(frame, []).append(target_id)
    return gt, gt_ids


def load_annotations(seq_name: str) -> dict[int, list[list[float]]]:
    """Chỉ trả về box (giữ API cũ cho các chỗ không cần track ID)."""
    return load_annotations_with_ids(seq_name)[0]


def expand_box(box: list[float], padding: float, img_w: int, img_h: int) -> list[float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    bw_exp = bw * (1.0 + padding)
    bh_exp = bh * (1.0 + padding)
    nx1 = max(0.0, cx - bw_exp / 2.0)
    ny1 = max(0.0, cy - bh_exp / 2.0)
    nx2 = min(float(img_w), cx + bw_exp / 2.0)
    ny2 = min(float(img_h), cy + bh_exp / 2.0)
    return [nx1, ny1, nx2, ny2]


def boxes_to_mask(boxes: list[list[float]], h: int, w: int) -> np.ndarray:
    """Rasterize union của các box thành mask uint8 (1 = thuộc vùng che).

    Dùng mask thay vì cộng diện tích từng box để box chồng nhau không bị
    double-count khi tính `blur_area_ratio` / `over_protection_ratio`.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        mask[y1:y2, x1:x2] = 1
    return mask


def blur_boxes(
    frame: np.ndarray,
    boxes: list[list[float]],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    if mask is None:
        mask = boxes_to_mask(boxes, h, w)
    if not mask.any():
        return frame.copy()
    if BLUR_MODE == "full":
        # Cách cũ: blur toàn frame rồi chỉ lấy pixel trong mask (~26 ms/frame
        # với 1360x765, kernel 51). Giữ lại để đối chứng PSNR.
        blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)
        out = frame.copy()
        bool_mask = mask.astype(bool)
        out[bool_mask] = blurred[bool_mask]
        return out
    # Mặc định: blur từng ROI, cùng kernel, cùng vùng ghi đè.
    return roi_gaussian_blur(frame, boxes, ksize=BLUR_KERNEL, mask=mask)


def compute_psnr_masked(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """PSNR chỉ trên các pixel thuộc mask (bool, shape HxW).

    Trả NaN nếu mask rỗng. Khi hai ảnh giống hệt nhau trên mask (MSE = 0),
    trả 100 dB thay vì +inf để trung bình có ý nghĩa (quy ước thường dùng).
    """
    if mask is None or not mask.any():
        return float("nan")
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    squared = diff * diff
    values = squared[mask]
    mse = float(np.mean(values, dtype=np.float64))
    if mse <= 0.0:
        return 100.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    # float32 đủ chính xác cho hiệu ảnh uint8 (giá trị <= 255, bình phương <= 65025
    # biểu diễn chính xác), còn trung bình được tích luỹ ở float64 -> MSE trùng
    # khớp với cách tính float64 cũ nhưng nhanh hơn ~2x do ít bộ nhớ hơn.
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse = float(np.mean(diff * diff, dtype=np.float64))
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 10.0 * math.log10((max_pixel ** 2) / mse)


def psnr_components(
    img1: np.ndarray, img2: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    """Tính 1 lần diff, trả về (PSNR toàn frame, PSNR nền, PSNR vùng che).

    Thay cho 3 lần quét toàn frame trước đây (`compute_psnr` + 2 lần
    `compute_psnr_masked`). Vì `img2` chỉ khác `img1` bên trong `mask`, tổng
    bình phương sai khác ở nền là 0 và có thể suy ra từ tổng toàn frame.

    Quy ước giữ nguyên như bản cũ:
    - PSNR toàn frame = +inf khi ảnh giống hệt (caller bỏ qua bằng isfinite).
    - PSNR trên một vùng = 100 dB khi vùng đó giống hệt.
    - PSNR nền = NaN khi mask phủ toàn ảnh (không còn pixel nền).
    """
    h, w = img1.shape[:2]
    channels = img1.shape[2] if img1.ndim == 3 else 1
    total_values = int(h) * int(w) * int(channels)

    mask_bool = mask.astype(bool) if mask is not None else np.zeros((h, w), dtype=bool)
    diff = img1.astype(np.float32) - img2
    squared = diff * diff

    if mask_bool.any():
        protected = squared[mask_bool]
        ssd_prot = float(np.sum(protected, dtype=np.float64))
        n_prot = int(protected.size)
    else:
        ssd_prot = 0.0
        n_prot = 0
    ssd_total = float(np.sum(squared, dtype=np.float64))
    n_bg = total_values - n_prot
    ssd_bg = ssd_total - ssd_prot  # chính xác: các số hạng là số nguyên <= 65025

    def _psnr(ssd: float, n: int) -> float:
        if n <= 0:
            return float("nan")
        if ssd <= 0.0:
            return 100.0
        mse = ssd / float(n)
        return 10.0 * math.log10((255.0 ** 2) / mse)

    psnr_full = _psnr(ssd_total, total_values)
    if ssd_total <= 0.0:
        psnr_full = float("inf")
    return psnr_full, _psnr(ssd_bg, n_bg), _psnr(ssd_prot, n_prot)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_metrics(
    gt_all: dict[str, dict[int, list[list[float]]]],
    pred_all: dict[str, dict[int, list[list[float]]]],
    pred_ids_all: dict[str, dict[int, list[int]]],
    pred_scores_all: dict[str, dict[int, list[float]]],
    gt_ids_all: dict[str, dict[int, list[int]]],
    detector_all: dict[str, dict[int, list[list[float]]]],
    frames_all: dict[str, int],
    latency_samples: list[float],
    detection_calls: int,
    detection_intervals: list[float],
    blocked_frames: int,
    released_frames: int,
    protected_gt_boxes: int,
    contained_gt_boxes: int,
    total_protect_gt_boxes: int,
    psnr_sum: float,
    psnr_count: int,
    ssim_sum: float,
    ssim_count: int,
    per_frame_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def box_metrics(
        predictions: dict[str, dict[int, list[list[float]] | list[tuple[list[float], float]]]],
        only_predicted_frames: bool = False,
    ) -> tuple[float, float, float, int, int]:
        total_gt = total_tp = total_fp = 0
        for seq, seq_gt in gt_all.items():
            seq_pred = predictions.get(seq, {})
            for frame_number, gts in seq_gt.items():
                # Với YOLO: chỉ chấm trên frame thực sự gọi detect, nếu không
                # scheduler giãn lịch sẽ làm recall detector thấp giả tạo.
                if only_predicted_frames and frame_number not in seq_pred:
                    continue
                raw_preds = seq_pred.get(frame_number, [])
                # Chuẩn hóa về list of box (loại bỏ conf nếu có)
                preds = []
                for p in raw_preds:
                    if isinstance(p, tuple):
                        preds.append(p[0])
                    else:
                        preds.append(p)
                total_gt += len(gts)
                if not gts or not preds:
                    total_fp += len(preds)
                    continue
                ious = np.array(
                    [[iou(pred, gt) for gt in gts] for pred in preds],
                    dtype=np.float64,
                )
                match_cost = np.where(ious >= 0.5, -1.0 - ious * 1e-6, 0.0)
                pred_indices, gt_indices = linear_sum_assignment(match_cost)
                matched = sum(ious[p, g] >= 0.5 for p, g in zip(pred_indices, gt_indices))
                total_tp += int(matched)
                total_fp += len(preds) - int(matched)
        total_fn = total_gt - total_tp
        precision = total_tp / max(1, total_tp + total_fp)
        recall = total_tp / max(1, total_gt)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        return precision, recall, f1, total_gt, total_tp + total_fp

    detection_precision, detection_recall, detection_f1, total_gt, total_pred = box_metrics(
        detector_all, only_predicted_frames=True
    )
    # Bản toàn cục: mọi frame có GT, kể cả frame scheduler bỏ qua detect.
    (
        detection_precision_global,
        detection_recall_global,
        detection_f1_global,
        total_gt_global,
        total_pred_global,
    ) = box_metrics(detector_all, only_predicted_frames=False)
    tracking_precision, tracking_recall, tracking_f1, _, total_tracking_pred = box_metrics(
        pred_all
    )
    # Tổng GT trên mọi frame (kể cả frame scheduler bỏ qua detect), dùng cho
    # blocked/effective coverage để không phụ thuộc lịch detect.
    total_gt_all = sum(
        len(boxes) for seq_gt in gt_all.values() for boxes in seq_gt.values()
    )

    avg_psnr = psnr_sum / max(1, psnr_count) if psnr_count > 0 else 0.0
    avg_ssim = ssim_sum / max(1, ssim_count) if ssim_count > 0 else 0.0

    # AP@0.5 và AP@0.5:0.95 (COCO/VOC style) cho detector
    detection_aps = average_precision_coco(
        detector_all, gt_all, iou_thresholds=[0.5, 0.75]
    )
    # average_precision_coco trả về AP cho mỗi iou threshold; chúng ta lấy AP@0.5
    # và trung bình các ngưỡng 0.5:0.05:0.95 (COCO-style).
    ap50 = detection_aps.get(0.5, 0.0)
    # AP@0.5:0.95: tính trung bình AP tại các ngưỡng 0.5, 0.55, ..., 0.95
    ap_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    detection_aps_all = average_precision_coco(
        detector_all, gt_all, iou_thresholds=ap_thresholds
    )
    ap50_95 = float(np.mean([detection_aps_all.get(t, 0.0) for t in ap_thresholds]))

    # MOTA/MOTP fallback (per-frame matching). IDF1/HOTA thật lấy từ TrackEval;
    # nếu TrackEval lỗi thì ghi N/A (NaN), tuyệt đối không ghi 0 giả.
    tracking_mot = mota_motp(gt_all, pred_all)
    mot_metrics: dict[str, float] = {}
    try:
        from mot_eval import evaluate_trackeval

        mot_metrics = evaluate_trackeval(
            gt_all, gt_ids_all, pred_all, pred_ids_all, pred_scores_all
        )
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        print(f"[Bước 5] Cảnh báo: không chạy được TrackEval: {exc}")
    if mot_metrics:
        tracking_mot = {
            "mota": mot_metrics["mota"],
            "motp": mot_metrics["motp"],
        }
    else:
        print(
            "[Bước 5] Cảnh báo: IDF1/HOTA/HOTA không khả dụng -> ghi N/A. "
            "Kiểm tra trackeval hoặc annotation target_id."
        )

    total_frames = sum(frames_all.values())
    output_gate_block_rate = blocked_frames / max(1, total_frames)
    output_gate_release_rate = released_frames / max(1, total_frames)
    detection_calls_per_100_frames = 100.0 * detection_calls / max(1, total_frames)
    avg_detection_interval = float(np.mean(detection_intervals)) if detection_intervals else 0.0
    # Tỷ lệ GT box người được protect box phủ với IoU >= 0.5.
    protect_box_coverage = protected_gt_boxes / max(1, total_protect_gt_boxes)
    # Tỷ lệ GT box được protect box che >= 90% diện tích (đúng bản chất ROI padding).
    protect_box_containment = contained_gt_boxes / max(1, total_protect_gt_boxes)
    blocked_gt_boxes = max(0, total_gt_all - total_protect_gt_boxes)
    effective_privacy_coverage = (
        protected_gt_boxes + blocked_gt_boxes
    ) / max(1, total_gt_all)

    fps_wall = (
        1000.0 * len(latency_samples) / max(1e-9, sum(latency_samples))
        if latency_samples else 0.0
    )
    latency_arr = np.array(latency_samples) if latency_samples else np.array([0.0])

    metrics = {
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "detection_f1": detection_f1,
        "detection_precision_global": detection_precision_global,
        "detection_recall_global": detection_recall_global,
        "detection_f1_global": detection_f1_global,
        "detection_gt_boxes_global": total_gt_global,
        "detection_pred_boxes_global": total_pred_global,
        "tracking_precision": tracking_precision,
        "tracking_recall": tracking_recall,
        "tracking_f1": tracking_f1,
        "avg_psnr_db": avg_psnr,
        "avg_ssim": avg_ssim,
        "detection_ap50": ap50,
        "detection_ap50_95": ap50_95,
        "tracking_mota": tracking_mot["mota"],
        "tracking_motp": tracking_mot["motp"],
        "tracking_idf1": mot_metrics.get("idf1", float("nan")),
        "tracking_idp": mot_metrics.get("idp", float("nan")),
        "tracking_idr": mot_metrics.get("idr", float("nan")),
        "tracking_hota": mot_metrics.get("hota", float("nan")),
        "tracking_deta": mot_metrics.get("deta", float("nan")),
        "tracking_assa": mot_metrics.get("assa", float("nan")),
        "tracking_idsw": mot_metrics.get("idsw", float("nan")),
        "tracking_frag": mot_metrics.get("frag", float("nan")),
        "tracking_mt": mot_metrics.get("mt", float("nan")),
        "tracking_ml": mot_metrics.get("ml", float("nan")),
        "tracking_tp": mot_metrics.get("clr_tp", float("nan")),
        "tracking_fp": mot_metrics.get("clr_fp", float("nan")),
        "tracking_fn": mot_metrics.get("clr_fn", float("nan")),
        "tracking_eval_sequences": mot_metrics.get("mot_eval_sequences", float("nan")),
        "output_gate_block_rate": output_gate_block_rate,
        "output_gate_release_rate": output_gate_release_rate,
        "protect_box_coverage_iou50": protect_box_coverage,
        "protect_box_containment90": protect_box_containment,
        "blocked_gt_boxes": blocked_gt_boxes,
        "effective_privacy_coverage": effective_privacy_coverage,
        "protected_gt_boxes": protected_gt_boxes,
        "contained_gt_boxes": contained_gt_boxes,
        "total_protect_gt_boxes": total_protect_gt_boxes,
        "detection_calls_per_100_frames": detection_calls_per_100_frames,
        "average_detection_interval_frames": avg_detection_interval,
        "fps_wall": fps_wall,
        "latency_p50_ms": float(np.percentile(latency_arr, 50)),
        "latency_p95_ms": float(np.percentile(latency_arr, 95)),
        "latency_p99_ms": float(np.percentile(latency_arr, 99)),
        "detection_calls": detection_calls,
        "total_gt_boxes": total_gt,
        "total_gt_boxes_all": total_gt_all,
        "total_pred_boxes": total_pred,
        "total_tracking_pred_boxes": total_tracking_pred,
    }

    # Metric bổ sung từ per-frame records: privacy/utility + gate quality.
    # Ghi NaN (không ghi 0 giả) khi thiếu dữ liệu, ví dụ checkpoint cũ không
    # sinh per-frame records.
    if per_frame_records:
        metrics.update(privacy_utility_metrics(per_frame_records))
        metrics.update(gate_quality_metrics(per_frame_records))

    return metrics


def crop_quality_stats(
    gray: np.ndarray,
    box: list[float],
    resize: int = CROP_RESIZE,
) -> tuple[float, float, float]:
    """Trả về (lap_var, brightness, contrast) của crop, giống hệt lúc train.

    Notebook train cắt crop theo box của TRACKER rồi resize cạnh dài về
    CROP_RESIZE trước khi tính Laplacian, nên runtime phải làm y như vậy;
    nếu không `lap_var_log` / `crop_contrast` sẽ lệch phân phối.
    """
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(x1 + 1, min(x2, w)))
    y2 = int(max(y1 + 1, min(y2, h)))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, 0.5, 0.0
    if resize and max(crop.shape[:2]) > resize:
        s = resize / max(crop.shape[:2])
        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
            interpolation=cv2.INTER_AREA,
        )
    lap = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    bright = float(crop.mean() / 255.0)
    contrast = float(crop.std() / 255.0)
    return lap, bright, contrast


class TrackHistory:
    def __init__(self, track_id: int, first_frame: int):
        self.track_id = track_id
        self.first_frame = first_frame
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.prev_frame = None
        self.conf_history: list[float] = []
        self.center_history: list[tuple[float, float]] = []
        self.lap_history: list[float] = []
        self.box_history: list[list[float]] = []
        self.area_history: list[float] = []
        self.brightness_history: list[float] = []
        self.contrast_history: list[float] = []
        self.box: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.conf: float = 0.5

    def update(
        self,
        frame: int,
        box: list[float],
        conf: float,
        lap_var: float,
        brightness: float = 0.5,
        contrast: float = 0.0,
    ) -> None:
        self.age = max(1, frame - self.first_frame + 1)
        self.hits += 1
        self.time_since_update = 0 if self.prev_frame is None else max(0, frame - self.prev_frame - 1)
        self.prev_frame = frame
        self.box = box
        self.conf = conf
        self.conf_history.append(conf)
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        self.center_history.append((cx, cy))
        self.lap_history.append(lap_var)
        self.box_history.append(list(box))
        self.area_history.append(max(box[2] - box[0], 1e-6) * max(box[3] - box[1], 1e-6))
        self.brightness_history.append(float(brightness))
        self.contrast_history.append(float(contrast))
        # Giới hạn bộ nhớ lịch sử với track rất dài.
        keep = 64
        for attr in (
            "conf_history", "center_history", "lap_history", "box_history",
            "area_history", "brightness_history", "contrast_history",
        ):
            hist = getattr(self, attr)
            if len(hist) > keep:
                setattr(self, attr, hist[-keep:])

    def mark_missed(self, frame: int) -> None:
        self.time_since_update += 1
        self.age = max(1, frame - self.first_frame + 1)

    @property
    def track_state_code(self) -> float:
        if self.age == 1:
            return 0.0
        if self.time_since_update == 0:
            return 1.0
        return 2.0

    def compute_features(
        self,
        frame: int,
        img_w: int,
        img_h: int,
        all_boxes_frame: list[list[float]],
        prev_det_count: int | None,
        ewma_det: float | None,
        r_ref: float,
    ) -> dict[str, float]:
        D = math.hypot(img_w, img_h)
        box = self.box
        conf = self.conf
        bw = max(box[2] - box[0], 1e-6)
        bh = max(box[3] - box[1], 1e-6)
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0

        f_conf = conf
        # conf_history đã chứa conf hiện tại (update() gọi trước compute_features).
        c5 = self.conf_history[-CONF_WINDOW:] if CONF_WINDOW > 1 else [conf]
        f_conf_std5 = float(np.std(c5)) if len(c5) > 1 else 0.0
        f_conf_delta = 0.0 if len(self.conf_history) < 2 else conf - self.conf_history[-2]
        f_conf_mean5 = float(np.mean(c5)) if c5 else conf
        f_conf_slope5 = 0.0
        if len(c5) >= 2:
            x = np.arange(len(c5), dtype=np.float64)
            xm = float(x.mean())
            denom = float(((x - xm) ** 2).sum())
            if denom > 0:
                y = np.asarray(c5, dtype=np.float64)
                f_conf_slope5 = float(((x - xm) * (y - y.mean())).sum() / denom)
        f_track_age = float(self.age)
        f_hit_ratio = min(1.0, self.hits / max(self.age, 1))
        f_time_since_update = float(self.time_since_update)
        f_track_state = self.track_state_code
        f_min_side_px = float(min(bw, bh))
        f_aspect_dev = float(abs(bw / bh - r_ref))
        f_area_norm = float((bw * bh) / max(img_w * img_h, 1.0))
        f_height_norm = float(bh / max(img_h, 1.0))
        f_center_y_norm = float(cy / max(img_h, 1.0))
        f_dist_edge = float(max(0.0, min(box[0], box[1], img_w - box[2], img_h - box[3]) / D))
        f_is_trunc = 1.0 if min(box[0], box[1], img_w - box[2], img_h - box[3]) <= TRUNC_MARGIN_PX else 0.0
        f_speed_norm = 0.0
        if self.prev_frame is not None and len(self.center_history) >= 2:
            f_speed_norm = float(np.hypot(cx - self.center_history[-2][0], cy - self.center_history[-2][1]) / D)
        cen_win = self.center_history[-(PRED_UNCERT_WINDOW - 1):] + [(cx, cy)] if PRED_UNCERT_WINDOW > 1 else [(cx, cy)]
        f_pred_uncert = 0.0
        if len(cen_win) > 1:
            arr = np.array(cen_win)
            f_pred_uncert = float(np.sqrt(arr[:, 0].var() + arr[:, 1].var()) / D)
        f_iou_prev = 0.0
        f_scale_change = 0.0
        if len(self.box_history) >= 2:
            f_iou_prev = float(iou(self.box_history[-2], box))
            prev_area = self.area_history[-2]
            if prev_area > 0:
                f_scale_change = float(np.clip(math.log(max(self.area_history[-1], 1e-6) / prev_area), -5.0, 5.0))
        f_lap_var_log = float(math.log1p(self.lap_history[-1])) if self.lap_history else 0.0
        f_brightness = float(self.brightness_history[-1]) if self.brightness_history else 0.5
        f_crop_contrast = float(self.contrast_history[-1]) if self.contrast_history else 0.0
        f_motion_blur_ratio = 0.0
        if len(self.lap_history) >= 2:
            L_prev = self.lap_history[-2]
            L_cur = self.lap_history[-1]
            f_motion_blur_ratio = float((L_prev - L_cur) / (L_prev + L_cur + 1e-6))
        f_nearest_dist_norm = 0.0
        f_n_neighbors_r2_norm = 0.0
        f_overlap_max = 0.0
        if len(all_boxes_frame) > 1:
            centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in all_boxes_frame])
            my_idx = None
            for i, b in enumerate(all_boxes_frame):
                if iou(b, box) > 0.9:
                    my_idx = i
                    break
            if my_idx is not None:
                dists = np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)
                dists[my_idx] = np.inf
                f_nearest_dist_norm = float(np.clip(dists.min() / D, 0, 1))
                radius = NEIGHBOR_RADIUS_FRAC * D
                # dists[my_idx] đã là inf nên không trừ 1 (khớp notebook).
                n_nb = int((dists <= radius).sum())
                f_n_neighbors_r2_norm = float(min(n_nb / NEIGHBOR_K, 1.0))
                ious = []
                for j, b2 in enumerate(all_boxes_frame):
                    if j == my_idx:
                        continue
                    ious.append(iou(box, b2))
                f_overlap_max = float(max(ious)) if ious else 0.0
        N_t = len(all_boxes_frame)
        if prev_det_count is None:
            f_frame_det_delta = 0.0
        else:
            f_frame_det_delta = float((N_t - prev_det_count) / (prev_det_count + 1))
        expected = N_t if ewma_det is None else ewma_det
        f_n_det_expected_ratio = float(N_t / (expected + 1.0))

        return {
            "conf": f_conf,
            "conf_std5": f_conf_std5,
            "conf_delta": f_conf_delta,
            "conf_mean5": f_conf_mean5,
            "conf_slope5": f_conf_slope5,
            "lap_var_log": f_lap_var_log,
            "brightness": f_brightness,
            "motion_blur_ratio": f_motion_blur_ratio,
            "crop_contrast": f_crop_contrast,
            "track_age": f_track_age,
            "hit_ratio": f_hit_ratio,
            "time_since_update": f_time_since_update,
            "speed_norm": f_speed_norm,
            "pred_uncert": f_pred_uncert,
            "iou_prev": f_iou_prev,
            "scale_change": f_scale_change,
            "min_side_px": f_min_side_px,
            "area_norm": f_area_norm,
            "height_norm": f_height_norm,
            "center_y_norm": f_center_y_norm,
            "dist_edge": f_dist_edge,
            "nearest_dist_norm": f_nearest_dist_norm,
            "n_neighbors_r2_norm": f_n_neighbors_r2_norm,
            "overlap_max": f_overlap_max,
            "n_det_expected_ratio": f_n_det_expected_ratio,
            # 4 đặc trưng chẩn đoán (không dùng cho model 25-feature,
            # nhưng giữ để nạp được model 21-feature cũ).
            "track_state": f_track_state,
            "aspect_dev": f_aspect_dev,
            "is_trunc": f_is_trunc,
            "frame_det_delta": f_frame_det_delta,
        }


def classify_risk_level(risk: float, rules: dict[str, Any]) -> str:
    if risk >= rules.get("risk_critical", 0.75):
        return "critical"
    if risk >= rules.get("risk_high", 0.50):
        return "high"
    if risk >= rules.get("risk_low", 0.25):
        return "medium"
    return "low"


def decide_roi_padding(risk: float, rules: dict[str, Any]) -> float:
    """Padding vùng bảo vệ theo mức rủi ro (ROI expansion thích ứng).

    Đây là phần `roi_expand_*` trong config trước đây bị bỏ qua vì runtime
    luôn dùng PROTECT_PADDING = 0.0.
    """
    level = classify_risk_level(risk, rules)
    default = {"low": 0.10, "medium": 0.20, "high": 0.30, "critical": 0.50}[level]
    padding = float(rules.get(f"roi_expand_{level}", default))
    return max(PROTECT_PADDING, padding)


def resolve_gate_threshold(
    rules: dict[str, Any], tau: float | None = None
) -> float:
    """Ngưỡng gate phải nằm cùng không gian xác suất với model runtime.

    Runtime dùng model đã calibrate (isotonic, dải xác suất thực tế ~0..0.354),
    nên ngưỡng phải được chọn trên chính phân bố frame-risk của split val
    sequence-disjoint (`config/gate_threshold_val.json`). `tau_val.json` là
    ngưỡng theo track, dùng trực tiếp ở mức frame (max trên các track) sẽ chặn
    gần như toàn bộ frame, nên chỉ còn là fallback.

    Thứ tự ưu tiên: `NCKH_GATE_THRESH` > `config/gate_threshold_val.json` >
    `tau` > `output_gate_block_threshold`.
    """
    override = os.getenv("NCKH_GATE_THRESH")
    if override:
        return float(override)
    if GATE_THRESHOLD_PATH.exists():
        try:
            payload = json.loads(GATE_THRESHOLD_PATH.read_text(encoding="utf-8"))
            value = payload.get("threshold")
            if value is not None:
                return float(value)
        except Exception:
            print(
                f"[Bước 5] Cảnh báo: không đọc được {GATE_THRESHOLD_PATH}, "
                "fallback sang tau/config."
            )
    if tau is not None:
        return float(tau)
    return float(rules.get("output_gate_block_threshold", 0.75))


def gate_threshold_source() -> str:
    if os.getenv("NCKH_GATE_THRESH"):
        return "env NCKH_GATE_THRESH"
    if GATE_THRESHOLD_PATH.exists():
        return "frame-level val calibration"
    return "tau_val (track-level fallback)"


def should_block_output_direct(
    risk: float, rules: dict[str, Any], tau: float | None = None
) -> bool:
    """Chặn frame nếu risk vượt ngưỡng gate (xem resolve_gate_threshold).

    Không dùng `output_gate_block_threshold` = 0.526 làm mặc định vì giá trị
    này nằm ngoài dải xác suất calibrated, khiến gate không bao giờ chặn.
    """
    return risk >= resolve_gate_threshold(rules, tau)


def should_block_containment(
    gt_boxes: list[list[float]],
    protect_boxes: list[list[float]],
    threshold: float,
) -> bool:
    """Chặn frame nếu có bất kỳ GT box nào không được protect box chứa đủ mức threshold.

    Dùng làm cổng bổ sung để đảm bảo không phát hành frame có người chưa được che.
    Tính bằng ma trận containment (vectorized) thay cho 2 vòng lặp lồng nhau.
    """
    if not gt_boxes:
        return False
    if not protect_boxes:
        return True
    mat = containment_matrix(gt_boxes, protect_boxes)
    return bool((mat.max(axis=1) < float(threshold)).any())


VIDEO_CANVAS = (1280, 720)


def fit_to_canvas(frame: np.ndarray, canvas_w: int | None = None, canvas_h: int | None = None) -> np.ndarray:
    """Đưa frame về canvas cố định (giữ tỉ lệ, pad đen).

    VisDrone2019-VID-test-dev có 7 độ phân giải khác nhau (680x382 ... 2720x1530),
    nên một file video demo duy nhất bắt buộc phải chuẩn hoá kích thước.
    """
    canvas_w = VIDEO_CANVAS[0] if canvas_w is None else canvas_w
    canvas_h = VIDEO_CANVAS[1] if canvas_h is None else canvas_h
    h, w = frame.shape[:2]
    scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1))
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    new_w -= new_w % 2
    new_h -= new_h % 2
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    top = (canvas_h - new_h) // 2
    left = (canvas_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def even_frame(frame: np.ndarray) -> np.ndarray:
    """Cắt 1 px nếu rộng/cao lẻ.

    mp4v/FFmpeg yêu cầu kích thước chẵn; VisDrone có frame 1360x765 nên nếu
    không cắt sẽ sinh hàng loạt cảnh báo "Failed to write frame".
    """
    h, w = frame.shape[:2]
    out = frame[: h - (h % 2), : w - (w % 2)]
    return np.ascontiguousarray(out)


def draw_demo_overlay(frame: np.ndarray, block: bool, frame_risk: float, n_protect: int) -> np.ndarray:
    """Vẽ nhãn trạng thái lên frame demo (không ảnh hưởng metric)."""
    label = f"BLOCKED  risk={frame_risk:.2f}" if block else f"RELEASED  risk={frame_risk:.2f}"
    color = (0, 0, 255) if block else (0, 200, 0)
    cv2.rectangle(frame, (0, 0), (470, 46), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame, f"protect_boxes={n_protect}", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def tracker_predict_only(tracker: Any) -> np.ndarray:
    """Lan truyền track qua frame không gọi detect.

    Không được gọi `tracker.update(empty_boxes, frame)`: ByteTracker sẽ đánh
    dấu toàn bộ `tracked_stracks` thành lost và `_format_output()` trả mảng
    rỗng, làm mất ID ở mọi frame skip. Ở đây chỉ chạy Kalman predict cho các
    track đang active, giữ nguyên ID và vị trí dự đoán.
    """
    tracker.frame_id += 1
    stracks = [t for t in tracker.tracked_stracks if t.is_activated]
    if stracks:
        tracker.multi_predict(stracks)
    return tracker._format_output()


def run_sequence(
    seq_name: str,
    model: Any,
    risk_model: Any,
    tau: float,
    coordinator_rules: dict[str, Any],
    writer: cv2.VideoWriter | None,
    max_frames: int,
    device: str = "cpu",
    half: bool = False,
    risk_labels_lookup: dict | None = None,
    backend: str = "pytorch",
    ort_session: Any = None,
    ort_input_name: str | None = None,
) -> dict[str, Any]:
    seq_dir = SEQUENCES_DIR / seq_name
    images = limit_frames(sorted(seq_dir.glob("*.jpg")), max_frames)
    gt, gt_ids = load_annotations_with_ids(seq_name)
    seq_wall_start = time.perf_counter()

    pred_boxes_by_frame: dict[int, list[list[float]]] = {}
    pred_ids_by_frame: dict[int, list[int]] = {}
    pred_scores_by_frame: dict[int, list[float]] = {}
    gt_ids_by_frame: dict[int, list[int]] = {}
    detector_boxes_by_frame: dict[int, list[list[float]]] = {}
    gt_boxes_by_frame: dict[int, list[list[float]]] = {}
    risks: list[float] = []
    risk_y_true: list[int] = []
    risk_y_prob: list[float] = []
    latency_samples: list[float] = []
    detection_calls = 0
    detection_intervals: list[float] = []
    blocked = 0
    released = 0
    unsafe = 0
    protected_gt_boxes = 0
    total_protect_gt_boxes = 0
    contained_gt_boxes = 0

    psnr_sum = 0.0
    psnr_count = 0
    ssim_sum = 0.0
    ssim_count = 0
    # Bản ghi mức frame dùng cho metric privacy/utility + gate quality.
    per_frame_records: list[dict[str, Any]] = []

    frame_idx = 0
    last_detect_frame = -10000
    prev_det_count: int | None = None
    prev_n_tracks: int | None = None
    ewma_det: float | None = None
    prev_frame_risk_level: str | None = None

    track_histories: dict[int, TrackHistory] = {}
    # Danh sách đặc trưng lấy theo payload model: model 25-feature mới dùng FEATURES,
    # model 21-feature cũ vẫn tự fallback về LEGACY_FEATURES.
    feature_names = resolve_feature_names(risk_model)
    risk_standardizer = SequenceStandardizer(len(feature_names))

    tracker_args = SimpleNamespace(
        track_high_thresh=TRACK_HIGH_THRESH,
        track_low_thresh=TRACK_LOW_THRESH,
        new_track_thresh=TRACK_NEW_THRESH,
        track_buffer=TRACK_BUFFER_FRAMES,
        match_thresh=TRACK_MATCH_THRESH,
        fuse_score=True,
    )
    tracker = BYTETracker(tracker_args)

    all_gt_boxes = [box for boxes in gt.values() for box in boxes]
    if all_gt_boxes:
        r_ref = float(np.median([max(b[2] - b[0], 1e-6) / max(b[3] - b[1], 1e-6) for b in all_gt_boxes]))
    else:
        r_ref = 0.4

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        frame_number = int(img_path.stem)
        wall_start = time.perf_counter()

        frame_risk = 0.0
        if prev_frame_risk_level is not None:
            det_interval = {
                "critical": coordinator_rules.get("detect_interval_critical", 1),
                "high": coordinator_rules.get("detect_interval_high", 1),
                "medium": coordinator_rules.get("detect_interval_medium", 3),
                "low": coordinator_rules.get("detect_interval_low", 5),
            }.get(prev_frame_risk_level, coordinator_rules.get("detect_interval_low", 5))
        else:
            det_interval = coordinator_rules.get("detect_interval_low", 5)
        should_detect = (
            TRACK_EVAL
            or frame_idx < WARMUP_FRAMES
            or (frame_idx - last_detect_frame) >= det_interval
        )
        detection_intervals.append(1.0 if TRACK_EVAL else float(det_interval))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if should_detect:
            detection_calls += 1
            last_detect_frame = frame_idx
            if backend == "onnx" and ort_session is not None:
                detections_np = run_onnx_detect(
                    ort_session, frame, ort_input_name,
                    CONF_THRES, IOU_THRES, IMG_SIZE,
                )
                # Tạo đối tượng results giả tương thích với ultralytics Boxes cho tracker.update
                class _OnnxBoxes:
                    def __init__(self, det: np.ndarray):
                        if len(det) > 0:
                            self.conf = det[:, 4].astype(np.float32)
                            self.cls = det[:, 5].astype(np.float32)
                            self.xyxy = det[:, :4].astype(np.float32)
                            # Tính xywh từ xyxy (center_x, center_y, w, h)
                            x1, y1, x2, y2 = self.xyxy[:, 0], self.xyxy[:, 1], self.xyxy[:, 2], self.xyxy[:, 3]
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            w = x2 - x1
                            h = y2 - y1
                            self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
                            self.xywhr = np.stack([cx, cy, w, h, np.zeros_like(cx)], axis=1).astype(np.float32)
                        else:
                            self.conf = np.empty((0,), dtype=np.float32)
                            self.cls = np.empty((0,), dtype=np.float32)
                            self.xyxy = np.empty((0, 4), dtype=np.float32)
                            self.xywh = np.empty((0, 4), dtype=np.float32)
                            self.xywhr = np.empty((0, 5), dtype=np.float32)

                    def cpu(self):
                        return self

                    @property
                    def boxes(self):
                        return self

                    def __getitem__(self, inds):
                        if isinstance(inds, np.ndarray):
                            if inds.dtype == bool:
                                inds = np.flatnonzero(inds)
                            else:
                                inds = inds.tolist()
                        if len(inds) > 0:
                            det = np.column_stack(
                                [
                                    self.xyxy[inds],
                                    self.conf[inds],
                                    self.cls[inds],
                                ]
                            )
                            return _OnnxBoxes(det)
                        return _OnnxBoxes(np.empty((0, 6), dtype=np.float32))

                    def __len__(self):
                        return len(self.conf)

                results_boxes = _OnnxBoxes(detections_np)
                results = type(
                    "Results",
                    (),
                    {
                        "__getitem__": lambda self, idx: results_boxes,
                    },
                )()
            else:
                results = model(
                    frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES,
                    device=device, max_det=MAX_DET, verbose=False,
                    **({"half": True} if half else {}),
                )
                cpu_boxes = results[0].cpu().boxes
                xyxy = cpu_boxes.xyxy.numpy()
                confs = cpu_boxes.conf.numpy()
                detections_np = (
                    np.column_stack((xyxy, confs, np.zeros_like(confs))).astype(np.float32)
                    if len(xyxy) else np.empty((0, 6), dtype=np.float32)
                )
        else:
            detections_np = np.empty((0, 6), dtype=np.float32)

        if should_detect:
            tracker_out = tracker.update(results[0].cpu().boxes, frame)
        else:
            # Không gọi update(empty) vì sẽ đánh lost toàn bộ track (xem
            # tracker_predict_only). Chỉ predict vị trí cho frame bị skip.
            tracker_out = tracker_predict_only(tracker)

        track_boxes: list[list[float]] = []
        track_confs: list[float] = []
        track_ids: list[int] = []

        if len(tracker_out) > 0:
            for row in tracker_out:
                x1, y1, x2, y2, tid, score, cls = row[:7]
                det_idx = int(row[7]) if len(row) > 7 else -1
                track_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                track_confs.append(float(score))
                track_ids.append(int(tid))

                if int(tid) not in track_histories:
                    track_histories[int(tid)] = TrackHistory(int(tid), frame_idx)
                th = track_histories[int(tid)]

                # Thống kê crop tính trên box của TRACKER ở frame hiện tại,
                # đúng bằng dữ liệu lúc train (features_frame.csv dùng box track),
                # và tính mỗi frame để brightness/contrast/lap khớp notebook.
                lap_var, bright_cur, contrast_cur = crop_quality_stats(
                    gray, [float(x1), float(y1), float(x2), float(y2)]
                )
                th.update(
                    frame_idx, [float(x1), float(y1), float(x2), float(y2)],
                    float(score), lap_var, bright_cur, contrast_cur,
                )

        current_ids = set(track_ids)
        for tid, th in track_histories.items():
            if tid not in current_ids:
                th.mark_missed(frame_idx)

        # Đếm số track mỗi frame (khớp notebook: N = số dòng tracker trong frame),
        # dùng EWMA của quá khứ nên phải lấy giá trị TRƯỚC khi cập nhật.
        N_t = len(track_boxes)
        prev_det_count = prev_n_tracks
        ewma_prev = ewma_det
        ewma_det = N_t if ewma_det is None else EWMA_ALPHA * N_t + (1 - EWMA_ALPHA) * ewma_det
        prev_n_tracks = N_t

        all_boxes_frame = track_boxes
        frame_risk = 0.0

        # Gom đặc trưng của toàn bộ track đang hoạt động trong frame rồi dự đoán
        # rủi ro 1 lần (batch) thay vì gọi predict_proba cho từng track.
        # Lưu ý: SequenceStandardizer vẫn cập nhật online tuần tự từng vector
        # (đúng như lúc train), chỉ phần suy luận model được batch.
        frame_rows: list[np.ndarray] = []
        for tid, th in track_histories.items():
            if tid not in current_ids:
                continue

            feat = th.compute_features(
                frame=frame_idx,
                img_w=w,
                img_h=h,
                all_boxes_frame=all_boxes_frame,
                prev_det_count=prev_det_count,
                ewma_det=ewma_prev,
                r_ref=r_ref,
            )

            X = np.array([feat[f] for f in feature_names], dtype=np.float32)
            frame_rows.append(risk_standardizer.transform(X).reshape(-1))

        if frame_rows:
            probs = predict_risk_batch(risk_model, np.stack(frame_rows))
            # Chỉ ghi nhận cặp (y_true, y_prob) khi có ground truth label khớp
            # theo (sequence, frame) từ risk_dataset.csv.
            frame_key = (seq_name, frame_idx)
            matched_label: int | None = None
            if risk_labels_lookup and frame_key in risk_labels_lookup:
                # Nếu có bất kỳ artifact track nào trong frame này có label=1,
                # coi frame đó là rủi ro (conservative).
                matched_label = int(max(risk_labels_lookup[frame_key]))
            for risk_prob in probs:
                risk_prob = float(risk_prob)
                if matched_label is not None:
                    risk_y_true.append(matched_label)
                    risk_y_prob.append(risk_prob)
                if risk_prob > frame_risk:
                    frame_risk = risk_prob

        risks.append(frame_risk)
        prev_frame_risk_level = classify_risk_level(frame_risk, coordinator_rules)

        candidate_boxes = detections_np[:, :4].tolist() if should_detect else track_boxes
        roi_padding = decide_roi_padding(frame_risk, coordinator_rules)
        protect_boxes_frame = [
            expand_box(box, roi_padding, w, h) for box in candidate_boxes
        ]

        block = should_block_output_direct(frame_risk, coordinator_rules, tau)

        # Cổng bổ sung dùng GT (oracle) CHỈ để tham chiếu upper-bound, không
        # dùng cho kết quả deployable. Bật bằng NCKH_ORACLE_GATE=1.
        if (
            ORACLE_GATE
            and not block
            and coordinator_rules.get("enable_containment_gate", False)
        ):
            gts_for_gate = gt.get(frame_number, [])
            if gts_for_gate:
                block = should_block_containment(
                    gts_for_gate,
                    protect_boxes_frame,
                    coordinator_rules.get("containment_block_threshold", 0.9),
                )

        # ===== Đo coverage cho MỌI frame =====
        # `frame_unsafe` = nếu phát hành frame này thì còn GT chưa được che đủ.
        # Nhãn này chỉ dùng để ĐO (gate quality), không tham gia quyết định gate.
        gts = gt.get(frame_number, [])
        gt_ids_frame = gt_ids.get(frame_number, [])
        iou_cov = (
            iou_matrix(gts, candidate_boxes)
            if (gts and candidate_boxes)
            else np.zeros((len(gts), 0), dtype=np.float64)
        )
        cont_cov = (
            containment_matrix(gts, protect_boxes_frame)
            if (gts and protect_boxes_frame)
            else np.zeros((len(gts), 0), dtype=np.float64)
        )
        protected_any = (
            (iou_cov >= 0.5).any(axis=1)
            if iou_cov.size
            else np.zeros(len(gts), dtype=bool)
        )
        contained_any = (
            (cont_cov >= 0.9).any(axis=1)
            if cont_cov.size
            else np.zeros(len(gts), dtype=bool)
        )
        frame_unsafe = bool(len(gts) > 0 and not bool(contained_any.all()))

        # Mask union của vùng che, dùng lại cho blur + đo diện tích.
        protect_mask = boxes_to_mask(protect_boxes_frame, h, w)
        blur_area_px = int(cv2.countNonZero(protect_mask))
        gt_covered_px = 0
        if gts and blur_area_px > 0:
            gt_mask = boxes_to_mask(gts, h, w)
            gt_covered_px = int(cv2.countNonZero(cv2.bitwise_and(gt_mask, protect_mask)))

        psnr_bg = float("nan")
        psnr_prot = float("nan")

        if block:
            blocked += 1
            out_frame = np.zeros_like(frame)
        else:
            released += 1
            out_frame = blur_boxes(frame, protect_boxes_frame, mask=protect_mask)
            # GT chỉ dùng để ĐO coverage/unsafe sau khi đã quyết định phát hành.
            if gts:
                total_protect_gt_boxes += len(gts)
                protected_gt_boxes += int(protected_any.sum())
                contained_gt_boxes += int(contained_any.sum())
            if frame_unsafe:
                unsafe += 1
            # PSNR: một lần diff cho cả toàn frame / nền / vùng che.
            try:
                psnr_full_val, psnr_bg, psnr_prot = psnr_components(
                    frame, out_frame, protect_mask
                )
                if math.isfinite(psnr_full_val):
                    psnr_sum += psnr_full_val
                    psnr_count += 1
            except Exception:
                pass

        per_frame_records.append(
            {
                "sequence": seq_name,
                "frame_number": int(frame_number),
                "block": bool(block),
                "risk": float(frame_risk),
                "risk_level": str(prev_frame_risk_level),
                "n_gt": len(gts),
                "gt_ids": [int(i) for i in gt_ids_frame],
                "gt_contained": [bool(v) for v in contained_any],
                "unsafe": bool(frame_unsafe),
                "blur_area_px": blur_area_px,
                "frame_area_px": int(h * w),
                "gt_covered_area_px": gt_covered_px,
                "psnr_bg": float(psnr_bg),
                "psnr_prot": float(psnr_prot),
            }
        )

        if writer is not None:
            if DEMO_OVERLAY:
                # Video demo: luôn ghi khung ĐÃ CHE (blur) kể cả khi gate chắn,
                # kèm nhãn trạng thái để minh hoạ quyết định của hệ thống.
                # Metric vẫn tính trên `out_frame` (frame bị chắn = đen) nên
                # video demo không làm sai lệch số liệu.
                demo = out_frame if not block else blur_boxes(
                    frame, protect_boxes_frame, mask=protect_mask
                )
                demo = fit_to_canvas(demo)
                demo = draw_demo_overlay(demo, block, frame_risk, len(protect_boxes_frame))
                writer.write(demo)
            else:
                writer.write(fit_to_canvas(out_frame))

        if should_detect:
            detector_boxes_by_frame[frame_number] = [
                ([float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(b[4]))
                for b in detections_np[:, :5].tolist()
            ]
        pred_boxes_by_frame[frame_number] = track_boxes
        pred_ids_by_frame[frame_number] = track_ids
        pred_scores_by_frame[frame_number] = track_confs
        gt_boxes_by_frame[frame_number] = gt.get(frame_number, [])
        gt_ids_by_frame[frame_number] = gt_ids.get(frame_number, [])

        # Chỉ đo chất lượng vùng che trên frame ĐÃ PHÁT HÀNH (không tính frame bị chắn đen).
        # PSNR đã tính ở trên (1 lần diff dùng chung), ở đây chỉ còn SSIM.
        if not block:
            if frame_idx % SSIM_INTERVAL == 0:
                try:
                    ssim_val = compute_ssim(frame, out_frame)
                    ssim_sum += ssim_val
                    ssim_count += 1
                except Exception:
                    pass

        latency_ms = (time.perf_counter() - wall_start) * 1000.0
        latency_samples.append(latency_ms)
        frame_idx += 1

        # Log tiến độ để không còn cảm giác "treo" (không cần thêm thư viện).
        if PROGRESS_EVERY > 0 and (frame_idx % PROGRESS_EVERY == 0 or frame_idx == len(images)):
            _elapsed = time.perf_counter() - seq_wall_start
            _fps = frame_idx / max(_elapsed, 1e-9)
            _eta = (len(images) - frame_idx) / max(_fps, 1e-9)
            print(
                f"[Bước 5]   {seq_name}: {frame_idx}/{len(images)} frame | "
                f"{_fps:.2f} fps | {latency_ms:.0f} ms/frame | "
                f"ETA sequence {_eta / 60.0:.1f} phút",
                flush=True,
            )

    return {
        "pred_boxes_by_frame": pred_boxes_by_frame,
        "pred_ids_by_frame": pred_ids_by_frame,
        "pred_scores_by_frame": pred_scores_by_frame,
        "detector_boxes_by_frame": detector_boxes_by_frame,
        "gt_boxes_by_frame": gt_boxes_by_frame,
        "gt_ids_by_frame": gt_ids_by_frame,
        "risks": risks,
        "risk_y_true": risk_y_true,
        "risk_y_prob": risk_y_prob,
        "latency_samples": latency_samples,
        "detection_calls": detection_calls,
        "detection_intervals": detection_intervals,
        "blocked": blocked,
        "released": released,
        "unsafe": unsafe,
        "protected_gt_boxes": protected_gt_boxes,
        "contained_gt_boxes": contained_gt_boxes,
        "total_protect_gt_boxes": total_protect_gt_boxes,
        "psnr_sum": psnr_sum,
        "psnr_count": psnr_count,
        "ssim_sum": ssim_sum,
        "ssim_count": ssim_count,
        "per_frame_records": per_frame_records,
    }


def _format_metric(value: Any, digits: int = 4) -> str:
    """Format số cho log, hiển thị N/A khi metric không khả dụng."""
    try:
        if value is None:
            return "N/A"
        val = float(value)
        if math.isnan(val):
            return "N/A"
        return f"{val:.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def main() -> None:
    global IMG_SIZE, CONF_THRES, IOU_THRES, MAX_FRAMES_PER_SEQ, WARMUP_FRAMES
    global TRACK_HIGH_THRESH, TRACK_LOW_THRESH, TRACK_NEW_THRESH, TRACK_MATCH_THRESH
    global TRACK_BUFFER_FRAMES, MAX_DET, WRITE_VIDEO, PROGRESS_EVERY, VIDEO_CANVAS, DEMO_OVERLAY

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    OUT_GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    # Đồng nhất các tham số inference từ config dùng chung (cùng Kaggle)
    eval_cfg = load_eval_config()
    IMG_SIZE = int(eval_cfg.get("img_size", IMG_SIZE))
    CONF_THRES = float(eval_cfg.get("conf_thres", CONF_THRES))
    IOU_THRES = float(eval_cfg.get("iou_thres", IOU_THRES))
    MAX_FRAMES_PER_SEQ = int(os.getenv("NCKH_MAX_FRAMES", eval_cfg.get("max_frames_per_seq", MAX_FRAMES_PER_SEQ)))
    WARMUP_FRAMES = int(eval_cfg.get("warmup_frames", WARMUP_FRAMES))
    MAX_DET = int(eval_cfg.get("max_det", MAX_DET))
    # ByteTrack: ngưỡng tune trên val sequence-disjoint, đồng bộ conf detector 0.15
    TRACK_HIGH_THRESH = float(os.getenv("NCKH_TRACK_HIGH", eval_cfg.get("track_high_thresh", TRACK_HIGH_THRESH)))
    TRACK_LOW_THRESH = float(os.getenv("NCKH_TRACK_LOW", eval_cfg.get("track_low_thresh", TRACK_LOW_THRESH)))
    TRACK_NEW_THRESH = float(os.getenv("NCKH_TRACK_NEW", eval_cfg.get("new_track_thresh", TRACK_NEW_THRESH)))
    TRACK_MATCH_THRESH = float(os.getenv("NCKH_TRACK_MATCH", eval_cfg.get("track_match_thresh", TRACK_MATCH_THRESH)))
    TRACK_BUFFER_FRAMES = int(os.getenv("NCKH_TRACK_BUFFER", eval_cfg.get("track_buffer", TRACK_BUFFER_FRAMES)))
    WRITE_VIDEO = bool(eval_cfg.get("write_video", WRITE_VIDEO)) and os.getenv("NCKH_WRITE_VIDEO", "1") == "1"
    PROGRESS_EVERY = int(os.getenv("NCKH_PROGRESS_EVERY", str(PROGRESS_EVERY)))
    print(
        f"[Bước 5] Eval config: img_size={IMG_SIZE}, conf_thres={CONF_THRES}, "
        f"iou_thres={IOU_THRES}, max_det={MAX_DET}, "
        f"max_frames_per_seq={MAX_FRAMES_PER_SEQ or 'all'}, "
        f"warmup_frames={WARMUP_FRAMES}"
    )
    print(
        f"[Bước 5] ByteTrack: high={TRACK_HIGH_THRESH}, low={TRACK_LOW_THRESH}, "
        f"new={TRACK_NEW_THRESH}, match={TRACK_MATCH_THRESH}, buffer={TRACK_BUFFER_FRAMES} "
        f"(đồng bộ conf detector + tune trên val)"
    )
    print(
        f"[Bước 5] Chế độ: track_eval={TRACK_EVAL} "
        f"({'detect mọi frame' if TRACK_EVAL else 'skip-frame + lan truyền track'}), "
        f"oracle_gate={ORACLE_GATE}, data_root={DATA_ROOT}"
    )
    print(f"[Bước 5] Ghi video demo: {WRITE_VIDEO} -> {OUT_VIDEO}")

    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình XGBoost: {RISK_MODEL_PATH}")

    backend = os.getenv("NCKH_BACKEND", str(eval_cfg.get("backend", DETECTOR_BACKEND))).lower()
    det_device = DEVICE
    detector_half = resolve_half(backend, det_device)
    ort_session = None
    ort_input_name = None

    # Chỉ kiểm tra file .pt khi backend là pytorch
    if backend == "pytorch":
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy mô hình detect: {WEIGHTS_PATH}")

    if backend == "onnx" and _HAS_ORT and ONNX_MODEL_PATH.exists():
        available_providers = list(ort.get_available_providers())
        if torch.cuda.is_available() and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            # Máy có CUDA mà onnxruntime vẫn chỉ có CPU provider -> đang cài bản
            # onnxruntime CPU, dễ tưởng nhầm là GPU chậm. Chỉ báo khi thật sự
            # PyTorch thấy CUDA (nếu PyTorch cũng không thấy thì đã cảnh báo ở trên).
            if torch.cuda.is_available():
                print(
                    "[Bước 5] Cảnh báo: onnxruntime không có CUDAExecutionProvider "
                    f"(đang có: {available_providers}). Cài onnxruntime-gpu và gỡ "
                    "onnxruntime CPU để dùng GPU cho backend ONNX."
                )
        ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
        ort_input_name = ort_session.get_inputs()[0].name
        # Lấy kích thước input mà model ONNX yêu cầu (H, W) để đặt IMG_SIZE khớp
        onnx_input_shape = ort_session.get_inputs()[0].shape
        if len(onnx_input_shape) >= 4:
            onnx_h = onnx_input_shape[2]
            onnx_w = onnx_input_shape[3]
            if isinstance(onnx_h, int) and isinstance(onnx_w, int) and onnx_h > 0 and onnx_w > 0:
                IMG_SIZE = max(onnx_h, onnx_w)
        print(
            f"[Bước 5] Detector backend: onnx | session: {ONNX_MODEL_PATH} | "
            f"providers yêu cầu: {providers} | providers thực tế: "
            f"{ort_session.get_providers()} | imgsz={IMG_SIZE}"
        )
    elif backend == "onnx":
        print(
            f"[Bước 5] Cảnh báo: onnxruntime không có hoặc không tìm thấy {ONNX_MODEL_PATH}, "
            "fallback PyTorch."
        )
        backend = "pytorch"
    elif backend == "openvino":
        print(
            "[Bước 5] Cảnh báo: backend openvino chưa triển khai, fallback PyTorch."
        )
        backend = "pytorch"

    model = None
    if backend == "pytorch":
        model = YOLO(str(WEIGHTS_PATH))
        model.to(det_device)
        print(
            f"[Bước 5] Detector backend: pytorch | device: {det_device} | "
            f"gpu: {runtime_info()['gpu_name']} | half: {detector_half} | "
            f"imgsz: {IMG_SIZE}"
        )

    risk_model = load_risk_model()
    tau = load_tau()
    risk_labels_lookup = load_risk_labels_lookup()

    coordinator_rules = {}
    if COORDINATOR_RULES.exists():
        with COORDINATOR_RULES.open("r", encoding="utf-8") as f:
            coordinator_rules = json.load(f)
    else:
        coordinator_rules = {
            "risk_low": 0.25,
            "risk_high": 0.50,
            "risk_critical": 0.75,
            "detect_interval_low": 2,
            "detect_interval_medium": 2,
            "detect_interval_high": 1,
            "detect_interval_critical": 1,
            "roi_expand_low": 0.20,
            "roi_expand_medium": 0.30,
            "roi_expand_high": 0.50,
            "roi_expand_critical": 0.70,
            "output_gate_block_threshold": 0.526,
            "containment_block_threshold": 0.9,
            "enable_containment_gate": True,
        }

    gate_threshold = resolve_gate_threshold(coordinator_rules, tau)
    print(
        f"[Bước 5] Output gate threshold: {gate_threshold:.4f} "
        f"({gate_threshold_source()})"
    )

    sequences = sorted(
        [d.name for d in SEQUENCES_DIR.iterdir() if d.is_dir()]
    )
    seq_filter = [s.strip() for s in os.getenv("NCKH_SEQ", "").split(",") if s.strip()]
    if seq_filter:
        sequences = [s for s in sequences if s in seq_filter]
    if MAX_SEQS > 0:
        sequences = sequences[:MAX_SEQS]

    all_gt: dict[str, dict[int, list[list[float]]]] = {}
    all_gt_ids: dict[str, dict[int, list[int]]] = {}
    all_pred: dict[str, dict[int, list[list[float]]]] = {}
    all_pred_ids: dict[str, dict[int, list[int]]] = {}
    all_pred_scores: dict[str, dict[int, list[float]]] = {}
    all_detector: dict[str, dict[int, list[list[float]]]] = {}
    risk_dump_rows: list[tuple[str, int, float]] = []
    all_frame_records: list[dict[str, Any]] = []
    all_frames: dict[str, int] = {}
    all_risks: dict[str, list[float]] = {}
    all_latency: list[float] = []
    all_intervals: list[float] = []
    total_detection_calls = 0
    total_blocked = 0
    total_released = 0
    total_unsafe = 0
    total_protected_gt_boxes = 0
    total_contained_gt_boxes = 0
    total_protect_gt_boxes = 0
    total_psnr_sum = 0.0
    total_psnr_count = 0
    total_ssim_sum = 0.0
    total_ssim_count = 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    video_fps = float(eval_cfg.get("video_fps", 30.0))
    VIDEO_CANVAS = (
        int(eval_cfg.get("video_width", 1280)),
        int(eval_cfg.get("video_height", 720)),
    )
    if WRITE_VIDEO:
        writer = cv2.VideoWriter(
            str(OUT_VIDEO), fourcc, video_fps, VIDEO_CANVAS
        )
        if not writer.isOpened():
            print("[Bước 5] Cảnh báo: không mở được VideoWriter, tắt ghi video.")
            writer.release()
            writer = None
        else:
            print(
                f"[Bước 5] Video demo: {OUT_VIDEO} | canvas {VIDEO_CANVAS[0]}x{VIDEO_CANVAS[1]} "
                f"| {video_fps:.0f} fps"
            )
    run_wall_start = time.perf_counter()
    total_frames_planned = 0
    for _seq in sequences:
        total_frames_planned += len(limit_frames(sorted((SEQUENCES_DIR / _seq).glob("*.jpg")), MAX_FRAMES_PER_SEQ))
    frames_done = 0
    print(
        f"[Bước 5] Bắt đầu: {len(sequences)} sequence | {total_frames_planned} frame | "
        f"backend={backend} | device={det_device}",
        flush=True,
    )

    for seq_idx, seq in enumerate(sequences, start=1):
        seq_dir = SEQUENCES_DIR / seq
        images = limit_frames(sorted(seq_dir.glob("*.jpg")), MAX_FRAMES_PER_SEQ)
        if not images:
            continue

        seq_start = time.perf_counter()
        print(
            f"[Bước 5] [{seq_idx}/{len(sequences)}] {seq}: {len(images)} frame "
            f"(đã xong {frames_done}/{total_frames_planned} frame)",
            flush=True,
        )

        result = run_sequence(
            seq, model, risk_model, tau, coordinator_rules, writer,
            MAX_FRAMES_PER_SEQ, det_device, detector_half,
            risk_labels_lookup,
            backend=backend,
            ort_session=ort_session,
            ort_input_name=ort_input_name,
        )

        frames_done += len(result["pred_boxes_by_frame"])
        if os.getenv("NCKH_DUMP_RISK"):
            for _i, _r in enumerate(result["risks"]):
                risk_dump_rows.append((seq, _i, float(_r)))
        _seq_elapsed = time.perf_counter() - seq_start
        _overall_fps = frames_done / max(time.perf_counter() - run_wall_start, 1e-9)
        _eta_all = (total_frames_planned - frames_done) / max(_overall_fps, 1e-9)
        print(
            f"[Bước 5] [{seq_idx}/{len(sequences)}] {seq} xong trong {_seq_elapsed:.1f}s | "
            f"detection calls={result['detection_calls']} | blocked={result['blocked']} | "
            f"released={result['released']} | tốc độ chung {_overall_fps:.2f} fps | "
            f"ETA toàn bộ {_eta_all / 60.0:.1f} phút",
            flush=True,
        )

        all_gt[seq] = result["gt_boxes_by_frame"]
        all_gt_ids[seq] = result["gt_ids_by_frame"]
        all_pred[seq] = result["pred_boxes_by_frame"]
        all_pred_ids[seq] = result["pred_ids_by_frame"]
        all_pred_scores[seq] = result["pred_scores_by_frame"]
        all_detector[seq] = result["detector_boxes_by_frame"]
        all_frames[seq] = len(result["pred_boxes_by_frame"])
        all_risks[seq] = result["risks"]
        all_latency.extend(result["latency_samples"])
        all_intervals.extend(result["detection_intervals"])
        total_detection_calls += result["detection_calls"]
        total_blocked += result["blocked"]
        total_released += result["released"]
        total_unsafe += result["unsafe"]
        total_protected_gt_boxes += result["protected_gt_boxes"]
        total_contained_gt_boxes += result["contained_gt_boxes"]
        total_protect_gt_boxes += result["total_protect_gt_boxes"]
        total_psnr_sum += result["psnr_sum"]
        total_psnr_count += result["psnr_count"]
        total_ssim_sum += result["ssim_sum"]
        total_ssim_count += result["ssim_count"]
        all_frame_records.extend(result["per_frame_records"])

    if writer is not None:
        writer.release()

    if risk_dump_rows:
        _dump_path = Path(os.environ["NCKH_DUMP_RISK"])
        _dump_path.parent.mkdir(parents=True, exist_ok=True)
        with _dump_path.open("w", newline="", encoding="utf-8") as _f:
            _w = csv.writer(_f)
            _w.writerow(["sequence", "frame_index", "frame_risk"])
            _w.writerows(risk_dump_rows)
        print(f"[Bước 5] Đã lưu frame risk: {_dump_path} ({len(risk_dump_rows)} frame)")

    metrics = compute_metrics(
        all_gt, all_pred, all_pred_ids, all_pred_scores, all_gt_ids, all_detector, all_frames,
        all_latency, total_detection_calls,
        all_intervals, total_blocked, total_released,
        total_protected_gt_boxes, total_contained_gt_boxes, total_protect_gt_boxes,
        total_psnr_sum, total_psnr_count,
        total_ssim_sum, total_ssim_count,
        all_frame_records,
    )

    policy_name = "integrated_oracle" if ORACLE_GATE else "integrated_deployable"
    rows = []
    for k, v in metrics.items():
        rows.append({"metric_group": "system", "policy": policy_name, "sequence": "all",
                     "repetition": 0, "metric_name": k, "metric_value": v})
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    gate_log = {
        "total_blocked": total_blocked,
        "total_released": total_released,
        "total_unsafe": total_unsafe,
        "gate_mode": "oracle" if ORACLE_GATE else "deployable",
        "gate_threshold": gate_threshold,
        "track_eval": TRACK_EVAL,
        # Thông tin runtime để báo cáo không trộn số CPU với số GPU.
        "backend": backend,
        "device": det_device,
        "half": detector_half,
        "blur_mode": BLUR_MODE,
        "blur_kernel": BLUR_KERNEL,
        **runtime_info(),
    }
    with OUT_GATE_LOG.open("w", encoding="utf-8") as f:
        json.dump(gate_log, f, indent=2, ensure_ascii=False)

    print(f"[Bước 5] Đã lưu gate log: {OUT_GATE_LOG}")
    if WRITE_VIDEO and writer is not None and OUT_VIDEO.exists():
        print(f"[Bước 5] Đã lưu video demo đã che: {OUT_VIDEO} ({OUT_VIDEO.stat().st_size / 2**20:.1f} MB)")
    else:
        print("[Bước 5] Bỏ qua ghi video (WRITE_VIDEO=False), vẫn tính metric trên frame đã che.")
    print(f"[Bước 5] Đã lưu chỉ số đánh giá: {OUT_METRICS}")
    print(f"[Bước 5] YOLO precision: {_format_metric(metrics['detection_precision'])}")
    print(f"[Bước 5] YOLO recall: {_format_metric(metrics['detection_recall'])}")
    print(f"[Bước 5] YOLO F1: {_format_metric(metrics['detection_f1'])}")
    print(f"[Bước 5] YOLO recall (toàn cục): {_format_metric(metrics['detection_recall_global'])}")
    print(f"[Bước 5] Tracking precision: {_format_metric(metrics['tracking_precision'])}")
    print(f"[Bước 5] Tracking recall: {_format_metric(metrics['tracking_recall'])}")
    print(f"[Bước 5] Tracking F1: {_format_metric(metrics['tracking_f1'])}")
    print(f"[Bước 5] Tracking MOTA: {_format_metric(metrics['tracking_mota'])}")
    print(f"[Bước 5] Tracking MOTP: {_format_metric(metrics['tracking_motp'])}")
    print(f"[Bước 5] Tracking IDF1: {_format_metric(metrics['tracking_idf1'])}")
    print(f"[Bước 5] Tracking HOTA: {_format_metric(metrics['tracking_hota'])}")
    print(f"[Bước 5] Tracking IDSW: {_format_metric(metrics['tracking_idsw'], 0)}")
    print(f"[Bước 5] Protect-box coverage @ IoU 0.5: {metrics['protect_box_coverage_iou50']:.4f}")
    print(f"[Bước 5] Protect-box containment @ 90%: {metrics['protect_box_containment90']:.4f}")
    print(f"[Bước 5] Avg PSNR (dB): {metrics['avg_psnr_db']:.2f}")
    print(f"[Bước 5] Avg SSIM: {metrics['avg_ssim']:.4f}")
    print(f"[Bước 5] Output gate block rate: {metrics['output_gate_block_rate']:.4f}")
    print(f"[Bước 5] Output gate release rate: {metrics['output_gate_release_rate']:.4f}")
    print(f"[Bước 5] ULFR (unsafe leak frame rate): {_format_metric(metrics.get('ulfr'))}")
    print(f"[Bước 5] Unsafe release rate: {_format_metric(metrics.get('unsafe_release_rate'))}")
    print(f"[Bước 5] Identity leak rate: {_format_metric(metrics.get('identity_leak_rate'))}")
    print(f"[Bước 5] TTP p50/p95 (frames): {_format_metric(metrics.get('ttp_p50_frames'), 1)} / "
          f"{_format_metric(metrics.get('ttp_p95_frames'), 1)}")
    print(f"[Bước 5] Blur-area ratio: {_format_metric(metrics.get('blur_area_ratio'))}")
    print(f"[Bước 5] Over-protection ratio: {_format_metric(metrics.get('over_protection_ratio'))}")
    print(f"[Bước 5] PSNR background / protected (dB): "
          f"{_format_metric(metrics.get('psnr_bg_db'), 2)} / "
          f"{_format_metric(metrics.get('psnr_protected_db'), 2)}")
    print(f"[Bước 5] Gate precision/recall: {_format_metric(metrics.get('gate_precision'))} / "
          f"{_format_metric(metrics.get('gate_recall'))}")
    print(f"[Bước 5] Frame risk ROC-AUC / ECE: {_format_metric(metrics.get('risk_roc_auc'))} / "
          f"{_format_metric(metrics.get('risk_ece'))}")
    print(f"[Bước 5] Detection calls / 100 frames: {metrics['detection_calls_per_100_frames']:.2f}")
    print(f"[Bước 5] FPS wall: {metrics['fps_wall']:.2f}")


if __name__ == "__main__":
    main()