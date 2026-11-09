"""Bước 5: Tích hợp hệ thống hoàn chỉnh - phát hiện, theo dõi, ước lượng rủi ro,
điều phối thích ứng, che đối tượng và kiểm soát đầu ra.

Chỉ sử dụng các chỉ số được công nhận trong literature:
- Detection: Precision, Recall, F1 (ground-truth person box, IoU >= 0.5)
- Tracking: MOTA, MOTP, IDF1, IDSW, Frag
- Risk model: ROC AUC, PR AUC, Accuracy, Precision/Recall/F1, Brier Score, ECE
- Scheduler/Gate: Detection calls/100 frames, Average detection interval,
  Output gate block rate, Output gate release rate
- Privacy protection: PSNR (dB), SSIM
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
    average_precision_coco,
    precision_recall_f1_one_to_one,
    mota_motp,
    expected_calibration_error,
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
DATA_ROOT = Path("VisDrone2019-VID-test-dev")
SEQUENCES_DIR = DATA_ROOT / "sequences"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
OUT_VIDEO = Path("outputs/protected_video.mp4")
OUT_GATE_LOG = Path(os.getenv("NCKH_OUT_GATE_LOG", "results/output_gate_log.json"))
OUT_METRICS = Path(os.getenv("NCKH_OUT_METRICS", "results/system_metrics_unified.csv"))
COORDINATOR_RULES = Path("config/coordinator_rules.json")
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
IMG_SIZE = 1024
# Ghi video ra file là tùy chọn: tắt mặc định để tránh lỗi FFmpeg
# "Failed to write frame" trên môi trường không có encoder tương thích.
# Metrics (PSNR/SSIM) chỉ được tính trên frame ĐÃ PHÁT HÀNH (released), không tính
# trên frame bị gate chắn (mảng đen) để PSNR/SSIM phản ánh chất lượng vùng che thực tế.
WRITE_VIDEO = False
CONF_THRES = 0.25
IOU_THRES = 0.7
TRACK_HIGH_THRESH = 0.10
TRACK_LOW_THRESH = 0.01
TRACK_NEW_THRESH = 0.08
TRACK_MATCH_THRESH = 0.8
PROTECT_PADDING = 0.15
# Cho phép chạy nhanh để so sánh backend mà không sửa code:
#   NCKH_BACKEND=onnx NCKH_MAX_SEQS=3 NCKH_MAX_FRAMES=40 python step5_integrated_system.py
DETECTOR_BACKEND = os.getenv("NCKH_BACKEND", "onnx")
ONNX_MODEL_PATH = Path(os.getenv("NCKH_ONNX_MODEL", "final_extracted/artifacts/yolo26s.onnx"))
MAX_SEQS = int(os.getenv("NCKH_MAX_SEQS", "0"))  # 0 = tất cả sequence
MAX_FRAMES_PER_SEQ = int(os.getenv("NCKH_MAX_FRAMES", "200"))
WARMUP_FRAMES = 30

CONF_WINDOW = 5
PRED_UNCERT_WINDOW = 5
EWMA_ALPHA = 0.3
TRACK_BUFFER_FRAMES = 60
TRUNC_MARGIN_PX = 2
NEIGHBOR_RADIUS_FRAC = 0.05
NEIGHBOR_K = 5
CROP_RESIZE = 224
SSIM_INTERVAL = 5

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[Bước 5] Sử dụng thiết bị: {DEVICE} (CUDA: {torch.cuda.is_available()})")


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
        "conf_thres": 0.25,
        "iou_thres": 0.70,
        "img_size": 1024,
        "device": "cpu",
        "half": False,
        "iou_match_threshold": 0.5,
        "exclude_occlusion_2": True,
        "category_person": 1,
        "max_frames_per_seq": 200,
        "warmup_frames": 30,
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

    Mô hình ONNX đã xuất từ Ultralytics với output (1, 300, 6) =
    [cx, cy, w, h, conf, class] trong không gian ảnh gốc (đã un-scale trong graph).
    Chuyển về [x1, y1, x2, y2] trước khi trả về.
    """
    h, w = frame.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    top = (img_size - new_h) // 2
    left = (img_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)

    outputs = ort_session.run(None, {input_name: blob})
    # Output shape: (1, 300, 6) -> [cx, cy, w, h, conf, class] trong không gian ảnh gốc
    preds = outputs[0][0]  # (300, 6)

    # Lọc theo conf
    mask = preds[:, 4] >= conf_thres
    if not mask.any():
        return np.empty((0, 6), dtype=np.float32)

    dets = preds[mask]
    cx = dets[:, 0]
    cy = dets[:, 1]
    bw = dets[:, 2]
    bh = dets[:, 3]
    x1 = np.clip(cx - bw / 2.0, 0, w)
    y1 = np.clip(cy - bh / 2.0, 0, h)
    x2 = np.clip(cx + bw / 2.0, 0, w)
    y2 = np.clip(cy + bh / 2.0, 0, h)
    confs = dets[:, 4]
    classes = dets[:, 5]

    # NMS bổ sung (nếu muốn chặt hơn) - iou_thres dùng để lọc box trùng lặp
    order = np.argsort(-confs)
    keep = []
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ovr_list = [
            iou(
                [boxes_xyxy[i, 0], boxes_xyxy[i, 1], boxes_xyxy[i, 2], boxes_xyxy[i, 3]],
                [boxes_xyxy[j, 0], boxes_xyxy[j, 1], boxes_xyxy[j, 2], boxes_xyxy[j, 3]],
            )
            for j in rest
        ]
        ovr = np.array(ovr_list)
        order = rest[ovr < iou_thres]

    det_out = np.stack(
        [
            x1[keep],
            y1[keep],
            x2[keep],
            y2[keep],
            confs[keep],
            classes[keep],
        ],
        axis=1,
    )
    return det_out.astype(np.float32)


def predict_risk(risk_model: Any, X: np.ndarray) -> float:
    """Dự đoán xác suất rủi ro cho một vector đặc trưng.

    Hỗ trợ cả mô hình XGBoost thô và payload đã calibrate
    {'model': XGBClassifier, 'calibrator': _RiskCalibrator, ...}.
    """
    if isinstance(risk_model, dict) and "model" in risk_model:
        raw_prob = float(risk_model["model"].predict_proba(X)[0, 1])
        cal = risk_model.get("calibrator")
        if cal is not None and "X_thresholds_" in cal and "y_values_" in cal:
            return float(
                np.clip(
                    np.interp(raw_prob, np.asarray(cal["X_thresholds_"]), np.asarray(cal["y_values_"])),
                    0.0,
                    1.0,
                )
            )
        return raw_prob
    return float(risk_model.predict_proba(X)[0, 1])


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


def load_annotations(seq_name: str) -> dict[int, list[list[float]]]:
    ann_path = ANNOTATIONS_DIR / f"{seq_name}.txt"
    gt: dict[int, list[list[float]]] = {}
    if not ann_path.exists():
        return gt
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            frame = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            category = int(parts[7])
            occlusion = int(parts[9])
            if category != 1 or occlusion == 2:
                continue
            gt.setdefault(frame, []).append([x, y, x + w, y + h])
    return gt


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


def blur_boxes(frame: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        mask[y1:y2, x1:x2] = 1
    if not mask.any():
        return frame.copy()
    blurred = cv2.GaussianBlur(frame, (51, 51), 0)
    out = frame.copy()
    out[mask.astype(bool)] = blurred[mask.astype(bool)]
    return out


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 10.0 * math.log10((max_pixel ** 2) / mse)


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

    # MOTA, MOTP cho tracking (per-frame matching, IDSW = 0 do không có GT track ID)
    tracking_mot = mota_motp(gt_all, pred_all)

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
        "tracking_precision": tracking_precision,
        "tracking_recall": tracking_recall,
        "tracking_f1": tracking_f1,
        "avg_psnr_db": avg_psnr,
        "avg_ssim": avg_ssim,
        "detection_ap50": ap50,
        "detection_ap50_95": ap50_95,
        "tracking_mota": tracking_mot["mota"],
        "tracking_motp": tracking_mot["motp"],
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


def should_block_output_direct(
    risk: float, rules: dict[str, Any], tau: float | None = None
) -> bool:
    """Chặn frame nếu risk vượt ngưỡng config.

    `tau` chỉ dùng làm fallback khi config thiếu ngưỡng. Không dùng tau làm
    gate chính vì OOF tau (0.54) không transfer sang VisDrone test-dev
    (domain shift): áp tau trực tiếp làm chặn ~47% frame.
    """
    config_threshold = rules.get("output_gate_block_threshold")
    if config_threshold is None:
        config_threshold = tau if tau is not None else 0.75
    return risk >= float(config_threshold)


def should_block_containment(
    gt_boxes: list[list[float]],
    protect_boxes: list[list[float]],
    threshold: float,
) -> bool:
    """Chặn frame nếu có bất kỳ GT box nào không được protect box chứa đủ mức threshold.

    Dùng làm cổng bổ sung để đảm bảo không phát hành frame có người chưa được che.
    """
    for gb in gt_boxes:
        contained = False
        for cb in protect_boxes:
            if containment(gb, cb) >= threshold:
                contained = True
                break
        if not contained:
            return True
    return False


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
    images = sorted(seq_dir.glob("*.jpg"))[:max_frames]
    gt = load_annotations(seq_name)

    pred_boxes_by_frame: dict[int, list[list[float]]] = {}
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
            frame_idx < WARMUP_FRAMES
            or (frame_idx - last_detect_frame) >= det_interval
        )
        detection_intervals.append(float(det_interval))

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
                results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, device=device, half=half, verbose=False)
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
            class _EmptyBoxes:
                def __init__(self):
                    self.conf = np.empty((0,), dtype=np.float32)
                    self.cls = np.empty((0,), dtype=np.float32)
                    self.xywh = np.empty((0, 4), dtype=np.float32)
                    self.xywhr = np.empty((0, 5), dtype=np.float32)

                def __getitem__(self, idx):
                    return _EmptyBoxes()

                def __len__(self):
                    return 0

            empty_boxes = _EmptyBoxes()
            tracker_out = tracker.update(empty_boxes, frame)

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
            X_norm = risk_standardizer.transform(X).reshape(1, -1)
            risk_prob = predict_risk(risk_model, X_norm)
            # Chỉ ghi nhận cặp (y_true, y_prob) khi có ground truth label khớp
            # theo (sequence, frame) từ risk_dataset.csv.
            matched_label: int | None = None
            if risk_labels_lookup:
                frame_key = (seq_name, frame_idx)
                if frame_key in risk_labels_lookup:
                    # Nếu có bất kỳ artifact track nào trong frame này có label=1,
                    # coi frame đó là rủi ro (conservative).
                    matched_label = int(max(risk_labels_lookup[frame_key]))
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

        # Cổng bổ sung: chặn nếu có GT box nào chưa được che đủ mức containment.
        if (
            not block
            and coordinator_rules.get("enable_containment_gate", False)
        ):
            gts_for_gate = gt.get(frame_number, [])
            if gts_for_gate:
                block = should_block_containment(
                    gts_for_gate,
                    protect_boxes_frame,
                    coordinator_rules.get("containment_block_threshold", 0.9),
                )

        if block:
            blocked += 1
            out_frame = np.zeros_like(frame)
        else:
            released += 1
            out_frame = blur_boxes(frame, protect_boxes_frame)
            gts = gt.get(frame_number, [])
            frame_unsafe = False
            for gb in gts:
                total_protect_gt_boxes += 1
                protected_any = False
                contained_any = False
                # IoU coverage: đo trên box detector/tracker TRƯỚC padding,
                # đúng protocol trong README.
                for pb in candidate_boxes:
                    if iou(gb, pb) >= 0.5:
                        protected_any = True
                        break
                # Containment: đo trên box ĐÃ padding (vùng thực sự bị che).
                for cb in protect_boxes_frame:
                    if containment(gb, cb) >= 0.9:
                        contained_any = True
                        break
                if protected_any:
                    protected_gt_boxes += 1
                if contained_any:
                    contained_gt_boxes += 1
                else:
                    frame_unsafe = True
            if frame_unsafe:
                unsafe += 1

        if writer is not None:
            writer.write(out_frame)

        if should_detect:
            detector_boxes_by_frame[frame_number] = [
                ([float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(b[4]))
                for b in detections_np[:, :5].tolist()
            ]
        pred_boxes_by_frame[frame_number] = track_boxes
        gt_boxes_by_frame[frame_number] = gt.get(frame_number, [])

        # Chỉ đo chất lượng vùng che trên frame ĐÃ PHÁT HÀNH (không tính frame bị chắn đen).
        if not block:
            try:
                psnr_val = compute_psnr(frame, out_frame)
                if math.isfinite(psnr_val):
                    psnr_sum += psnr_val
                    psnr_count += 1
            except Exception:
                pass

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

    return {
        "pred_boxes_by_frame": pred_boxes_by_frame,
        "detector_boxes_by_frame": detector_boxes_by_frame,
        "gt_boxes_by_frame": gt_boxes_by_frame,
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
    }


def main() -> None:
    global IMG_SIZE, CONF_THRES, IOU_THRES, MAX_FRAMES_PER_SEQ, WARMUP_FRAMES

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    OUT_GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    # Đồng nhất các tham số inference từ config dùng chung (cùng Kaggle)
    eval_cfg = load_eval_config()
    IMG_SIZE = int(eval_cfg.get("img_size", IMG_SIZE))
    CONF_THRES = float(eval_cfg.get("conf_thres", CONF_THRES))
    IOU_THRES = float(eval_cfg.get("iou_thres", IOU_THRES))
    MAX_FRAMES_PER_SEQ = int(eval_cfg.get("max_frames_per_seq", MAX_FRAMES_PER_SEQ))
    WARMUP_FRAMES = int(eval_cfg.get("warmup_frames", WARMUP_FRAMES))
    print(
        f"[Bước 5] Eval config: img_size={IMG_SIZE}, conf_thres={CONF_THRES}, "
        f"iou_thres={IOU_THRES}, max_frames_per_seq={MAX_FRAMES_PER_SEQ}, "
        f"warmup_frames={WARMUP_FRAMES}"
    )

    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình XGBoost: {RISK_MODEL_PATH}")

    backend = DETECTOR_BACKEND.lower()
    det_device = DEVICE
    detector_half = False
    ort_session = None
    ort_input_name = None

    # Chỉ kiểm tra file .pt khi backend là pytorch
    if backend == "pytorch":
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy mô hình detect: {WEIGHTS_PATH}")

    if backend == "onnx" and _HAS_ORT and ONNX_MODEL_PATH.exists():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
        ort_input_name = ort_session.get_inputs()[0].name
        # Lấy kích thước input mà model ONNX yêu cầu (H, W) để đặt IMG_SIZE khớp
        onnx_input_shape = ort_session.get_inputs()[0].shape
        if len(onnx_input_shape) >= 4:
            onnx_h = onnx_input_shape[2]
            onnx_w = onnx_input_shape[3]
            if isinstance(onnx_h, int) and isinstance(onnx_w, int) and onnx_h > 0 and onnx_w > 0:
                IMG_SIZE = max(onnx_h, onnx_w)
        print(f"[Bước 5] Detector backend: onnx | session: {ONNX_MODEL_PATH} | providers: {providers} | imgsz={IMG_SIZE}")
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
        model.to(DEVICE)
        print(f"[Bước 5] Detector backend: pytorch | device: {DEVICE} | half: {detector_half}")

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

    sequences = sorted(
        [d.name for d in SEQUENCES_DIR.iterdir() if d.is_dir()]
    )
    if MAX_SEQS > 0:
        sequences = sequences[:MAX_SEQS]

    all_gt: dict[str, dict[int, list[list[float]]]] = {}
    all_pred: dict[str, dict[int, list[list[float]]]] = {}
    all_detector: dict[str, dict[int, list[list[float]]]] = {}
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

    for seq in sequences:
        seq_dir = SEQUENCES_DIR / seq
        images = sorted(seq_dir.glob("*.jpg"))[:MAX_FRAMES_PER_SEQ]
        if not images:
            continue

        if WRITE_VIDEO and writer is None:
            first_frame = cv2.imread(str(images[0]))
            if first_frame is None:
                continue
            h, w = first_frame.shape[:2]
            writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, 30.0, (w, h))
            if not writer.isOpened():
                print("[Bước 5] Cảnh báo: không mở được VideoWriter, tắt ghi video.")
                writer.release()
                writer = None

        result = run_sequence(
            seq, model, risk_model, tau, coordinator_rules, writer,
            MAX_FRAMES_PER_SEQ, det_device, detector_half,
            risk_labels_lookup,
            backend=backend,
            ort_session=ort_session,
            ort_input_name=ort_input_name,
        )

        all_gt[seq] = result["gt_boxes_by_frame"]
        all_pred[seq] = result["pred_boxes_by_frame"]
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

    if writer is not None:
        writer.release()

    metrics = compute_metrics(
        all_gt, all_pred, all_detector, all_frames,
        all_latency, total_detection_calls,
        all_intervals, total_blocked, total_released,
        total_protected_gt_boxes, total_contained_gt_boxes, total_protect_gt_boxes,
        total_psnr_sum, total_psnr_count,
        total_ssim_sum, total_ssim_count,
    )

    rows = []
    for k, v in metrics.items():
        rows.append({"metric_group": "system", "policy": "integrated", "sequence": "all",
                     "repetition": 0, "metric_name": k, "metric_value": v})
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    gate_log = {
        "total_blocked": total_blocked,
        "total_released": total_released,
        "total_unsafe": total_unsafe,
    }
    with OUT_GATE_LOG.open("w", encoding="utf-8") as f:
        json.dump(gate_log, f, indent=2, ensure_ascii=False)

    print(f"[Bước 5] Đã lưu gate log: {OUT_GATE_LOG}")
    if WRITE_VIDEO and writer is not None:
        print(f"[Bước 5] Đã lưu video đã che: {OUT_VIDEO}")
    else:
        print("[Bước 5] Bỏ qua ghi video (WRITE_VIDEO=False), vẫn tính metric trên frame đã che.")
    print(f"[Bước 5] Đã lưu chỉ số đánh giá: {OUT_METRICS}")
    print(f"[Bước 5] YOLO precision: {metrics['detection_precision']:.4f}")
    print(f"[Bước 5] YOLO recall: {metrics['detection_recall']:.4f}")
    print(f"[Bước 5] YOLO F1: {metrics['detection_f1']:.4f}")
    print(f"[Bước 5] Tracking precision: {metrics['tracking_precision']:.4f}")
    print(f"[Bước 5] Tracking recall: {metrics['tracking_recall']:.4f}")
    print(f"[Bước 5] Tracking F1: {metrics['tracking_f1']:.4f}")
    print(f"[Bước 5] Protect-box coverage @ IoU 0.5: {metrics['protect_box_coverage_iou50']:.4f}")
    print(f"[Bước 5] Protect-box containment @ 90%: {metrics['protect_box_containment90']:.4f}")
    print(f"[Bước 5] Avg PSNR (dB): {metrics['avg_psnr_db']:.2f}")
    print(f"[Bước 5] Avg SSIM: {metrics['avg_ssim']:.4f}")
    print(f"[Bước 5] Output gate block rate: {metrics['output_gate_block_rate']:.4f}")
    print(f"[Bước 5] Output gate release rate: {metrics['output_gate_release_rate']:.4f}")
    print(f"[Bước 5] Detection calls / 100 frames: {metrics['detection_calls_per_100_frames']:.2f}")
    print(f"[Bước 5] FPS wall: {metrics['fps_wall']:.2f}")


if __name__ == "__main__":
    main()