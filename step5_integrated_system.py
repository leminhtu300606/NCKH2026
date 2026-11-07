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
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

WEIGHTS_PATH = Path("models/best.pt")
RISK_MODEL_PATH = Path("models/xgboost_risk.json")
RISK_CALIBRATED_PATH = Path("models/xgboost_risk_calibrated.joblib")
DATA_ROOT = Path("VisDrone2019-VID-test-dev")
SEQUENCES_DIR = DATA_ROOT / "sequences"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
OUT_VIDEO = Path("outputs/protected_video.mp4")
OUT_GATE_LOG = Path("results/output_gate_log.json")
OUT_METRICS = Path("results/system_metrics_unified.csv")
COORDINATOR_RULES = Path("config/coordinator_rules.json")
TAU_PATH = Path("zip_extract_full/artifacts/tau.json")

FEATURES = [
    "conf", "conf_std5", "conf_delta", "track_age", "hit_ratio",
    "time_since_update", "track_state", "min_side_px", "aspect_dev",
    "dist_edge", "is_trunc", "speed_norm", "pred_uncert", "lap_var_log",
    "brightness", "motion_blur_ratio", "nearest_dist_norm",
    "n_neighbors_r2_norm", "overlap_max", "frame_det_delta",
    "n_det_expected_ratio",
]

IMG_SIZE = 640
# Ghi video ra file là tùy chọn: tắt mặc định để tránh lỗi FFmpeg
# "Failed to write frame" trên môi trường không có encoder tương thích.
# Metrics (PSNR/SSIM) vẫn được tính trên frame đã che trong bộ nhớ.
WRITE_VIDEO = False
CONF_THRES = 0.05  # Giảm ngưỡng để tăng recall (phát hiện nhiều người hơn → bảo hộ nhiều hơn)
IOU_THRES = 0.5
MAX_FRAMES_PER_SEQ = 200
WARMUP_FRAMES = 30

CONF_WINDOW = 5
PRED_UNCERT_WINDOW = 5
EWMA_ALPHA = 0.3
TRACK_BUFFER_FRAMES = 30
TRUNC_MARGIN_PX = 2
NEIGHBOR_RADIUS_FRAC = 0.05
NEIGHBOR_K = 5
CROP_RESIZE = 224
SSIM_INTERVAL = 5

DEVICE = "0" if torch.cuda.is_available() else "cpu"
print(f"[Bước 5] Sử dụng thiết bị: {DEVICE} (CUDA: {torch.cuda.is_available()})")


def load_tau() -> float:
    if TAU_PATH.exists():
        with TAU_PATH.open("r", encoding="utf-8") as f:
            return float(json.load(f).get("tau", 0.4449612498283386))
    return 0.4449612498283386


def load_risk_model() -> Any:
    """Tải model calibration nếu hợp lệ, ngược lại fallback về XGBoost thô.

    File `models/xgboost_risk_calibrated.joblib` trong artifact có thể bị hỏng
    (joblib báo "input stream corrupted"). Trường hợp đó chỉ ghi cảnh báo ngắn
    gọn và dùng `models/xgboost_risk.json`.
    """
    if RISK_CALIBRATED_PATH.exists() and _HAS_JOBLIB:
        try:
            cal = joblib.load(str(RISK_CALIBRATED_PATH))
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


def predict_risk(risk_model: Any, X: np.ndarray) -> float:
    if _HAS_JOBLIB and hasattr(risk_model, "predict_proba"):
        return float(risk_model.predict_proba(X)[0, 1])
    return float(risk_model.predict_proba(X)[0, 1])


NORM_MIN_SAMPLES = 50


class SequenceStandardizer:
    """Chuẩn hóa z-score online theo sequence, chỉ dùng dữ liệu quá khứ.

    Khớp với cách huấn luyện trong `step_train_risk_model.py`:
    - `min_n` vector đầu tiên giữ nguyên giá trị thô;
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
            ignore = int(parts[9])
            if category != 1 or ignore == 1:
                continue
            gt.setdefault(frame, []).append([x, y, x + w, y + h])
    return gt


def iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def containment(gt_box: list[float], cover_box: list[float]) -> float:
    """Tỷ lệ diện tích GT box nằm trong cover box (0..1).

    Phù hợp để đo mức che phủ vì protect box được mở rộng (ROI padding) nên
    IoU sẽ bị giảm dù GT vẫn được che hoàn toàn.
    """
    x1 = max(gt_box[0], cover_box[0])
    y1 = max(gt_box[1], cover_box[1])
    x2 = min(gt_box[2], cover_box[2])
    y2 = min(gt_box[3], cover_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = max(0, gt_box[2] - gt_box[0]) * max(0, gt_box[3] - gt_box[1])
    return inter / gt_area if gt_area > 0 else 0.0


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
    out = frame.copy()
    h, w = out.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        crop = out[y1:y2, x1:x2]
        if crop.size > 0:
            out[y1:y2, x1:x2] = cv2.GaussianBlur(crop, (51, 51), 0)
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


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(prob[mask].mean()) - float(y_true[mask].mean()))
    return ece


def compute_metrics(
    gt_all: dict[str, list[list[float]]],
    pred_all: dict[str, list[list[float]]],
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
    risk_y_true: list[int] | None = None,
    risk_y_prob: list[float] | None = None,
) -> dict[str, Any]:
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for seq in gt_all:
        gts = gt_all[seq]
        preds = pred_all.get(seq, [])
        total_gt += len(gts)
        matched_gt = set()
        tp = 0
        fp = 0
        for pb in preds:
            best_iou = 0.0
            best_idx = -1
            for i, gb in enumerate(gts):
                if i in matched_gt:
                    continue
                iou_val = iou(pb, gb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            if best_iou >= 0.5 and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn = len(gts) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    detection_precision = total_tp / max(1, total_tp + total_fp)
    detection_recall = total_tp / max(1, total_tp + total_fn)
    detection_f1 = 2 * detection_precision * detection_recall / max(1e-9, detection_precision + detection_recall)

    avg_psnr = psnr_sum / max(1, psnr_count) if psnr_count > 0 else 0.0
    avg_ssim = ssim_sum / max(1, ssim_count) if ssim_count > 0 else 0.0

    total_frames = sum(frames_all.values())
    output_gate_block_rate = blocked_frames / max(1, total_frames)
    output_gate_release_rate = released_frames / max(1, total_frames)
    detection_calls_per_100_frames = 100.0 * detection_calls / max(1, total_frames)
    avg_detection_interval = float(np.mean(detection_intervals)) if detection_intervals else 0.0
    # Tỷ lệ GT box người được protect box phủ với IoU >= 0.5.
    protect_box_coverage = protected_gt_boxes / max(1, total_protect_gt_boxes)
    # Tỷ lệ GT box được protect box che >= 90% diện tích (đúng bản chất ROI padding).
    protect_box_containment = contained_gt_boxes / max(1, total_protect_gt_boxes)

    fps_wall = len(latency_samples) / max(1e-9, sum(latency_samples)) if latency_samples else 0.0
    latency_arr = np.array(latency_samples) if latency_samples else np.array([0.0])

    risk_auc = None
    risk_pr_auc = None
    risk_accuracy = None
    risk_precision = None
    risk_recall = None
    risk_f1 = None
    risk_brier = None
    risk_ece = None

    if risk_y_true is not None and risk_y_prob is not None and len(risk_y_true) > 0:
        y = np.array(risk_y_true, dtype=int)
        p = np.array(risk_y_prob, dtype=float)
        if len(np.unique(y)) > 1:
            from sklearn.metrics import (roc_auc_score, average_precision_score,
                                         accuracy_score, precision_score,
                                         recall_score, f1_score,
                                         brier_score_loss)
            risk_auc = float(roc_auc_score(y, p))
            risk_pr_auc = float(average_precision_score(y, p))
            y_hat = (p >= 0.5).astype(int)
            risk_accuracy = float(accuracy_score(y, y_hat))
            risk_precision = float(precision_score(y, y_hat, zero_division=0))
            risk_recall = float(recall_score(y, y_hat, zero_division=0))
            risk_f1 = float(f1_score(y, y_hat, zero_division=0))
            risk_brier = float(brier_score_loss(y, p))
            risk_ece = expected_calibration_error(y, p)

    metrics = {
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "detection_f1": detection_f1,
        "avg_psnr_db": avg_psnr,
        "avg_ssim": avg_ssim,
        "output_gate_block_rate": output_gate_block_rate,
        "output_gate_release_rate": output_gate_release_rate,
        "protect_box_coverage_iou50": protect_box_coverage,
        "protect_box_containment90": protect_box_containment,
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
        "total_pred_boxes": total_tp + total_fp,
        "risk_auc": risk_auc,
        "risk_pr_auc": risk_pr_auc,
        "risk_accuracy": risk_accuracy,
        "risk_precision": risk_precision,
        "risk_recall": risk_recall,
        "risk_f1": risk_f1,
        "risk_brier": risk_brier,
        "risk_ece": risk_ece,
    }
    return metrics


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
        self.box: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.conf: float = 0.5

    def update(self, frame: int, box: list[float], conf: float, lap_var: float) -> None:
        self.age += 1
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

    def mark_missed(self, frame: int) -> None:
        self.time_since_update += 1

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
        c5 = self.conf_history[-(CONF_WINDOW - 1):] + [conf] if CONF_WINDOW > 1 else [conf]
        f_conf_std5 = float(np.std(c5)) if len(c5) > 1 else 0.0
        f_conf_delta = 0.0 if len(self.conf_history) < 2 else conf - self.conf_history[-2]
        f_track_age = float(self.age)
        f_hit_ratio = min(1.0, self.hits / max(self.age, 1))
        f_time_since_update = float(self.time_since_update)
        f_track_state = self.track_state_code
        f_min_side_px = float(min(bw, bh))
        f_aspect_dev = float(abs(bw / bh - r_ref))
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
        f_lap_var_log = float(math.log1p(self.lap_history[-1])) if self.lap_history else 0.0
        f_brightness = 0.5
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
                n_nb = int((dists <= radius).sum()) - 1
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
            "track_age": f_track_age,
            "hit_ratio": f_hit_ratio,
            "time_since_update": f_time_since_update,
            "track_state": f_track_state,
            "min_side_px": f_min_side_px,
            "aspect_dev": f_aspect_dev,
            "dist_edge": f_dist_edge,
            "is_trunc": f_is_trunc,
            "speed_norm": f_speed_norm,
            "pred_uncert": f_pred_uncert,
            "lap_var_log": f_lap_var_log,
            "brightness": f_brightness,
            "motion_blur_ratio": f_motion_blur_ratio,
            "nearest_dist_norm": f_nearest_dist_norm,
            "n_neighbors_r2_norm": f_n_neighbors_r2_norm,
            "overlap_max": f_overlap_max,
            "frame_det_delta": f_frame_det_delta,
            "n_det_expected_ratio": f_n_det_expected_ratio,
        }


def classify_risk_level(risk: float, rules: dict[str, Any]) -> str:
    if risk >= rules.get("risk_critical", 0.75):
        return "critical"
    if risk >= rules.get("risk_high", 0.50):
        return "high"
    if risk >= rules.get("risk_low", 0.25):
        return "medium"
    return "low"


def should_block_output_direct(risk: float, rules: dict[str, Any]) -> bool:
    return risk >= rules.get("output_gate_block_threshold", 0.75)


def run_sequence(
    seq_name: str,
    model: YOLO,
    risk_model: Any,
    tau: float,
    coordinator_rules: dict[str, Any],
    writer: cv2.VideoWriter | None,
    max_frames: int,
) -> dict[str, Any]:
    seq_dir = SEQUENCES_DIR / seq_name
    images = sorted(seq_dir.glob("*.jpg"))[:max_frames]
    gt = load_annotations(seq_name)

    pred_boxes: list[list[float]] = []
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
    ewma_det: float | None = None
    prev_frame_risk_level: str | None = None

    track_histories: dict[int, TrackHistory] = {}
    risk_standardizer = SequenceStandardizer(len(FEATURES))

    tracker_args = SimpleNamespace(
        track_high_thresh=CONF_THRES,
        track_low_thresh=0.05,
        new_track_thresh=CONF_THRES,
        track_buffer=TRACK_BUFFER_FRAMES,
        match_thresh=0.7,
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

        if should_detect:
            results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, device=DEVICE, verbose=False)
            detection_calls += 1
            last_detect_frame = frame_idx
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            det_array = []
            lap_vars: dict[int, float] = {}
            for r in results:
                for box in r.boxes or []:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    det_array.append([x1, y1, x2, y2, conf, 0.0])
                    idx = len(det_array) - 1
                    crop = gray[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))]
                    lap_vars[idx] = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size > 0 else 0.0

            detections_np = np.array(det_array, dtype=np.float32) if det_array else np.empty((0, 6), dtype=np.float32)
        else:
            detections_np = np.empty((0, 6), dtype=np.float32)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_vars = {}

        if should_detect:
            tracker_out = tracker.update(results[0].boxes, frame)
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
                x1, y1, x2, y2, tid, score, cls, det_idx = row
                track_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                track_confs.append(float(score))
                track_ids.append(int(tid))

                if int(tid) not in track_histories:
                    track_histories[int(tid)] = TrackHistory(int(tid), frame_idx)
                th = track_histories[int(tid)]

                lap_var = 0.0
                if should_detect and len(detections_np) > 0:
                    best_iou = 0.0
                    best_idx = -1
                    for i, det in enumerate(detections_np):
                        iou_val = iou([float(x1), float(y1), float(x2), float(y2)],
                                      [float(det[0]), float(det[1]), float(det[2]), float(det[3])])
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_idx = i
                    if best_idx >= 0 and best_idx in lap_vars:
                        lap_var = lap_vars[best_idx]
                else:
                    lap_var = th.lap_history[-1] if th.lap_history else 0.0

                th.update(frame_idx, [float(x1), float(y1), float(x2), float(y2)], float(score), lap_var)

        current_ids = set(track_ids)
        for tid, th in track_histories.items():
            if tid not in current_ids:
                th.mark_missed(frame_idx)

        if should_detect:
            N_t = len(detections_np)
            expected = N_t if ewma_det is None else ewma_det
            ewma_det = N_t if ewma_det is None else EWMA_ALPHA * N_t + (1 - EWMA_ALPHA) * ewma_det
            prev_det_count = N_t
        else:
            prev_det_count = len(track_boxes)

        all_boxes_frame = track_boxes
        frame_risk = 0.0

        max_risk_prob = 0.0
        for tid, th in track_histories.items():
            if tid not in current_ids:
                continue

            feat = th.compute_features(
                frame=frame_idx,
                img_w=w,
                img_h=h,
                all_boxes_frame=all_boxes_frame,
                prev_det_count=prev_det_count if should_detect else None,
                ewma_det=ewma_det,
                r_ref=r_ref,
            )

            if should_detect and tid in track_ids:
                idx = track_ids.index(tid)
                x1, y1, x2, y2 = [int(v) for v in track_boxes[idx]]
                x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w)); y2 = max(y1 + 1, min(y2, h))
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    feat["brightness"] = float(gray_crop.mean() / 255.0)

            X = np.array([feat[f] for f in FEATURES], dtype=np.float32)
            X_norm = risk_standardizer.transform(X).reshape(1, -1)
            risk_prob = predict_risk(risk_model, X_norm)
            if risk_prob > max_risk_prob:
                max_risk_prob = risk_prob

        risk_y_prob.append(max_risk_prob)
        frame_risk = max_risk_prob

        risks.append(frame_risk)
        prev_frame_risk_level = classify_risk_level(frame_risk, coordinator_rules)

        roi_padding = {
            "critical": coordinator_rules.get("roi_expand_critical", 0.50),
            "high": coordinator_rules.get("roi_expand_high", 0.30),
            "medium": coordinator_rules.get("roi_expand_medium", 0.20),
            "low": coordinator_rules.get("roi_expand_low", 0.10),
        }.get(prev_frame_risk_level, coordinator_rules.get("roi_expand_low", 0.10))
        protect_boxes_frame = []
        for box in track_boxes:
            pbox = expand_box(box, roi_padding, w, h)
            protect_boxes_frame.append(pbox)

        block = should_block_output_direct(frame_risk, coordinator_rules)
        frame_unsafe = False
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
                for pb in protect_boxes_frame:
                    if not protected_any and iou(gb, pb) >= 0.5:
                        protected_any = True
                    if containment(gb, pb) >= 0.9:
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
            risk_y_true.append(1 if (block or frame_unsafe) else 0)

        if writer is not None:
            writer.write(out_frame)

        pred_boxes.extend(track_boxes)

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
        "pred_boxes": pred_boxes,
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
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    OUT_GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình detect: {WEIGHTS_PATH}")
    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình XGBoost: {RISK_MODEL_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    model.to(DEVICE)
    risk_model = load_risk_model()
    tau = load_tau()

    coordinator_rules = {}
    if COORDINATOR_RULES.exists():
        with COORDINATOR_RULES.open("r", encoding="utf-8") as f:
            coordinator_rules = json.load(f)
    else:
        coordinator_rules = {
            "risk_low": 0.25,
            "risk_high": 0.50,
            "risk_critical": 0.75,
            "detect_interval_low": 5,
            "detect_interval_medium": 3,
            "detect_interval_high": 1,
            "detect_interval_critical": 1,
            "roi_expand_low": 0.10,
            "roi_expand_medium": 0.20,
            "roi_expand_high": 0.30,
            "roi_expand_critical": 0.50,
            "output_gate_block_threshold": 0.75,
        }

    sequences = sorted(
        [d.name for d in SEQUENCES_DIR.iterdir() if d.is_dir()]
    )

    all_gt: dict[str, list[list[float]]] = {}
    all_pred: dict[str, list[list[float]]] = {}
    all_frames: dict[str, int] = {}
    all_risks: dict[str, list[float]] = {}
    all_latency: list[float] = []
    all_intervals: list[float] = []
    all_risk_y_true: list[int] = []
    all_risk_y_prob: list[float] = []
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

        result = run_sequence(seq, model, risk_model, tau, coordinator_rules, writer, MAX_FRAMES_PER_SEQ)

        gt = load_annotations(seq)
        all_gt[seq] = [box for boxes in gt.values() for box in boxes]
        all_pred[seq] = result["pred_boxes"]
        all_frames[seq] = len(images)
        all_risks[seq] = result["risks"]
        all_latency.extend(result["latency_samples"])
        all_intervals.extend(result["detection_intervals"])
        all_risk_y_true.extend(result["risk_y_true"])
        all_risk_y_prob.extend(result["risk_y_prob"])
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
        all_gt, all_pred, all_frames,
        all_latency, total_detection_calls,
        all_intervals, total_blocked, total_released,
        total_protected_gt_boxes, total_contained_gt_boxes, total_protect_gt_boxes,
        total_psnr_sum, total_psnr_count,
        total_ssim_sum, total_ssim_count,
        all_risk_y_true, all_risk_y_prob,
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
    print(f"[Bước 5] Detection precision: {metrics['detection_precision']:.4f}")
    print(f"[Bước 5] Detection recall: {metrics['detection_recall']:.4f}")
    print(f"[Bước 5] Detection F1: {metrics['detection_f1']:.4f}")
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

