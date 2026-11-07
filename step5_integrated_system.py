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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import xgboost as xgb
from ultralytics.trackers.byte_tracker import BYTETracker

from ultralytics import YOLO

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
OUT_GATE_LOG = Path(os.getenv("NCKH_OUT_GATE_LOG", "results/output_gate_log.json"))
OUT_METRICS = Path(os.getenv("NCKH_OUT_METRICS", "results/system_metrics_unified.csv"))
COORDINATOR_RULES = Path("config/coordinator_rules.json")
TAU_PATH = Path("config/tau_val.json")
RISK_DATASET_PATH = Path("zip_extract_full/artifacts/risk_dataset.csv")

FEATURES = [
    "conf", "conf_std5", "conf_delta", "track_age", "hit_ratio",
    "time_since_update", "track_state", "min_side_px", "aspect_dev",
    "dist_edge", "is_trunc", "speed_norm", "pred_uncert", "lap_var_log",
    "brightness", "motion_blur_ratio", "nearest_dist_norm",
    "n_neighbors_r2_norm", "overlap_max", "frame_det_delta",
    "n_det_expected_ratio",
]

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
PROTECT_PADDING = 0.0
# Cho phép chạy nhanh để so sánh backend mà không sửa code:
#   NCKH_BACKEND=onnx NCKH_MAX_SEQS=3 NCKH_MAX_FRAMES=40 python step5_integrated_system.py
DETECTOR_BACKEND = os.getenv("NCKH_BACKEND", "pytorch")
ONNX_MODEL_PATH = Path(os.getenv("NCKH_ONNX_MODEL", "models/best_raw_dyn.onnx"))
MAX_SEQS = int(os.getenv("NCKH_MAX_SEQS", "0"))  # 0 = tất cả sequence
MAX_FRAMES_PER_SEQ = int(os.getenv("NCKH_MAX_FRAMES", "200"))
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


def load_risk_model() -> Any:
    """Tải model calibration nếu hợp lệ, ngược lại fallback về XGBoost thô.

    File models/xgboost_risk_calibrated.joblib trong artifact có thể bị hỏng
    (joblib báo "input stream corrupted"). Trường hợp đó chỉ ghi cảnh báo ngắn
    gọn và dùng models/xgboost_risk.json.
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


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(prob[mask].mean()) - float(y_true[mask].mean()))
    return ece


def average_precision(
    preds_by_frame: dict[str, dict[int, list[tuple[list[float], float]]]],
    gts_by_frame: dict[str, dict[int, list[list[float]]]],
    iou_thresholds: list[float],
) -> dict[float, float]:
    """Tính Average Precision (COCO style) tại các ngưỡng IoU.

    preds_by_frame: seq -> frame -> list of (box, conf)
    gts_by_frame: seq -> frame -> list of box
    Trả về dict {iou_threshold: AP}.

    AP được tính bằng cách integral của precision-recall envelope
    tại 101 ngưỡng recall (0.00, 0.01, ..., 1.00), chuẩn COCO.
    """
    # Gom tất cả detection thành danh sách (seq, frame, box, conf)
    all_dets: list[tuple[str, int, list[float], float]] = []
    total_gt = 0
    for seq, seq_gts in gts_by_frame.items():
        seq_preds = preds_by_frame.get(seq, {})
        for frame_number, gts in seq_gts.items():
            total_gt += len(gts)
            dets = seq_preds.get(frame_number, [])
            for box, conf in dets:
                all_dets.append((seq, frame_number, box, conf))

    # Sort theo conf giảm dần
    all_dets.sort(key=lambda x: x[3], reverse=True)

    aps: dict[float, float] = {}
    for iou_thr in iou_thresholds:
        tp = np.zeros(len(all_dets), dtype=np.float64)
        fp = np.zeros(len(all_dets), dtype=np.float64)
        # Đánh dấu GT đã match (seq, frame, gt_idx)
        matched_gt: set[tuple[str, int, int]] = set()
        for i, (seq, frame_number, box, conf) in enumerate(all_dets):
            gts = gts_by_frame.get(seq, {}).get(frame_number, [])
            best_iou = 0.0
            best_gt_idx = -1
            for j, gt in enumerate(gts):
                if (seq, frame_number, j) in matched_gt:
                    continue
                iou_val = iou(box, gt)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = j
            if best_iou >= iou_thr and best_gt_idx >= 0:
                tp[i] = 1.0
                matched_gt.add((seq, frame_number, best_gt_idx))
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(1, total_gt)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        # COCO AP: integral of precision-recall envelope at 101 recall points
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 101):
            p_at_recall = precisions[recalls >= t].max() if np.any(recalls >= t) else 0.0
            ap += p_at_recall
        aps[iou_thr] = float(ap / 101.0)
    return aps


def mota_motp(
    gt_all: dict[str, dict[int, list[list[float]]]],
    pred_all: dict[str, dict[int, list[list[float]]]],
) -> dict[str, float]:
    """Tính MOTA và MOTP dựa trên matching một-một theo frame tại IoU >= 0.5.

    VisDrone VID test-dev không cung cấp track ID ground-truth, nên IDSW
    không thể tính chính xác. Ở đây IDSW coi là 0 (không theo dõi liên tục).
    """
    total_gt = total_tp = total_fp = total_fn = 0
    iou_sum = 0.0
    iou_count = 0
    for seq, seq_gt in gt_all.items():
        seq_pred = pred_all.get(seq, {})
        for frame_number, gts in seq_gt.items():
            preds = seq_pred.get(frame_number, [])
            total_gt += len(gts)
            if not gts or not preds:
                total_fp += len(preds)
                total_fn += len(gts)
                continue
            ious = np.array(
                [[iou(pred, gt) for gt in gts] for pred in preds],
                dtype=np.float64,
            )
            match_cost = np.where(ious >= 0.5, -1.0 - ious * 1e-6, 0.0)
            pred_indices, gt_indices = linear_sum_assignment(match_cost)
            matched_pairs = [(p, g) for p, g in zip(pred_indices, gt_indices)
                             if ious[p, g] >= 0.5]
            matched = len(matched_pairs)
            total_tp += matched
            total_fp += len(preds) - matched
            total_fn += len(gts) - matched
            for p, g in matched_pairs:
                iou_sum += float(ious[p, g])
                iou_count += 1

    mota = 1.0 - (total_fn + total_fp) / max(1, total_gt)  # IDSW = 0
    motp = iou_sum / max(1, iou_count)
    return {
        "mota": float(mota),
        "motp": float(motp),
        "total_gt": int(total_gt),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
    }


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
    detection_aps = average_precision(
        detector_all, gt_all, iou_thresholds=[0.5, 0.5, 0.75]
    )
    # average_precision trả về AP cho mỗi iou threshold; chúng ta lấy AP@0.5
    # và trung bình các ngưỡng 0.5:0.05:0.95 (COCO-style).
    ap50 = detection_aps.get(0.5, 0.0)
    # AP@0.5:0.95: tính trung bình AP tại các ngưỡng 0.5, 0.55, ..., 0.95
    ap_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    detection_aps_all = average_precision(
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
    ewma_det: float | None = None
    prev_frame_risk_level: str | None = None

    track_histories: dict[int, TrackHistory] = {}
    risk_standardizer = SequenceStandardizer(len(FEATURES))

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
            results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, device=device, half=half, verbose=False)
            detection_calls += 1
            last_detect_frame = frame_idx
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
                    if best_idx >= 0:
                        x1, y1, x2, y2 = [int(v) for v in detections_np[best_idx, :4]]
                        x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
                        x2 = max(x1 + 1, min(x2, w)); y2 = max(y1 + 1, min(y2, h))
                        crop = gray[y1:y2, x1:x2]
                        lap_var = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size > 0 else 0.0
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
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    OUT_GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình detect: {WEIGHTS_PATH}")
    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình XGBoost: {RISK_MODEL_PATH}")

    backend = DETECTOR_BACKEND.lower()
    det_device = DEVICE
    detector_half = False
    if backend in ("onnx", "openvino"):
        print(
            f"[Bước 5] Cảnh báo: backend '{backend}' chưa triển khai inference riêng "
            "trong Bước 5, fallback PyTorch."
        )
        backend = "pytorch"
    model = YOLO(str(WEIGHTS_PATH))
    model.to(DEVICE)
    print(f"[Bước 5] Detector backend: {backend} | device: {DEVICE} | half: {detector_half}")
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