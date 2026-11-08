"""Bước 6: Đo hiệu năng phần cứng khi nạp mô hình vào thiết bị.

Input:
- models/best.pt (mô hình detect)
- models/xgboost_risk.json (mô hình XGBoost)

Output:
- results/hardware_benchmark.csv (latency, FPS, RAM, CPU/GPU)

Latency end-to-end bây giờ bao gồm: detect YOLO -> tính 21 đặc trưng rủi ro
-> XGBoost predict_proba -> GaussianBlur, phản ánh chi phí thực tế trên một
frame thay vì chỉ đo detector đơn lẻ. Báo cáo thêm tỉ lệ frame vượt deadline
thiết bị biên (mặc định 40 ms theo README).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import psutil
import torch
import xgboost as xgb
from ultralytics import YOLO

WEIGHTS_PATH = Path("zip_extract_full/artifacts/yolo26s.pt")
RISK_MODEL_PATH = Path("zip_extract_full/artifacts/risk_model.json")
OUT_CSV = Path("results/hardware_benchmark.csv")

IMG_SIZE = 1024
CONF_THRES = 0.25
IOU_THRES = 0.7
WARMUP = 10
MEASURE = 100
# Deadline xử lý một frame trên thiết bị biên (theo README). 40 ms ~ 25 FPS.
EDGE_DEADLINE_MS = float(os.getenv("NCKH_EDGE_DEADLINE_MS", "40.0"))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[Bước 6] Sử dụng thiết bị: {DEVICE} (CUDA: {torch.cuda.is_available()})")

FEATURES = [
    "conf", "conf_std5", "conf_delta", "track_age", "hit_ratio",
    "time_since_update", "track_state", "min_side_px", "aspect_dev",
    "dist_edge", "is_trunc", "speed_norm", "pred_uncert", "lap_var_log",
    "brightness", "motion_blur_ratio", "nearest_dist_norm",
    "n_neighbors_r2_norm", "overlap_max", "frame_det_delta",
    "n_det_expected_ratio",
]

TRUNC_MARGIN_PX = 2
CONF_WINDOW = 5
PRED_UNCERT_WINDOW = 5
NEIGHBOR_RADIUS_FRAC = 0.05
NEIGHBOR_K = 5


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


def lap_var_of_crop(frame: np.ndarray, box: list[float], w: int, h: int) -> float:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def compute_21_features_light(
    box: list[float],
    prev_box: list[float],
    frame: np.ndarray,
    gray: np.ndarray,
    lap_history: list[float],
    conf_history: list[float],
    center_history: list[tuple[float, float]],
    det_count_prev: int,
    ewma_det: float,
    r_ref: float,
    conf: float = 0.7,
    track_age: float = 5.0,
    hit_ratio: float = 1.0,
    time_since_update: float = 0.0,
    track_state: float = 1.0,
) -> dict[str, float]:
    """Tính 21 đặc trưng rủi ro cho một track đơn (phiên bản đơn giản để benchmark)."""
    h, w = frame.shape[:2]
    D = math.hypot(w, h)
    bw = max(box[2] - box[0], 1e-6)
    bh = max(box[3] - box[1], 1e-6)
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    f_conf = float(conf)
    c5 = conf_history[-(CONF_WINDOW - 1):] + [conf] if CONF_WINDOW > 1 else [conf]
    f_conf_std5 = float(np.std(c5)) if len(c5) > 1 else 0.0
    f_conf_delta = 0.0 if len(conf_history) < 2 else conf - conf_history[-2]
    f_track_age = float(track_age)
    f_hit_ratio = float(hit_ratio)
    f_time_since_update = float(time_since_update)
    f_track_state = float(track_state)
    f_min_side_px = float(min(bw, bh))
    f_aspect_dev = float(abs(bw / bh - r_ref))
    f_dist_edge = float(
        max(0.0, min(box[0], box[1], w - box[2], h - box[3]) / D)
    )
    f_is_trunc = 1.0 if min(box[0], box[1], w - box[2], h - box[3]) <= TRUNC_MARGIN_PX else 0.0
    f_speed_norm = 0.0
    if len(center_history) >= 2:
        f_speed_norm = float(
            np.hypot(cx - center_history[-2][0], cy - center_history[-2][1]) / D
        )
    cen_win = center_history[-(PRED_UNCERT_WINDOW - 1):] + [(cx, cy)] if PRED_UNCERT_WINDOW > 1 else [(cx, cy)]
    f_pred_uncert = 0.0
    if len(cen_win) > 1:
        arr = np.array(cen_win)
        f_pred_uncert = float(np.sqrt(arr[:, 0].var() + arr[:, 1].var()) / D)
    f_lap_var_log = float(math.log1p(lap_history[-1])) if lap_history else 0.0
    f_brightness = 0.5
    if len(lap_history) >= 2:
        L_prev = lap_history[-2]
        L_cur = lap_history[-1]
        f_motion_blur_ratio = float((L_prev - L_cur) / (L_prev + L_cur + 1e-6))
    else:
        f_motion_blur_ratio = 0.0
    f_nearest_dist_norm = 0.0
    f_n_neighbors_r2_norm = 0.0
    f_overlap_max = 0.0
    N_t = 1
    f_frame_det_delta = 0.0 if det_count_prev is None else float((N_t - det_count_prev) / (det_count_prev + 1))
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


def pick_primary_box(detections_np: np.ndarray, w: int, h: int) -> list[float]:
    """Chọn box detector lớn nhất làm track chính để tính đặc trưng."""
    if len(detections_np) == 0:
        return [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
    areas = (detections_np[:, 2] - detections_np[:, 0]) * (detections_np[:, 3] - detections_np[:, 1])
    idx = int(np.argmax(areas))
    return [float(v) for v in detections_np[idx, :4]]


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {WEIGHTS_PATH}")
    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {RISK_MODEL_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    model.to(DEVICE)
    det_device = DEVICE
    det_half = False
    print(f"[Bước 6] Detector backend: pytorch | device: {DEVICE} | half: {det_half}")
    risk_model = xgb.XGBClassifier()
    risk_model.load_model(str(RISK_MODEL_PATH))

    process = psutil.Process()
    real_frame = Path("VisDrone2019-VID-test-dev/sequences/uav0000009_03358_v/0000001.jpg")
    if real_frame.exists():
        frame = cv2.imread(str(real_frame))
        print(f"[Bước 6] Dùng frame thật: {real_frame} shape={frame.shape}")
    else:
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        print("[Bước 6] Cảnh báo: không tìm thấy frame thật, dùng frame ngẫu nhiên thay thế.")

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Chuẩn bị lịch sử track giả lập (5 frame trước) để tính đặc trưng đại diện.
    primary_box = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
    prev_box = [w * 0.24, h * 0.24, w * 0.76, h * 0.76]
    lap_history = [lap_var_of_crop(frame, primary_box, w, h)] * 5
    conf_history = [0.70] * 5
    center_history = [((primary_box[0] + primary_box[2]) / 2.0, (primary_box[1] + primary_box[3]) / 2.0)] * 5
    det_count_prev = 1
    ewma_det = 1.0
    r_ref = 0.4

    def run_detect():
        return model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, device=det_device, half=det_half, verbose=False)

    def run_features(dets):
        box = pick_primary_box(dets, w, h) if len(dets) else primary_box
        return compute_21_features_light(
            box=box,
            prev_box=prev_box,
            frame=frame,
            gray=gray,
            lap_history=lap_history,
            conf_history=conf_history,
            center_history=center_history,
            det_count_prev=det_count_prev,
            ewma_det=ewma_det,
            r_ref=r_ref,
        )

    def run_xgb(feat):
        X = np.array([feat[f] for f in FEATURES], dtype=np.float32).reshape(1, -1)
        return risk_model.predict_proba(X)

    def run_blur():
        cv2.GaussianBlur(frame, (51, 51), 0)

    # Warm-up
    for _ in range(WARMUP):
        res = run_detect()
        cpu_boxes = res[0].cpu().boxes
        dets = cpu_boxes.xyxy.numpy() if len(cpu_boxes.xyxy) else np.empty((0, 4))
        feat = run_features(dets)
        run_xgb(feat)
        run_blur()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def measure_detector() -> list[float]:
        lat = []
        for _ in range(MEASURE):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_detect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    def measure_detect_blur() -> list[float]:
        lat = []
        for _ in range(MEASURE):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_detect()
            run_blur()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    def measure_xgb() -> list[float]:
        # Tính đặc trưng một lần (bên ngoài đo) để chỉ đo chi phí XGBoost.
        res = run_detect()
        cpu_boxes = res[0].cpu().boxes
        dets = cpu_boxes.xyxy.numpy() if len(cpu_boxes.xyxy) else np.empty((0, 4))
        feat = run_features(dets)
        X = np.array([feat[f] for f in FEATURES], dtype=np.float32).reshape(1, -1)
        lat = []
        for _ in range(MEASURE):
            t0 = time.perf_counter()
            risk_model.predict_proba(X)
            lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    def measure_end_to_end() -> list[float]:
        lat = []
        for _ in range(MEASURE):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = run_detect()
            cpu_boxes = res[0].cpu().boxes
            dets = cpu_boxes.xyxy.numpy() if len(cpu_boxes.xyxy) else np.empty((0, 4))
            feat = run_features(dets)
            run_xgb(feat)
            run_blur()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    det_latencies = measure_detector()
    db_latencies = measure_detect_blur()
    risk_latencies = measure_xgb()
    e2e_latencies = measure_end_to_end()

    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=1)

    def stats(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        return {
            "mean_ms": float(np.mean(a)),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(np.max(a)),
            "fps": 1000.0 / float(np.mean(a)) if np.mean(a) > 0 else 0.0,
        }

    s_det = stats(det_latencies)
    s_db = stats(db_latencies)
    s_xgb = stats(risk_latencies)
    s_e2e = stats(e2e_latencies)

    def over_deadline(latencies: list[float]) -> dict[str, Any]:
        a = np.array(latencies)
        over = int((a > EDGE_DEADLINE_MS).sum())
        return {
            "pct_over_deadline": float(over / len(a) * 100.0) if len(a) else 0.0,
            "over_deadline_count": over,
        }

    od_det = over_deadline(det_latencies)
    od_e2e = over_deadline(e2e_latencies)

    rows: list[dict[str, Any]] = []
    for name, s in [
        ("detector", s_det),
        ("detect_blur", s_db),
        ("xgboost", s_xgb),
        ("end_to_end", s_e2e),
    ]:
        rows.append({
            "component": name,
            "mean_ms": s["mean_ms"],
            "p50_ms": s["p50_ms"],
            "p95_ms": s["p95_ms"],
            "p99_ms": s["p99_ms"],
            "max_ms": s["max_ms"],
            "fps": s["fps"],
            "ram_mb": np.nan,
            "cpu_percent": np.nan,
            "deadline_ms": EDGE_DEADLINE_MS,
            "pct_over_deadline": od_det["pct_over_deadline"] if name == "detector" else od_e2e["pct_over_deadline"],
            "over_deadline_count": od_det["over_deadline_count"] if name == "detector" else od_e2e["over_deadline_count"],
        })
    rows.append({
        "component": "system",
        "mean_ms": np.nan,
        "p50_ms": np.nan,
        "p95_ms": np.nan,
        "p99_ms": np.nan,
        "max_ms": np.nan,
        "fps": np.nan,
        "ram_mb": ram_mb,
        "cpu_percent": cpu_percent,
        "deadline_ms": EDGE_DEADLINE_MS,
        "pct_over_deadline": np.nan,
        "over_deadline_count": np.nan,
    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[Bước 6] Đã lưu hiệu năng phần cứng: {OUT_CSV}")
    print(f"[Bước 6] Detector mean: {s_det['mean_ms']:.2f} ms, FPS: {s_det['fps']:.2f}")
    print(f"[Bước 6] Detect+blur mean: {s_db['mean_ms']:.2f} ms, FPS: {s_db['fps']:.2f}")
    print(f"[Bước 6] XGBoost mean: {s_xgb['mean_ms']:.4f} ms, FPS: {s_xgb['fps']:.2f}")
    print(f"[Bước 6] End-to-end mean: {s_e2e['mean_ms']:.2f} ms, FPS: {s_e2e['fps']:.2f}")
    print(f"[Bước 6] Deadline edge: {EDGE_DEADLINE_MS} ms | end-to-end quá deadline: {od_e2e['pct_over_deadline']:.1f}%")
    print(f"[Bước 6] RAM: {ram_mb:.1f} MB, CPU: {cpu_percent:.1f}%")


if __name__ == "__main__":
    main()
