"""Bước 6: Đo hiệu năng phần cứng khi nạp mô hình vào thiết bị.

Input:
- models/best.pt (mô hình detect)
- models/xgboost_risk.json (mô hình XGBoost)

Output:
- results/hardware_benchmark.csv (latency, FPS, RAM, CPU/GPU)
"""

from __future__ import annotations

import json
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

WEIGHTS_PATH = Path("models/best.pt")
RISK_MODEL_PATH = Path("models/xgboost_risk.json")
OUT_CSV = Path("results/hardware_benchmark.csv")

IMG_SIZE = 1024
CONF_THRES = 0.05
IOU_THRES = 0.7
WARMUP = 10
MEASURE = 100

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[Bước 6] Sử dụng thiết bị: {DEVICE} (CUDA: {torch.cuda.is_available()})")

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
    print(f"[Bước 6] Detector backend: pytorch | device: {DEVICE} | half: False")
    risk_model = xgb.XGBClassifier()
    risk_model.load_model(str(RISK_MODEL_PATH))

    FEATURES = [
        "conf", "conf_std5", "conf_delta", "track_age", "hit_ratio",
        "time_since_update", "track_state", "min_side_px", "aspect_dev",
        "dist_edge", "is_trunc", "speed_norm", "pred_uncert", "lap_var_log",
        "brightness", "motion_blur_ratio", "nearest_dist_norm",
        "n_neighbors_r2_norm", "overlap_max", "frame_det_delta",
        "n_det_expected_ratio",
    ]

    process = psutil.Process()
    # Dùng frame thật để latency phản ánh dữ liệu triển khai; fallback frame đen.
    real_frame = Path("VisDrone2019-VID-test-dev/sequences/uav0000009_03358_v/0000001.jpg")
    if real_frame.exists():
        frame = cv2.imread(str(real_frame))
        print(f"[Bước 6] Dùng frame thật: {real_frame} shape={frame.shape}")
    else:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        print("[Bước 6] Cảnh báo: không tìm thấy frame thật, dùng frame đen tổng hợp.")

    # Warm-up
    for _ in range(WARMUP):
        _ = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, device=det_device, half=det_half, verbose=False)
        X = np.zeros((1, len(FEATURES)))
        _ = risk_model.predict_proba(X)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Đo latency detector
    det_latencies: list[float] = []
    for _ in range(MEASURE):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, device=det_device, half=det_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        det_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Đo latency detector + blur (không risk) để có mốc so sánh riêng
    db_latencies: list[float] = []
    for _ in range(MEASURE):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, device=det_device, half=det_half, verbose=False)
        cv2.GaussianBlur(frame, (51, 51), 0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        db_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Đo latency XGBoost
    risk_latencies: list[float] = []
    for _ in range(MEASURE):
        X = np.zeros((1, len(FEATURES)))
        t0 = time.perf_counter()
        _ = risk_model.predict_proba(X)
        risk_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Đo end-to-end (detect + risk + blur)
    e2e_latencies: list[float] = []
    for _ in range(MEASURE):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, device=det_device, half=det_half, verbose=False)
        X = np.zeros((1, len(FEATURES)))
        _ = risk_model.predict_proba(X)
        cv2.GaussianBlur(frame, (51, 51), 0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e_latencies.append((time.perf_counter() - t0) * 1000.0)

    # RAM và CPU
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=1)

    rows: list[dict[str, Any]] = []
    for name, arr in [
        ("detector", det_latencies),
        ("detect_blur", db_latencies),
        ("xgboost", risk_latencies),
        ("end_to_end", e2e_latencies),
    ]:
        arr_np = np.array(arr)
        rows.append({
            "component": name,
            "mean_ms": float(np.mean(arr_np)),
            "p50_ms": float(np.percentile(arr_np, 50)),
            "p95_ms": float(np.percentile(arr_np, 95)),
            "p99_ms": float(np.percentile(arr_np, 99)),
            "max_ms": float(np.max(arr_np)),
            "fps": 1000.0 / float(np.mean(arr_np)) if np.mean(arr_np) > 0 else 0.0,
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
    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[Bước 6] Đã lưu hiệu năng phần cứng: {OUT_CSV}")
    print(f"[Bước 6] Detector mean: {rows[0]['mean_ms']:.2f} ms, FPS: {rows[0]['fps']:.2f}")
    print(f"[Bước 6] Detect+blur mean: {rows[1]['mean_ms']:.2f} ms, FPS: {rows[1]['fps']:.2f}")
    print(f"[Bước 6] End-to-end mean: {rows[3]['mean_ms']:.2f} ms, FPS: {rows[3]['fps']:.2f}")
    print(f"[Bước 6] RAM: {ram_mb:.1f} MB, CPU: {cpu_percent:.1f}%")


if __name__ == "__main__":
    main()