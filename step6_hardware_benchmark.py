"""Bước 6: Đo hiệu năng phần cứng khi nạp mô hình vào thiết bị.

Input:
- final_extracted/best.pt (mô hình detect)
- final_extracted/xgboost_risk.json (mô hình XGBoost)

Output:
- results/hardware_benchmark.csv (latency, FPS, RAM, CPU/GPU)

Latency end-to-end bây giờ bao gồm: detect YOLO -> tính 25 đặc trưng rủi ro
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

from perf_utils import (
    DEFAULT_BLUR_KERNEL,
    configure_threads,
    resolve_device,
    resolve_half,
    roi_gaussian_blur,
    runtime_info,
    warn_if_cuda_missing,
)

WEIGHTS_PATH = Path("final_extracted/artifacts/yolo26s.pt")
RISK_MODEL_PATH = Path("final_extracted/artifacts/risk_model.json")
OUT_CSV = Path("results/hardware_benchmark.csv")

IMG_SIZE = 1024
# Đọc cùng config với Bước 5 để benchmark phần cứng dùng đúng operating point
# đã huấn luyện (notebook: imgsz=1024, conf/iou từ track_config.json).
_EVAL_CFG_PATH = Path("config/eval_config.json")
_EVAL_CFG: dict = {}
if _EVAL_CFG_PATH.exists():
    with _EVAL_CFG_PATH.open("r", encoding="utf-8") as _f:
        _EVAL_CFG = json.load(_f)
IMG_SIZE = int(_EVAL_CFG.get("img_size", IMG_SIZE))
CONF_THRES = float(_EVAL_CFG.get("conf_thres", 0.15))
IOU_THRES = float(_EVAL_CFG.get("iou_thres", 0.5))
MAX_DET = int(_EVAL_CFG.get("max_det", 1000))
WARMUP = 10
MEASURE = 100
# Deadline xử lý một frame trên thiết bị biên (theo README). 40 ms ~ 25 FPS.
EDGE_DEADLINE_MS = float(os.getenv("NCKH_EDGE_DEADLINE_MS", "40.0"))
BLUR_KERNEL = int(os.getenv("NCKH_BLUR_KERNEL", str(DEFAULT_BLUR_KERNEL)))

DEVICE = resolve_device()
configure_threads()
print(f"[Bước 6] Sử dụng thiết bị: {DEVICE} (CUDA: {torch.cuda.is_available()})")
if not torch.cuda.is_available():
    warn_if_cuda_missing("[Bước 6]")
print(
    f"[Bước 6] Blur: ROI (kernel {BLUR_KERNEL}) | cv_threads: {cv2.getNumThreads()} | "
    f"torch_threads: {torch.get_num_threads()}"
)

# ===== HỢP ĐỒNG ĐẶC TRƯNG (schema v2) =====
# 25 đặc trưng khớp với notebook train + step5 runtime.
FEATURES = [
    "conf", "conf_std5", "conf_delta", "conf_mean5", "conf_slope5",
    "lap_var_log", "brightness", "motion_blur_ratio", "crop_contrast",
    "track_age", "hit_ratio", "time_since_update",
    "speed_norm", "pred_uncert", "iou_prev", "scale_change",
    "min_side_px", "area_norm", "height_norm", "center_y_norm", "dist_edge",
    "nearest_dist_norm", "n_neighbors_r2_norm", "overlap_max",
    "n_det_expected_ratio",
]
# Model 21-feature cũ vẫn benchmark được (đặc trưng chẩn đoán vẫn được tính).
LEGACY_FEATURES = [
    "conf", "conf_std5", "conf_delta", "track_age", "hit_ratio",
    "time_since_update", "track_state", "min_side_px", "aspect_dev",
    "dist_edge", "is_trunc", "speed_norm", "pred_uncert", "lap_var_log",
    "brightness", "motion_blur_ratio", "nearest_dist_norm",
    "n_neighbors_r2_norm", "overlap_max", "frame_det_delta",
    "n_det_expected_ratio",
]


def resolve_feature_names(model: xgb.XGBClassifier) -> list[str]:
    """Chọn danh sách đặc trưng theo số chiều model (25 mới / 21 cũ)."""
    n_in = getattr(model, "n_features_in_", None)
    if n_in == len(LEGACY_FEATURES):
        return list(LEGACY_FEATURES)
    return list(FEATURES)


TRUNC_MARGIN_PX = 2
CONF_WINDOW = 5
PRED_UNCERT_WINDOW = 5
NEIGHBOR_RADIUS_FRAC = 0.05
NEIGHBOR_K = 5
CROP_RESIZE = 224


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


def compute_risk_features(
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
    """Tính 25 đặc trưng rủi ro cho một track đơn (bản rút gọn để benchmark)."""
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
    f_conf_mean5 = float(np.mean(c5)) if c5 else float(conf)
    f_conf_slope5 = 0.0
    if len(c5) >= 2:
        x = np.arange(len(c5), dtype=np.float64)
        xm = float(x.mean())
        denom = float(((x - xm) ** 2).sum())
        if denom > 0:
            y = np.asarray(c5, dtype=np.float64)
            f_conf_slope5 = float(((x - xm) * (y - y.mean())).sum() / denom)
    f_track_age = float(track_age)
    f_hit_ratio = float(hit_ratio)
    f_time_since_update = float(time_since_update)
    f_track_state = float(track_state)
    f_min_side_px = float(min(bw, bh))
    f_aspect_dev = float(abs(bw / bh - r_ref))
    f_area_norm = float((bw * bh) / max(w * h, 1.0))
    f_height_norm = float(bh / max(h, 1.0))
    f_center_y_norm = float(cy / max(h, 1.0))
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
    f_iou_prev = float(iou(box, prev_box)) if prev_box is not None else 0.0
    f_scale_change = 0.0
    if prev_box is not None:
        prev_area = max(prev_box[2] - prev_box[0], 1e-6) * max(prev_box[3] - prev_box[1], 1e-6)
        f_scale_change = float(np.clip(math.log(max(bw * bh, 1e-6) / prev_area), -5.0, 5.0))
    f_lap_var_log = float(math.log1p(lap_history[-1])) if lap_history else 0.0

    # Thống kê crop giống lúc train: cắt theo box track rồi resize về CROP_RESIZE.
    x1 = max(0, min(int(box[0]), w - 1)); y1 = max(0, min(int(box[1]), h - 1))
    x2 = max(x1 + 1, min(int(box[2]), w)); y2 = max(y1 + 1, min(int(box[3]), h))
    crop = gray[y1:y2, x1:x2]
    if crop.size and max(crop.shape[:2]) > CROP_RESIZE:
        s = CROP_RESIZE / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
    f_brightness = float(crop.mean() / 255.0) if crop.size else 0.5
    f_crop_contrast = float(crop.std() / 255.0) if crop.size else 0.0
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
        # chẩn đoán (model 21-feature cũ)
        "track_state": f_track_state,
        "aspect_dev": f_aspect_dev,
        "is_trunc": f_is_trunc,
        "frame_det_delta": f_frame_det_delta,
    }


def pick_primary_box(detections_np: np.ndarray, w: int, h: int) -> list[float]:
    """Chọn box detector lớn nhất làm track chính để tính đặc trưng."""
    if len(detections_np) == 0:
        return [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
    areas = (detections_np[:, 2] - detections_np[:, 0]) * (detections_np[:, 3] - detections_np[:, 1])
    idx = int(np.argmax(areas))
    return [float(v) for v in detections_np[idx, :4]]


def tile_area_ratio(boxes: list[list[float]], w: int, h: int, ksize: int) -> float:
    """Tỷ lệ diện tích frame mà blur ROI thực sự xử lý (box + margin kernel/2).

    Đây là đại lượng quyết định chi phí blur ROI; box chồng nhau không trừ
    (tỷ lệ là ước lượng, không dùng để tính metric).
    """
    half = ksize // 2
    total = 0
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        total += (min(w, x2 + half) - max(0, x1 - half)) * (min(h, y2 + half) - max(0, y1 - half))
    return min(1.0, total / float(w * h)) if w > 0 and h > 0 else 0.0


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {WEIGHTS_PATH}")
    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {RISK_MODEL_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    model.to(DEVICE)
    det_device = DEVICE
    det_half = resolve_half("pytorch", det_device)
    print(
        f"[Bước 6] Detector backend: pytorch | device: {det_device} | "
        f"gpu: {runtime_info()['gpu_name']} | half: {det_half} | imgsz: {IMG_SIZE} | "
        f"conf: {CONF_THRES} | iou: {IOU_THRES} | max_det: {MAX_DET}"
    )
    risk_model = xgb.XGBClassifier()
    risk_model.load_model(str(RISK_MODEL_PATH))
    feature_names = resolve_feature_names(risk_model)
    print(f"[Bước 6] Risk model dùng {len(feature_names)} đặc trưng (schema v2).")

    process = psutil.Process()
    real_frame = Path(
        os.getenv(
            "NCKH_BENCH_FRAME",
            "VisDrone2019-VID-test-dev/sequences/uav0000009_03358_v/0000001.jpg",
        )
    )
    if real_frame.exists():
        frame = cv2.imread(str(real_frame))
        print(f"[Bước 6] Dùng frame thật: {real_frame} shape={frame.shape}")
        bench_frame_path = str(real_frame)
    else:
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        print("[Bước 6] Cảnh báo: không tìm thấy frame thật, dùng frame ngẫu nhiên thay thế.")
        bench_frame_path = "(frame ngẫu nhiên)"

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def run_detect():
        return model(
            frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES,
            device=det_device, max_det=MAX_DET, verbose=False,
            **({"half": True} if det_half else {}),
        )

    def det_boxes(res) -> np.ndarray:
        cpu_boxes = res[0].cpu().boxes
        return cpu_boxes.xyxy.numpy() if len(cpu_boxes.xyxy) else np.empty((0, 4))

    # Frame mặc định có thể không có người -> blur ROI gần như bằng 0 và phép đo
    # không đại diện. Nếu vậy thử vài frame rải rác trong sequence và chọn frame
    # đầu tiên có detection. Có thể ép frame bằng NCKH_BENCH_FRAME.
    if real_frame.exists() and os.getenv("NCKH_BENCH_FRAME") is None and len(det_boxes(run_detect())) == 0:
        files = sorted(real_frame.parent.glob("*.jpg"))
        stride = max(1, len(files) // 12)
        for cand in files[::stride][:12]:
            if cand.name == real_frame.name:
                continue
            alt = cv2.imread(str(cand))
            if alt is None:
                continue
            frame = alt
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(det_boxes(run_detect())) > 0:
                bench_frame_path = str(cand)
                print(
                    f"[Bước 6] Frame mặc định không có detection, dùng {cand} "
                    f"shape={frame.shape}"
                )
                break
        else:
            print("[Bước 6] Cảnh báo: không tìm được frame có detection, "
                  "giữ frame hiện tại; chi phí blur sẽ thấp hơn thực tế.")
    print(f"[Bước 6] Frame benchmark: {bench_frame_path} shape={frame.shape}")

    # Chuẩn bị lịch sử track giả lập (5 frame trước) để tính đặc trưng đại diện.
    primary_box = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
    prev_box = [w * 0.24, h * 0.24, w * 0.76, h * 0.76]
    lap_history = [lap_var_of_crop(frame, primary_box, w, h)] * 5
    conf_history = [0.70] * 5
    center_history = [((primary_box[0] + primary_box[2]) / 2.0, (primary_box[1] + primary_box[3]) / 2.0)] * 5
    det_count_prev = 1
    ewma_det = 1.0
    r_ref = 0.4

    def run_features(dets):
        box = pick_primary_box(dets, w, h) if len(dets) else primary_box
        return compute_risk_features(
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
        X = np.array([feat[f] for f in feature_names], dtype=np.float32).reshape(1, -1)
        return risk_model.predict_proba(X)

    def boxes_for_blur(dets: np.ndarray) -> list[list[float]]:
        """Box cần che: box detector; nếu không có thì dùng box trung tâm."""
        if len(dets):
            return [[float(v) for v in b[:4]] for b in dets]
        return [list(primary_box)]

    def run_blur_roi(boxes: list[list[float]]):
        # Đúng như step5: GaussianBlur từng ROI với cùng kernel, không blur toàn frame.
        roi_gaussian_blur(frame, boxes, ksize=BLUR_KERNEL)

    def run_blur_full():
        # Cách cũ, chỉ đo để đối chứng chi phí.
        cv2.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)

    # Box mẫu để đo riêng chi phí blur (không lẫn thời gian detect).
    sample_dets = det_boxes(run_detect())
    sample_boxes = boxes_for_blur(sample_dets)
    sample_area_ratio = tile_area_ratio(sample_boxes, w, h, BLUR_KERNEL)
    sample_box_count = len(sample_dets)

    # Warm-up
    for _ in range(WARMUP):
        res = run_detect()
        dets = det_boxes(res)
        feat = run_features(dets)
        run_xgb(feat)
        run_blur_roi(boxes_for_blur(dets))
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

    def measure_blur_full() -> list[float]:
        lat = []
        for _ in range(MEASURE):
            t0 = time.perf_counter()
            run_blur_full()
            lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    def measure_blur_roi() -> list[float]:
        lat = []
        for _ in range(MEASURE):
            t0 = time.perf_counter()
            run_blur_roi(sample_boxes)
            lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    def measure_detect_blur() -> list[float]:
        lat = []
        for _ in range(MEASURE):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = run_detect()
            run_blur_roi(boxes_for_blur(det_boxes(res)))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    def measure_xgb() -> list[float]:
        # Tính đặc trưng một lần (bên ngoài đo) để chỉ đo chi phí XGBoost.
        res = run_detect()
        feat = run_features(det_boxes(res))
        X = np.array([feat[f] for f in feature_names], dtype=np.float32).reshape(1, -1)
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
            dets = det_boxes(res)
            feat = run_features(dets)
            run_xgb(feat)
            run_blur_roi(boxes_for_blur(dets))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    det_latencies = measure_detector()
    blur_full_latencies = measure_blur_full()
    blur_roi_latencies = measure_blur_roi()
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
    s_blur_full = stats(blur_full_latencies)
    s_blur_roi = stats(blur_roi_latencies)
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

    info = runtime_info()
    rows: list[dict[str, Any]] = []
    for name, s in [
        ("detector", s_det),
        ("blur_full", s_blur_full),
        ("blur_roi", s_blur_roi),
        ("detect_blur", s_db),
        ("xgboost", s_xgb),
        ("end_to_end", s_e2e),
    ]:
        od = od_det if name == "detector" else od_e2e
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
            "pct_over_deadline": od["pct_over_deadline"],
            "over_deadline_count": od["over_deadline_count"],
            "device": det_device,
            "gpu_name": info["gpu_name"],
            "backend": "pytorch",
            "half": det_half,
            "blur_mode": "roi" if name in {"blur_roi", "detect_blur", "end_to_end"} else (
                "full" if name == "blur_full" else "none"
            ),
            "blur_kernel": BLUR_KERNEL,
            "torch_version": info["torch_version"],
            "torch_cuda": info["torch_cuda"],
            "cv_threads": info["cv_threads"],
            "torch_threads": info["torch_threads"],
            "blur_area_ratio": sample_area_ratio if name in {"blur_roi", "detect_blur", "end_to_end"} else np.nan,
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
        "device": det_device,
        "gpu_name": info["gpu_name"],
        "backend": "pytorch",
        "half": det_half,
        "blur_mode": "roi",
        "blur_kernel": BLUR_KERNEL,
        "torch_version": info["torch_version"],
        "torch_cuda": info["torch_cuda"],
        "cv_threads": info["cv_threads"],
        "torch_threads": info["torch_threads"],
        "blur_area_ratio": np.nan,
    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[Bước 6] Đã lưu hiệu năng phần cứng: {OUT_CSV}")
    print(f"[Bước 6] Detector mean: {s_det['mean_ms']:.2f} ms, FPS: {s_det['fps']:.2f}")
    print(f"[Bước 6] Blur full-frame mean: {s_blur_full['mean_ms']:.2f} ms")
    print(f"[Bước 6] Blur ROI mean: {s_blur_roi['mean_ms']:.2f} ms "
          f"({sample_box_count} box, {sample_area_ratio * 100:.1f}% diện tích frame)")
    print(f"[Bước 6] Detect+blur mean: {s_db['mean_ms']:.2f} ms, FPS: {s_db['fps']:.2f}")
    print(f"[Bước 6] XGBoost mean: {s_xgb['mean_ms']:.4f} ms, FPS: {s_xgb['fps']:.2f}")
    print(f"[Bước 6] End-to-end mean: {s_e2e['mean_ms']:.2f} ms, FPS: {s_e2e['fps']:.2f}")
    print(f"[Bước 6] Deadline edge: {EDGE_DEADLINE_MS} ms | end-to-end quá deadline: {od_e2e['pct_over_deadline']:.1f}%")
    print(f"[Bước 6] RAM: {ram_mb:.1f} MB, CPU: {cpu_percent:.1f}%")


if __name__ == "__main__":
    main()
