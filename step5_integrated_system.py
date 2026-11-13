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

import json
import logging
import math
import os
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Tắt cảnh báo spam từ ultralytics/YOLO (không ảnh hưởng kết quả)
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
warnings.filterwarnings("ignore", message=".*Use 'quantize' instead.*")

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics.trackers.byte_tracker import BYTETracker

from ultralytics import YOLO

# Gốc dự án + tiện ích đường dẫn (thay cho core.config đã xóa).
ROOT = Path(__file__).resolve().parent


def project_path(value: str | Path) -> Path:
    """Trả đường dẫn tuyệt đối: đường dẫn tương đối tính từ ROOT."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


# Ultralytics phát cảnh báo qua `logging` (không phải `warnings`) nên phải hạ
# level logger.
logging.getLogger("ultralytics").setLevel(
    os.getenv("NCKH_ULTRALYTICS_LOG", "ERROR").upper()
)

from metrics_unified import (
    iou,
    iou_matrix,
    containment_matrix,
    average_precision_coco,
    privacy_utility_metrics,
    gate_quality_metrics,
)
from perf_utils import (
    AsyncVideoWriter,
    FastOpenVINODetector,
    FramePrefetcher,
    StageProfiler,
    configure_threads,
    parse_tiles,
    resolve_device,
    roi_gaussian_blur,
    run_tiled_detect,
    runtime_info,
    warn_if_cuda_missing,
)

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# Mọi đường dẫn tương đối được neo vào ROOT (không phụ thuộc CWD).


# Một chế độ chạy duy nhất: scheduler thích ứng + output gate luôn bật.
# Không còn NCKH_TRACK_EVAL / NCKH_ORACLE_GATE: tracking chấm trên toàn frame,
# AP chấm trên frame có gọi detect (kèm AP_tracked toàn frame để đối chứng).
RISK_CALIBRATED_PATH = project_path(
    "final_extracted/artifacts/risk_model_calibrated.joblib"
)
# Split duy nhất của toàn hệ thống là val (sequence-disjoint, dùng cho cả hiệu
# chuẩn gate và báo cáo). Giá trị thật đọc từ CONFIG["data_root"] trong run();
# đây chỉ là default khi thiếu cấu hình.
DATA_ROOT = project_path(os.getenv("NCKH_DATA_ROOT", "visdrone_val/VisDrone2019-VID-val"))
SEQUENCES_DIR = DATA_ROOT / "sequences"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
OUT_VIDEO = project_path(os.getenv("NCKH_OUT_VIDEO", "outputs/protected_video.mp4"))
# Một bộ output duy nhất cho cả hệ thống (không còn hậu tố _eval/_unified).
# Giá trị thật đọc từ CONFIG trong run().
OUT_GATE_LOG = project_path(
    os.getenv("NCKH_OUT_GATE_LOG", "results/output_gate_log.json")
)
OUT_METRICS = project_path(
    os.getenv("NCKH_OUT_METRICS", "results/system_metrics.csv")
)
OUT_FRAME_RECORDS = project_path(
    os.getenv("NCKH_OUT_FRAME_RECORDS", "results/val_frame_records.json")
)
# Nguồn cấu hình duy nhất của dự án (CONFIG ở trên).
RISK_DATASET_PATH = project_path("final_extracted/artifacts/risk_dataset.csv")

# ===== HỢP ĐỒNG ĐẶC TRƯNG (schema v2) =====
# 25 đặc trưng này phải khớp 1:1 với notebook4234ae1bfe.ipynb (nguồn cấu hình
# duy nhất của dự án) ở cả thứ tự lẫn cách tính.
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
    - Model thô (không có `features`) thì dựa vào `n_features_in_` (25 mới / 21 cũ).
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

# Giá trị mặc định; sẽ được override bởi CONFIG trong run()
# để đảm bảo đồng nhất chỉ số giữa hệ thống và notebook huấn luyện Kaggle.
# Các giá trị dưới đây lấy đúng theo notebook4234ae1bfe.ipynb (CELL 2/13 CONFIG):
#   - IMGSZ = 1024, MAX_DET = 1000
#   - TRACK_CONF / TRACK_IOU = 0.05 / 0.5
#   - MOT_IOU_TH = 0.5 (ghép detection/tracking với GT)
#   - PERSON_CATS = (1, 2), không lọc occlusion
#   - ByteTrack: high/low/new = 0.35/0.05/0.25, match = 0.70, buffer = 45
IMG_SIZE = 1024
# Video demo: bật mặc định (outputs/protected_video.mp4). Tắt bằng
# NCKH_WRITE_VIDEO=0 hoặc "output.write_video": false trong system_config.json.
# nếu môi trường không có encoder tương thích.
WRITE_VIDEO = os.getenv("NCKH_WRITE_VIDEO", "1") == "1"
CONF_THRES = 0.05
IOU_THRES = 0.5
MATCH_IOU_THRESHOLD = 0.5
COVERAGE_IOU_THRESHOLD = 0.5
CONTAINMENT_THRESHOLD = 0.9
TRACK_HIGH_THRESH = 0.35
TRACK_LOW_THRESH = 0.05
TRACK_NEW_THRESH = 0.25
TRACK_MATCH_THRESH = 0.70
TRACK_BUFFER_FRAMES = 45
# Camera motion compensation (CMC) cho ByteTrack. UAV di chuyển nên đây là đòn
# bẩy tiềm năng cho IDF1/HOTA, nhưng ablation trên val cho thấy GMC làm giảm
# mọi chỉ số tracking -> tắt trong config ("tracker.gmc": false).
# Ultralytics GMC hỗ trợ: sparseOptFlow (rẻ nhất), orb, sift, ecc.
GMC_ENABLE = False
GMC_METHOD = "sparseOptFlow"
GMC_DOWNSCALE = 2
# Xuất cả track chưa xác nhận (xem run_sequence). Mặc định tắt = ByteTrack chuẩn.
TRACK_ALL = False
MAX_DET = 1000

# ===== Protocol nhãn GT (đồng bộ notebook4234ae1bfe) =====
# Notebook dùng PERSON_CATS = (1, 2) (1=pedestrian, 2=people) và KHÔNG lọc
# cột occlusion. Giá trị thật được nạp từ CONFIG trong run();
# các biến này chỉ là default khi thiếu cấu hình.
PERSON_CATEGORIES: tuple[int, ...] = (1, 2)
EXCLUDE_OCCLUSION_2 = False
# Video demo: ghi khung đã che + nhãn BLOCKED/RELEASED (không ảnh hưởng metric).
DEMO_OVERLAY = os.getenv("NCKH_DEMO_OVERLAY", "1") == "1"
PROTECT_PADDING = 0.15
# Chạy nhanh trên subset bằng NCKH_MAX_SEQS / NCKH_MAX_FRAMES (không đổi protocol).
# OpenVINO model do Ultralytics export (thư mục *_openvino_model) là detector duy
# nhất của dự án. Đường dẫn thật lấy từ CONFIG["openvino_model"].
# Tên thư mục phải kết thúc bằng `_openvino_model` để Ultralytics nhận đúng.
OPENVINO_MODEL_DIR = project_path(
    "final_extracted/artifacts/yolo26s_1024_openvino_model"
)
# Chia frame thành lưới tile trước khi đưa vào detector. Probe trên val cho
# thấy tiling không đáng so với protocol gốc (chi phí 1,7x, precision giảm),
# nên mặc định "1x1" = nguyên frame, đúng notebook4234ae1bfe. Giá trị thật lấy
# từ CONFIG["det_tiles"].
DET_TILES = "1x1"
TILE_OVERLAP = 0.15
MAX_SEQS = int(os.getenv("NCKH_MAX_SEQS", "0"))  # 0 = tất cả sequence
MAX_FRAMES_PER_SEQ = int(os.getenv("NCKH_MAX_FRAMES", "0"))  # 0 = tất cả frame
WARMUP_FRAMES = 30
PROGRESS_EVERY = int(os.getenv("NCKH_PROGRESS_EVERY", "25"))  # 0 = tắt log tiến độ

CONF_WINDOW = 5
PRED_UNCERT_WINDOW = 5
EWMA_ALPHA = 0.3
TRUNC_MARGIN_PX = 2
NEIGHBOR_RADIUS_FRAC = 0.05
NEIGHBOR_K = 5
CROP_RESIZE = 224
SSIM_INTERVAL = 5

# ===== Tăng tốc vòng lặp frame (không đổi bất kỳ metric nào) =====
# NCKH_DECODE_THREADS=0  -> đọc JPEG tuần tự (như cũ)
# NCKH_SSIM_THREADS=0    -> tính SSIM ngay trong luồng chính (như cũ)
# NCKH_ENCODE_QUEUE=0    -> ghi video đồng bộ (như cũ)
# NCKH_PROFILE=1         -> in bảng thời gian từng stage ở cuối run
# NCKH_STRICT_METRICS=1  -> tính PSNR/SSIM bằng cả cách cũ để đối chiếu
DECODE_THREADS = int(os.getenv("NCKH_DECODE_THREADS", "3"))
DECODE_MAX_PENDING = int(os.getenv("NCKH_DECODE_PENDING", "16"))
SSIM_WORKERS = int(os.getenv("NCKH_SSIM_THREADS", "2"))
SSIM_MAX_PENDING = int(os.getenv("NCKH_SSIM_PENDING", "8"))
ENCODE_QUEUE = int(os.getenv("NCKH_ENCODE_QUEUE", "8"))
PROFILE = os.getenv("NCKH_PROFILE", "0") == "1"
STRICT_METRICS = os.getenv("NCKH_STRICT_METRICS", "0") == "1"
# Chạy detect ở thread riêng để chồng lấn GPU với CPU (mặc định bật).
# NCKH_ASYNC_DETECT=0 để quay lại vòng lặp tuần tự (đối chứng FPS).
ASYNC_DETECT = os.getenv("NCKH_ASYNC_DETECT", "1").strip() != "0"
# Dùng đường detect OpenVINO tối giản thay cho predictor của Ultralytics
# (mặc định bật). NCKH_FAST_DETECT=0 để quay lại đường Ultralytics.
FAST_DETECT = os.getenv("NCKH_FAST_DETECT", "1").strip() != "0"
PROFILE_OUT = project_path(
    os.getenv("NCKH_PROFILE_OUT", "results/stage_profile.json")
)

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
# Kernel mặc định đồng bộ notebook4234ae1bfe (ANON_METHOD='blur', ANON_KSIZE=15).
# Giá trị thật được nạp từ CONFIG trong run().
BLUR_KERNEL = int(os.getenv("NCKH_BLUR_KERNEL", "15"))

DEVICE = resolve_device()
configure_threads()
print(f"[Bước 5] Sử dụng thiết bị: {DEVICE} (CUDA: {torch.cuda.is_available()})")
if not torch.cuda.is_available():
    warn_if_cuda_missing("[Bước 5]")
print(
    f"[Bước 5] Blur mode: {BLUR_MODE} | kernel: {BLUR_KERNEL} | "
    f"cv_threads: {cv2.getNumThreads()} | torch_threads: {torch.get_num_threads()}"
)
print(
    f"[Bước 5] Tăng tốc: decode_threads={DECODE_THREADS} | "
    f"ssim_threads={SSIM_WORKERS} | encode_queue={ENCODE_QUEUE} | "
    f"profile={PROFILE} | strict_metrics={STRICT_METRICS}"
)


GATE_THRESHOLD_CACHED: float | None = None

# ===== CẤU HÌNH DUY NHẤT CỦA DỰ ÁN =====
# Thay cho config/system_config.json + core/config.py đã xóa. Mọi script dùng
# chung dict này; đổi tham số tại đây là đổi cho toàn hệ thống.
CONFIG: dict[str, Any] = {
    "notebook_source": "notebook4234ae1bfe.ipynb",
    "data_root": "visdrone_val/VisDrone2019-VID-val",
    "person_categories": [1, 2],
    "exclude_occlusion_2": False,
    "backend": "openvino",
    "openvino_model": "final_extracted/artifacts/yolo26s_1024_openvino_model",
    # Thiết bị OpenVINO: "intel:CPU", "intel:GPU.0" (iGPU), "intel:GPU.1" (dGPU).
    # Tự fallback về CPU nếu thiết bị không có trên máy.
    "openvino_device": "intel:GPU.0",
    "img_size": 1024,
    "conf_thres": 0.05,
    "iou_thres": 0.5,
    "max_det": 1000,
    "det_tiles": "1x1",
    "tile_overlap": 0.15,
    "max_frames_per_seq": 0,
    "warmup_frames": 30,
    "track_high_thresh": 0.35,
    "track_low_thresh": 0.05,
    "new_track_thresh": 0.25,
    # Ablation val (nhóm 3): nới association (0.70 -> 0.80) và giữ track mất
    # lâu hơn (45 -> 60) cải thiện IDF1 0.4492 -> 0.5224, HOTA 0.3578 -> 0.3955,
    # IDSW 973 -> 486, coverage 0.7732 -> 0.7747; ULFR +0.36pp, identity leak
    # không đổi. FPS gần như giữ nguyên trong sai số đo (9.45 -> 9.05 fps_stream).
    "track_match_thresh": 0.80,
    "track_buffer": 60,
    "gmc": False,
    "gmc_method": "sparseOptFlow",
    "gmc_downscale": 2,
    "track_all": False,
    "blur_mode": "roi",
    "blur_kernel": 15,
    "coverage_iou_threshold": 0.5,
    "containment_threshold": 0.9,
    "tau": 0.18806509945750452,
    "gate_threshold": 0.2151402278702892,
    "gate_target_block_rate": 0.40,
    "gate_risk_scale": 0.5,
    "iou_match_threshold": 0.5,
    "out_metrics": "results/system_metrics.csv",
    "out_gate_log": "results/output_gate_log.json",
    "out_frame_records": "results/val_frame_records.json",
    "out_video": "outputs/protected_video.mp4",
    "write_video": True,
    "video_fps": 30.0,
    "video_width": 1280,
    "video_height": 720,
    "scheduler": {
        "risk_low": 0.25,
        "risk_high": 0.5,
        "risk_critical": 0.75,
        "detect_interval_low": 1,
        "detect_interval_medium": 1,
        "detect_interval_high": 1,
        "detect_interval_critical": 1,
        "roi_expand_low": 0.2,
        "roi_expand_medium": 0.3,
        "roi_expand_high": 0.5,
        "roi_expand_critical": 0.7,
        "hybrid_max_skip": 0,
        "hybrid_max_miss": 0,
        "hybrid_crowd_tracks": 0,
    },
    # ===== Output gate =====
    # `gate_threshold` được hiệu chuẩn trên val và ghi bởi
    # `python run_pipeline.py calibrate --apply`. Kết quả hiệu chuẩn đầy đủ nằm
    # trong `results/gate_calibration.json` (nguồn duy nhất), không hardcode
    # trong CONFIG để tránh số cũ bị trích dẫn nhầm.
    #
    # `gate_score` = leak_score + gate_risk_scale * frame_risk, trong đó
    # leak_score là proxy rò rỉ coverage chỉ dùng tín hiệu runtime (không GT):
    # tracker rơi track, detection không được xuất, detection conf thấp, đám đông.
    #
    # `no_track` mặc định là hằng số 0.5 (tắt làm mềm: `gate_no_track_ref` = 0).
    # Đã thử làm mềm theo số người kỳ vọng trên val: coverage gần như không đổi
    # nhưng block dồn hết vào sequence đông nhất và identity leak tăng (3 -> 9
    # identity trên val), nên giữ nguyên hành vi cũ.
    "gate_leak_weights": {
        "no_track": 0.50,   # n_tracks == 0 trong khi vẫn có bằng chứng về người
        "missing": 0.00,    # tỷ lệ người kỳ vọng không được track xuất ra
        "low_conf": 0.00,   # tỷ lệ detection conf thấp (0 trên VisDrone)
        "crowd": 0.10,      # quy mô đám đông (tie-breaker giữa frame cùng dropout)
    },
    # 0 = tắt làm mềm `no_track`. Giữ 0 vì thực nghiệm trên val cho thấy làm mềm
    # làm giảm identity leak protection (block dồn hết vào một sequence đông).
    "gate_no_track_ref": 0.0,
    "gate_crowd_ref": 168.0,
    "gate_low_conf_thresh": 0.25,
}



def load_tau() -> float:
    """Ngưỡng tau dự phòng (track-level) nằm trong CONFIG["tau"]."""
    return float(CONFIG.get("tau", 0.5))


def resolve_openvino_device(requested: str) -> str:
    """Chọn thiết bị OpenVINO có thật trên máy, fallback về CPU.

    `requested` nhận cả dạng `intel:GPU.0` (Ultralytics) lẫn `GPU.0`/`CPU`.
    Trả chuỗi mà Ultralytics OpenVINO backend hiểu (`intel:<device>`).
    """
    text = str(requested or "CPU").strip()
    suffix = text.split(":")[-1] if ":" in text else text
    suffix = suffix or "CPU"
    try:
        import openvino as ov

        available = list(ov.Core().available_devices)
    except Exception:
        available = ["CPU"]
    if suffix.upper() == "CPU" or suffix not in available:
        if suffix.upper() != "CPU":
            print(
                f"[Bước 5] Cảnh báo: thiết bị OpenVINO '{suffix}' không có "
                f"(available={available}); fallback CPU."
            )
        return "intel:CPU"
    return f"intel:{suffix}"


def load_gate_calibration() -> dict[str, Any]:
    """Đọc kết quả hiệu chuẩn gate từ `results/gate_calibration.json`.

    Đây là nguồn duy nhất của số hiệu chuẩn; CONFIG chỉ giữ `gate_threshold`
    đang dùng. Thiếu file thì trả {} (report/verify sẽ cảnh báo).
    """
    path = project_path(
        str(CONFIG.get("gate_calibration", "results/gate_calibration.json"))
    )
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Bước 5] Cảnh báo: không đọc được {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def config_fingerprint() -> dict[str, Any]:
    """Hash CONFIG + git SHA để report/verify phát hiện số liệu cũ."""
    import hashlib

    payload = json.dumps(CONFIG, sort_keys=True, ensure_ascii=False, default=str)
    return {
        "config_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "git_sha": _git_sha(),
    }


def _git_sha() -> str | None:
    """Đọc git SHA hiện tại mà không gọi subprocess (đọc .git trực tiếp)."""
    try:
        git = ROOT / ".git"
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head[5:].strip()
            ref_path = git / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
            packed = git / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0][:12]
            return None
        return head[:12] or None
    except Exception:
        return None


def resolve_person_categories(cfg: dict[str, Any]) -> tuple[int, ...]:
    """Trả về tập category VisDrone được coi là person.

    Ưu tiên khóa mới `person_categories` (notebook4234ae1bfe: `[1, 2]`).
    Nếu config cũ chỉ có `category_person` thì fallback về `(category_person,)`
    để không phá vỡ các file config đã có.
    """
    raw = cfg.get("person_categories")
    if raw is None:
        legacy = cfg.get("category_person")
        raw = [legacy] if legacy is not None else list(PERSON_CATEGORIES)
    if isinstance(raw, (int, str)):
        raw = [raw]
    cats = tuple(int(c) for c in raw)
    if not cats:
        raise ValueError("person_categories rỗng: cần ít nhất một category VisDrone")
    return cats


def load_risk_model() -> Any:
    """Tải mô hình risk đã calibration (artifact duy nhất của dự án).

    `final_extracted/artifacts/risk_model_calibrated.joblib` là model duy nhất;
    không còn fallback sang `risk_model.json` đã xóa.
    """
    if not RISK_CALIBRATED_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình risk đã calibration: {RISK_CALIBRATED_PATH}"
        )
    if not _HAS_JOBLIB:
        raise RuntimeError(
            "Cần joblib để đọc risk_model_calibrated.joblib (xem requirements.txt)."
        )
    cal = joblib.load(str(RISK_CALIBRATED_PATH))
    if isinstance(cal, dict) and "model" in cal and "calibrator" in cal:
        print(
            f"[Bước 5] Đã tải mô hình đã calibrate (ECE raw={cal.get('ece_raw', float('nan')):.4f} "
            f"-> cal={cal.get('ece_calibrated', float('nan')):.4f}): {RISK_CALIBRATED_PATH}"
        )
    else:
        print(f"[Bước 5] Đã tải mô hình đã calibration: {RISK_CALIBRATED_PATH}")
    return cal


class _DetBoxes:
    """Shim tối thiểu tương thích `ultralytics.engine.results.Boxes`.

    ByteTracker chỉ cần .xyxy/.conf/.cls/.xywh và __len__/__getitem__, nên có thể
    dùng chung cho OpenVINO và đường tiling thay vì tạo class mới
    trong mỗi frame (bản cũ định nghĩa lại class trong vòng lặp).

    Cố ý **không** khai báo `.xywhr`: Ultralytics ByteTrack coi sự tồn tại của
    `xywhr` là box xoay (OBB) và sẽ hoán đổi w/h rồi trả kết quả 9 cột
    `[x, y, w, h, angle, id, score, cls, idx]`, làm hỏng cả toạ độ lẫn track ID.
    """

    __slots__ = ("xyxy", "conf", "cls", "xywh")

    def __init__(self, det: np.ndarray):
        det = np.asarray(det, dtype=np.float32)
        det = det.reshape(-1, 6) if det.size else np.empty((0, 6), dtype=np.float32)
        if len(det) > 0:
            self.xyxy = det[:, :4].astype(np.float32)
            self.conf = det[:, 4].astype(np.float32)
            self.cls = det[:, 5].astype(np.float32)
            x1, y1, x2, y2 = self.xyxy.T
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = x2 - x1, y2 - y1
            self.xywh = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)
        else:
            self.xyxy = np.empty((0, 4), dtype=np.float32)
            self.conf = np.empty((0,), dtype=np.float32)
            self.cls = np.empty((0,), dtype=np.float32)
            self.xywh = np.empty((0, 4), dtype=np.float32)

    def cpu(self) -> "_DetBoxes":
        return self

    def numpy(self) -> "_DetBoxes":
        return self

    @property
    def boxes(self) -> "_DetBoxes":
        return self

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, inds: Any) -> "_DetBoxes":
        if isinstance(inds, np.ndarray) and inds.dtype == bool:
            inds = np.flatnonzero(inds)
        xyxy = self.xyxy[inds].reshape(-1, 4)
        conf = self.conf[inds].reshape(-1)
        cls = self.cls[inds].reshape(-1)
        return _DetBoxes(np.column_stack([xyxy, conf, cls]))


class _DetResults:
    """Giả lập `list[Results]` để dùng lại nhánh xử lý của Ultralytics."""

    __slots__ = ("boxes",)

    def __init__(self, det: np.ndarray):
        self.boxes = _DetBoxes(det)

    def __getitem__(self, _idx: int) -> "_DetBoxes":
        return self.boxes


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
            score = int(parts[6])
            category = int(parts[7])
            occlusion = int(parts[9])
            # Protocol đồng bộ notebook4234ae1bfe: bỏ box score 0 / kích thước
            # không hợp lệ, giữ cat 1+2, không lọc occlusion.
            if score == 0 or w <= 0 or h <= 0:
                continue
            if category not in PERSON_CATEGORIES:
                continue
            if EXCLUDE_OCCLUSION_2 and occlusion == 2:
                continue
            gt.setdefault(frame, []).append([x, y, x + w, y + h])
            gt_ids.setdefault(frame, []).append(target_id)
    return gt, gt_ids


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
        # Cách cũ: blur toàn frame rồi chỉ lấy pixel trong mask (chi phí cao hơn
        # ROI). Giữ lại để đối chứng PSNR với cùng BLUR_KERNEL.
        blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)
        out = frame.copy()
        # copyTo nhanh hơn fancy indexing khoảng 30x trên mask lớn.
        mask_u8 = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
        cv2.copyTo(blurred, mask_u8, out)
        return out
    # Mặc định: blur từng ROI, cùng kernel, cùng vùng ghi đè.
    return roi_gaussian_blur(frame, boxes, ksize=BLUR_KERNEL, mask=mask)


def _mask_as_bool(mask: np.ndarray | None) -> np.ndarray | None:
    """Xem mask uint8/bool 0-1 như bool mà không copy (giống perf_utils)."""
    if mask is None:
        return None
    if mask.dtype == np.bool_:
        return mask
    if mask.dtype == np.uint8:
        return mask.view(np.bool_)
    return mask.astype(bool)


def psnr_components(
    img1: np.ndarray, img2: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    """Tính (PSNR toàn frame, PSNR nền, PSNR vùng che) bằng **một** phép SSD.

    Tăng tốc: `cv2.norm(..., NORM_L2SQR)` cộng bình phương sai khác trong C++
    (SIMD). Đo trên frame 1360x765: **0,2-0,45 ms** thay vì 39 ms của bản numpy
    float32. Giá trị giống hệt vì hiệu hai ảnh uint8 là số nguyên <= 255, bình
    phương <= 65025 biểu diễn chính xác trong float64.

    Vì `blur_boxes` chỉ ghi trong `mask` (xem `perf_utils.roi_gaussian_blur`),
    `img1` và `img2` giống nhau tuyệt đối ở ngoài mask, nên:
      - SSD nền = 0                          -> PSNR nền = 100 dB
      - SSD toàn frame = SSD vùng che        -> không phải quét lại toàn ảnh

    Quy ước giữ nguyên như bản cũ:
    - PSNR toàn frame = +inf khi ảnh giống hệt (caller bỏ qua bằng isfinite).
    - PSNR trên một vùng = 100 dB khi vùng đó giống hệt.
    - PSNR nền = NaN khi mask phủ toàn ảnh (không còn pixel nền).

    `NCKH_STRICT_METRICS=1`: tính lại bằng bản numpy cũ, assert hai bên bằng
    nhau tuyệt đối rồi dùng giá trị bản cũ (chế độ đối chứng, chậm hơn).
    """
    h, w = img1.shape[:2]
    channels = img1.shape[2] if img1.ndim == 3 else 1
    total_values = int(h) * int(w) * int(channels)

    mask_bool = _mask_as_bool(mask)
    if mask_bool is None:
        mask_bool = np.zeros((h, w), dtype=bool)
    n_mask_px = int(np.count_nonzero(mask_bool))
    n_prot = n_mask_px * int(channels)
    n_bg = total_values - n_prot

    ssd_total = float(cv2.norm(img1, img2, cv2.NORM_L2SQR))

    if STRICT_METRICS:
        diff = img1.astype(np.float32) - img2
        squared = diff * diff
        ssd_total_ref = float(np.sum(squared, dtype=np.float64))
        ssd_prot_ref = (
            float(np.sum(squared[mask_bool], dtype=np.float64)) if n_mask_px else 0.0
        )
        if ssd_total_ref != ssd_total:
            raise AssertionError(
                f"cv2.norm lệch bản numpy: {ssd_total} != {ssd_total_ref}"
            )
        if ssd_total_ref != ssd_prot_ref:
            raise AssertionError(
                "blur ghi ra ngoài mask nên công thức rút gọn PSNR không còn đúng "
                f"(ssd_total={ssd_total_ref}, ssd_prot={ssd_prot_ref})"
            )
        ssd_total = ssd_total_ref

    if n_prot > 0:
        ssd_prot = ssd_total
        ssd_bg = 0.0
    else:
        # Không có vùng che: giữ đúng hành vi tổng quát của bản cũ.
        ssd_prot = 0.0
        ssd_bg = ssd_total

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


SSIM_KSIZE = 11
SSIM_SIGMA = 1.5
SSIM_RADIUS = SSIM_KSIZE // 2
SSIM_HALO = SSIM_RADIUS + 1


def _ssim_map(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """SSIM map trên 2 ảnh xám float64 (giữ nguyên công thức bản cũ)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ksize = (SSIM_KSIZE, SSIM_KSIZE)
    mu1 = cv2.GaussianBlur(gray1, ksize, SSIM_SIGMA)
    mu2 = cv2.GaussianBlur(gray2, ksize, SSIM_SIGMA)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(gray1 ** 2, ksize, SSIM_SIGMA) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 ** 2, ksize, SSIM_SIGMA) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, ksize, SSIM_SIGMA) - mu1_mu2
    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM trung bình toàn ảnh — bản tham chiếu, giữ nguyên hành vi cũ."""
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(
        _ssim_map(img1.astype(np.float64), img2.astype(np.float64)).mean()
    )


def compute_ssim_masked(
    img1: np.ndarray, img2: np.ndarray, mask: np.ndarray | None
) -> float:
    """SSIM trung bình toàn ảnh nhưng chỉ tính map trong vùng mask nở 5 px.

    Tại sao **chính xác**, không phải xấp xỉ: ngoài `dilate(mask, 11x11)`, cửa
    sổ Gaussian 11x11 của mọi pixel không chạm vào mask; mà mask là vùng duy
    nhất `img2` khác `img1`, nên ở đó `mu1 == mu2` và
    `sigma1 == sigma2 == sigma12`, kéo theo map = 1.0 đúng tuyệt đối. Vì vậy:

        mean toàn ảnh = (tổng map trong vùng + số pixel ngoài vùng) / (H*W)

    Crop thêm `SSIM_HALO` px mỗi phía để biên crop không ảnh hưởng map trong
    vùng; nếu crop chạm biên ảnh thì biên crop trùng biên ảnh nên cách xử lý
    border của OpenCV giống hệt bản tính toàn frame.

    Kết quả trùng `compute_ssim` tới sai số làm tròn float64 (~1e-16).
    """
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    h, w = gray1.shape[:2]
    n_px = int(h) * int(w)

    mask_bool = _mask_as_bool(mask)
    if mask_bool is None or not mask_bool.any():
        # Không có vùng che -> hai ảnh giống hệt -> mọi pixel SSIM = 1.
        return 1.0

    dilated = cv2.dilate(
        mask_bool.astype(np.uint8),
        np.ones((SSIM_KSIZE, SSIM_KSIZE), dtype=np.uint8),
    )
    rows = np.flatnonzero(dilated.any(axis=1))
    cols = np.flatnonzero(dilated.any(axis=0))
    y0 = max(0, int(rows[0]) - SSIM_HALO)
    y1 = min(h, int(rows[-1]) + 1 + SSIM_HALO)
    x0 = max(0, int(cols[0]) - SSIM_HALO)
    x1 = min(w, int(cols[-1]) + 1 + SSIM_HALO)

    ssim_map = _ssim_map(
        gray1[y0:y1, x0:x1].astype(np.float64),
        gray2[y0:y1, x0:x1].astype(np.float64),
    )
    inside = dilated[y0:y1, x0:x1].astype(bool)
    n_inside = int(np.count_nonzero(inside))
    total = float(np.sum(ssim_map[inside], dtype=np.float64))
    total += float(n_px - n_inside)
    return total / float(n_px)


def ssim_for_frame(
    frame: np.ndarray, out_frame: np.ndarray, mask: np.ndarray | None
) -> float:
    """SSIM dùng cho pipeline; strict mode đối chiếu với bản toàn frame."""
    value = compute_ssim_masked(frame, out_frame, mask)
    if STRICT_METRICS:
        ref = compute_ssim(frame, out_frame)
        if abs(value - ref) > 1e-9:
            raise AssertionError(
                f"SSIM chỉ-trong-vùng lệch bản toàn frame: {value!r} vs {ref!r}"
            )
    return value


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
                match_cost = np.where(
                    ious >= MATCH_IOU_THRESHOLD,
                    -1.0 - ious * 1e-6,
                    0.0,
                )
                pred_indices, gt_indices = linear_sum_assignment(match_cost)
                matched = sum(
                    ious[p, g] >= MATCH_IOU_THRESHOLD
                    for p, g in zip(pred_indices, gt_indices)
                )
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

    # AP của detector chỉ chấm trên các frame thực sự gọi detect, để scheduler
    # không bị tính là FN. AP_tracked chấm trên track output ở mọi frame
    # (đối chứng cho thấy phần bị bỏ detect được Kalman lấp ở mức nào).
    gt_detected = {
        seq: {
            fr: gt_all.get(seq, {}).get(fr, [])
            for fr in detector_all.get(seq, {})
        }
        for seq in gt_all
    }
    ap_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    detection_aps = average_precision_coco(
        detector_all, gt_detected, iou_thresholds=[0.5, 0.75]
    )
    ap50 = detection_aps.get(0.5, 0.0)
    detection_aps_all = average_precision_coco(
        detector_all, gt_detected, iou_thresholds=ap_thresholds
    )
    ap50_95 = float(np.mean([detection_aps_all.get(t, 0.0) for t in ap_thresholds]))

    tracked_scored: dict[str, dict[int, list[Any]]] = {}
    for seq, seq_preds in pred_all.items():
        seq_scores = pred_scores_all.get(seq, {})
        tracked_scored[seq] = {
            fr: list(zip(boxes, seq_scores.get(fr) or [0.0] * len(boxes)))
            for fr, boxes in seq_preds.items()
        }
    tracked_aps = average_precision_coco(
        tracked_scored, gt_all, iou_thresholds=[0.5, 0.75]
    )
    tracked_aps_all = average_precision_coco(
        tracked_scored, gt_all, iou_thresholds=ap_thresholds
    )
    ap50_tracked = float(tracked_aps.get(0.5, 0.0))
    ap50_95_tracked = float(
        np.mean([tracked_aps_all.get(t, 0.0) for t in ap_thresholds])
    )
    frames_detected = sum(len(v) for v in detector_all.values())

    # Metric tracking chỉ lấy từ TrackEval (nguồn duy nhất, không fallback):
    # MOTA/MOTP/IDF1/HOTA/IDSW/Frag đều do `mot_eval.py` chấm trên target_id.
    # Nếu TrackEval lỗi thì toàn bộ cột tracking ghi N/A (NaN), không ghi 0 giả.
    mot_metrics: dict[str, float] = {}
    try:
        from mot_eval import evaluate_trackeval

        mot_metrics = evaluate_trackeval(
            gt_all, gt_ids_all, pred_all, pred_ids_all, pred_scores_all
        )
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        print(f"[Bước 5] Cảnh báo: không chạy được TrackEval: {exc}")
    if not mot_metrics:
        print(
            "[Bước 5] Cảnh báo: TrackEval không khả dụng -> toàn bộ metric tracking "
            "(MOTA/MOTP/IDF1/HOTA/IDSW/Frag) ghi N/A. Không có fallback per-frame; "
            "kiểm tra package `trackeval` hoặc annotation target_id."
        )

    total_frames = sum(frames_all.values())
    output_gate_block_rate = blocked_frames / max(1, total_frames)
    output_gate_release_rate = released_frames / max(1, total_frames)
    detection_calls_per_100_frames = 100.0 * detection_calls / max(1, total_frames)
    avg_detection_interval = float(np.mean(detection_intervals)) if detection_intervals else 0.0
    # `total_protect_gt_boxes` chỉ cộng GT của frame ĐÃ PHÁT HÀNH, nên đây là
    # mẫu số đúng cho coverage của frame phát hành.
    released_gt_boxes = total_protect_gt_boxes
    blocked_gt_boxes = max(0, total_gt_all - released_gt_boxes)
    # Headline privacy: trong các frame đã phát hành, bao nhiêu GT được che đủ?
    coverage_released_iou50 = protected_gt_boxes / max(1, released_gt_boxes)
    coverage_released_containment90 = contained_gt_boxes / max(1, released_gt_boxes)
    # Coverage tính cả frame bị chặn (frame đen được coi là bảo vệ hoàn toàn).
    coverage_incl_blocked_iou50 = (
        protected_gt_boxes + blocked_gt_boxes
    ) / max(1, total_gt_all)
    coverage_incl_blocked_containment90 = (
        contained_gt_boxes + blocked_gt_boxes
    ) / max(1, total_gt_all)
    blocked_gt_share = blocked_gt_boxes / max(1, total_gt_all)
    released_gt_share = released_gt_boxes / max(1, total_gt_all)
    # Alias cũ giữ để report/notebook cũ không vỡ.
    protect_box_coverage = coverage_released_iou50
    protect_box_containment = coverage_released_containment90
    effective_privacy_coverage = coverage_incl_blocked_iou50

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
        "detection_ap50_tracked": ap50_tracked,
        "detection_ap50_95_tracked": ap50_95_tracked,
        "frames_detected": frames_detected,
        "tracking_mota": mot_metrics.get("mota", float("nan")),
        "tracking_motp": mot_metrics.get("motp", float("nan")),
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
        "coverage_released_iou50": coverage_released_iou50,
        "coverage_released_containment90": coverage_released_containment90,
        "coverage_incl_blocked_iou50": coverage_incl_blocked_iou50,
        "coverage_incl_blocked_containment90": coverage_incl_blocked_containment90,
        "blocked_gt_share": blocked_gt_share,
        "released_gt_share": released_gt_share,
        "blocked_gt_boxes": blocked_gt_boxes,
        "released_gt_boxes": released_gt_boxes,
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
    # `meanStdDev` tính mean và std trong một lượt C++ (thay vì 2 lượt numpy).
    mean_val, std_val = cv2.meanStdDev(crop)
    bright = float(mean_val[0][0] / 255.0)
    contrast = float(std_val[0][0] / 255.0)
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
        neighbor: dict[str, float] | None = None,
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
        if neighbor is not None:
            # Thống kê lân cận đã tính sẵn một lần cho cả frame (vector hoá).
            f_nearest_dist_norm = float(neighbor["nearest_dist_norm"])
            f_n_neighbors_r2_norm = float(neighbor["n_neighbors_r2_norm"])
            f_overlap_max = float(neighbor["overlap_max"])
        elif len(all_boxes_frame) > 1:
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


def frame_neighbor_stats(
    boxes: list[list[float]], img_w: int, img_h: int
) -> list[dict[str, float]]:
    """Thống kê lân cận cho MỌI box trong frame, vector hoá một lần.

    Bản cũ tính trong `TrackHistory.compute_features` cho từng track: mỗi track
    quét toàn bộ box và gọi `iou()` bằng Python -> O(T^2) lời gọi Python mỗi
    frame (~20 ms khi T ~ 40). Ở đây ma trận khoảng cách và IoU được tính một
    lần bằng numpy.

    Giữ đúng quy ước cũ: `my_idx` là box **đầu tiên** có IoU > 0.9 với box của
    track (thường là chính nó, nhưng nếu có box trùng trước đó thì là box đó),
    rồi loại `my_idx` khỏi khoảng cách/lân cận/chồng lấn.
    """
    n = len(boxes)
    if n <= 1:
        return [
            {"nearest_dist_norm": 0.0, "n_neighbors_r2_norm": 0.0, "overlap_max": 0.0}
            for _ in range(n)
        ]

    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    D = math.hypot(img_w, img_h)
    cx = (b[:, 0] + b[:, 2]) * 0.5
    cy = (b[:, 1] + b[:, 3]) * 0.5
    dist = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])

    widths = b[:, 2] - b[:, 0]
    heights = b[:, 3] - b[:, 1]
    area = widths * heights
    ix1 = np.maximum(b[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(b[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(b[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    union = area[:, None] + area[None, :] - inter
    iou_mat = inter / np.maximum(union, 1e-12)
    np.fill_diagonal(iou_mat, 1.0)  # self-IoU đúng bằng 1, tránh sai số

    rows = np.arange(n)
    match = np.argmax(iou_mat > 0.9, axis=1)  # box đầu tiên có IoU > 0.9
    dist_masked = dist.copy()
    dist_masked[rows, match] = np.inf
    iou_masked = iou_mat.copy()
    iou_masked[rows, match] = -np.inf

    radius = NEIGHBOR_RADIUS_FRAC * D
    nearest = np.clip(dist_masked.min(axis=1) / D, 0.0, 1.0)
    n_nb = (dist_masked <= radius).sum(axis=1)

    return [
        {
            "nearest_dist_norm": float(nearest[i]),
            "n_neighbors_r2_norm": float(min(n_nb[i] / NEIGHBOR_K, 1.0)),
            "overlap_max": float(iou_masked[i].max()),
        }
        for i in range(n)
    ]


def compute_leak_score(
    *,
    n_tracks: int,
    n_det: int,
    n_low_conf: int,
    ewma_det: float | None,
    prev_n_tracks: int | None,
    weights: dict[str, Any] | None = None,
    crowd_ref: float = 168.0,
    no_track_ref: float = 0.0,
) -> float:
    """Proxy rủi ro hở GT mức frame, chỉ dùng tín hiệu runtime (không GT).

    Vì sao cần: gate cũ dùng thẳng xác suất re-ID của risk model làm "leak risk".
    Hai đại lượng này khác nhau — risk cao nghĩa "dễ bị nhận dạng lại", còn rò
    rỉ coverage nghĩa "có người trong frame nhưng không được che". Detector bỏ
    sót người (đặc biệt frame đông, vật thể nhỏ) tạo ra `risk = 0` dù frame
    vẫn có GT, nên gate cũ tha phần lớn các frame này.

    Bốn tín hiệu dưới đây bắt được phần lớn rò rỉ mà không cần GT:

    1. `no_track`: tracker không còn track nào nhưng vừa có detection / vừa có
       người ở frame trước -> dropout. Được làm mềm theo số người kỳ vọng
       (`no_track_ref`) để không biến gate thành quy tắc nhị phân cứng.
    2. `missing`: số detection vượt số track được xuất ra -> người bị bỏ khỏi
       output (bị lọc bởi ngưỡng conf hoặc association).
    3. `low_conf`: tỷ lệ detection có conf thấp -> detector không chắc, dễ miss.
    4. `crowd`: frame càng đông thì một lần miss càng làm hở nhiều GT box.

    Trả về score >= 0 (không clip để còn cộng với risk); caller quyết định
    ngưỡng. Đây là proxy, không phải xác suất đã calibration.
    """
    w = {
        "no_track": 0.50,
        "missing": 0.10,
        "low_conf": 0.00,
        "crowd": 0.00,
    }
    if weights:
        for key in w:
            if key in weights:
                try:
                    w[key] = float(weights[key])
                except (TypeError, ValueError):
                    pass

    n_tracks = max(0, int(n_tracks))
    n_det = max(0, int(n_det))
    n_low_conf = max(0, int(n_low_conf))
    expected = max(
        float(n_det),
        float(n_tracks),
        float(ewma_det or 0.0),
        float(prev_n_tracks or 0),
    )

    # `no_track` có thể được làm mềm theo bằng chứng về người (`no_track_ref` > 0):
    # 1 người kỳ vọng không tương đương 80 người kỳ vọng. Mặc định tắt (0) vì
    # thực nghiệm trên val cho thấy làm mềm làm block dồn vào một sequence đông
    # và tăng identity leak; giữ tham số để có thể bật lại khi có bằng chứng mới.
    no_track = 1.0 if (n_tracks == 0 and expected > 0.0) else 0.0
    if no_track_ref > 0:
        no_track *= min(1.0, expected / float(no_track_ref))
    missing_ratio = max(0.0, expected - n_tracks) / max(expected, 1.0)
    low_conf_ratio = n_low_conf / float(n_det) if n_det > 0 else 0.0
    crowd = min(1.0, n_det / max(float(crowd_ref), 1.0))

    return (
        w["no_track"] * no_track
        + w["missing"] * missing_ratio
        + w["low_conf"] * low_conf_ratio
        + w["crowd"] * crowd
    )


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
    """Ngưỡng gate nằm trong cùng không gian xác suất với model runtime.

    Ngưỡng được hiệu chuẩn trên chính phân bố frame-risk của val và ghi vào
    `CONFIG["gate_threshold"]`. `CONFIG["tau"]` chỉ còn là fallback (track-level,
    chặn gần như toàn bộ frame nếu dùng trực tiếp ở mức frame).

    Không có biến môi trường ghi đè: thứ tự cố định là
    `CONFIG["gate_threshold"]` > `tau` (fallback).
    """
    if GATE_THRESHOLD_CACHED is not None:
        return float(GATE_THRESHOLD_CACHED)
    if tau is not None:
        return float(tau)
    return float(rules.get("output_gate_block_threshold", 0.75))


def gate_threshold_source() -> str:
    if GATE_THRESHOLD_CACHED is not None:
        return "CONFIG (gate_threshold)"
    return "tau fallback (privacy.tau)"


def should_block_output_direct(
    risk: float, rules: dict[str, Any], tau: float | None = None
) -> bool:
    """Chặn frame nếu risk vượt ngưỡng gate (xem resolve_gate_threshold).

    Không dùng `output_gate_block_threshold` = 0.526 làm mặc định vì giá trị
    này nằm ngoài dải xác suất calibrated, khiến gate không bao giờ chặn.
    """
    return risk >= resolve_gate_threshold(rules, tau)


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


class _DemoRenderer:
    """Render frame demo (blur cho frame bị chắn + resize + nhãn trạng thái).

    Dùng class thay vì lambda để đóng gói tham số của từng frame một cách
    tường minh — lambda trong vòng lặp sẽ bắt biến theo tham chiếu và đọc nhầm
    giá trị của frame sau.
    """

    __slots__ = ("block", "risk", "n_protect", "boxes", "mask")

    def __init__(
        self,
        block: bool,
        risk: float,
        n_protect: int,
        boxes: list[list[float]],
        mask: np.ndarray,
    ) -> None:
        self.block = bool(block)
        self.risk = float(risk)
        self.n_protect = int(n_protect)
        self.boxes = boxes
        self.mask = mask

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        demo = blur_boxes(frame, self.boxes, mask=self.mask) if self.block else frame
        demo = fit_to_canvas(demo)
        return draw_demo_overlay(demo, self.block, self.risk, self.n_protect)


class _CanvasRenderer:
    """Chỉ resize frame đã che về canvas video (không vẽ nhãn)."""

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return fit_to_canvas(frame)


def submit_demo_frame(
    writer: Any,
    frame: np.ndarray,
    out_frame: np.ndarray,
    block: bool,
    frame_risk: float,
    protect_boxes: list[list[float]],
    protect_mask: np.ndarray,
) -> None:
    """Đưa frame demo sang writer.

    - Frame được phát hành (`not block`): ghi đúng `out_frame` (đã che).
    - Frame bị gate chắn: video demo vẫn cho xem khung đã che (không phải khung
      đen) để minh hoạ quyết định, nên blur lại `frame` gốc.

    Với `AsyncVideoWriter`, phần blur/resize/vẽ nhãn **chạy trong thread ghi
    video** nên không nằm trên đường tới hạn. Metric vẫn tính trên `out_frame`
    ở luồng chính nên không đổi.
    """
    if writer is None:
        return
    base = frame if block else out_frame
    if isinstance(writer, AsyncVideoWriter):
        render = (
            _DemoRenderer(block, frame_risk, len(protect_boxes), protect_boxes, protect_mask)
            if DEMO_OVERLAY
            else _CanvasRenderer()
        )
        writer.write(base, render=render)
        return
    if DEMO_OVERLAY:
        demo = fit_to_canvas(blur_boxes(base, protect_boxes, mask=protect_mask) if block else base)
        writer.write(draw_demo_overlay(demo, block, frame_risk, len(protect_boxes)))
    else:
        writer.write(fit_to_canvas(base))


def detect_frame(
    model: Any,
    frame: np.ndarray,
    img_size: int,
    conf: float,
    iou: float,
    max_det: int,
    tiles: str,
    overlap: float,
    device: str,
) -> np.ndarray:
    """Gọi detector, tự chọn đường nhanh hoặc đường Ultralytics.

    `FastOpenVINODetector` chỉ hỗ trợ `1x1` (đúng cấu hình đang dùng); mọi
    trường hợp khác (kể cả khi fast path lỗi) đi qua `run_tiled_detect`.
    """
    if isinstance(model, FastOpenVINODetector):
        return model(frame, conf, iou, max_det)
    return run_tiled_detect(
        model, frame, img_size, conf, iou, max_det,
        tiles=tiles, overlap=overlap,
        predict_kwargs=dict(device=device, verbose=False),
    )


def _detect_job(
    model: Any,
    frame: np.ndarray,
    img_size: int,
    conf: float,
    iou: float,
    max_det: int,
    tiles: str,
    overlap: float,
    device: str,
) -> tuple[np.ndarray, float]:
    """Một lần detect độc lập, chạy được trong thread riêng.

    Trả `(detections, thời_gian_ms)` để thread chính ghi vào profiler mà không
    phải đo gián tiếp qua thời gian chờ.
    """
    t0 = time.perf_counter()
    det = detect_frame(model, frame, img_size, conf, iou, max_det, tiles, overlap, device)
    return det, (time.perf_counter() - t0) * 1000.0


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
    writer: Any,
    max_frames: int,
    device: str = "cpu",
    risk_labels_lookup: dict | None = None,
    gate_risk_scale: float = 0.5,
    gate_leak_weights: dict[str, Any] | None = None,
    gate_crowd_ref: float = 168.0,
    gate_low_conf_thresh: float = 0.25,
    gate_no_track_ref: float = 0.0,
) -> dict[str, Any]:
    seq_dir = SEQUENCES_DIR / seq_name
    images = limit_frames(sorted(seq_dir.glob("*.jpg")), max_frames)
    gt, gt_ids = load_annotations_with_ids(seq_name)
    seq_wall_start = time.perf_counter()

    # --- Tăng tốc: SSIM chạy nền, không chặn luồng chính (kết quả như cũ) ---
    profiler = StageProfiler(PROFILE)
    ssim_executor = (
        ThreadPoolExecutor(max_workers=SSIM_WORKERS) if SSIM_WORKERS > 0 else None
    )
    ssim_pending: deque[Any] = deque()

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
    # Số detection/conf thấp của frame gần nhất có gọi detect (dùng cho
    # leak-score khi scheduler bỏ detect một frame).
    last_n_det = 0
    last_n_low_conf = 0

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
    if GMC_ENABLE and str(GMC_METHOD).lower() not in {"", "0", "none"}:
        # Ultralytics BYTETracker gọi `self.gmc.apply(...)` trong hook
        # `_pre_first_associate`, nên chỉ cần gắn GMC vào instance.
        try:
            from ultralytics.trackers.utils.gmc import GMC

            tracker.gmc = GMC(method=GMC_METHOD, downscale=GMC_DOWNSCALE)
        except Exception as exc:  # pragma: no cover - phụ thuộc ultralytics
            print(f"[Bước 5] Cảnh báo: không bật được GMC ({GMC_METHOD}): {exc}")

    all_gt_boxes = [box for boxes in gt.values() for box in boxes]
    if all_gt_boxes:
        r_ref = float(np.median([max(b[2] - b[0], 1e-6) / max(b[3] - b[1], 1e-6) for b in all_gt_boxes]))
    else:
        r_ref = 0.4

    if DECODE_THREADS > 0:
        frame_iter: Any = FramePrefetcher(
            images, workers=DECODE_THREADS, max_pending=DECODE_MAX_PENDING
        )
    else:
        frame_iter = ((path, cv2.imread(str(path))) for path in images)

    # ===== Chồng lấn detect (GPU) với xử lý CPU =====
    # Detector chạy trên iGPU ~40 ms/frame; tracker + risk + blur + PSNR cũng
    # ~40 ms trên CPU. Chạy tuần tự mất ~80 ms/frame; chạy chồng lấn chỉ còn
    # ~max(40, 40). Điều kiện an toàn: scheduler luôn detect mọi frame (mọi
    # `detect_interval_*=1`, hybrid tắt) — khi đó việc submit detect cho frame
    # kế tiếp trước khi xử lý frame hiện tại không thể làm đổi quyết định nào.
    always_detect = (
        ASYNC_DETECT
        and all(
            int(coordinator_rules.get(k, 1) or 1) == 1
            for k in (
                "detect_interval_low",
                "detect_interval_medium",
                "detect_interval_high",
                "detect_interval_critical",
            )
        )
        and int(coordinator_rules.get("hybrid_max_skip", 0) or 0) == 0
        and int(coordinator_rules.get("hybrid_max_miss", 0) or 0) == 0
        and int(coordinator_rules.get("hybrid_crowd_tracks", 0) or 0) == 0
    )
    det_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="detect")
        if always_detect
        else None
    )
    if det_executor is not None:
        print("[Bước 5] Detect bất đồng bộ: GPU chạy trước 1 frame (overlap CPU).")

    def _pull_frame() -> tuple[Any, np.ndarray] | None:
        for path, fr in frame_iter:
            if fr is not None:
                return path, fr
        return None

    def _submit_detect(item: tuple[Any, np.ndarray] | None) -> Any:
        if item is None or det_executor is None:
            return None
        return det_executor.submit(
            _detect_job, model, item[1], IMG_SIZE, CONF_THRES, IOU_THRES,
            MAX_DET, DET_TILES, TILE_OVERLAP, device,
        )

    lookahead = _pull_frame()
    lookahead_future = _submit_detect(lookahead)

    try:
      while lookahead is not None:
        img_path, frame = lookahead
        cur_detect_future = lookahead_future
        # Kéo frame kế tiếp và cho GPU chạy detect ngay, trước khi xử lý frame
        # hiện tại trên CPU.
        lookahead = _pull_frame()
        lookahead_future = _submit_detect(lookahead)
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

        # ----- Scheduler lai (K2): chặn trần bỏ detect + cứu track mất cập nhật -----
        # Risk model trên val có AUC thấp nên lịch theo risk có thể bỏ detect quá
        # lâu. Ba luật dưới đây chỉ THÊM detect (không bao giờ bỏ bớt), mặc định
        # 0 = tắt để giữ nguyên hành vi cũ cho tới khi đo trên val.
        max_skip = int(coordinator_rules.get("hybrid_max_skip", 0) or 0)
        max_miss = int(coordinator_rules.get("hybrid_max_miss", 0) or 0)
        crowd_tracks = int(coordinator_rules.get("hybrid_crowd_tracks", 0) or 0)
        frames_since_detect = frame_idx - last_detect_frame

        force_detect = False
        force_reason = ""
        if max_skip > 0 and frames_since_detect > max_skip:
            force_detect = True
            force_reason = "max_skip"
        # Track đang active nhưng lâu không được cập nhật bởi detect -> sai số
        # Kalman tích luỹ, phải detect lại để neo ID.
        active_stracks = [t for t in tracker.tracked_stracks if t.is_activated]
        if max_miss > 0 and active_stracks and not force_detect:
            last_track_update = max(
                int(getattr(t, "frame_id", frame_idx)) for t in active_stracks
            )
            if frame_idx - last_track_update >= max_miss:
                force_detect = True
                force_reason = "max_miss"
        # Đám đông: càng nhiều track thì lỗi lan truyền càng lớn.
        if crowd_tracks > 0 and len(active_stracks) >= crowd_tracks and not force_detect:
            force_detect = True
            force_reason = "crowd"

        should_detect = (
            frame_idx < WARMUP_FRAMES
            or frames_since_detect >= det_interval
            or force_detect
        )
        detection_intervals.append(float(det_interval))

        # Ảnh xám chỉ dùng cho thống kê crop của track -> tạo lazy (nhiều frame
        # không có track nên bỏ được 1 ms cvtColor).
        gray: np.ndarray | None = None

        if should_detect:
            detection_calls += 1
            last_detect_frame = frame_idx
            if cur_detect_future is not None:
                # Kết quả đã được GPU tính trong lúc CPU xử lý frame trước.
                detections_np, _det_ms = cur_detect_future.result()
                profiler.add("detect", _det_ms)
            else:
                _t_detect = profiler.tick()
                detections_np = detect_frame(
                    model, frame, IMG_SIZE, CONF_THRES, IOU_THRES, MAX_DET,
                    DET_TILES, TILE_OVERLAP, device,
                )
                profiler.mark("detect", _t_detect)
            # Shim dùng chung (thay class định nghĩa mỗi frame).
            results = _DetResults(detections_np)
        else:
            detections_np = np.empty((0, 6), dtype=np.float32)
            # Nhánh này chỉ xảy ra khi scheduler bỏ detect; khi đó không có
            # future nào được submit (always_detect=False).
            if cur_detect_future is not None:
                cur_detect_future.result()

        _t_track = profiler.tick()
        if should_detect:
            tracker_out = tracker.update(results[0].cpu().boxes, frame)
            if TRACK_ALL:
                # ByteTrack chuẩn chỉ xuất track đã xác nhận (is_activated), nên
                # người xuất hiện 1 frame bị mất hoàn toàn khỏi output. Chế độ
                # này xuất cả track chưa xác nhận: recall/IDF1 tăng, đổi lại FP
                # và IDSW có thể tăng. Chỉ chọn sau khi ablation trên val.
                extra = [
                    t.result for t in tracker.tracked_stracks if not t.is_activated
                ]
                if extra:
                    extra_arr = np.asarray(extra, dtype=np.float32)
                    tracker_out = (
                        np.vstack([tracker_out, extra_arr])
                        if len(tracker_out)
                        else extra_arr
                    )
        else:
            # Không gọi update(empty) vì sẽ đánh lost toàn bộ track (xem
            # tracker_predict_only). Chỉ predict vị trí cho frame bị skip.
            tracker_out = tracker_predict_only(tracker)

        track_boxes: list[list[float]] = []
        track_confs: list[float] = []
        track_ids: list[int] = []
        track_index_by_tid: dict[int, int] = {}

        if len(tracker_out) > 0:
            for row in tracker_out:
                x1, y1, x2, y2, tid, score, cls = row[:7]
                det_idx = int(row[7]) if len(row) > 7 else -1
                track_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                track_confs.append(float(score))
                track_ids.append(int(tid))
                track_index_by_tid[int(tid)] = len(track_boxes) - 1

                if int(tid) not in track_histories:
                    track_histories[int(tid)] = TrackHistory(int(tid), frame_idx)
                th = track_histories[int(tid)]

                # Thống kê crop tính trên box của TRACKER ở frame hiện tại,
                # đúng bằng dữ liệu lúc train (features_frame.csv dùng box track),
                # và tính mỗi frame để brightness/contrast/lap khớp notebook.
                if gray is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        profiler.mark("track", _t_track)

        # Đếm số track mỗi frame (khớp notebook: N = số dòng tracker trong frame),
        # dùng EWMA của quá khứ nên phải lấy giá trị TRƯỚC khi cập nhật.
        N_t = len(track_boxes)
        prev_det_count = prev_n_tracks
        ewma_prev = ewma_det
        ewma_det = N_t if ewma_det is None else EWMA_ALPHA * N_t + (1 - EWMA_ALPHA) * ewma_det
        prev_n_tracks = N_t

        # Tín hiệu cho leak-score: số detection thô và số detection conf thấp.
        # Frame bỏ detect dùng lại giá trị của frame gần nhất (detector không chạy).
        if should_detect:
            n_det_frame = int(len(detections_np))
            n_low_conf_frame = (
                int(np.count_nonzero(detections_np[:, 4] < gate_low_conf_thresh))
                if n_det_frame > 0
                else 0
            )
            last_n_det = n_det_frame
            last_n_low_conf = n_low_conf_frame
        else:
            n_det_frame = last_n_det
            n_low_conf_frame = last_n_low_conf

        all_boxes_frame = track_boxes
        frame_risk = 0.0

        # Gom đặc trưng của toàn bộ track đang hoạt động trong frame rồi dự đoán
        # rủi ro 1 lần (batch) thay vì gọi predict_proba cho từng track.
        # Lưu ý: SequenceStandardizer vẫn cập nhật online tuần tự từng vector
        # (đúng như lúc train), chỉ phần suy luận model được batch.
        _t_risk = profiler.tick()
        # Thống kê lân cận cho cả frame một lần (thay O(T^2) lời gọi Python).
        neighbor_stats = frame_neighbor_stats(all_boxes_frame, w, h)
        frame_rows: list[np.ndarray] = []
        for tid, th in track_histories.items():
            if tid not in current_ids:
                continue

            _idx = track_index_by_tid.get(tid)
            feat = th.compute_features(
                frame=frame_idx,
                img_w=w,
                img_h=h,
                all_boxes_frame=all_boxes_frame,
                prev_det_count=prev_det_count,
                ewma_det=ewma_prev,
                r_ref=r_ref,
                neighbor=neighbor_stats[_idx] if _idx is not None else None,
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
        profiler.mark("risk", _t_risk)

        _t_gate = profiler.tick()
        candidate_boxes = detections_np[:, :4].tolist() if should_detect else track_boxes
        roi_padding = decide_roi_padding(frame_risk, coordinator_rules)
        protect_boxes_frame = [
            expand_box(box, roi_padding, w, h) for box in candidate_boxes
        ]

        # ===== OutputGate: leak proxy + risk re-ID =====
        # Gate cũ chỉ dùng `frame_risk` (xác suất re-ID), nên tha toàn bộ frame
        # detector bỏ sót người (risk = 0). `leak_score` bổ sung tín hiệu
        # dropout/detection không được track/conf thấp/đám đông. Quyết định vẫn
        # KHÔNG dùng GT; GT chỉ dùng để ĐO sau khi đã quyết định.
        leak_score = compute_leak_score(
            n_tracks=N_t,
            n_det=n_det_frame,
            n_low_conf=n_low_conf_frame,
            ewma_det=ewma_prev,
            prev_n_tracks=prev_det_count,
            weights=gate_leak_weights,
            crowd_ref=gate_crowd_ref,
            no_track_ref=gate_no_track_ref,
        )
        gate_score = float(leak_score + gate_risk_scale * frame_risk)
        block = should_block_output_direct(gate_score, coordinator_rules, tau)

        # ===== Đo coverage cho MỌI frame =====
        # `frame_unsafe` = nếu phát hành frame này thì còn GT chưa được che đủ.
        # Nhãn này chỉ dùng để ĐO (gate quality), không tham gia quyết định gate.
        gts = gt.get(frame_number, [])
        gt_ids_frame = gt_ids.get(frame_number, [])
        # Đo trên đúng vùng sẽ blur (`protect_boxes_frame`, đã padding) cho
        # containment; IoU dùng box thô của detector/tracker (định nghĩa cũ,
        # không phụ thuộc padding) để giữ khả năng so sánh với báo cáo trước.
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
            (iou_cov >= COVERAGE_IOU_THRESHOLD).any(axis=1)
            if iou_cov.size
            else np.zeros(len(gts), dtype=bool)
        )
        contained_any = (
            (cont_cov >= CONTAINMENT_THRESHOLD).any(axis=1)
            if cont_cov.size
            else np.zeros(len(gts), dtype=bool)
        )
        frame_unsafe = bool(len(gts) > 0 and not bool(contained_any.all()))
        profiler.mark("gate", _t_gate)

        # Mask union của vùng che, dùng lại cho blur + đo diện tích.
        _t_mask = profiler.tick()
        protect_mask = boxes_to_mask(protect_boxes_frame, h, w)
        blur_area_px = int(cv2.countNonZero(protect_mask))
        gt_covered_px = 0
        if gts and blur_area_px > 0:
            gt_mask = boxes_to_mask(gts, h, w)
            gt_covered_px = int(cv2.countNonZero(cv2.bitwise_and(gt_mask, protect_mask)))
        profiler.mark("mask", _t_mask)

        psnr_bg = float("nan")
        psnr_prot = float("nan")

        if block:
            blocked += 1
            out_frame = np.zeros_like(frame)
        else:
            released += 1
            _t_blur = profiler.tick()
            out_frame = blur_boxes(frame, protect_boxes_frame, mask=protect_mask)
            profiler.mark("blur", _t_blur)
            # GT chỉ dùng để ĐO coverage/unsafe sau khi đã quyết định phát hành.
            if gts:
                total_protect_gt_boxes += len(gts)
                protected_gt_boxes += int(protected_any.sum())
                contained_gt_boxes += int(contained_any.sum())
            if frame_unsafe:
                unsafe += 1
            # PSNR: cv2.norm tính SSD toàn frame trong C++ (~0,3 ms); vì blur chỉ
            # ghi trong mask nên SSD toàn frame = SSD vùng che, PSNR nền = 100 dB.
            _t_psnr = profiler.tick()
            try:
                psnr_full_val, psnr_bg, psnr_prot = psnr_components(
                    frame, out_frame, protect_mask
                )
                if math.isfinite(psnr_full_val):
                    psnr_sum += psnr_full_val
                    psnr_count += 1
            except Exception:
                pass
            profiler.mark("psnr", _t_psnr)

        per_frame_records.append(
            {
                "sequence": seq_name,
                "frame_number": int(frame_number),
                "block": bool(block),
                "risk": float(frame_risk),
                "risk_level": str(prev_frame_risk_level),
                "leak_score": float(leak_score),
                "gate_score": float(gate_score),
                "n_tracks": int(N_t),
                "prev_n_tracks": int(prev_det_count or 0),
                "n_det": int(n_det_frame),
                "n_low_conf": int(n_low_conf_frame),
                "ewma_det": float(ewma_prev or 0.0),
                "n_gt": len(gts),
                "gt_ids": [int(i) for i in gt_ids_frame],
                "gt_contained": [bool(v) for v in contained_any],
                # Cùng định nghĩa với `protect_box_coverage_iou50` để
                # `run_pipeline.py calibrate` hiệu chuẩn đúng metric report đang báo.
                "gt_iou_protected": [bool(v) for v in protected_any],
                "unsafe": bool(frame_unsafe),
                "blur_area_px": blur_area_px,
                "frame_area_px": int(h * w),
                "gt_covered_area_px": gt_covered_px,
                "psnr_bg": float(psnr_bg),
                "psnr_prot": float(psnr_prot),
            }
        )

        _t_enc = profiler.tick()
        submit_demo_frame(
            writer, frame, out_frame, block, frame_risk,
            protect_boxes_frame, protect_mask,
        )
        profiler.mark("encode", _t_enc)

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
        # PSNR đã tính ở trên (1 phép SSD dùng chung); SSIM là stage đắt nhất nên
        # chạy ở thread nền. Hàng đợi FIFO + lấy kết quả cũ nhất trước giữ đúng
        # thứ tự cộng dồn như bản tuần tự (kết quả không đổi).
        if not block and frame_idx % SSIM_INTERVAL == 0:
            if ssim_executor is None:
                _t_ssim = profiler.tick()
                try:
                    ssim_sum += ssim_for_frame(frame, out_frame, protect_mask)
                    ssim_count += 1
                except Exception:
                    pass
                profiler.mark("ssim", _t_ssim)
            else:
                if len(ssim_pending) >= SSIM_MAX_PENDING:
                    _t_wait = profiler.tick()
                    try:
                        ssim_sum += float(ssim_pending.popleft().result())
                        ssim_count += 1
                    except Exception:
                        pass
                    profiler.mark("ssim_wait", _t_wait)
                ssim_pending.append(
                    ssim_executor.submit(
                        ssim_for_frame, frame, out_frame, protect_mask
                    )
                )

        latency_ms = (time.perf_counter() - wall_start) * 1000.0
        latency_samples.append(latency_ms)
        profiler.add("frame", latency_ms)
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

    finally:
        # Đợi detect cuối cùng (nếu có) rồi đóng thread để không rò tài nguyên.
        if lookahead_future is not None:
            try:
                lookahead_future.result()
            except Exception:
                pass
        if det_executor is not None:
            det_executor.shutdown(wait=True)

    # ----- Dọn các stage chạy nền, cộng nốt SSIM còn lại theo đúng thứ tự -----
    while ssim_pending:
        _t_wait = profiler.tick()
        try:
            ssim_sum += float(ssim_pending.popleft().result())
            ssim_count += 1
        except Exception:
            pass
        profiler.mark("ssim_wait", _t_wait)
    if ssim_executor is not None:
        ssim_executor.shutdown(wait=True)

    if isinstance(frame_iter, FramePrefetcher):
        frame_iter.close()
        profiler.add_total("decode_wait", frame_iter.wait_ms, frame_iter.frames_read)

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
        "profile": profiler,
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


def _parse_main_args(argv: list[str] | None = None) -> Any:
    """Tham số CLI tối thiểu; vẫn cho phép override bằng biến NCKH_* như cũ."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bước 5: hệ thống tích hợp trên val (hoặc test-dev)."
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Ghi đè data.root, ví dụ VisDrone2019-VID-test-dev.",
    )
    parser.add_argument("--out-metrics", default=None)
    parser.add_argument("--out-gate-log", default=None)
    parser.add_argument("--out-frame-records", default=None)
    parser.add_argument("--out-video", default=None)
    parser.add_argument("--no-video", action="store_true", help="Không ghi video demo.")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    """Chạy toàn bộ pipeline trên split trong CONFIG (mặc định val)."""
    global IMG_SIZE, CONF_THRES, IOU_THRES, MAX_FRAMES_PER_SEQ, WARMUP_FRAMES
    global MATCH_IOU_THRESHOLD, COVERAGE_IOU_THRESHOLD, CONTAINMENT_THRESHOLD
    global TRACK_HIGH_THRESH, TRACK_LOW_THRESH, TRACK_NEW_THRESH, TRACK_MATCH_THRESH
    global TRACK_BUFFER_FRAMES, MAX_DET, WRITE_VIDEO, PROGRESS_EVERY, VIDEO_CANVAS, DEMO_OVERLAY
    global PERSON_CATEGORIES, EXCLUDE_OCCLUSION_2
    global BLUR_MODE, BLUR_KERNEL
    global GMC_ENABLE, GMC_METHOD, GMC_DOWNSCALE
    global TRACK_ALL
    global DATA_ROOT, SEQUENCES_DIR, ANNOTATIONS_DIR, DET_TILES, TILE_OVERLAP
    global GATE_THRESHOLD_CACHED
    global OPENVINO_MODEL_DIR
    global OUT_METRICS, OUT_GATE_LOG, OUT_VIDEO, OUT_FRAME_RECORDS

    # Tham số CLI chỉ ghi đè qua biến môi trường, không tạo thêm nguồn cấu hình.
    args = _parse_main_args(argv)
    if args.data_root:
        os.environ["NCKH_DATA_ROOT"] = args.data_root
    if args.out_metrics:
        os.environ["NCKH_OUT_METRICS"] = args.out_metrics
    if args.out_gate_log:
        os.environ["NCKH_OUT_GATE_LOG"] = args.out_gate_log
    if args.out_frame_records:
        os.environ["NCKH_OUT_FRAME_RECORDS"] = args.out_frame_records
    if args.out_video:
        os.environ["NCKH_OUT_VIDEO"] = args.out_video
    if args.no_video:
        os.environ["NCKH_WRITE_VIDEO"] = "0"

    # Cấu hình duy nhất của dự án (thay config/system_config.json đã xóa).
    eval_cfg = CONFIG
    # Một bộ output duy nhất, đường dẫn lấy từ CONFIG.
    OUT_METRICS = project_path(
        os.getenv("NCKH_OUT_METRICS", str(eval_cfg.get("out_metrics", OUT_METRICS)))
    )
    OUT_GATE_LOG = project_path(
        os.getenv("NCKH_OUT_GATE_LOG", str(eval_cfg.get("out_gate_log", OUT_GATE_LOG)))
    )
    OUT_VIDEO = project_path(
        os.getenv("NCKH_OUT_VIDEO", str(eval_cfg.get("out_video", OUT_VIDEO)))
    )
    OUT_FRAME_RECORDS = project_path(
        os.getenv(
            "NCKH_OUT_FRAME_RECORDS",
            str(eval_cfg.get("out_frame_records", OUT_FRAME_RECORDS)),
        )
    )

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    OUT_GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_FRAME_RECORDS.parent.mkdir(parents=True, exist_ok=True)

    # Split duy nhất của toàn hệ thống (CONFIG["data_root"]).
    DATA_ROOT = project_path(
        os.getenv("NCKH_DATA_ROOT", str(eval_cfg.get("data_root", DATA_ROOT)))
    )
    SEQUENCES_DIR = DATA_ROOT / "sequences"
    ANNOTATIONS_DIR = DATA_ROOT / "annotations"
    DET_TILES = str(eval_cfg.get("det_tiles", DET_TILES)).strip().lower()
    TILE_OVERLAP = float(eval_cfg.get("tile_overlap", TILE_OVERLAP))
    parse_tiles(DET_TILES)  # validate sớm: sai định dạng thì báo ngay
    if not SEQUENCES_DIR.is_dir():
        raise FileNotFoundError(
            f"Không thấy sequence dir: {SEQUENCES_DIR}. Kiểm tra NCKH_DATA_ROOT "
            "hoặc CONFIG['data_root']."
        )
    IMG_SIZE = int(eval_cfg.get("img_size", IMG_SIZE))
    CONF_THRES = float(eval_cfg.get("conf_thres", CONF_THRES))
    IOU_THRES = float(eval_cfg.get("iou_thres", IOU_THRES))
    MAX_FRAMES_PER_SEQ = int(os.getenv("NCKH_MAX_FRAMES", eval_cfg.get("max_frames_per_seq", MAX_FRAMES_PER_SEQ)))
    WARMUP_FRAMES = int(eval_cfg.get("warmup_frames", WARMUP_FRAMES))
    MAX_DET = int(eval_cfg.get("max_det", MAX_DET))
    # Cùng một định nghĩa IoU cho detection/tracking/coverage trên mọi báo cáo.
    MATCH_IOU_THRESHOLD = float(
        eval_cfg.get("iou_match_threshold", MATCH_IOU_THRESHOLD)
    )
    COVERAGE_IOU_THRESHOLD = float(
        eval_cfg.get("coverage_iou_threshold", COVERAGE_IOU_THRESHOLD)
    )
    CONTAINMENT_THRESHOLD = float(
        eval_cfg.get("containment_threshold", CONTAINMENT_THRESHOLD)
    )
    # Protocol nhãn GT: đồng bộ notebook4234ae1bfe (cat 1+2, giữ occlusion 2)
    PERSON_CATEGORIES = resolve_person_categories(eval_cfg)
    EXCLUDE_OCCLUSION_2 = bool(eval_cfg.get("exclude_occlusion_2", EXCLUDE_OCCLUSION_2))
    # ByteTrack: đúng tham số notebook4234ae1bfe (high/low/new .35/.05/.25, match .70, buffer 45)
    TRACK_HIGH_THRESH = float(eval_cfg.get("track_high_thresh", TRACK_HIGH_THRESH))
    TRACK_LOW_THRESH = float(eval_cfg.get("track_low_thresh", TRACK_LOW_THRESH))
    TRACK_NEW_THRESH = float(eval_cfg.get("new_track_thresh", TRACK_NEW_THRESH))
    TRACK_MATCH_THRESH = float(eval_cfg.get("track_match_thresh", TRACK_MATCH_THRESH))
    TRACK_BUFFER_FRAMES = int(eval_cfg.get("track_buffer", TRACK_BUFFER_FRAMES))
    # CMC: tắt theo cấu hình (ablation trên val cho thấy GMC làm giảm IDF1/HOTA).
    GMC_ENABLE = bool(eval_cfg.get("gmc", False))
    GMC_METHOD = str(eval_cfg.get("gmc_method", GMC_METHOD)).strip()
    GMC_DOWNSCALE = int(eval_cfg.get("gmc_downscale", GMC_DOWNSCALE))
    TRACK_ALL = bool(eval_cfg.get("track_all", False))
    WRITE_VIDEO = bool(eval_cfg.get("write_video", WRITE_VIDEO)) and os.getenv("NCKH_WRITE_VIDEO", "1") == "1"
    PROGRESS_EVERY = int(os.getenv("NCKH_PROGRESS_EVERY", str(PROGRESS_EVERY)))
    # Che đối tượng: notebook4234ae1bfe dùng ANON_METHOD='blur', ANON_KSIZE=15.
    cfg_blur_mode = str(eval_cfg.get("blur_mode", BLUR_MODE)).strip().lower()
    if cfg_blur_mode not in {"roi", "full"}:
        print(
            f"[Bước 5] Cảnh báo: blur_mode='{cfg_blur_mode}' trong config không hợp lệ, "
            "dùng 'roi'."
        )
        cfg_blur_mode = "roi"
    BLUR_MODE = os.getenv("NCKH_BLUR_MODE", cfg_blur_mode).strip().lower()
    if BLUR_MODE not in {"roi", "full"}:
        print(
            f"[Bước 5] Cảnh báo: NCKH_BLUR_MODE='{BLUR_MODE}' không hợp lệ, dùng 'roi'."
        )
        BLUR_MODE = "roi"
    BLUR_KERNEL = int(
        os.getenv("NCKH_BLUR_KERNEL", eval_cfg.get("blur_kernel", BLUR_KERNEL))
    )
    if BLUR_KERNEL % 2 == 0:  # GaussianBlur yêu cầu kernel lẻ, giống notebook
        BLUR_KERNEL += 1
    print(
        f"[Bước 5] Eval config: img_size={IMG_SIZE}, conf_thres={CONF_THRES}, "
        f"iou_thres={IOU_THRES}, max_det={MAX_DET}, "
        f"max_frames_per_seq={MAX_FRAMES_PER_SEQ or 'all'}, "
        f"warmup_frames={WARMUP_FRAMES}"
    )
    print(
        f"[Bước 5] Detect: tiles={DET_TILES}, tile_overlap={TILE_OVERLAP:g} "
        f"({'nguyên frame' if DET_TILES == '1x1' else 'cắt lưới + NMS gộp'}) "
        f"| split={DATA_ROOT.name}"
    )
    print(
        f"[Bước 5] Metric thresholds: match_iou={MATCH_IOU_THRESHOLD}, "
        f"coverage_iou={COVERAGE_IOU_THRESHOLD}, containment={CONTAINMENT_THRESHOLD}"
    )
    print(
        f"[Bước 5] Blur: mode={BLUR_MODE}, kernel={BLUR_KERNEL} "
        "(đồng bộ notebook4234ae1bfe: ANON_METHOD=blur, ANON_KSIZE=15)"
    )
    print(
        f"[Bước 5] GT protocol: person_categories={PERSON_CATEGORIES}, "
        f"exclude_occlusion_2={EXCLUDE_OCCLUSION_2} (đồng bộ notebook4234ae1bfe)"
    )
    print(
        f"[Bước 5] ByteTrack: high={TRACK_HIGH_THRESH}, low={TRACK_LOW_THRESH}, "
        f"new={TRACK_NEW_THRESH}, match={TRACK_MATCH_THRESH}, buffer={TRACK_BUFFER_FRAMES} "
        f"(đồng bộ notebook4234ae1bfe)"
    )
    print(
        f"[Bước 5] Chế độ: scheduler thích ứng + output gate (một chế độ duy nhất), "
        f"data_root={DATA_ROOT}"
    )
    print(
        f"[Bước 5] CMC: {'on - ' + GMC_METHOD if GMC_ENABLE else 'off'} "
        f"('tracker.gmc' trong config) | downscale={GMC_DOWNSCALE}"
    )
    print(f"[Bước 5] Ghi video demo: {WRITE_VIDEO} -> {OUT_VIDEO}")

    # Detector duy nhất của dự án: OpenVINO model đã export.
    OPENVINO_MODEL_DIR = project_path(
        str(eval_cfg.get("openvino_model", OPENVINO_MODEL_DIR))
    )
    if not OPENVINO_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy OpenVINO model: {OPENVINO_MODEL_DIR}. "
            "Đặt 'openvino_model' trong CONFIG trỏ tới thư mục *_openvino_model."
        )
    det_device = resolve_openvino_device(
        str(eval_cfg.get("openvino_device", "intel:CPU"))
    )
    backend = "openvino"
    detector_half = False
    # Ultralytics lưu imgsz lúc export trong metadata.yaml; dùng đúng kích thước
    # đó để tránh preprocess lệch và đo FPS sai.
    meta_path = OPENVINO_MODEL_DIR / "metadata.yaml"
    if meta_path.exists():
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            ov_imgsz = meta.get("imgsz")
            if isinstance(ov_imgsz, (list, tuple)) and ov_imgsz:
                IMG_SIZE = int(max(int(v) for v in ov_imgsz))
            elif isinstance(ov_imgsz, (int, str)):
                IMG_SIZE = int(ov_imgsz)
        except Exception as exc:  # pragma: no cover - metadata hỏng
            print(
                "[Bước 5] Cảnh báo: không đọc được metadata OpenVINO "
                f"({meta_path}): {exc}; giữ imgsz={IMG_SIZE}"
            )

    model: Any
    fast_detector = None
    if DET_TILES == "1x1" and FAST_DETECT:
        try:
            fast_detector = FastOpenVINODetector(
                OPENVINO_MODEL_DIR, det_device, IMG_SIZE
            )
            model = fast_detector
            # NMS của Ultralytics trên tensor nhỏ chạy 4 thread tốn ~14 ms wall
            # (thread pool đồng bộ liên tục); 1 thread chỉ ~2 ms và không tranh
            # CPU với blur/mask ở luồng chính.
            torch.set_num_threads(1)
            print(
                "[Bước 5] Detector: fast OpenVINO path "
                "(preprocess numpy + NMS 1 thread, tương đương bit-for-bit)."
            )
        except Exception as exc:
            print(
                f"[Bước 5] Cảnh báo: không dựng được fast detector ({exc}); "
                "dùng đường Ultralytics."
            )
            fast_detector = None
    if fast_detector is None:
        model = YOLO(str(OPENVINO_MODEL_DIR))
    print(
        f"[Bước 5] Detector backend: openvino | weights: {OPENVINO_MODEL_DIR} | "
        f"device: {det_device} | gpu: {runtime_info()['gpu_name']} | imgsz: {IMG_SIZE}"
    )

    # Warmup bắt buộc: lần inference đầu trên iGPU bao gồm compile kernel của
    # OpenVINO (đo được ~20 s trên Iris Xe). Nếu để nó rơi vào vòng lặp đo thì
    # `fps_wall` bị kéo sai hoàn toàn (200 frame: 5.3 fps thay vì ~12 fps).
    _warm = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _warm_times: list[float] = []
    for _ in range(3):
        _tw = time.perf_counter()
        detect_frame(
            model, _warm, IMG_SIZE, CONF_THRES, IOU_THRES, MAX_DET,
            DET_TILES, TILE_OVERLAP, det_device,
        )
        _warm_times.append((time.perf_counter() - _tw) * 1000.0)
    print(
        "[Bước 5] Warmup detector: "
        + ", ".join(f"{v:.0f} ms" for v in _warm_times)
        + f" (lần cuối {_warm_times[-1]:.1f} ms)"
    )

    risk_model = load_risk_model()
    tau = load_tau()
    risk_labels_lookup = load_risk_labels_lookup()

    # Scheduler + ngưỡng gate đều nằm trong CONFIG.
    coordinator_rules = dict(CONFIG.get("scheduler", {}))
    if not coordinator_rules:
        raise ValueError("Thiếu khối 'scheduler' trong CONFIG.")
    print(
        "[Bước 5] Scheduler lai (K2): "
        f"max_skip={coordinator_rules.get('hybrid_max_skip', 0)} | "
        f"max_miss={coordinator_rules.get('hybrid_max_miss', 0)} | "
        f"crowd_tracks={coordinator_rules.get('hybrid_crowd_tracks', 0)} "
        "(0 = tắt)"
    )

    gate_value = eval_cfg.get("gate_threshold")
    if gate_value is None:
        raise ValueError(
            "Thiếu gate_threshold trong CONFIG. "
            "Chạy 'python run_pipeline.py calibrate' trước."
        )
    GATE_THRESHOLD_CACHED = float(gate_value)
    GATE_RISK_SCALE = float(eval_cfg.get("gate_risk_scale", 0.5))
    GATE_LEAK_WEIGHTS = dict(eval_cfg.get("gate_leak_weights", {}) or {})
    GATE_CROWD_REF = float(eval_cfg.get("gate_crowd_ref", 168.0))
    GATE_LOW_CONF_THRESH = float(eval_cfg.get("gate_low_conf_thresh", 0.25))
    GATE_NO_TRACK_REF = float(eval_cfg.get("gate_no_track_ref", 0.0))

    gate_threshold = resolve_gate_threshold(coordinator_rules, tau)
    print(
        f"[Bước 5] Output gate threshold: {gate_threshold:.4f} "
        f"({gate_threshold_source()}) | score = leak_score + "
        f"{GATE_RISK_SCALE:g} * frame_risk | leak_weights={GATE_LEAK_WEIGHTS}"
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
    profile_all = StageProfiler(PROFILE)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    video_fps = float(eval_cfg.get("video_fps", 30.0))
    VIDEO_CANVAS = (
        int(eval_cfg.get("video_width", 1280)),
        int(eval_cfg.get("video_height", 720)),
    )
    if WRITE_VIDEO:
        if ENCODE_QUEUE > 0:
            writer = AsyncVideoWriter(
                OUT_VIDEO, fourcc, video_fps, VIDEO_CANVAS, max_pending=ENCODE_QUEUE
            )
        else:
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
            MAX_FRAMES_PER_SEQ, det_device, risk_labels_lookup,
            gate_risk_scale=GATE_RISK_SCALE,
            gate_leak_weights=GATE_LEAK_WEIGHTS,
            gate_crowd_ref=GATE_CROWD_REF,
            gate_low_conf_thresh=GATE_LOW_CONF_THRESH,
            gate_no_track_ref=GATE_NO_TRACK_REF,
        )

        frames_done += len(result["pred_boxes_by_frame"])
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
        profile_all.merge(result["profile"])

    # FPS streaming thật: số frame xong / thời gian tường. Đo 2 mốc để tách chi
    # phí encode video khỏi đường tới hạn (fps_wall trong CSV vẫn giữ nguyên
    # định nghĩa cũ: 1000 * N / tổng latency từng frame).
    stream_loop_end = time.perf_counter()
    if writer is not None:
        writer.release()
    stream_wall_end = time.perf_counter()
    wall_loop_s = max(stream_loop_end - run_wall_start, 1e-9)
    wall_all_s = max(stream_wall_end - run_wall_start, 1e-9)
    fps_stream = frames_done / wall_loop_s
    fps_stream_with_encode = frames_done / wall_all_s
    print(
        f"[Bước 5] FPS streaming (xử lý frame): {fps_stream:.2f} "
        f"({frames_done} frame / {wall_loop_s:.1f}s)",
        flush=True,
    )
    print(
        f"[Bước 5] FPS streaming (gồm encode video): {fps_stream_with_encode:.2f} "
        f"({frames_done} frame / {wall_all_s:.1f}s)",
        flush=True,
    )

    if PROFILE:
        table = profile_all.format_table()
        if table:
            print("[Bước 5] Profile từng stage (ms/frame; 'frame' = toàn bộ frame):")
            print(table)
        PROFILE_OUT.parent.mkdir(parents=True, exist_ok=True)
        with PROFILE_OUT.open("w", encoding="utf-8") as f:
            json.dump(profile_all.summary(), f, indent=2, ensure_ascii=False)
        print(f"[Bước 5] Đã lưu profile: {PROFILE_OUT}")

    metrics = compute_metrics(
        all_gt, all_pred, all_pred_ids, all_pred_scores, all_gt_ids, all_detector, all_frames,
        all_latency, total_detection_calls,
        all_intervals, total_blocked, total_released,
        total_protected_gt_boxes, total_contained_gt_boxes, total_protect_gt_boxes,
        total_psnr_sum, total_psnr_count,
        total_ssim_sum, total_ssim_count,
        all_frame_records,
    )

    policy_name = "integrated_system"
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
        "gate_mode": "system",
        "gate_threshold": gate_threshold,
        "gate_threshold_source": gate_threshold_source(),
        "gate_target_block_rate": eval_cfg.get("gate_target_block_rate"),
        "gate_score": {
            "definition": "leak_score + gate_risk_scale * frame_risk",
            "risk_scale": GATE_RISK_SCALE,
            "leak_weights": GATE_LEAK_WEIGHTS,
            "crowd_ref": GATE_CROWD_REF,
            "low_conf_thresh": GATE_LOW_CONF_THRESH,
        },
        # Kết quả hiệu chuẩn đầy đủ: `results/gate_calibration.json`.
        "calibration": load_gate_calibration(),
        # Protocol đầy đủ đã dùng để sinh số liệu. Report đọc khối này để
        # cảnh báo nếu code/config đã đổi so với lúc chạy.
        "evaluation_config": {
            "notebook_source": eval_cfg.get(
                "notebook_source", "notebook4234ae1bfe.ipynb"
            ),
            "img_size": IMG_SIZE,
            "conf_thres": CONF_THRES,
            "iou_thres": IOU_THRES,
            "max_det": MAX_DET,
            "iou_match_threshold": MATCH_IOU_THRESHOLD,
            "coverage_iou_threshold": COVERAGE_IOU_THRESHOLD,
            "containment_threshold": CONTAINMENT_THRESHOLD,
            "person_categories": list(PERSON_CATEGORIES),
            "exclude_occlusion_2": EXCLUDE_OCCLUSION_2,
            "track_high_thresh": TRACK_HIGH_THRESH,
            "track_low_thresh": TRACK_LOW_THRESH,
            "new_track_thresh": TRACK_NEW_THRESH,
            "track_match_thresh": TRACK_MATCH_THRESH,
            "track_buffer": TRACK_BUFFER_FRAMES,
            "gmc": GMC_METHOD if GMC_ENABLE else None,
            "track_all": TRACK_ALL,
            "hybrid_max_skip": int(
                coordinator_rules.get("hybrid_max_skip", 0) or 0
            ),
            "hybrid_max_miss": int(
                coordinator_rules.get("hybrid_max_miss", 0) or 0
            ),
            "hybrid_crowd_tracks": int(
                coordinator_rules.get("hybrid_crowd_tracks", 0) or 0
            ),
            "det_tiles": DET_TILES,
            "tile_overlap": TILE_OVERLAP,
            "data_root": str(DATA_ROOT),
            "blur_mode": BLUR_MODE,
            "blur_kernel": BLUR_KERNEL,
            "gate_risk_scale": GATE_RISK_SCALE,
            "gate_leak_weights": GATE_LEAK_WEIGHTS,
            "gate_crowd_ref": GATE_CROWD_REF,
            "gate_low_conf_thresh": GATE_LOW_CONF_THRESH,
            "gate_no_track_ref": GATE_NO_TRACK_REF,
            "openvino_device": det_device,
            **config_fingerprint(),
        },
        # Thông tin runtime để báo cáo không trộn số CPU với số GPU.
        "backend": backend,
        "device": det_device,
        "half": detector_half,
        "blur_mode": BLUR_MODE,
        "blur_kernel": BLUR_KERNEL,
        # Cấu hình tăng tốc, để báo cáo không trộn số "async" với số "đồng bộ".
        "decode_threads": DECODE_THREADS,
        "ssim_threads": SSIM_WORKERS,
        "encode_queue": ENCODE_QUEUE,
        # FPS streaming thật (frames_done / thời gian tường), tách khỏi fps_wall
        # trong CSV để báo cáo không trộn hai định nghĩa.
        "frames_done": frames_done,
        "wall_loop_s": round(wall_loop_s, 3),
        "wall_total_s": round(wall_all_s, 3),
        "fps_stream": round(fps_stream, 4),
        "fps_stream_with_encode": round(fps_stream_with_encode, 4),
        "detector_weights": str(OPENVINO_MODEL_DIR),
        **runtime_info(),
        # `runtime_info()["device"]` là thiết bị PyTorch (cpu/cuda). Detector
        # thực tế chạy OpenVINO, nên ghi đè bằng thiết bị OpenVINO đã chọn.
        "device": det_device,
        "openvino_device": det_device,
    }
    with OUT_GATE_LOG.open("w", encoding="utf-8") as f:
        json.dump(gate_log, f, indent=2, ensure_ascii=False)

    # Dump per-frame records để hiệu chuẩn lại gate trên val
    # (`python run_pipeline.py calibrate`) và kiểm tra coverage/unsafe mà không
    # phải chạy lại pipeline. Đường dẫn mặc định lấy từ CONFIG["out_frame_records"].
    with OUT_FRAME_RECORDS.open("w", encoding="utf-8") as f:
        json.dump(all_frame_records, f, ensure_ascii=False)
    print(
        f"[Bước 5] Đã lưu per-frame records: {OUT_FRAME_RECORDS} "
        f"({len(all_frame_records)} frame)"
    )

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
    print(f"[Bước 5] Coverage released @ IoU 0.5: {metrics['coverage_released_iou50']:.4f}")
    print(f"[Bước 5] Coverage released @ containment 90%: "
          f"{metrics['coverage_released_containment90']:.4f}  (headline)")
    print(f"[Bước 5] Coverage incl. blocked @ containment 90%: "
          f"{metrics['coverage_incl_blocked_containment90']:.4f}")
    print(f"[Bước 5] Blocked-GT share: {metrics['blocked_gt_share']:.4f}")
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
