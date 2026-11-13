"""Tổng hợp metric của hệ thống từ kết quả của MỘT chế độ chạy duy nhất.

Input:
- results/system_metrics.csv        (Bước 5: scheduler thích ứng + output gate)
- results/output_gate_log.json      (Bước 5: gate log + calibration)
- results/stage_profile.json        (tùy chọn, sinh bởi NCKH_PROFILE=1)
- CONFIG trong step5_integrated_system.py (nguồn cấu hình duy nhất)

Output:
- results/system_report.md   (bảng markdown)
- in bảng tóm tắt ra màn hình

Module này là thư viện: `run_pipeline.py report` gọi `build_report(argv, CONFIG)`.
Không có hàm `main` riêng.

Script không tự sinh số. Nếu file metric được tạo bởi protocol khác với
CONFIG hiện tại (hoặc thiếu khối `evaluation_config` trong gate log), report sẽ
in cảnh báo ngay đầu file thay vì trình bày số cũ như số hiện hành.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

# Gốc dự án + tiện ích đường dẫn (thay cho core.config đã xóa).
ROOT = Path(__file__).resolve().parent


def project_path(value: str | Path) -> Path:
    """Trả đường dẫn tuyệt đối: đường dẫn tương đối tính từ ROOT."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


# Các khóa protocol được đối chiếu giữa gate log và CONFIG hiện hành.
PROTOCOL_KEYS = (
    "notebook_source",
    "img_size",
    "conf_thres",
    "iou_thres",
    "max_det",
    "iou_match_threshold",
    "coverage_iou_threshold",
    "containment_threshold",
    "person_categories",
    "exclude_occlusion_2",
    "track_high_thresh",
    "track_low_thresh",
    "new_track_thresh",
    "track_match_thresh",
    "track_buffer",
    "track_all",
    "det_tiles",
    "tile_overlap",
    "blur_mode",
    "blur_kernel",
    "gate_risk_scale",
)


def _same_value(old: Any, new: Any) -> bool:
    if isinstance(old, (list, tuple)) or isinstance(new, (list, tuple)):
        return [str(x) for x in (old or [])] == [str(x) for x in (new or [])]
    if isinstance(old, bool) or isinstance(new, bool):
        return bool(old) == bool(new)
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return abs(float(old) - float(new)) <= 1e-9
    return str(old) == str(new)


def protocol_mismatches(
    meta: dict[str, Any] | None, config: dict[str, Any]
) -> list[str]:
    """Liệt kê khác biệt giữa `evaluation_config` trong gate log và CONFIG."""
    if not meta:
        return ["gate log thiếu khối `evaluation_config` (chạy lại evaluate)."]
    issues: list[str] = []
    for key in PROTOCOL_KEYS:
        if key not in meta or key not in config:
            continue
        if not _same_value(meta[key], config[key]):
            issues.append(f"{key}: gate log={meta[key]!r} vs CONFIG={config[key]!r}")
    if meta.get("data_root") and config.get("data_root"):
        if Path(str(meta["data_root"])).name != Path(str(config["data_root"])).name:
            issues.append(
                f"data_root: gate log={meta['data_root']!r} vs "
                f"CONFIG={config['data_root']!r}"
            )
    return issues


# Một bộ output duy nhất (không còn hậu tố _eval/_unified).
SYSTEM_CSV = project_path("results/system_metrics.csv")
PROFILE_JSON = project_path("results/stage_profile.json")
GATE_JSON = project_path("results/output_gate_log.json")
OUT_MD = project_path("results/system_report.md")


def protocol_summary(
    config: dict[str, Any], data_root: str | None = None
) -> dict[str, str]:
    """Tạo chuỗi mô tả protocol GT/tracker từ CONFIG."""
    cfg = dict(config)
    if data_root:
        cfg["data_root"] = data_root
    cats = cfg.get("person_categories", [1, 2])
    if isinstance(cats, (int, str)):
        cats = [cats]
    person_cats = "+".join(str(int(c)) for c in cats)
    exclude_occ2 = bool(cfg.get("exclude_occlusion_2", False))
    occlusion = "occlusion level 2 excluded" if exclude_occ2 else "all occlusion levels kept"
    iou_match = float(cfg.get("iou_match_threshold", 0.5))
    coverage_iou = float(cfg.get("coverage_iou_threshold", 0.5))
    containment = float(cfg.get("containment_threshold", 0.9))
    tracker = (
        f"high={float(cfg.get('track_high_thresh', 0.35)):.2f}, "
        f"low={float(cfg.get('track_low_thresh', 0.05)):.2f}, "
        f"new={float(cfg.get('new_track_thresh', 0.25)):.2f}, "
        f"match={float(cfg.get('track_match_thresh', 0.70)):.2f}, "
        f"buffer={int(cfg.get('track_buffer', 45))}"
    )
    return {
        "notebook": str(cfg.get("notebook_source", "notebook4234ae1bfe.ipynb")),
        "split": str(cfg.get("data_root", "visdrone_val/VisDrone2019-VID-val"))
        .replace("\\", "/")
        .rstrip("/")
        .split("/")[-1],
        "img_size": str(int(cfg.get("img_size", 1024))),
        "det_tiles": str(cfg.get("det_tiles", "1x1")),
        "tile_overlap": f"{float(cfg.get('tile_overlap', 0.15)):.2f}",
        "conf": f"{float(cfg.get('conf_thres', 0.05)):.2f}",
        "max_det": str(int(cfg.get("max_det", 1000))),
        "person_cats": person_cats,
        "occlusion": occlusion,
        "iou_match": f"{iou_match:.2f}",
        "coverage_iou": f"{coverage_iou:.2f}",
        "containment": f"{containment:.2f}",
        "blur": (
            f"{str(cfg.get('blur_mode', 'roi'))}, kernel "
            f"{int(cfg.get('blur_kernel', 15))}"
        ),
        "tracker": tracker,
        "gate_threshold": cfg.get("gate_threshold"),
        "gate_target_block_rate": cfg.get("gate_target_block_rate"),
    }

# metric_name -> (nhãn hiển thị, số chữ số thập phân)
DETECTION_METRICS = [
    ("detection_precision", "YOLO precision (frames có detect)", 4),
    ("detection_recall", "YOLO recall (frames có detect)", 4),
    ("detection_f1", "YOLO F1 (frames có detect)", 4),
    ("detection_precision_global", "YOLO precision (toàn cục, mọi frame)", 4),
    ("detection_recall_global", "YOLO recall (toàn cục, mọi frame)", 4),
    ("detection_f1_global", "YOLO F1 (toàn cục, mọi frame)", 4),
    ("detection_ap50", "AP @ IoU 0.5 (frames có detect)", 4),
    ("detection_ap50_95", "AP @ IoU 0.5:0.95 (frames có detect)", 4),
    ("detection_ap50_tracked", "AP @ IoU 0.5 (track output, mọi frame)", 4),
    ("detection_ap50_95_tracked", "AP @ IoU 0.5:0.95 (track output, mọi frame)", 4),
    ("frames_detected", "Số frame thực sự gọi detect", 0),
]

TRACKING_METRICS = [
    ("tracking_precision", "ByteTrack precision", 4),
    ("tracking_recall", "ByteTrack recall", 4),
    ("tracking_f1", "ByteTrack F1", 4),
    ("tracking_mota", "MOTA", 4),
    ("tracking_motp", "MOTP", 4),
    ("tracking_idf1", "IDF1", 4),
    ("tracking_idp", "IDP", 4),
    ("tracking_idr", "IDR", 4),
    ("tracking_hota", "HOTA", 4),
    ("tracking_deta", "DetA", 4),
    ("tracking_assa", "AssA", 4),
    ("tracking_idsw", "ID switches (IDSW)", 0),
    ("tracking_frag", "Fragmentation (Frag)", 0),
    ("tracking_mt", "Mostly tracked (MT)", 0),
    ("tracking_ml", "Mostly lost (ML)", 0),
    ("tracking_tp", "TrackEval TP", 0),
    ("tracking_fp", "TrackEval FP", 0),
    ("tracking_fn", "TrackEval FN", 0),
    ("tracking_eval_sequences", "Số sequence đánh giá", 0),
]

PRIVACY_METRICS = [
    ("coverage_released_containment90", "Coverage @ containment 90% (released frames, headline)", 4),
    ("coverage_incl_blocked_containment90", "Coverage @ containment 90% incl. blocked frames", 4),
    ("coverage_released_iou50", "Coverage @ IoU 0.5 (released frames)", 4),
    ("coverage_incl_blocked_iou50", "Coverage @ IoU 0.5 incl. blocked frames", 4),
    ("blocked_gt_share", "Share of GT boxes on blocked frames", 4),
    ("ulfr", "ULFR (unsafe leak frame rate, all GT frames)", 4),
    ("unsafe_release_rate", "Unsafe release rate (released GT frames)", 4),
    ("leak_share_zero_track", "Leak share: zero-track frames", 4),
    ("leak_share_low_risk", "Leak share: tracked, risk < 0.25", 4),
    ("leak_share_medium_risk", "Leak share: tracked, risk >= 0.25", 4),
    ("identity_leak_rate", "Identity leak rate (GT target_id)", 4),
    ("ttp_p50_frames", "TTP p50 (frames)", 1),
    ("ttp_p95_frames", "TTP p95 (frames)", 1),
    ("ttp_censored_rate", "TTP censored rate", 4),
    ("blur_area_ratio", "Blur-area ratio (released frames)", 4),
    ("over_protection_ratio", "Over-protection ratio (blur outside GT)", 4),
    ("psnr_bg_db", "PSNR background (utility, dB)", 2),
    ("psnr_protected_db", "PSNR protected region (obfuscation, dB)", 2),
    ("avg_psnr_db", "PSNR whole frame (dB)", 2),
    ("avg_ssim", "SSIM whole frame", 4),
]

GATE_QUALITY_METRICS = [
    ("gate_precision", "Gate precision (unsafe frames)", 4),
    ("gate_recall", "Gate recall (unsafe frames)", 4),
    ("gate_f1", "Gate F1", 4),
    ("gate_balanced_accuracy", "Gate balanced accuracy", 4),
    ("false_block_rate", "False block rate (safe frames blocked)", 4),
    ("gate_leak_rate", "Gate leak rate (unsafe frames released)", 4),
    ("gate_score_roc_auc", "Gate-score ROC-AUC (leak proxy + risk)", 4),
    ("gate_score_pr_auc", "Gate-score PR-AUC", 4),
    ("gate_score_brier", "Gate-score Brier", 4),
    ("frame_leak_base_rate", "Unsafe base rate (constant baseline)", 4),
    ("frame_leak_brier_baseline", "Brier of constant baseline", 4),
    ("frame_leak_pr_auc_baseline", "PR-AUC of constant baseline", 4),
    ("frame_leak_roc_auc", "Re-ID frame-risk ROC-AUC (reference only)", 4),
    ("frame_leak_pr_auc", "Re-ID frame-risk PR-AUC", 4),
    ("frame_leak_brier", "Re-ID frame-risk Brier", 4),
    ("frame_leak_ece", "Re-ID frame-risk ECE", 4),
    ("risk_score_p50", "Frame risk p50", 4),
    ("risk_score_p95", "Frame risk p95", 4),
    ("risk_level_low_share", "Risk level low share", 4),
    ("risk_level_medium_share", "Risk level medium share", 4),
    ("risk_level_high_share", "Risk level high share", 4),
    ("risk_level_critical_share", "Risk level critical share", 4),
]

SYSTEM_METRICS = [
    ("output_gate_block_rate", "Output gate block rate", 4),
    ("output_gate_release_rate", "Output gate release rate", 4),
    ("detection_calls_per_100_frames", "Detection calls / 100 frames", 2),
    ("average_detection_interval_frames", "Avg detection interval (frames)", 2),
    ("fps_wall", "FPS (wall)", 4),
    ("latency_p50_ms", "Latency p50 (ms)", 2),
    ("latency_p95_ms", "Latency p95 (ms)", 2),
    ("latency_p99_ms", "Latency p99 (ms)", 2),
]

COUNT_METRICS = [
    ("detection_calls", "Detection calls", 0),
    ("total_gt_boxes", "GT boxes (frames with detection)", 0),
    ("total_gt_boxes_all", "GT boxes (all frames)", 0),
    ("detection_gt_boxes_global", "GT boxes (global detection eval)", 0),
    ("total_pred_boxes", "YOLO predicted boxes", 0),
    ("detection_pred_boxes_global", "YOLO predicted boxes (global)", 0),
    ("total_tracking_pred_boxes", "ByteTrack output boxes", 0),
    ("protected_gt_boxes", "Protected GT boxes", 0),
    ("blocked_gt_boxes", "GT boxes on fully blocked frames", 0),
    ("total_protect_gt_boxes", "Evaluated GT boxes (released frames)", 0),
    ("frames_with_gt", "Frames with GT", 0),
    ("released_frames_with_gt", "Released frames with GT", 0),
    ("unsafe_frames", "Unsafe released frames", 0),
    ("unsafe_released_gt_boxes", "Unsafe GT boxes on released frames", 0),
    ("n_gt_identities", "GT identities (target_id)", 0),
    ("n_leaked_identities", "Leaked GT identities", 0),
    ("gate_tp", "Gate TP (unsafe frames blocked)", 0),
    ("gate_fp", "Gate FP (safe frames blocked)", 0),
    ("gate_tn", "Gate TN (safe frames released)", 0),
    ("gate_fn", "Gate FN (unsafe frames released)", 0),
]

# Stage trong results/stage_profile.json (sinh bởi `NCKH_PROFILE=1` ở Bước 5).
PROFILE_LABELS = {
    "frame": "Cả frame (đường tới hạn)",
    "decode_wait": "Chờ prefetch JPEG",
    "detect": "Detect (chỉ frame gọi detect)",
    "track": "ByteTrack",
    "risk": "Đặc trưng + LogisticRegression risk",
    "gate": "Gate + đo coverage",
    "mask": "Rasterize mask",
    "blur": "Blur vùng che",
    "psnr": "PSNR (1 phép SSD)",
    "ssim": "SSIM (1 frame / 5)",
    "ssim_wait": "Chờ SSIM nền",
    "encode": "Đẩy frame sang writer",
}


def fmt(value: Any, digits: int) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, str):
        return value
    if digits == 0:
        return f"{int(round(float(value))):,}"
    return f"{float(value):.{digits}f}"


def load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "metric_name" not in df or "metric_value" not in df:
        return None
    return dict(zip(df["metric_name"], df["metric_value"]))


def load_json(path: Path) -> dict[str, Any] | None:
    """Đọc JSON, trả None nếu thiếu hoặc hỏng (không làm hỏng report)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def add_metric_table(
    lines: list[str],
    title: str,
    metrics: dict[str, Any],
    groups: list[tuple[str, list[tuple[str, str, int]]]],
) -> None:
    lines += ["", f"### {title}", "", "| Metric | Value |", "|---|---|"]
    for group_name, items in groups:
        rows = [(k, label, d) for k, label, d in items if k in metrics]
        if not rows:
            continue
        lines.append(f"| **{group_name}** | |")
        for key, label, digits in rows:
            lines.append(f"| {label} | {fmt(metrics[key], digits)} |")


def _fmt_threshold(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{float(value):.10f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def build_report(
    argv: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Sinh báo cáo markdown từ kết quả của `run_pipeline.py evaluate`."""
    from datetime import date

    if config is None:
        raise ValueError("build_report cần CONFIG của dự án (từ step5).")

    parser = argparse.ArgumentParser(description="Tổng hợp báo cáo hệ thống.")
    parser.add_argument("--metrics", default=None, help="CSV metric (mặc định results/system_metrics.csv).")
    parser.add_argument("--gate-log", default=None, help="Gate log JSON (mặc định results/output_gate_log.json).")
    parser.add_argument("--out", default=None, help="File markdown đầu ra.")
    parser.add_argument(
        "--split-label", default="val (selection)",
        help="Nhãn split trong báo cáo, ví dụ 'test-dev (held-out)'.",
    )
    args = parser.parse_args(argv)
    system_csv = project_path(args.metrics) if args.metrics else SYSTEM_CSV
    gate_json = project_path(args.gate_log) if args.gate_log else GATE_JSON
    out_md = project_path(args.out) if args.out else OUT_MD
    split_label = args.split_label

    lines: list[str] = ["# Báo cáo đánh giá toàn hệ thống", ""]

    system_metrics = load_metrics(system_csv)
    if system_metrics is None:
        raise FileNotFoundError(
            f"Chưa có {system_csv}. Hãy chạy "
            "`python run_pipeline.py evaluate` trước."
        )

    # Đối chiếu metadata protocol ghi trong gate log với CONFIG hiện hành.
    cfg = dict(config)
    gate = load_json(gate_json)
    meta = (gate or {}).get("evaluation_config")

    warnings: list[str] = []
    source_name = system_csv.name
    for issue in protocol_mismatches(meta, cfg):
        warnings.append(
            f"Số liệu trong `{source_name}` không khớp protocol "
            "hiện hành: " + issue
        )
    if warnings:
        lines += [
            "> **CẢNH BÁO: số liệu bên dưới không còn hiệu lực để trích dẫn.**",
            ">",
        ]
        lines += [f"> - {w}" for w in warnings]
        lines += [
            ">",
            "> Chạy lại `python run_pipeline.py evaluate` rồi "
            "`python run_pipeline.py report` trước khi dùng số.",
            "",
        ]

    proto = protocol_summary(cfg, (meta or {}).get("data_root"))
    n_frames = system_metrics.get("detection_calls")
    avg_interval = system_metrics.get("average_detection_interval_frames")
    if avg_interval is not None and float(avg_interval) <= 1.0000001:
        run_mode = (
            "- Chế độ chạy duy nhất: detect **mỗi frame** (mọi "
            "`detect_interval_*` = 1) + output gate luôn bật; cơ chế scheduler "
            "thích ứng vẫn có trong `CONFIG` nhưng không bỏ frame nào."
        )
    else:
        run_mode = (
            "- Chế độ chạy duy nhất: scheduler thích ứng (skip-frame) + output gate "
            "luôn bật; frame bị bỏ detect dùng Kalman predict của ByteTrack."
        )
    lines += [
        f"- Ngày tổng hợp: {date.today().isoformat()}",
        f"- Nguồn cấu hình: `CONFIG` trong `step5_integrated_system.py`, đồng bộ "
        f"`{proto['notebook']}` (CELL 2/13 CONFIG)",
        run_mode,
        f"- Operating point: imgsz {proto['img_size']}, conf {proto['conf']}, "
        f"max_det {proto['max_det']}, blur {proto['blur']}",
        f"- Split đánh giá: `{proto['split']}` (person category "
        f"{proto['person_cats']}, {proto['occlusion']}, IoU {proto['iou_match']})",
        f"- Detection calls: {fmt(n_frames, 0)} frame",
        f"- Thiết bị chạy: `{(gate or {}).get('device', 'N/A')}` "
        f"(backend `{(gate or {}).get('backend', 'openvino')}`).",
        f"- Gate: ngưỡng {_fmt_threshold(proto['gate_threshold'])} trên "
        "`gate_score = leak_score + "
        f"{float(cfg.get('gate_risk_scale', 0.5)):g} * frame_risk` "
        f"(mục tiêu chặn {fmt(proto['gate_target_block_rate'], 2)} frame của val).",
        "",
    ]

    # CSV chỉ lưu metric_name/metric_value; policy là cột metadata nên không
    # nằm trong dict vừa đọc — hệ thống chỉ có một policy duy nhất.
    policy = str(system_metrics.get("policy") or "integrated_system")
    lines += [
        "## 1. Chỉ số hệ thống (một chế độ chạy)",
        "",
        f"Policy: `{policy}`. Detection AP được chấm trên các frame thực sự gọi "
        "detect (để scheduler không bị tính là false negative); "
        "`detection_ap50_tracked` chấm trên track output ở mọi frame để đối "
        "chứng phần bị bỏ detect được Kalman lấp ở mức nào. Tracking "
        "(MOTA/MOTP/IDF1/HOTA/IDSW/Frag) chấm trên toàn bộ frame bằng TrackEval.",
    ]
    add_metric_table(
        lines,
        "Chỉ số",
        system_metrics,
        [
            ("Detection", DETECTION_METRICS),
            ("Tracking (TrackEval)", TRACKING_METRICS),
            ("Privacy & utility", PRIVACY_METRICS),
            ("Gate & risk quality", GATE_QUALITY_METRICS),
            ("Scheduler / latency", SYSTEM_METRICS),
            ("Counts", COUNT_METRICS),
        ],
    )

    if gate is not None:
        lines += ["", "## 2. Output gate (Bước 5)", "", "| Metric | Value |", "|---|---|"]
        for key, label in [
            ("total_blocked", "Frames blocked"),
            ("total_released", "Frames released"),
            ("total_unsafe", "Frames with unprotected GT (unsafe)"),
            ("gate_mode", "Gate mode"),
            ("gate_threshold", "Gate threshold (calibrated risk)"),
            ("gate_threshold_source", "Gate threshold source"),
            ("gate_target_block_rate", "Target block rate (val)"),
            ("backend", "Detector backend"),
            ("device", "Device"),
            ("half", "FP16 (quantize=16)"),
            ("fps_stream", "FPS streaming (xử lý frame)"),
            ("fps_stream_with_encode", "FPS streaming (gồm encode video)"),
            ("wall_loop_s", "Wall time xử lý (s)"),
        ]:
            if key in gate:
                value = gate[key]
                if key == "gate_threshold":
                    value = _fmt_threshold(value)
                elif isinstance(value, float):
                    value = f"{value:.10f}".rstrip("0").rstrip(".")
                lines.append(f"| {label} | {value} |")

        if isinstance(meta, dict) and meta:
            lines += ["", "### Protocol của run này", "", "| Field | Value |", "|---|---|"]
            for key in [
                "notebook_source",
                "img_size",
                "conf_thres",
                "iou_thres",
                "max_det",
                "iou_match_threshold",
                "coverage_iou_threshold",
                "containment_threshold",
                "person_categories",
                "exclude_occlusion_2",
                "track_high_thresh",
                "track_low_thresh",
                "new_track_thresh",
                "track_match_thresh",
                "track_buffer",
                "gmc",
                "track_all",
                "det_tiles",
                "tile_overlap",
                "data_root",
                "blur_mode",
                "blur_kernel",
            ]:
                if key in meta:
                    lines.append(f"| `{key}` | {meta[key]} |")
        else:
            lines += [
                "",
                "> Gate log thiếu khối `evaluation_config`: run này được sinh bởi "
                "bản code trước khi đồng bộ `notebook4234ae1bfe.ipynb`.",
            ]

    # Hiệu chuẩn: ưu tiên bản ghi trong gate log (đúng run), fallback config.
    calib = (gate or {}).get("calibration") or cfg.get("calibration") or {}
    if calib:
        threshold = calib.get("threshold", calib.get("gate_threshold"))
        if threshold is None:
            threshold = proto.get("gate_threshold")
        lines += [
            "",
            "### Hiệu chuẩn ngưỡng gate",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Threshold | {_fmt_threshold(threshold)} |",
            f"| Gate score | {calib.get('score_key', 'N/A')} |",
            f"| Method | {calib.get('method', 'N/A')} |",
            f"| Target block rate (val) | {calib.get('target_block_rate', 'N/A')} |",
            f"| Achieved block rate (val) | {calib.get('achieved_block_rate_val', 'N/A')} |",
            f"| Achieved coverage released @ containment90 | "
            f"{calib.get('achieved_coverage_released_containment90', 'N/A')} |",
            f"| Achieved coverage incl. blocked @ containment90 | "
            f"{calib.get('achieved_coverage_incl_blocked_containment90', 'N/A')} |",
            f"| Achieved blocked-GT share | "
            f"{calib.get('achieved_blocked_gt_share', 'N/A')} |",
            f"| Achieved gate recall | {calib.get('achieved_gate_recall', 'N/A')} |",
            f"| Achieved ULFR | {calib.get('achieved_ulfr', 'N/A')} |",
            f"| Achieved effective coverage (legacy) | "
            f"{calib.get('achieved_effective_coverage_val', 'N/A')} |",
            f"| Val frames / sequences | {calib.get('n_val_frames', 'N/A')} / "
            f"{calib.get('n_val_sequences', 'N/A')} |",
            f"| Calibration split | {calib.get('calibration_split', 'N/A')} |",
            f"| Frame risk definition | {calib.get('metric', 'N/A')} |",
            f"| Source | {calib.get('source', 'N/A')} |",
            "",
            f"- {calib.get('note', '')}",
        ]
        curve = calib.get("curve") or []
        if curve:
            lines += [
                "",
                "### Đường cong privacy-utility (từ per-frame records của val)",
                "",
                "| Block rate | Frames blocked | Coverage released (containment90) | "
                "Coverage incl. blocked | Blocked-GT share | Threshold |",
                "|---|---|---|---|---|---|",
            ]
            for row in curve:
                cov_rel = row.get("coverage_released")
                blocked_gt = row.get("blocked_gt_share")
                lines.append(
                    f"| {row['block_rate']:.4f} | {row['frames_blocked']} | "
                    f"{cov_rel:.4f} | {row['effective_coverage']:.4f} | "
                    f"{blocked_gt:.4f} | {row['threshold']:.6f} |"
                    if cov_rel is not None and blocked_gt is not None
                    else f"| {row['block_rate']:.4f} | {row['frames_blocked']} | "
                    f"- | {row['effective_coverage']:.4f} | - | "
                    f"{row['threshold']:.6f} |"
                )
            lines += [
                "",
                "> Block rate = tỷ lệ frame bị chặn đen. **Coverage released** chỉ tính "
                "trên frame đã phát hành (headline); **Coverage incl. blocked** cộng "
                "frame bị chặn như bảo vệ hoàn toàn (thước đo cũ, dễ gây hiểu nhầm). "
                "Đọc cả hai cùng block rate và blocked-GT share.",
            ]
        loo = calib.get("leave_one_sequence_out") or {}
        if loo.get("rows"):
            lines += [
                "",
                "### Kiểm tra leave-one-sequence-out",
                "",
                "| Sequence bị bỏ | Threshold | Effective coverage | Block rate |",
                "|---|---|---|---|",
            ]
            for row in loo["rows"]:
                lines.append(
                    f"| {row['held_out']} | {row['threshold']:.6f} | "
                    f"{row['effective_coverage']:.4f} | {row['block_rate']:.4f} |"
                )
            lines += [
                "",
                f"- Threshold trung bình **{loo['threshold_mean']:.6f}** "
                f"(std {loo['threshold_std']:.6f}, khoảng "
                f"{loo['threshold_min']:.6f}–{loo['threshold_max']:.6f}).",
                "> Val được dùng cho cả hiệu chuẩn và báo cáo, nên bảng này kiểm tra "
                "ngưỡng có bị một sequence chi phối hay không.",
            ]

    if PROFILE_JSON.exists():
        # Bảng này giải thích FPS: cộng các stage trên đường tới hạn (trừ các
        # stage chạy nền) rồi so với 1000 / mean("frame").
        try:
            profile = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            profile = {}
        if profile:
            lines += [
                "",
                "## 3. Phân rã thời gian từng stage (Bước 5, `NCKH_PROFILE=1`)",
                "",
                "| Stage | n | Mean (ms) | p50 (ms) | p95 (ms) | Max (ms) |",
                "|---|---|---|---|---|---|",
            ]
            order = list(PROFILE_LABELS)
            for name in sorted(
                profile, key=lambda k: (order.index(k) if k in order else len(order), k)
            ):
                stats = profile[name]
                label = PROFILE_LABELS.get(name, name)
                lines.append(
                    f"| {label} | {fmt(stats.get('count'), 0)} | "
                    f"{fmt(stats.get('mean'), 2)} | {fmt(stats.get('p50'), 2)} | "
                    f"{fmt(stats.get('p95'), 2)} | {fmt(stats.get('max'), 2)} |"
                )
            frame_stats = profile.get("frame")
            if frame_stats and float(frame_stats.get("mean") or 0) > 0:
                fps = 1000.0 / float(frame_stats["mean"])
                lines += [
                    "",
                    f"- `Cả frame` là thời gian xử lý một frame ở luồng chính; "
                    f"1000 / mean = **{fps:.2f} FPS** (khớp `fps_wall` khi đo trên cùng run).",
                    "- `Chờ prefetch JPEG` và `Chờ SSIM nền` là thời gian luồng chính "
                    "phải đợi; nếu gần 0 nghĩa là các stage nền đã chạy kịp, "
                    "không nằm trên đường tới hạn.",
                    "- `Đẩy frame sang writer` chỉ là chi phí đưa frame vào hàng đợi; "
                    "blur demo + resize + vẽ nhãn + encode mp4 chạy trong thread ghi video.",
                    "- `Detect` chỉ tính trên frame thật sự gọi detect, nên n xấp xỉ "
                    "`detection_calls`; FPS riêng của detector là 1000 / mean của dòng này.",
                ]

    lines += [
        "",
        "## Định nghĩa metric bổ sung",
        "",
        "- **Coverage released @ containment90** (headline) = số GT box được protect box (đã padding) che >= 90% diện tích / số GT box trong các frame ĐÃ PHÁT HÀNH. "
        "Đây là metric khớp với nhãn `unsafe`; `coverage_incl_blocked_*` cộng thêm frame bị chặn (coi là bảo vệ hoàn toàn) và luôn cao hơn.",
        "- **Blocked-GT share** = tỷ lệ GT box nằm trong frame bị chặn. Block rate và blocked-GT share khác nhau vì frame đông người thường bị chặn nhiều hơn.",
        "- **ULFR** = số frame released còn >= 1 GT box chưa được che >= 90% / tổng số frame có GT. "
        "Frame bị gate chặn (đen) không tính là hở, nên ULFR là chỉ số coverage-only; đọc cùng block rate.",
        "- **Unsafe release rate** = số frame released còn GT hở / số frame released có GT.",
        "- **Leak share** = phân rã các frame unsafe được phát hành theo nguyên nhân: zero-track (không có track), risk thấp (<0.25), risk trung bình (>=0.25).",
        "- **Identity leak rate** = số `target_id` không được bảo vệ ở bất kỳ frame nào / tổng `target_id`. "
        "Frame bị chặn được coi là bảo vệ hoàn toàn cho mục đích này.",
        "- **TTP p50/p95** = số frame từ lúc identity xuất hiện đến lần đầu được bảo vệ; identity không bao giờ được bảo vệ là censored.",
        "- **Blur-area ratio** = trung bình tỷ lệ diện tích frame bị blur (union mask, không double-count) trên frame released.",
        "- **Over-protection ratio** = 1 - (diện tích GT được phủ / diện tích đã blur). Cao hơn nghĩa là blur lan ra ngoài người nhiều (mất utility).",
        "- **PSNR background / protected** = PSNR giới hạn theo mask nền và mask vùng che. PSNR vùng che thấp hơn nghĩa là che mạnh hơn; "
        "PSNR nền cao nghĩa là giữ được chi tiết cảnh. Giá trị 100 dB là quy ước khi hai ảnh giống hệt trên mask.",
        "- **Gate precision/recall** coi positive = frame nguy hiểm (nếu phát hành thì hở GT), dự đoán positive = frame bị chặn. "
        "`false_block_rate` = frame an toàn bị chặn; `gate_leak_rate` = frame nguy hiểm được phát hành.",
        "- **Gate-score ROC-AUC/PR-AUC/Brier** đo score mà gate thực sự dùng (`leak_score + scale * frame_risk`) theo nhãn unsafe. "
        "Baseline hằng số (`frame_leak_base_rate`, `frame_leak_brier_baseline`) cho biết score có hơn đoán theo base rate không.",
        "- **Re-ID frame-risk ROC-AUC/PR-AUC/Brier/ECE** chỉ là tham chiếu cho model risk (LogisticRegression, 25 feature), KHÔNG phải metric của gate.",
        "- **`detection_ap50_tracked`** dùng chính box track output + confidence của track, chấm trên "
        "toàn bộ frame. Chênh lệch với `detection_ap50` cho biết mức mất mát do bỏ gọi detect.",
        "",
        "## Protocol",
        "",
        f"- **Split đang báo cáo: `{split_label}`**. Tham số (detector/tracker/gate) "
        "được chọn trên `val` (7 sequence, 2.846 frame); `test-dev` (17 sequence, "
        "6.635 frame) là split held-out dùng để báo cáo và không tham gia chọn tham số.",
        f"- Dữ liệu gốc của run này: `{proto['split']}` (sequence-disjoint). Số liệu "
        "**không so trực tiếp** với bảng test-dev trong literature VisDrone.",
        f"- Cấu hình chuẩn: `CONFIG` trong `step5_integrated_system.py`, đồng bộ `{proto['notebook']}` "
        f"(CELL 2/13 CONFIG): imgsz {proto['img_size']}, conf {proto['conf']}, "
        f"max_det {proto['max_det']}, blur {proto['blur']}.",
        f"- Detector: tiles `{proto['det_tiles']}`, overlap {proto['tile_overlap']} "
        "(`1x1` = nguyên frame, đúng protocol notebook). Tiling đã được probe trên "
        "val và không đem lại lợi ích đủ để đổi operating point.",
        f"- Person metric: VisDrone person category {proto['person_cats']}, {proto['occlusion']}, "
        f"one-to-one matching at IoU >= {proto['iou_match']}.",
        "- YOLO detection metrics use raw detector boxes; tracking metrics use tracker output boxes.",
        "- Detection AP@0.5 and AP@0.5:0.95 computed in COCO style with 101-point interpolation over detections pooled across the frames where the detector was actually called.",
        "- `detection_ap50_tracked` / `detection_ap50_95_tracked` dùng box track output trên toàn bộ frame để đo ảnh hưởng của skip-frame.",
        "- MOTA/MOTP/IDF1/HOTA/IDSW/Frag are computed with TrackEval (official MOTChallenge evaluator) using VisDrone `target_id` per sequence; frame numbers are remapped to a continuous 1..N timeline.",
        "- Tracking IDs are only unique within a sequence, so evaluation is done per sequence and then aggregated by TrackEval `combine_sequences`.",
        "- If TrackEval is unavailable, **all** tracking metrics (MOTA/MOTP/IDF1/HOTA/IDSW/Frag) are reported as `-` (N/A); there is no per-frame fallback.",
        "- The system skips detector calls; detection precision/recall on detected frames and global values over all frames are reported separately.",
        f"- Protection coverage uses IoU >= {proto['coverage_iou']} and containment >= {proto['containment']}, both measured on the padded blur region (`protect_boxes_frame`).",
        "- Headline privacy metric is containment90 on released frames; `coverage_incl_blocked_*` is reported separately because blacked-out frames are counted as fully protected there.",
        "- Low detector confidence is used to prioritize coverage; false positives can cause substantial extra blur and lower precision.",
        f"- ByteTrack thresholds ({proto['tracker']}) match the training notebook "
        f"`{proto['notebook']}`; they are applied unchanged.",
        f"- The output-gate threshold ({_fmt_threshold(proto['gate_threshold'])}) is calibrated at frame level on the sequence-disjoint val split (target block rate {fmt(proto['gate_target_block_rate'], 2)}) and stored in `CONFIG['gate_threshold']`; full calibration output is in `results/gate_calibration.json`. The gate score is `leak_score + scale * frame_risk`, where `leak_score` uses only runtime signals (track dropout, detections not exported as tracks, low-confidence detections, crowd size). Ground truth is never used to decide whether a frame is blocked; ground-truth containment is only used afterwards to label unsafe frames for measurement.",
        "- Tham số được chọn trên `val`; bảng leave-one-sequence-out của val vẫn "
        "được giữ làm bằng chứng độ ổn định của ngưỡng, không dùng để tính metric "
        "test-dev.",
    ]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nĐã lưu báo cáo: {out_md}")
