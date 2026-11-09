"""Tổng hợp metric đánh giá toàn hệ thống từ các file kết quả.

Input:
- results/system_metrics_unified.csv      (Bước 5)
- results/hardware_benchmark.csv          (Bước 6)
- results/output_gate_log.json            (Bước 5)
- results/risk_model_training_summary.json (huấn luyện lại XGBoost)

Output:
- results/system_report.md   (bảng markdown)
- in bảng tóm tắt ra màn hình

Chạy: python report_system_metrics.py
"""

from __future__ import annotations

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

METRICS_CSV = Path("results/system_metrics_unified.csv")
HARDWARE_CSV = Path("results/hardware_benchmark.csv")
GATE_JSON = Path("results/output_gate_log.json")
OUT_MD = Path("results/system_report.md")

# metric_name -> (nhãn hiển thị, số chữ số thập phân)
SYSTEM_METRICS = [
    ("detection_precision", "YOLO detection precision", 4),
    ("detection_recall", "YOLO detection recall", 4),
    ("detection_f1", "YOLO detection F1", 4),
    ("tracking_precision", "ByteTrack precision", 4),
    ("tracking_recall", "ByteTrack recall", 4),
    ("tracking_f1", "ByteTrack F1", 4),
    ("tracking_mota", "MOTA (Multi-Object Tracking Accuracy)", 4),
    ("tracking_motp", "MOTP (Multi-Object Tracking Precision)", 4),
    ("detection_ap50", "AP @ IoU 0.5", 4),
    ("detection_ap50_95", "AP @ IoU 0.5:0.95", 4),
    ("protect_box_coverage_iou50", "Protect-box coverage @ IoU 0.5 (released frames)", 4),
    ("protect_box_containment90", "Protect-box containment @ 90% (released frames)", 4),
    ("effective_privacy_coverage", "Effective privacy coverage (blocked frames blacked out)", 4),
    ("avg_psnr_db", "PSNR (dB)", 2),
    ("avg_ssim", "SSIM", 4),
    ("output_gate_block_rate", "Output gate block rate", 4),
    ("output_gate_release_rate", "Output gate release rate", 4),
    ("detection_calls_per_100_frames", "Detection calls / 100 frames", 2),
    ("average_detection_interval_frames", "Avg detection interval (frames)", 2),
    ("fps_wall", "FPS (wall)", 4),
    ("latency_p50_ms", "Latency p50 (ms)", 2),
    ("latency_p95_ms", "Latency p95 (ms)", 2),
    ("latency_p99_ms", "Latency p99 (ms)", 2),
    ("detection_calls", "Detection calls", 0),
    ("total_gt_boxes", "GT boxes (frames with detection)", 0),
    ("total_gt_boxes_all", "GT boxes (all frames)", 0),
    ("total_pred_boxes", "YOLO predicted boxes", 0),
    ("total_tracking_pred_boxes", "ByteTrack output boxes", 0),
    ("protected_gt_boxes", "Protected GT boxes", 0),
    ("blocked_gt_boxes", "GT boxes on fully blocked frames", 0),
    ("total_protect_gt_boxes", "Evaluated GT boxes (released frames)", 0),
]

HARDWARE_LABELS = {
    "detector": "Detector (YOLO)",
    "detect_blur": "Detect + blur",
    "xgboost": "XGBoost risk model",
    "end_to_end": "End-to-end (detect + risk + blur)",
    "system": "System",
}


def fmt(value: Any, digits: int) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, str):
        return value
    if digits == 0:
        return f"{int(round(float(value))):,}"
    return f"{float(value):.{digits}f}"


def main() -> None:
    lines: list[str] = ["# Báo cáo đánh giá toàn hệ thống", ""]
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Chưa có {METRICS_CSV}. Hãy chạy step5_integrated_system.py trước.")

    df = pd.read_csv(METRICS_CSV)
    metrics = dict(zip(df["metric_name"], df["metric_value"]))

    lines += ["## 1. Chất lượng hệ thống (Bước 5)", "", "| Metric | Value |", "|---|---|"]
    for key, label, digits in SYSTEM_METRICS:
        if key in metrics:
            lines.append(f"| {label} | {fmt(metrics[key], digits)} |")

    if GATE_JSON.exists():
        gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
        lines += ["", "## 2. Output gate (Bước 5)", "", "| Metric | Value |", "|---|---|"]
        for key, label in [
            ("total_blocked", "Frames blocked"),
            ("total_released", "Frames released"),
            ("total_unsafe", "Frames with unprotected GT (unsafe)"),
        ]:
            if key in gate:
                lines.append(f"| {label} | {fmt(gate[key], 0)} |")

    if HARDWARE_CSV.exists():
        hw = pd.read_csv(HARDWARE_CSV)
        lines += ["", "## 3. Hiệu năng phần cứng (Bước 6)", "",
                  "| Component | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | FPS |",
                  "|---|---|---|---|---|---|"]
        for _, row in hw.iterrows():
            name = HARDWARE_LABELS.get(str(row.get("component")), str(row.get("component")))
            lines.append(
                f"| {name} | {fmt(row.get('mean_ms'), 2)} | {fmt(row.get('p50_ms'), 2)} | "
                f"{fmt(row.get('p95_ms'), 2)} | {fmt(row.get('p99_ms'), 2)} | {fmt(row.get('fps'), 2)} |"
            )
        sys_row = hw[hw["component"] == "system"]
        if not sys_row.empty:
            ram = sys_row.iloc[0].get("ram_mb")
            cpu = sys_row.iloc[0].get("cpu_percent")
            lines += ["", f"- RAM: {fmt(ram, 1)} MB", f"- CPU: {fmt(cpu, 1)} %"]

    lines += [
        "",
        "## Protocol",
        "",
        "- Person metric: VisDrone category 1, occlusion level 2 excluded, one-to-one matching per frame at IoU >= 0.5.",
        "- YOLO detection metrics use raw detector boxes; ByteTrack metrics use tracker output boxes.",
        "- Detection AP@0.5 and AP@0.5:0.95 computed in COCO style with 101-point interpolation over detections pooled across all frames.",
        "- MOTA/MOTP computed with per-frame one-to-one matching at IoU >= 0.5; IDSW is 0 because VisDrone VID test-dev does not provide ground-truth track IDs.",
        "- Risk model evaluation is available only for sequences that overlap between risk_dataset.csv and VisDrone test-dev (11 of 17 sequences); risk metrics are not reported end-to-end.",
        "- Protection IoU coverage is conditional on released frames only; blocked frames are blacked out and are reported separately, not counted as IoU-covered boxes.",
        "- Low detector confidence is used to prioritize coverage; false positives can cause substantial extra blur and lower precision.",
        "- Thresholds were selected after a sweep on 715 labeled frames from these same 17 sequences; aggregate metrics are exploratory, not an independent holdout result.",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nĐã lưu báo cáo: {OUT_MD}")


if __name__ == "__main__":
    main()