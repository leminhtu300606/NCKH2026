"""Tổng hợp metric đánh giá toàn hệ thống từ các file kết quả.

Input:
- results/system_metrics_system.csv        (Bước 5, chế độ deployable: skip-frame)
- results/system_metrics_eval.csv          (Bước 5, chế độ eval: detect mọi frame)
- results/system_metrics_unified.csv       (file tổng hợp chính, fallback)
- results/hardware_benchmark.csv           (Bước 6)
- results/output_gate_log.json             (Bước 5)

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

UNIFIED_CSV = Path("results/system_metrics_unified.csv")
SYSTEM_CSV = Path("results/system_metrics_system.csv")
EVAL_CSV = Path("results/system_metrics_eval.csv")
HARDWARE_CSV = Path("results/hardware_benchmark.csv")
GATE_JSON = Path("results/output_gate_log.json")
GATE_THRESHOLD_PATH = Path("config/gate_threshold_val.json")
OUT_MD = Path("results/system_report.md")

# metric_name -> (nhãn hiển thị, số chữ số thập phân)
DETECTION_METRICS = [
    ("detection_precision", "YOLO precision (frames có detect)", 4),
    ("detection_recall", "YOLO recall (frames có detect)", 4),
    ("detection_f1", "YOLO F1 (frames có detect)", 4),
    ("detection_precision_global", "YOLO precision (toàn cục, mọi frame)", 4),
    ("detection_recall_global", "YOLO recall (toàn cục, mọi frame)", 4),
    ("detection_f1_global", "YOLO F1 (toàn cục, mọi frame)", 4),
    ("detection_ap50", "AP @ IoU 0.5", 4),
    ("detection_ap50_95", "AP @ IoU 0.5:0.95", 4),
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
    ("protect_box_coverage_iou50", "Protect-box coverage @ IoU 0.5 (released frames)", 4),
    ("protect_box_containment90", "Protect-box containment @ 90% (released frames)", 4),
    ("effective_privacy_coverage", "Effective privacy coverage (blocked frames blacked out)", 4),
    ("ulfr", "ULFR (unsafe leak frame rate, all GT frames)", 4),
    ("unsafe_release_rate", "Unsafe release rate (released GT frames)", 4),
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
    ("risk_roc_auc", "Frame-risk ROC-AUC", 4),
    ("risk_pr_auc", "Frame-risk PR-AUC", 4),
    ("risk_brier", "Frame-risk Brier", 4),
    ("risk_ece", "Frame-risk ECE", 4),
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


def load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "metric_name" not in df or "metric_value" not in df:
        return None
    return dict(zip(df["metric_name"], df["metric_value"]))


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


def main() -> None:
    from datetime import date

    lines: list[str] = ["# Báo cáo đánh giá toàn hệ thống", ""]

    system_metrics = load_metrics(SYSTEM_CSV)
    eval_metrics = load_metrics(EVAL_CSV)
    unified_metrics = load_metrics(UNIFIED_CSV)
    if system_metrics is None:
        system_metrics = unified_metrics
    if system_metrics is None and eval_metrics is None:
        raise FileNotFoundError(
            "Chưa có file metric nào trong results/. Hãy chạy step5_integrated_system.py trước."
        )

    n_eval_frames = None
    if eval_metrics is not None:
        n_eval_frames = eval_metrics.get("detection_calls")
    lines += [
        f"- Ngày tổng hợp: {date.today().isoformat()}",
        "- Split đánh giá: VisDrone2019-VID-test-dev (person-only, IoU 0.5, "
        "occlusion 2 bị loại)",
        f"- Test-dev: 17 sequence, {fmt(n_eval_frames, 0)} frame"
        if n_eval_frames is not None
        else "- Test-dev: 17 sequence",
        "- Val hiệu chuẩn: VisDrone2019-VID-val, 7 sequence, 2.846 frame "
        "(sequence-disjoint, không dùng để chọn ngưỡng test-dev)",
        "- Phần cứng: Intel i7-1165G7 (4C/8T), 16 GB RAM, CPU-only",
        "- Số liệu chính: chế độ eval (detect 100% frame) ở mục 1; mục 2 là cấu "
        "hình triển khai thật (skip-frame + output gate).",
        "",
    ]

    def policy_note(metrics: dict[str, Any] | None, default: str) -> str:
        if not metrics:
            return default
        return str(metrics.get("policy", metrics.get("_policy", default)))

    if eval_metrics is not None:
        eval_table = dict(eval_metrics)
        eval_table["output_gate_block_rate"] = "N/A (gate bỏ qua)"
        eval_table["output_gate_release_rate"] = "N/A (gate bỏ qua)"
        # Gate không bật ở chế độ eval nên confusion matrix vô nghĩa; các metric
        # calibration của frame-risk vẫn giữ vì vẫn đo được trên nhãn unsafe.
        for key in list(eval_table):
            if key.startswith("gate_") or key == "false_block_rate":
                eval_table[key] = "N/A (gate bỏ qua)"
        lines += [
            "## 1. Chỉ số đánh giá chuẩn (detect mọi frame)",
            "",
            "Policy: `integrated_deployable`, detect 100% frame. Đây là bộ số chính "
            "dùng trong báo cáo và là protocol chuẩn để so sánh MOTA/IDF1/HOTA với "
            "literature; không phải cấu hình tiết kiệm năng lượng. Chế độ này bỏ qua "
            "output gate (mọi frame được phát hành) nên block/release rate và "
            "confusion matrix của gate được ghi `N/A`, không phải 0.",
        ]
        add_metric_table(
            lines,
            "Chỉ số",
            eval_table,
            [
                ("Detection", DETECTION_METRICS),
                ("Tracking (TrackEval)", TRACKING_METRICS),
                ("Privacy & utility", PRIVACY_METRICS),
                ("Gate & risk quality", GATE_QUALITY_METRICS),
                ("Scheduler / latency", SYSTEM_METRICS),
                ("Counts", COUNT_METRICS),
            ],
        )

    if system_metrics is not None:
        policy = next(
            (str(v) for k, v in system_metrics.items() if k == "policy"),
            "integrated_deployable",
        )
        lines += [
            "",
            "## 2. Cấu hình triển khai (skip-frame + lan truyền track)",
            "",
            f"Policy: `{policy}`. Detect theo lịch thích ứng; frame bị bỏ qua dùng "
            "Kalman predict của ByteTrack (không gọi update với box rỗng). Số liệu "
            "ở mục này phản ánh hệ thống thật (tiết kiệm detect + có output gate), "
            "không dùng để so sánh trực tiếp với literature.",
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

    gate = None
    if GATE_JSON.exists():
        gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
        lines += ["", "## 3. Output gate (Bước 5)", "", "| Metric | Value |", "|---|---|"]
        for key, label in [
            ("total_blocked", "Frames blocked"),
            ("total_released", "Frames released"),
            ("total_unsafe", "Frames with unprotected GT (unsafe)"),
            ("gate_mode", "Gate mode"),
            ("track_eval", "Track eval mode"),
            ("gate_threshold", "Gate threshold (calibrated risk)"),
        ]:
            if key in gate:
                value = gate[key]
                if isinstance(value, float):
                    value = f"{value:.10f}".rstrip("0").rstrip(".")
                lines.append(f"| {label} | {value} |")

    if GATE_THRESHOLD_PATH.exists():
        calib = json.loads(GATE_THRESHOLD_PATH.read_text(encoding="utf-8"))
        lines += [
            "",
            "### Hiệu chuẩn ngưỡng gate",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Threshold | {calib.get('threshold')} |",
            f"| Target block rate | {calib.get('target_block_rate')} |",
            f"| Achieved block rate (val) | {calib.get('achieved_block_rate_val')} |",
            f"| Val frames / sequences | {calib.get('n_val_frames')} / {calib.get('n_val_sequences')} |",
            f"| Calibration split | {calib.get('calibration_split')} |",
            f"| Frame risk definition | {calib.get('metric')} |",
            "",
            f"- {calib.get('note', '')}",
        ]

    if HARDWARE_CSV.exists():
        hw = pd.read_csv(HARDWARE_CSV)
        lines += ["", "## 4. Hiệu năng phần cứng (Bước 6)", "",
                  "| Component | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | FPS | Deadline miss (%) |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, row in hw.iterrows():
            name = HARDWARE_LABELS.get(str(row.get("component")), str(row.get("component")))
            lines.append(
                f"| {name} | {fmt(row.get('mean_ms'), 2)} | {fmt(row.get('p50_ms'), 2)} | "
                f"{fmt(row.get('p95_ms'), 2)} | {fmt(row.get('p99_ms'), 2)} | "
                f"{fmt(row.get('max_ms'), 2)} | {fmt(row.get('fps'), 2)} | "
                f"{fmt(row.get('pct_over_deadline'), 2)} |"
            )
        sys_row = hw[hw["component"] == "system"]
        if not sys_row.empty:
            ram = sys_row.iloc[0].get("ram_mb")
            cpu = sys_row.iloc[0].get("cpu_percent")
            lines += ["", f"- RAM: {fmt(ram, 1)} MB", f"- CPU: {fmt(cpu, 1)} %"]

    lines += [
        "",
        "## Định nghĩa metric bổ sung",
        "",
        "- **ULFR** = số frame released còn >= 1 GT box chưa được che >= 90% / tổng số frame có GT. "
        "Frame bị gate chặn (đen) không tính là hở, nên ULFR là chỉ số coverage-only; đọc cùng block rate.",
        "- **Unsafe release rate** = số frame released còn GT hở / số frame released có GT.",
        "- **Identity leak rate** = số `target_id` không được bảo vệ ở bất kỳ frame nào / tổng `target_id`. "
        "Frame bị chặn được coi là bảo vệ hoàn toàn cho mục đích này.",
        "- **TTP p50/p95** = số frame từ lúc identity xuất hiện đến lần đầu được bảo vệ; identity không bao giờ được bảo vệ là censored.",
        "- **Blur-area ratio** = trung bình tỷ lệ diện tích frame bị blur (union mask, không double-count) trên frame released.",
        "- **Over-protection ratio** = 1 - (diện tích GT được phủ / diện tích đã blur). Cao hơn nghĩa là blur lan ra ngoài người nhiều (mất utility).",
        "- **PSNR background / protected** = PSNR giới hạn theo mask nền và mask vùng che. PSNR vùng che thấp hơn nghĩa là che mạnh hơn; "
        "PSNR nền cao nghĩa là giữ được chi tiết cảnh. Giá trị 100 dB là quy ước khi hai ảnh giống hệt trên mask.",
        "- **Gate precision/recall** coi positive = frame nguy hiểm (nếu phát hành thì hở GT), dự đoán positive = frame bị chặn. "
        "`false_block_rate` = frame an toàn bị chặn; `gate_leak_rate` = frame nguy hiểm được phát hành.",
        "- **Frame-risk ROC-AUC/PR-AUC/Brier/ECE** đo calibration của frame risk theo nhãn unsafe ở trên, "
        "KHÔNG phải metric chất lượng model XGBoost trên `risk_dataset.csv` (metric đó nằm trong artifact risk model).",
        "",
        "## Protocol",
        "",
        "- Person metric: VisDrone category 1, occlusion level 2 excluded, one-to-one matching at IoU >= 0.5.",
        "- YOLO detection metrics use raw detector boxes; tracking metrics use tracker output boxes.",
        "- Detection AP@0.5 and AP@0.5:0.95 computed in COCO style with 101-point interpolation over detections pooled across all evaluated frames (all frames in eval mode, detector-call frames in deployable mode).",
        "- MOTA/MOTP/IDF1/HOTA/IDSW/Frag are computed with TrackEval (official MOTChallenge evaluator) using VisDrone `target_id` per sequence; frame numbers are remapped to a continuous 1..N timeline.",
        "- Tracking IDs are only unique within a sequence, so evaluation is done per sequence and then aggregated by TrackEval `combine_sequences`.",
        "- If TrackEval is unavailable, tracking ID metrics are reported as `-` (N/A), not as zero.",
        "- In deployable mode, detection AP is computed only over frames where the detector was called; it is therefore not directly comparable to the eval-mode AP over all frames (which pools detections across 6,635 frames).",
        "- The deployable mode skips detector calls; detection precision/recall on detected frames and global values over all frames are reported separately.",
        "- Protection IoU coverage is conditional on released frames only; blocked frames are blacked out and are reported separately, not counted as IoU-covered boxes. ULFR and identity leak treat blocked frames as fully protected.",
        "- Low detector confidence is used to prioritize coverage; false positives can cause substantial extra blur and lower precision.",
        "- ByteTrack thresholds (high=0.30, low=0.05, new=0.25, match=0.95, buffer=30) were selected on a sequence-disjoint validation split (3 sequences, 120 frames each) and then applied unchanged to test-dev.",
        "- The output-gate threshold is calibrated at frame level on the sequence-disjoint val split (7 sequences, 2,846 frames) from `config/gate_threshold_val.json`; ground truth is never used to decide whether a frame is blocked. Ground-truth containment is only used afterwards to label unsafe frames for measurement.",
        "- The isotonic risk calibrator saturates at 0.35408 (p90 = p100), so the gate fires on 14.55% of val frames instead of the 5% target; this is reported as a limitation rather than tuned on test-dev.",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nĐã lưu báo cáo: {OUT_MD}")


if __name__ == "__main__":
    main()
