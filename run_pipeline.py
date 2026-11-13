"""Entrypoint duy nhất của dự án: một pipeline, một cấu hình, một bộ output.

Các lệnh:
    python run_pipeline.py all              # evaluate -> report (val)
    python run_pipeline.py evaluate         # Bước 5: chạy hệ thống trên val
    python run_pipeline.py evaluate --split test-dev   # báo cáo held-out
    python run_pipeline.py report           # tổng hợp results/system_report.md
    python run_pipeline.py report --split test-dev
    python run_pipeline.py calibrate        # hiệu chuẩn ngưỡng gate trên val
    python run_pipeline.py calibrate --target-coverage 0.80
    python run_pipeline.py verify           # kiểm tra CONFIG ↔ gate log ↔ metrics

Nguyên tắc split:
    - val      : chọn tham số (7 sequence, 2.846 frame).
    - test-dev : báo cáo held-out (17 sequence, 6.635 frame), không chọn tham số.

Cấu hình duy nhất: CONFIG trong step5_integrated_system.py.
Output:
    results/system_metrics.csv / results/system_metrics_testdev.csv
    results/output_gate_log.json / results/output_gate_log_testdev.json
    results/val_frame_records.json / results/testdev_frame_records.json
    results/gate_calibration.json
    results/system_report.md / results/system_report_testdev.md
    outputs/protected_video.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Đường dẫn output riêng cho split held-out, không ghi đè kết quả val.
TESTDEV_ENV = {
    "NCKH_DATA_ROOT": "VisDrone2019-VID-test-dev",
    "NCKH_OUT_METRICS": "results/system_metrics_testdev.csv",
    "NCKH_OUT_GATE_LOG": "results/output_gate_log_testdev.json",
    "NCKH_OUT_FRAME_RECORDS": "results/testdev_frame_records.json",
    "NCKH_WRITE_VIDEO": "0",
}

# Headline privacy criterion: containment >= 90% (khớp nhãn `unsafe`).
# Giữ fallback IoU cho records sinh bởi bản code cũ.
PROTECTED_KEYS = ("gt_contained", "gt_iou_protected")
# Các mức block rate dùng cho bảng đường cong privacy-utility trong báo cáo.
CURVE_BLOCK_RATES = (
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.50, 0.60, 0.75, 1.0,
)
DEFAULT_TARGET_BLOCK_RATE = 0.40


def _banner(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(title)
    print(f"{'=' * 64}\n", flush=True)


def _extract_split(argv: list[str]) -> tuple[str, list[str]]:
    """Tách `--split` khỏi argv để không chuyển tiếp xuống step5/report."""
    split = "val"
    if not argv or argv[0] not in {"evaluate", "report"}:
        return split, list(argv)
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--split":
            if i + 1 >= len(argv):
                raise SystemExit("--split cần giá trị 'val' hoặc 'test-dev'")
            split = argv[i + 1]
            i += 2
            continue
        if argv[i].startswith("--split="):
            split = argv[i].split("=", 1)[1]
            i += 1
            continue
        cleaned.append(argv[i])
        i += 1
    if split not in {"val", "test-dev"}:
        raise SystemExit(f"--split không hợp lệ: {split}")
    return split, cleaned


def _report_extra(split: str) -> list[str]:
    if split != "test-dev":
        return []
    return [
        "--metrics", "results/system_metrics_testdev.csv",
        "--gate-log", "results/output_gate_log_testdev.json",
        "--out", "results/system_report_testdev.md",
        "--split-label", "test-dev (held-out)",
    ]


def run_evaluate(split: str, extra: list[str]) -> int:
    """Chạy Bước 5 trên split đã chọn (mặc định val)."""
    from step5_integrated_system import run as run_system

    if split == "test-dev":
        os.environ.update(TESTDEV_ENV)
    _banner("Bước 5: Hệ thống tích hợp trên " + ("test-dev (held-out)" if split == "test-dev" else "val"))
    try:
        run_system(extra)
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def run_report(split: str, extra: list[str]) -> int:
    """Tổng hợp báo cáo markdown từ kết quả evaluate."""
    from report_system_metrics import build_report
    from step5_integrated_system import CONFIG

    cfg = dict(CONFIG)
    args = list(extra)
    if split == "test-dev":
        # Report so protocol với CONFIG của run tương ứng, không phải val.
        cfg["data_root"] = TESTDEV_ENV["NCKH_DATA_ROOT"]
        args = _report_extra(split) + args
    _banner("Tổng hợp báo cáo")
    try:
        build_report(args, cfg)
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


# ===== Hiệu chuẩn ngưỡng gate trên per-frame records =====

def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"Chưa có per-frame records: {path}. Chạy `python run_pipeline.py evaluate` trước."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Không đọc được {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise SystemExit(f"Per-frame records rỗng hoặc sai định dạng: {path}")
    return data


def _protected_counts(records: list[dict[str, Any]]) -> tuple[list[int], str]:
    """Số GT được che đủ trong từng frame, theo định nghĩa coverage của report."""
    for key in PROTECTED_KEYS:
        if all(key in r for r in records):
            return (
                [sum(1 for v in r.get(key, []) if v) for r in records],
                key,
            )
    raise SystemExit(
        "Per-frame records thiếu cả "
        f"{PROTECTED_KEYS}; chạy lại evaluate bằng bản code hiện tại."
    )


def _gate_scores(records: list[dict[str, Any]]) -> tuple[list[float], str]:
    """Tính lại `gate_score` từ per-frame components bằng CONFIG hiện hành.

    Nhờ vậy đổi trọng số leak trong CONFIG rồi chạy `calibrate` là đủ, không
    cần chạy lại `evaluate` chỉ để sinh lại score. Records cũ không có
    component thì fallback `gate_score`/`risk` đã ghi.
    """
    from step5_integrated_system import CONFIG, compute_leak_score

    required = ("n_tracks", "n_det", "n_low_conf", "ewma_det", "risk")
    if all(all(k in r for k in required) for r in records):
        weights = CONFIG.get("gate_leak_weights", {}) or {}
        crowd_ref = float(CONFIG.get("gate_crowd_ref", 168.0))
        no_track_ref = float(CONFIG.get("gate_no_track_ref", 0.0))
        scale = float(CONFIG.get("gate_risk_scale", 0.5))
        scores: list[float] = []
        for r in records:
            leak = compute_leak_score(
                n_tracks=int(r.get("n_tracks", 0)),
                n_det=int(r.get("n_det", 0)),
                n_low_conf=int(r.get("n_low_conf", 0)),
                ewma_det=float(r.get("ewma_det", 0.0) or 0.0),
                prev_n_tracks=int(r.get("prev_n_tracks", 0) or 0),
                weights=weights,
                crowd_ref=crowd_ref,
                no_track_ref=no_track_ref,
            )
            scores.append(float(leak + scale * float(r.get("risk", 0.0))))
        return scores, "gate_score (recomputed from CONFIG)"
    if all("gate_score" in r for r in records):
        return [float(r.get("gate_score", 0.0)) for r in records], "gate_score"
    return [float(r.get("risk", 0.0)) for r in records], "risk"


def _sweep_points(records: list[dict[str, Any]]) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Duyệt mọi ngưỡng duy nhất (giảm dần) và tính coverage tương ứng.

    Dùng `gate_score` (tính lại từ CONFIG nếu có component) để hiệu chuẩn đúng
    score mà gate thực sự dùng; fallback `risk` cho records cũ.
    """
    risks, score_key = _gate_scores(records)
    n_gt = [int(r.get("n_gt", 0)) for r in records]
    protected, protected_key = _protected_counts(records)
    order = sorted(range(len(records)), key=lambda i: -risks[i])
    total_gt = float(sum(n_gt))
    total_prot = float(sum(protected))

    cum_gt: list[int] = []
    cum_prot: list[int] = []
    cg = cp = 0
    for i in order:
        cg += n_gt[i]
        cp += protected[i]
        cum_gt.append(cg)
        cum_prot.append(cp)

    def _point(threshold: float, blocked: int) -> dict[str, float]:
        blocked_gt = float(cum_gt[blocked - 1]) if blocked > 0 else 0.0
        released_gt = max(total_gt - blocked_gt, 0.0)
        protected_released = (
            max(total_prot - float(cum_prot[blocked - 1]), 0.0)
            if blocked > 0
            else total_prot
        )
        return {
            "threshold": threshold,
            "blocked": float(blocked),
            "block_rate": blocked / max(len(records), 1),
            "effective_coverage": (blocked_gt + protected_released)
            / max(total_gt, 1e-9),
            "coverage_released": protected_released / max(released_gt, 1e-9),
            "blocked_gt_share": blocked_gt / max(total_gt, 1e-9),
        }

    n = len(records)
    points: list[dict[str, float]] = []
    # Ứng viên "không chặn frame nào": ngưỡng cao hơn max risk.
    max_risk = max(risks) if risks else 0.0
    points.append(_point(max_risk + 1e-12, 0))

    i = 0
    while i < n:
        j = i
        threshold = risks[order[i]]
        while j < n and risks[order[j]] == threshold:
            j += 1
        points.append(_point(threshold, j))
        i = j

    stats = {
        "n_frames": n,
        "n_frames_with_gt": sum(1 for v in n_gt if v > 0),
        "n_gt_boxes": int(total_gt),
        "total_protected": int(total_prot),
        "protected_key": protected_key,
        "score_key": score_key,
        "sequences": sorted({str(r.get("sequence", "")) for r in records if r.get("sequence")}),
    }
    return points, stats


def _select_point(
    points: list[dict[str, float]],
    target_block_rate: float | None = None,
    target_coverage: float | None = None,
) -> dict[str, float]:
    """Chọn điểm vận hành theo block rate hoặc coverage mục tiêu."""
    if target_coverage is not None:
        feasible = [p for p in points if p["effective_coverage"] + 1e-12 >= target_coverage]
        if not feasible:
            return points[-1]  # không đạt được coverage mục tiêu -> chặn tất cả
        return min(feasible, key=lambda p: p["block_rate"])
    feasible = [p for p in points if p["block_rate"] <= target_block_rate + 1e-12]
    if not feasible:
        return points[0]  # ngưỡng quá thấp -> không chặn frame nào
    return max(feasible, key=lambda p: p["block_rate"])


def _coverage_curve(points: list[dict[str, float]]) -> list[dict[str, float]]:
    curve: list[dict[str, float]] = []
    for rate in CURVE_BLOCK_RATES:
        point = _select_point(points, target_block_rate=rate)
        curve.append({
            "block_rate": round(point["block_rate"], 6),
            "frames_blocked": int(point["blocked"]),
            "effective_coverage": round(point["effective_coverage"], 6),
            "coverage_released": round(point.get("coverage_released", float("nan")), 6),
            "blocked_gt_share": round(point.get("blocked_gt_share", float("nan")), 6),
            "threshold": round(point["threshold"], 6),
        })
    return curve


def _point_metrics(
    records: list[dict[str, Any]], threshold: float
) -> dict[str, float]:
    """Metric đầy đủ tại một ngưỡng gate (tính lại từ per-frame records)."""
    scores, _ = _gate_scores(records)
    blocked = [float(s) >= float(threshold) for s in scores]
    released = [not b for b in blocked]
    total_gt = float(sum(int(r.get("n_gt", 0)) for r in records))
    released_gt = float(
        sum(int(r.get("n_gt", 0)) for r, rel in zip(records, released) if rel)
    )
    blocked_gt = max(0.0, total_gt - released_gt)
    prot_cont = float(
        sum(
            sum(1 for v in r.get("gt_contained", []) if v)
            for r, rel in zip(records, released)
            if rel
        )
    )
    prot_iou = float(
        sum(
            sum(1 for v in r.get("gt_iou_protected", []) if v)
            for r, rel in zip(records, released)
            if rel
        )
    )
    y = [bool(r.get("unsafe")) for r in records]
    tp = float(sum(1 for yy, bb in zip(y, blocked) if yy and bb))
    fp = float(sum(1 for yy, bb in zip(y, blocked) if not yy and bb))
    fn = float(sum(1 for yy, bb in zip(y, blocked) if yy and not bb))
    tn = float(sum(1 for yy, bb in zip(y, blocked) if not yy and not bb))
    unsafe_released = float(
        sum(
            1
            for r, yy, rel in zip(records, y, released)
            if rel and yy and int(r.get("n_gt", 0)) > 0
        )
    )
    frames_with_gt = float(sum(1 for r in records if int(r.get("n_gt", 0)) > 0))
    released_with_gt = float(
        sum(1 for r, rel in zip(records, released) if rel and int(r.get("n_gt", 0)) > 0)
    )
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "block_rate": float(sum(blocked)) / max(len(records), 1),
        "coverage_released_containment90": prot_cont / max(released_gt, 1e-9),
        "coverage_incl_blocked_containment90": (prot_cont + blocked_gt)
        / max(total_gt, 1e-9),
        "coverage_released_iou50": prot_iou / max(released_gt, 1e-9),
        "coverage_incl_blocked_iou50": (prot_iou + blocked_gt)
        / max(total_gt, 1e-9),
        "blocked_gt_share": blocked_gt / max(total_gt, 1e-9),
        "gate_precision": precision,
        "gate_recall": recall,
        "gate_f1": f1,
        "false_block_rate": fp / max(fp + tn, 1.0),
        "gate_leak_rate": fn / max(fn + tp, 1.0),
        "ulfr": unsafe_released / max(frames_with_gt, 1.0),
        "unsafe_release_rate": unsafe_released / max(released_with_gt, 1.0),
    }


def _leave_one_sequence_out(
    records: list[dict[str, Any]],
    target_block_rate: float | None,
    target_coverage: float | None,
) -> dict[str, Any]:
    sequences = sorted({str(r.get("sequence", "")) for r in records if r.get("sequence")})
    rows: list[dict[str, Any]] = []
    for held_out in sequences:
        subset = [r for r in records if str(r.get("sequence", "")) != held_out]
        if not subset:
            continue
        points, _ = _sweep_points(subset)
        point = _select_point(points, target_block_rate, target_coverage)
        rows.append({
            "held_out": held_out,
            "threshold": round(point["threshold"], 6),
            "effective_coverage": round(point["effective_coverage"], 6),
            "block_rate": round(point["block_rate"], 6),
        })
    thresholds = [r["threshold"] for r in rows]
    if not thresholds:
        return {"rows": []}
    mean = sum(thresholds) / len(thresholds)
    var = sum((t - mean) ** 2 for t in thresholds) / len(thresholds)
    return {
        "rows": rows,
        "threshold_mean": mean,
        "threshold_std": var ** 0.5,
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
    }


def _apply_threshold(threshold: float) -> None:
    """Ghi gate_threshold mới vào CONFIG trong step5_integrated_system.py."""
    path = ROOT / "step5_integrated_system.py"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r'("gate_threshold":\s*)[0-9.eE+\-]+')
    new_text, count = pattern.subn(lambda m: m.group(1) + repr(float(threshold)), text)
    if count == 0:
        raise SystemExit(
            "Không tìm thấy khóa \"gate_threshold\" trong step5_integrated_system.py."
        )
    path.write_text(new_text, encoding="utf-8")
    print(f"Đã cập nhật {count} khóa \"gate_threshold\" trong CONFIG.")


def run_calibration(extra: list[str]) -> int:
    from step5_integrated_system import CONFIG, project_path

    parser = argparse.ArgumentParser(
        prog="run_pipeline.py calibrate",
        description="Hiệu chuẩn ngưỡng gate trên per-frame records của val.",
    )
    parser.add_argument(
        "--records", default="results/val_frame_records.json",
        help="Per-frame records (mặc định results/val_frame_records.json).",
    )
    parser.add_argument(
        "--target-block-rate", type=float, default=None,
        help=f"Tỉ lệ frame bị chặn tối đa (mặc định {DEFAULT_TARGET_BLOCK_RATE}).",
    )
    parser.add_argument(
        "--target-coverage", type=float, default=None,
        help="Coverage tối thiểu cần đạt; thay cho --target-block-rate.",
    )
    parser.add_argument(
        "--out", default="results/gate_calibration.json",
        help="File JSON lưu kết quả hiệu chuẩn.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Ghi ngưỡng mới vào CONFIG trong step5_integrated_system.py.",
    )
    args = parser.parse_args(extra)

    if args.target_coverage is not None and args.target_block_rate is not None:
        parser.error("chỉ chọn một trong --target-coverage / --target-block-rate")
    target_coverage = args.target_coverage
    target_block_rate = args.target_block_rate
    if target_coverage is None and target_block_rate is None:
        target_block_rate = float(
            CONFIG.get("gate_target_block_rate", DEFAULT_TARGET_BLOCK_RATE)
        )
    if target_coverage is not None and not 0.0 <= target_coverage <= 1.0:
        parser.error("--target-coverage phải nằm trong [0, 1]")
    if target_block_rate is not None and not 0.0 <= target_block_rate <= 1.0:
        parser.error("--target-block-rate phải nằm trong [0, 1]")

    records_path = project_path(args.records)
    records = _load_records(records_path)
    points, stats = _sweep_points(records)
    chosen = _select_point(points, target_block_rate, target_coverage)
    loo = _leave_one_sequence_out(records, target_block_rate, target_coverage)

    chosen_metrics = _point_metrics(records, chosen["threshold"])
    result = {
        "threshold": chosen["threshold"],
        "gate_threshold": chosen["threshold"],
        "score_key": stats["score_key"],
        "method": "target_coverage" if target_coverage is not None else "target_block_rate",
        "target_block_rate": target_block_rate,
        "target_coverage": target_coverage,
        "achieved_block_rate_val": chosen["block_rate"],
        "achieved_effective_coverage_val": chosen["effective_coverage"],
        "achieved_coverage_released_containment90": chosen_metrics[
            "coverage_released_containment90"
        ],
        "achieved_coverage_incl_blocked_containment90": chosen_metrics[
            "coverage_incl_blocked_containment90"
        ],
        "achieved_blocked_gt_share": chosen_metrics["blocked_gt_share"],
        "achieved_gate_recall": chosen_metrics["gate_recall"],
        "achieved_gate_precision": chosen_metrics["gate_precision"],
        "achieved_ulfr": chosen_metrics["ulfr"],
        "achieved_unsafe_release_rate": chosen_metrics["unsafe_release_rate"],
        "achieved": chosen_metrics,
        "n_val_frames": stats["n_frames"],
        "n_val_frames_with_gt": stats["n_frames_with_gt"],
        "n_val_gt_boxes": stats["n_gt_boxes"],
        "n_val_sequences": len(stats["sequences"]),
        "protected_key": stats["protected_key"],
        "records_file": args.records.replace("\\", "/"),
        "curve": _coverage_curve(points),
        "leave_one_sequence_out": loo,
    }

    out_path = project_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _banner("Hiệu chuẩn ngưỡng gate")
    print(f"Records     : {records_path} ({stats['n_frames']} frame, "
          f"{len(stats['sequences'])} sequence)")
    if target_coverage is not None:
        print(f"Mục tiêu    : coverage >= {target_coverage:.4f}")
    else:
        print(f"Mục tiêu    : block rate <= {target_block_rate:.4f}")
    print(f"Ngưỡng chọn : {chosen['threshold']:.6f} (score = {stats['score_key']})")
    print(f"Block rate  : {chosen['block_rate']:.4f} "
          f"({int(chosen['blocked'])} frame)")
    print(f"Coverage released @ containment90 : "
          f"{chosen_metrics['coverage_released_containment90']:.4f}")
    print(f"Coverage incl. blocked @ containment90 : "
          f"{chosen_metrics['coverage_incl_blocked_containment90']:.4f}")
    print(f"Gate precision / recall : "
          f"{chosen_metrics['gate_precision']:.4f} / "
          f"{chosen_metrics['gate_recall']:.4f}")
    print(f"ULFR / unsafe release   : "
          f"{chosen_metrics['ulfr']:.4f} / "
          f"{chosen_metrics['unsafe_release_rate']:.4f}")
    print(f"Đã lưu      : {out_path}")

    if args.apply:
        _apply_threshold(chosen["threshold"])
    else:
        print(
            "Chưa ghi vào CONFIG. Thêm `--apply` để cập nhật gate_threshold, "
            "sau đó chạy lại `evaluate`."
        )
    return 0


# ===== Kiểm tra nhất quán CONFIG ↔ gate log ↔ metrics =====

def verify_consistency() -> int:
    """Kiểm tra CONFIG ↔ gate log ↔ metrics trước khi trích dẫn số."""
    from report_system_metrics import protocol_mismatches
    from step5_integrated_system import CONFIG, project_path

    cfg = CONFIG
    metrics_path = project_path(str(cfg.get("out_metrics", "results/system_metrics.csv")))
    gate_path = project_path(str(cfg.get("out_gate_log", "results/output_gate_log.json")))
    issues: list[str] = []

    if not metrics_path.exists():
        issues.append(f"Thiếu file metric: {metrics_path}")
    if not gate_path.exists():
        issues.append(f"Thiếu gate log: {gate_path}")
    else:
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"Gate log hỏng: {exc}")
            gate = {}
        issues.extend(protocol_mismatches(gate.get("evaluation_config"), cfg))

        # Ngưỡng gate trong log phải khớp CONFIG.
        log_threshold = gate.get("gate_threshold")
        cfg_threshold = cfg.get("gate_threshold")
        if log_threshold is not None and cfg_threshold is not None:
            if abs(float(log_threshold) - float(cfg_threshold)) > 1e-9:
                issues.append(
                    f"gate_threshold lệch: gate log {log_threshold} vs "
                    f"CONFIG {cfg_threshold} — chạy lại `evaluate`."
                )

        # Nguồn hiệu chuẩn duy nhất là results/gate_calibration.json (CONFIG chỉ
        # giữ ngưỡng đang dùng). Gate log phải khớp cả hai.
        calib_path = project_path("results/gate_calibration.json")
        calib: dict[str, Any] = {}
        if not calib_path.exists():
            issues.append(
                "Thiếu results/gate_calibration.json — chạy "
                "`python run_pipeline.py calibrate`."
            )
        else:
            try:
                calib = json.loads(calib_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"gate_calibration.json hỏng: {exc}")
        cal_threshold = calib.get("gate_threshold", calib.get("threshold"))
        if cal_threshold is not None and cfg_threshold is not None:
            if abs(float(cal_threshold) - float(cfg_threshold)) > 1e-9:
                issues.append(
                    f"gate_threshold lệch: gate_calibration.json "
                    f"{cal_threshold} vs CONFIG {cfg_threshold} — chạy lại "
                    "`calibrate --apply`."
                )
        log_cal = gate.get("calibration") or {}
        for key in ("gate_threshold", "threshold"):
            if key in log_cal and cal_threshold is not None:
                if abs(float(log_cal[key]) - float(cal_threshold)) > 1e-9:
                    issues.append(
                        f"`calibration.{key}` lệch: gate log {log_cal[key]} vs "
                        f"gate_calibration.json {cal_threshold} — chạy lại "
                        "`evaluate`."
                    )
        if calib.get("score_key") and not (
            (gate.get("evaluation_config") or {}).get("gate_risk_scale") is not None
        ):
            issues.append(
                "Gate log thiếu `gate_risk_scale` — số liệu sinh bởi code cũ, "
                "chạy lại `evaluate`."
            )

    if issues:
        print("KHÔNG NHẤT QUÁN:")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print("OK: CONFIG ↔ gate log ↔ metrics khớp protocol.")
    print(f" - metrics : {metrics_path}")
    print(f" - gate log: {gate_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("all", help="evaluate -> report (val)")
    sub.add_parser("evaluate", help="Bước 5: chạy hệ thống trên val hoặc test-dev")
    sub.add_parser("report", help="Tổng hợp báo cáo markdown")
    sub.add_parser("calibrate", help="Hiệu chuẩn ngưỡng gate trên val")
    sub.add_parser("verify", help="Kiểm tra CONFIG ↔ gate log ↔ metrics")
    return parser


def main() -> None:
    parser = _build_parser()
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return

    command = argv[0]
    if command not in {"all", "evaluate", "report", "calibrate", "verify"}:
        parser.error(f"lệnh không hợp lệ: {command!r}")
    if command == "all" and any(
        arg == "--split" or arg.startswith("--split=") for arg in argv
    ):
        parser.error(
            "`all` không nhận --split; hãy chạy `evaluate --split test-dev` "
            "rồi `report --split test-dev`."
        )

    split, rest = _extract_split(argv)
    extra = rest[1:]
    if extra and extra[0] == "--":
        extra = extra[1:]

    if command == "verify":
        sys.exit(verify_consistency())
    if command == "calibrate":
        sys.exit(run_calibration(extra))

    if command == "all":
        rc = run_evaluate("val", [])
        if rc != 0:
            print(f"\nLỖI: evaluate kết thúc với mã {rc}")
            sys.exit(rc)
        rc = run_report("val", [])
        if rc != 0:
            print(f"\nLỖI: report kết thúc với mã {rc}")
            sys.exit(rc)
    elif command == "evaluate":
        rc = run_evaluate(split, extra)
        if rc != 0:
            print(f"\nLỖI: evaluate kết thúc với mã {rc}")
            sys.exit(rc)
    elif command == "report":
        rc = run_report(split, extra)
        if rc != 0:
            print(f"\nLỖI: report kết thúc với mã {rc}")
            sys.exit(rc)

    if command in {"all", "evaluate"}:
        is_testdev = split == "test-dev"
        split_label = "test-dev (held-out)" if is_testdev else "val"
        out_metrics = os.environ.get(
            "NCKH_OUT_METRICS", "results/system_metrics.csv"
        )
        out_gate = os.environ.get(
            "NCKH_OUT_GATE_LOG", "results/output_gate_log.json"
        )
        out_records = os.environ.get(
            "NCKH_OUT_FRAME_RECORDS", "results/val_frame_records.json"
        )
        print("\n" + "=" * 64)
        print("HOÀN TẤT")
        print("=" * 64)
        print(f"Kết quả chính ({split_label}):")
        print(f" - {out_metrics:34s} (metric hệ thống trên {split_label})")
        print(f" - {out_gate:34s} (gate log + calibration)")
        print(f" - {out_records:34s} (per-frame records)")
        if is_testdev:
            print(" - results/system_report_testdev.md      (chạy `report --split test-dev`)")
        else:
            print(" - results/system_report.md              (báo cáo tổng hợp)")
            print(" - outputs/protected_video.mp4           (video đã che đối tượng)")
        print("Tùy chọn thêm:")
        print(" - python run_pipeline.py calibrate  (hiệu chuẩn lại ngưỡng gate)")
        if is_testdev:
            print(" - python run_pipeline.py report --split test-dev  (báo cáo held-out)")
        else:
            print(" - python run_pipeline.py evaluate --split test-dev  (số held-out)")


if __name__ == "__main__":
    main()
