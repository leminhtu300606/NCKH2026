"""Benchmark + parity check: PyTorch (.pt) vs ONNX FP32 cho detector person.

Đo:
- Latency/FPS của từng backend trên frame thật (cùng imgsz/conf/iou).
- Precision/Recall/F1 so với GT person (category 1, bỏ occlusion level 2).
- Độ lệch trực tiếp PT vs ONNX: số box, IoU khớp, chênh confidence.

Chạy:
  python bench_onnx_parity.py --onnx models/best.onnx --max-frames 50 --sequences 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

DATA_ROOT = Path("VisDrone2019-VID-test-dev")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def load_gt(seq: str) -> dict[int, list[list[float]]]:
    path = DATA_ROOT / "annotations" / f"{seq}.txt"
    gt: dict[int, list[list[float]]] = {}
    if not path.exists():
        return gt
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 10:
            continue
        frame = int(parts[0])
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        category = int(parts[7])
        occlusion = int(parts[9])
        if category != 1 or occlusion == 2:
            continue
        gt.setdefault(frame, []).append([x, y, x + w, y + h])
    return gt


def match(preds: np.ndarray, gts: np.ndarray, thr: float = 0.5) -> tuple[int, float]:
    """Ghép một-một, trả về (số cặp khớp, IoU trung bình của cặp khớp)."""
    if len(preds) == 0 or len(gts) == 0:
        return 0, 0.0
    ious = np.array([[iou(p, g) for g in gts] for p in preds], dtype=np.float64)
    cost = np.where(ious >= thr, -1.0 - ious * 1e-6, 0.0)
    pi, gi = linear_sum_assignment(cost)
    keep = [(p, g) for p, g in zip(pi, gi) if ious[p, g] >= thr]
    mean_iou = float(np.mean([ious[p, g] for p, g in keep])) if keep else 0.0
    return len(keep), mean_iou


def evaluate(model: YOLO, frames: list[tuple[Path, list[list[float]]]],
             imgsz: int, conf: float, iou_thr: float) -> dict[str, Any]:
    tp = fp = fn = 0
    n_boxes = 0
    confs: list[float] = []
    for path, gts in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        res = model(img, imgsz=imgsz, conf=conf, iou=iou_thr, device="cpu", verbose=False)[0]
        preds = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        n_boxes += len(preds)
        confs.extend(float(c) for c in scores)
        gt_arr = np.array(gts, dtype=np.float64).reshape(-1, 4)
        matched, _ = match(preds, gt_arr, 0.5)
        tp += matched
        fp += len(preds) - matched
        fn += len(gt_arr) - matched
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "pred_boxes": n_boxes,
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
    }


def latency(model: YOLO, img: np.ndarray, imgsz: int, conf: float,
            iou_thr: float, warmup: int, reps: int) -> dict[str, float]:
    for _ in range(warmup):
        model(img, imgsz=imgsz, conf=conf, iou=iou_thr, device="cpu", verbose=False)
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(img, imgsz=imgsz, conf=conf, iou=iou_thr, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": float(1000.0 / arr.mean()),
    }


def direct_delta(pt: YOLO, onnx: YOLO, frames: list[tuple[Path, list[list[float]]]],
                 imgsz: int, conf: float, iou_thr: float) -> dict[str, float]:
    n_pt = n_on = n_matched = 0
    ious: list[float] = []
    conf_deltas: list[float] = []
    for path, _ in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        bp = pt(img, imgsz=imgsz, conf=conf, iou=iou_thr, device="cpu", verbose=False)[0].boxes
        bo = onnx(img, imgsz=imgsz, conf=conf, iou=iou_thr, device="cpu", verbose=False)[0].boxes
        p = bp.xyxy.cpu().numpy()
        o = bo.xyxy.cpu().numpy()
        pc = bp.conf.cpu().numpy()
        oc = bo.conf.cpu().numpy()
        n_pt += len(p)
        n_on += len(o)
        if len(p) and len(o):
            ious_mat = np.array([[iou(a, b) for b in o] for a in p])
            cost = np.where(ious_mat >= 0.5, -1.0 - ious_mat * 1e-6, 0.0)
            pi, oi = linear_sum_assignment(cost)
            for a, b in zip(pi, oi):
                if ious_mat[a, b] >= 0.5:
                    n_matched += 1
                    ious.append(float(ious_mat[a, b]))
                    conf_deltas.append(abs(float(pc[a]) - float(oc[b])))
    return {
        "pt_boxes": n_pt,
        "onnx_boxes": n_on,
        "box_delta_pct": 100.0 * (n_on - n_pt) / max(1, n_pt),
        "matched": n_matched,
        "match_rate_vs_pt": n_matched / max(1, n_pt),
        "mean_matched_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_abs_conf_delta": float(np.mean(conf_deltas)) if conf_deltas else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="models/best.pt")
    ap.add_argument("--onnx", default="models/best.onnx")
    ap.add_argument("--sequences", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.0075)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default="results/onnx_parity.json")
    args = ap.parse_args()

    seq_root = DATA_ROOT / "sequences"
    seqs = sorted([d.name for d in seq_root.iterdir() if d.is_dir()])[: args.sequences]
    frames: list[tuple[Path, list[list[float]]]] = []
    for seq in seqs:
        gt = load_gt(seq)
        images = sorted((seq_root / seq).glob("*.jpg"))[: args.max_frames]
        for p in images:
            frames.append((p, gt.get(int(p.stem), [])))
    print(f"Sequences: {seqs}")
    print(f"Frames: {len(frames)}")

    pt = YOLO(args.pt)
    onnx = YOLO(args.onnx)

    first_img = cv2.imread(str(frames[0][0]))
    print("\n== Latency (1 frame thật) ==")
    lat_pt = latency(pt, first_img, args.imgsz, args.conf, args.iou, args.warmup, args.reps)
    lat_on = latency(onnx, first_img, args.imgsz, args.conf, args.iou, args.warmup, args.reps)
    speedup = lat_pt["mean_ms"] / lat_on["mean_ms"]

    print("\n== Accuracy vs GT (IoU 0.5) ==")
    acc_pt = evaluate(pt, frames, args.imgsz, args.conf, args.iou)
    acc_on = evaluate(onnx, frames, args.imgsz, args.conf, args.iou)

    print("\n== Direct PT vs ONNX ==")
    delta = direct_delta(pt, onnx, frames, args.imgsz, args.conf, args.iou)

    result = {
        "protocol": {
            "sequences": seqs, "n_frames": len(frames), "imgsz": args.imgsz,
            "conf": args.conf, "iou": args.iou, "device": "cpu",
        },
        "model_size_mb": {
            "pt": Path(args.pt).stat().st_size / 1e6,
            "onnx": Path(args.onnx).stat().st_size / 1e6,
        },
        "latency": {"pt": lat_pt, "onnx": lat_on, "speedup": speedup},
        "accuracy": {"pt": acc_pt, "onnx": acc_on},
        "delta": delta,
    }

    def row(label: str, a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> str:
        return label + " | " + " | ".join(
            f"{a.get(k, 0):.4f} -> {b.get(k, 0):.4f}" for k in keys
        )

    print(f"\nModel size: PT {result['model_size_mb']['pt']:.1f} MB | "
          f"ONNX {result['model_size_mb']['onnx']:.1f} MB")
    print(f"Latency: PT {lat_pt['mean_ms']:.1f} ms ({lat_pt['fps']:.2f} FPS) | "
          f"ONNX {lat_on['mean_ms']:.1f} ms ({lat_on['fps']:.2f} FPS) | "
          f"speedup {speedup:.2f}x")
    for k in ("precision", "recall", "f1", "pred_boxes", "mean_conf"):
        print(f"  {k:12s}: PT {acc_pt[k]:.4f} -> ONNX {acc_on[k]:.4f}")
    print("  delta:", json.dumps(delta, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
