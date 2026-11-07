"""Benchmark ONNX raw head + NMS ngoài (không INT8), so sánh công bằng với PyTorch.

Điểm quan trọng: Ultralytics dùng LetterBox `auto=True` (minimum rectangle) cho .pt,
nên input thật là 736x1280 chứ không phải 1280x1280. Bản ONNX static 1280x1280
phải tính 1.74x số pixel => chậm hơn không công bằng. Script này dùng đúng
LetterBox của Ultralytics và model ONNX dynamic để so sánh 1:1.

So sánh:
- PyTorch predictor (Ultralytics, rect letterbox, NMS nội bộ)
- ONNX Runtime raw graph (session.run)
- ONNX Runtime raw + IOBinding
- ONNX Runtime full pipeline (preprocess + graph + decode + NMS ngoài)
- ONNX static 1280x1280 full pipeline (để tham chiếu, không công bằng)

Chạy:
  python bench_onnx_raw.py --sequences 3 --max-frames 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

DATA_ROOT = Path("VisDrone2019-VID-test-dev")


def preprocess(
    img: np.ndarray, imgsz: int = 1280, stride: int = 32, auto: bool = True
) -> tuple[np.ndarray, float, float, float]:
    """Letterbox y hệt Ultralytics, trả về (NCHW float32, ratio, pad_w, pad_h)."""
    lb = LetterBox((imgsz, imgsz), auto=auto, stride=stride)
    params = lb.get_params({"img": img})
    padded = lb.apply_image({"img": img}, params)["img"]
    r = float(params["ratio"][0])
    pad_w = float(params["left"])
    pad_h = float(params["top"])
    x = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0
    return x[None], r, pad_w, pad_h


def decode(
    out: np.ndarray,
    conf: float,
    iou_thr: float,
    r: float,
    pad_w: float,
    pad_h: float,
    orig_w: int,
    orig_h: int,
    max_det: int = 300,
    max_nms: int = 30000,
) -> tuple[np.ndarray, np.ndarray]:
    a = out[0].T  # (N, 5) = cx, cy, w, h, score
    scores = a[:, 4]
    keep = scores > conf
    a = a[keep]
    scores = scores[keep]
    if len(a) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    if len(a) > max_nms:
        idx = np.argsort(-scores)[:max_nms]
        a, scores = a[idx], scores[idx]
    cx, cy, bw, bh = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / r
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    keep_idx = nms(
        torch.from_numpy(boxes).float(),
        torch.from_numpy(scores).float(),
        iou_thr,
    )[:max_det].numpy()
    return boxes[keep_idx], scores[keep_idx]


def make_session(path: str, intra: int, inter: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_mem_pattern = True
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def stat(times: list[float]) -> dict[str, float]:
    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": float(1000.0 / arr.mean()),
    }


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return float(inter / u) if u > 0 else 0.0


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
        if int(parts[7]) != 1 or int(parts[9]) == 2:
            continue
        gt.setdefault(frame, []).append([x, y, x + w, y + h])
    return gt


def match(preds: np.ndarray, gts: np.ndarray, thr: float = 0.5) -> int:
    if len(preds) == 0 or len(gts) == 0:
        return 0
    m = np.array([[iou(p, g) for g in gts] for p in preds])
    cost = np.where(m >= thr, -1.0 - m * 1e-6, 0.0)
    pi, gi = linear_sum_assignment(cost)
    return sum(m[p, g] >= thr for p, g in zip(pi, gi))


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(1e-9, p + r)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="models/best.pt")
    ap.add_argument("--onnx-dyn", default="models/best_raw_dyn.onnx")
    ap.add_argument("--onnx-static", default="models/best_raw.onnx")
    ap.add_argument("--sequences", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.0075)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default="results/onnx_raw_benchmark.json")
    args = ap.parse_args()

    seq_root = DATA_ROOT / "sequences"
    seqs = sorted([d.name for d in seq_root.iterdir() if d.is_dir()])[: args.sequences]
    frames: list[tuple[Path, list[list[float]]]] = []
    for seq in seqs:
        gt = load_gt(seq)
        images = sorted((seq_root / seq).glob("*.jpg"))
        if len(images) > args.max_frames:
            idx = np.linspace(0, len(images) - 1, args.max_frames).round().astype(int)
            images = [images[i] for i in sorted(set(idx.tolist()))]
        for p in images:
            frames.append((p, gt.get(int(p.stem), [])))

    pt = YOLO(args.pt)
    img0 = cv2.imread(str(frames[0][0]))
    h0, w0 = img0.shape[:2]
    x0, r0, pw0, ph0 = preprocess(img0, args.imgsz, auto=True)
    print(f"Sequences: {seqs} | frames: {len(frames)}")
    print(f"Frame 0: {w0}x{h0} -> tensor {tuple(x0.shape)} (rect letterbox, giống PT)")

    # --- PT predictor ---
    def pt_predict(image: np.ndarray):
        res = pt(image, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device="cpu", verbose=False)[0]
        return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()

    # --- ONNX dynamic thread sweep (graph only) ---
    sweep: dict[str, dict[str, float]] = {}
    best_intra = 8
    best_ms = float("inf")
    for intra in (2, 4, 8):
        s_sweep = make_session(args.onnx_dyn, intra)
        s_inp = s_sweep.get_inputs()[0].name
        for _ in range(args.warmup):
            s_sweep.run(None, {s_inp: x0})
        t: list[float] = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            s_sweep.run(None, {s_inp: x0})
            t.append((time.perf_counter() - t0) * 1000.0)
        sweep[f"intra_{intra}"] = stat(t)
        if sweep[f"intra_{intra}"]["mean_ms"] < best_ms:
            best_ms = sweep[f"intra_{intra}"]["mean_ms"]
            best_intra = intra
        del s_sweep

    sess = make_session(args.onnx_dyn, best_intra)
    inp = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    def onnx_graph(image: np.ndarray) -> np.ndarray:
        x, *_ = preprocess(image, args.imgsz, auto=True)
        return sess.run(None, {inp: x})[0]

    # --- ONNX dynamic + IOBinding (buffer tái sử dụng) ---
    io = sess.io_binding()
    iobuf: np.ndarray | None = None
    iobuf_shape: tuple[int, ...] | None = None

    def onnx_io_full(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal iobuf, iobuf_shape
        h, w = image.shape[:2]
        x, r, pw, ph = preprocess(image, args.imgsz, auto=True)
        if iobuf is None or iobuf_shape != x.shape:
            iobuf = np.empty_like(x)
            iobuf_shape = x.shape
            io.bind_cpu_input(inp, iobuf)
            io.bind_output(out_name)
        np.copyto(iobuf, x)
        sess.run_with_iobinding(io)
        o = io.copy_outputs_to_cpu()[0]
        return decode(o, args.conf, args.iou, r, pw, ph, w, h)

    # --- ONNX dynamic full pipeline (session.run) ---
    def onnx_full(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        x, r, pw, ph = preprocess(image, args.imgsz, auto=True)
        o = sess.run(None, {inp: x})[0]
        return decode(o, args.conf, args.iou, r, pw, ph, w, h)

    # --- OpenVINO static 736x1280 (FP32, không lượng tử hóa) ---
    ov_full = None
    try:
        from openvino_detector import OpenVinoDetector

        ov_det = OpenVinoDetector(args.onnx_dyn, static_shape=(736, 1280))

        def ov_full(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            b = ov_det(image, imgsz=args.imgsz, conf=args.conf, iou=args.iou)[0].cpu().boxes
            return b.xyxy.numpy(), b.conf.numpy()

        ov_full(img0)
    except Exception as exc:  # noqa: BLE001
        print(f"OpenVINO bỏ qua: {exc}")
        ov_full = None

    # --- Đo round-robin để giảm bias nhiệt/CPU ---
    backends: list[tuple[str, Any]] = [
        ("pt_predictor", pt_predict),
        ("onnx_dyn_graph_only", onnx_graph),
        ("onnx_dyn_iobinding_full", onnx_io_full),
        ("onnx_dyn_full_pipeline", onnx_full),
    ]
    if ov_full is not None:
        backends.append(("openvino_static_full", ov_full))

    for name, fn in backends:
        for _ in range(args.warmup):
            fn(img0)

    times: dict[str, list[float]] = {name: [] for name, _ in backends}
    for _ in range(args.reps):
        for name, fn in backends:
            t0 = time.perf_counter()
            fn(img0)
            times[name].append((time.perf_counter() - t0) * 1000.0)

    stats = {name: stat(vals) for name, vals in times.items()}
    pt_stat = stats["pt_predictor"]
    graph_stat = stats["onnx_dyn_graph_only"]
    io_stat = stats["onnx_dyn_iobinding_full"]
    full_stat = stats["onnx_dyn_full_pipeline"]

    # --- ONNX static 1280x1280 full pipeline (tham chiếu) ---
    static_stat: dict[str, float] | None = None
    if Path(args.onnx_static).exists():
        s_sess = make_session(args.onnx_static, best_intra)
        s_inp = s_sess.get_inputs()[0].name

        def onnx_static_full(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            h, w = image.shape[:2]
            x, r, pw, ph = preprocess(image, args.imgsz, auto=False)  # vuông
            o = s_sess.run(None, {s_inp: x})[0]
            return decode(o, args.conf, args.iou, r, pw, ph, w, h)

        for _ in range(args.warmup):
            onnx_static_full(img0)
        t_s: list[float] = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            onnx_static_full(img0)
            t_s.append((time.perf_counter() - t0) * 1000.0)
        static_stat = stat(t_s)

    # --- Parity vs GT + direct PT vs ONNX dynamic ---
    tp = fp = fn = 0
    tp_o = fp_o = fn_o = 0
    n_pt = n_on = n_match = 0
    n_gt_frames = 0
    conf_deltas: list[float] = []
    iou_matches: list[float] = []
    for path, gts in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gt_arr = np.array(gts, dtype=np.float64).reshape(-1, 4)
        pb, pc = pt_predict(img)
        ob, oc = onnx_full(img)
        if len(gt_arr):
            m_pt = match(pb, gt_arr)
            m_on = match(ob, gt_arr)
            tp += m_pt
            fp += len(pb) - m_pt
            fn += len(gt_arr) - m_pt
            tp_o += m_on
            fp_o += len(ob) - m_on
            fn_o += len(gt_arr) - m_on
            n_gt_frames += 1
        n_pt += len(pb)
        n_on += len(ob)
        if len(pb) and len(ob):
            ious = np.array([[iou(a, b) for b in ob] for a in pb])
            cost = np.where(ious >= 0.5, -1.0 - ious * 1e-6, 0.0)
            pi, oi = linear_sum_assignment(cost)
            for a, b in zip(pi, oi):
                if ious[a, b] >= 0.5:
                    n_match += 1
                    iou_matches.append(float(ious[a, b]))
                    conf_deltas.append(abs(float(pc[a]) - float(oc[b])))

    result: dict[str, Any] = {
        "protocol": {
            "sequences": seqs,
            "n_frames": len(frames),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": "cpu",
            "letterbox": "ultralytics LetterBox auto=True (rect), giống PT",
            "onnx_dynamic": args.onnx_dyn,
            "best_intra_op_threads": best_intra,
        },
        "model_size_mb": {
            "pt": Path(args.pt).stat().st_size / 1e6,
            "onnx_dyn": Path(args.onnx_dyn).stat().st_size / 1e6,
        },
        "latency_ms": {
            **stats,
            "onnx_static_square_full": static_stat,
            "thread_sweep": sweep,
        },
        "accuracy_vs_gt": {
            "gt_frames": n_gt_frames,
            "pt": prf(tp, fp, fn),
            "onnx_dyn": prf(tp_o, fp_o, fn_o),
        },
        "direct_pt_vs_onnx": {
            "pt_boxes": n_pt,
            "onnx_boxes": n_on,
            "box_delta_pct": 100.0 * (n_on - n_pt) / max(1, n_pt),
            "matched": n_match,
            "match_rate_vs_pt": n_match / max(1, n_pt),
            "mean_matched_iou": float(np.mean(iou_matches)) if iou_matches else 0.0,
            "mean_abs_conf_delta": float(np.mean(conf_deltas)) if conf_deltas else 0.0,
        },
        "speedup_vs_pt": {
            "onnx_dyn_graph_only": pt_stat["mean_ms"] / graph_stat["mean_ms"],
            "onnx_dyn_full_pipeline": pt_stat["mean_ms"] / full_stat["mean_ms"],
        },
    }
    if static_stat:
        result["speedup_vs_pt"]["onnx_static_square_full"] = pt_stat["mean_ms"] / static_stat["mean_ms"]
    for _name in ("onnx_dyn_iobinding_full", "openvino_static_full"):
        if _name in stats:
            result["speedup_vs_pt"][_name] = pt_stat["mean_ms"] / stats[_name]["mean_ms"]

    print("\n== Latency ==")
    rows: list[tuple[str, dict[str, float] | None]] = [(name, stats[name]) for name, _ in backends]
    rows.append(("onnx_static_square_full", static_stat))
    for k, s in rows:
        if s is None:
            print(f"  {k:26s}: (skip)")
            continue
        print(f"  {k:26s}: {s['mean_ms']:8.1f} ms  {s['fps']:5.2f} FPS  "
              f"(p50 {s['p50_ms']:.1f}, p95 {s['p95_ms']:.1f})")
    print("  thread sweep:", json.dumps(sweep, ensure_ascii=False))

    print("\n== Accuracy vs GT ==")
    print(f"  GT frames: {n_gt_frames}/{len(frames)}")
    for k in ("pt", "onnx_dyn"):
        v = result["accuracy_vs_gt"][k]
        print(f"  {k:10s}: P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f}")
    print("  direct:", json.dumps(result["direct_pt_vs_onnx"], ensure_ascii=False))
    print("\n== Speedup vs PT ==")
    print(" ", json.dumps({k: round(v, 2) for k, v in result["speedup_vs_pt"].items()}, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
