"""Wrapper TrackEval cho protocol đánh giá tracking của hệ thống.

TrackEval là evaluator chính thức của MOTChallenge. Module này nhận dữ liệu
trong bộ nhớ (dict theo sequence/frame) từ Bước 5, ghi ra định dạng MOT tạm
thời, chấm bằng `MotChallenge2DBox` + `CLEAR`/`HOTA`/`Identity`, rồi trả về
đúng bộ khóa mà `step5_integrated_system.py` đang dùng.

Vì sao phải ghi file tạm: TrackEval được thiết kế để đọc file, không có API
in-memory. Toàn bộ thư mục tạm bị xoá sau khi chấm.

Quy ước:
- GT dùng `target_id` của VisDrone (cột 2 trong annotation), class = 1.
- Prediction dùng ByteTrack `track_id`.
- `BENCHMARK='MOT15'` + `DO_PREPROC=False`: bỏ qua preprocessing distractor
  của MOTChallenge (dữ liệu đã được lọc person + occlusion level 2 từ trước),
  tránh việc TrackEval loại thêm box vì cột visibility.
- Frame được dời về dải 1..N nếu dữ liệu bắt đầu từ 0.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _frame_offset(frames: list[int]) -> int:
    """Trả offset để frame nhỏ nhất trở thành 1 (TrackEval yêu cầu timestep >= 1)."""
    if not frames:
        return 0
    min_frame = min(frames)
    return 1 - min_frame if min_frame < 1 else 0


def _write_mot_file(path: Path, rows: list[tuple[Any, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


def _gt_rows(
    boxes_by_frame: dict[int, list[list[float]]],
    ids_by_frame: dict[int, list[int]],
    offset: int,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for frame, boxes in boxes_by_frame.items():
        ids = ids_by_frame.get(frame, [])
        for idx, box in enumerate(boxes):
            if idx >= len(ids):
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            rows.append(
                (
                    int(frame) + offset,
                    int(ids[idx]),
                    x1,
                    y1,
                    max(1e-6, x2 - x1),
                    max(1e-6, y2 - y1),
                    1,
                    1,
                    1,
                )
            )
    return rows


def _pred_rows(
    boxes_by_frame: dict[int, list[list[float]]],
    ids_by_frame: dict[int, list[int]],
    scores_by_frame: dict[int, list[float]],
    offset: int,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for frame, boxes in boxes_by_frame.items():
        ids = ids_by_frame.get(frame, [])
        scores = scores_by_frame.get(frame, [])
        for idx, box in enumerate(boxes):
            if idx >= len(ids):
                continue
            score = float(scores[idx]) if idx < len(scores) else 1.0
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            rows.append(
                (
                    int(frame) + offset,
                    int(ids[idx]),
                    x1,
                    y1,
                    max(1e-6, x2 - x1),
                    max(1e-6, y2 - y1),
                    score,
                    1,
                    1,
                )
            )
    return rows


def _mean_scalar(value: Any) -> float:
    """HOTA/DetA/AssA là mảng theo 19 ngưỡng; báo cáo giá trị trung bình."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def evaluate_trackeval(
    gt_all: dict[str, dict[int, list[list[float]]]],
    gt_ids_all: dict[str, dict[int, list[int]]],
    pred_all: dict[str, dict[int, list[list[float]]]],
    pred_ids_all: dict[str, dict[int, list[int]]],
    pred_scores_all: dict[str, dict[int, list[float]]] | None = None,
    return_per_sequence: bool = False,
) -> dict[str, Any]:
    """Chấm MOTA/MOTP/IDF1/HOTA/... bằng TrackEval cho toàn bộ sequence.

    Trả về dict phẳng với các khóa step5 đang dùng. Nếu `return_per_sequence`
    thì thêm khóa `per_sequence` = {seq: {metric: value}} để tính phân tán/CI.
    Ném exception nếu TrackEval không cài hoặc dữ liệu không hợp lệ; step5 sẽ
    bắt và ghi N/A, tuyệt đối không ghi 0 giả.
    """
    from trackeval.datasets import MotChallenge2DBox
    from trackeval.eval import Evaluator
    from trackeval.metrics import CLEAR, HOTA, Identity

    pred_scores_all = pred_scores_all or {}
    sequences = sorted(set(gt_all) | set(pred_all))
    if not sequences:
        raise ValueError("Không có sequence nào để chấm TrackEval")

    # Số timestep phải phủ mọi frame của cả GT và prediction.
    seq_info: dict[str, int] = {}
    offsets: dict[str, int] = {}
    for seq in sequences:
        frames = list(gt_all.get(seq, {}).keys()) + list(pred_all.get(seq, {}).keys())
        offsets[seq] = _frame_offset(frames)
        max_frame = max(frames) + offsets[seq] if frames else 1
        seq_info[seq] = max(1, int(max_frame))

    with tempfile.TemporaryDirectory(prefix="nckh_trackeval_") as tmp:
        root = Path(tmp)
        tracker_name = "person"
        tracker_dir = root / "trackers" / tracker_name

        for seq in sequences:
            _write_mot_file(
                root / f"{seq}.txt",
                _gt_rows(
                    gt_all.get(seq, {}), gt_ids_all.get(seq, {}), offsets[seq]
                ),
            )
            _write_mot_file(
                tracker_dir / f"{seq}.txt",
                _pred_rows(
                    pred_all.get(seq, {}),
                    pred_ids_all.get(seq, {}),
                    pred_scores_all.get(seq, {}),
                    offsets[seq],
                ),
            )

        dataset_config = {
            "GT_FOLDER": str(root),
            "TRACKERS_FOLDER": str(root / "trackers"),
            "OUTPUT_FOLDER": str(root / "out"),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT15",
            "SPLIT_TO_EVAL": "all",
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": False,
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_SUB_FOLDER": "",
            "TRACKER_DISPLAY_NAMES": {tracker_name: tracker_name},
            "SEQ_INFO": seq_info,
            "GT_LOC_FORMAT": "{gt_folder}/{seq}.txt",
            "SKIP_SPLIT_FOL": True,
        }
        eval_config = {
            "USE_PARALLEL": False,
            "PRINT_CONFIG": False,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }

        dataset = MotChallenge2DBox(dataset_config)
        metrics = [
            CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
            HOTA({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
            Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        ]
        evaluator = Evaluator(eval_config)
        results, messages = evaluator.evaluate([dataset], metrics)

    dataset_name = dataset.get_name()
    tracker_res = (results.get(dataset_name) or {}).get(tracker_name)
    if not tracker_res:
        raise RuntimeError(f"TrackEval không trả kết quả: {messages}")

    combined = tracker_res.get("COMBINED_SEQ", {}).get("pedestrian", {})
    clear = combined.get("CLEAR", {})
    hota = combined.get("HOTA", {})
    identity = combined.get("Identity", {})
    if not clear or not hota or not identity:
        raise RuntimeError(f"TrackEval thiếu metric CLEAR/HOTA/Identity: {messages}")

    total_gt_ids = float(clear.get("MT", 0) + clear.get("ML", 0) + clear.get("PT", 0))

    # Các hàm nhỏ chỉ để tránh lặp khóa giữa bản combined và per-sequence.
    def block_c(res: dict[str, Any]) -> dict[str, Any]:
        return res.get("CLEAR", {})

    def block_h(res: dict[str, Any]) -> dict[str, Any]:
        return res.get("HOTA", {})

    def block_i(res: dict[str, Any]) -> dict[str, Any]:
        return res.get("Identity", {})

    out: dict[str, Any] = {
        "mota": float(clear.get("MOTA", float("nan"))),
        "motp": float(clear.get("MOTP", float("nan"))),
        "idf1": float(identity.get("IDF1", float("nan"))),
        "idp": float(identity.get("IDP", float("nan"))),
        "idr": float(identity.get("IDR", float("nan"))),
        "hota": _mean_scalar(hota.get("HOTA", float("nan"))),
        "deta": _mean_scalar(hota.get("DetA", float("nan"))),
        "assa": _mean_scalar(hota.get("AssA", float("nan"))),
        "idsw": float(clear.get("IDSW", float("nan"))),
        "frag": float(clear.get("Frag", float("nan"))),
        "mt": float(clear.get("MT", float("nan"))),
        "ml": float(clear.get("ML", float("nan"))),
        "clr_tp": float(clear.get("CLR_TP", float("nan"))),
        "clr_fp": float(clear.get("CLR_FP", float("nan"))),
        "clr_fn": float(clear.get("CLR_FN", float("nan"))),
        "mot_eval_sequences": float(len(seq_info)),
        "track_eval_total_gt_ids": total_gt_ids,
    }

    if return_per_sequence:
        per_sequence: dict[str, dict[str, float]] = {}
        for seq in sequences:
            seq_res = tracker_res.get(seq, {}).get("pedestrian", {})
            if not seq_res:
                continue
            c = block_c(seq_res)
            h = block_h(seq_res)
            i = block_i(seq_res)
            per_sequence[seq] = {
                "mota": float(c.get("MOTA", float("nan"))),
                "motp": float(c.get("MOTP", float("nan"))),
                "idf1": float(i.get("IDF1", float("nan"))),
                "hota": _mean_scalar(h.get("HOTA", float("nan"))),
                "deta": _mean_scalar(h.get("DetA", float("nan"))),
                "assa": _mean_scalar(h.get("AssA", float("nan"))),
                "idsw": float(c.get("IDSW", float("nan"))),
                "frag": float(c.get("Frag", float("nan"))),
                "clr_tp": float(c.get("CLR_TP", float("nan"))),
                "clr_fp": float(c.get("CLR_FP", float("nan"))),
                "clr_fn": float(c.get("CLR_FN", float("nan"))),
            }
        out["per_sequence"] = per_sequence

    return out
