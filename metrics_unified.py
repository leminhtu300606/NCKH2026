"""Các hàm chỉ số đánh giá dùng chung cho hệ thống (detection, tracking, risk).

Chỉ sử dụng các chỉ số được công nhận trong literature:
- Detection: IoU, Containment, AP (COCO/VOC style), Precision/Recall/F1
- Tracking: MOTA, MOTP, IDF1, HOTA, IDSW, Frag
- Risk calibration: Expected Calibration Error (ECE), Brier, ROC-AUC, PR-AUC
- Privacy/utility: ULFR, unsafe release, TTP, identity leak, blur-area ratio,
  over-protection ratio, PSNR tách vùng che / vùng nền
- Gate quality: confusion matrix của cổng phát hành so với nhãn "hở GT"
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    """Tính IoU giữa hai box [x1, y1, x2, y2]."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def containment(gt_box: list[float], protect_box: list[float]) -> float:
    """Tỷ lệ diện tích gt_box được protect_box che phủ (IoU(gt, protect) / area(gt)).

    Dùng để đánh giá mức độ che phủ của vùng bảo vệ (ROI padding) đối với GT box.
    """
    x1 = max(float(gt_box[0]), float(protect_box[0]))
    y1 = max(float(gt_box[1]), float(protect_box[1]))
    x2 = min(float(gt_box[2]), float(protect_box[2]))
    y2 = min(float(gt_box[3]), float(protect_box[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_gt = max(0.0, float(gt_box[2]) - float(gt_box[0])) * max(0.0, float(gt_box[3]) - float(gt_box[1]))
    if area_gt <= 0:
        return 0.0
    return inter / area_gt


def _normalize_preds(
    raw_preds: list[Any],
) -> tuple[list[list[float]], list[float]]:
    """Chuẩn hóa danh sách dự đoán về (boxes, scores).

    Mỗi phần tử có thể là [x1,y1,x2,y2] hoặc ([x1,y1,x2,y2], conf).
    """
    boxes: list[list[float]] = []
    scores: list[float] = []
    for p in raw_preds:
        if isinstance(p, tuple):
            boxes.append(list(p[0]))
            scores.append(float(p[1]) if len(p) > 1 else 0.0)
        else:
            boxes.append(list(p))
            scores.append(0.0)
    return boxes, scores


def average_precision_coco(
    detector_all: dict[str, dict[int, list[Any]]],
    gt_all: dict[str, dict[int, list[Any]]],
    iou_thresholds: list[float],
) -> dict[float, float]:
    """Tính AP theo phong cách COCO/VOC cho detector.

    detector_all[seq][frame] = danh sách ([x1,y1,x2,y2], conf) hoặc [x1,y1,x2,y2]
    gt_all[seq][frame] = danh sách [x1,y1,x2,y2]
    Trả về dict {iou_threshold: AP}.
    """
    results: dict[float, float] = {}
    for thr in iou_thresholds:
        all_scores: list[float] = []
        all_matches: list[bool] = []
        total_gt = 0
        for seq, seq_gt in gt_all.items():
            seq_det = detector_all.get(seq, {})
            for frame_number, gts in seq_gt.items():
                raw_preds = seq_det.get(frame_number, [])
                preds, scores = _normalize_preds(raw_preds)
                total_gt += len(gts)
                if not gts:
                    all_scores.extend(scores)
                    all_matches.extend([False] * len(preds))
                    continue
                if not preds:
                    continue
                # Vector hoá IoU (n_pred, n_gt) thay cho 2 vòng lặp Python;
                # giá trị giống hệt iou() gọi từng cặp.
                ious = iou_matrix(preds, gts)
                matched_gt = [False] * len(gts)
                for pred_idx in range(len(preds)):
                    best_gt = -1
                    best_iou = thr
                    for gt_idx in range(len(gts)):
                        if not matched_gt[gt_idx] and ious[pred_idx, gt_idx] >= best_iou:
                            best_iou = float(ious[pred_idx, gt_idx])
                            best_gt = gt_idx
                    if best_gt >= 0:
                        matched_gt[best_gt] = True
                        all_scores.append(scores[pred_idx])
                        all_matches.append(True)
                    else:
                        all_scores.append(scores[pred_idx])
                        all_matches.append(False)
        if total_gt == 0 or not all_scores:
            results[thr] = 0.0
            continue
        scores_arr = np.array(all_scores, dtype=np.float64)
        matches_arr = np.array(all_matches, dtype=bool)
        order = np.argsort(-scores_arr)
        matches_sorted = matches_arr[order]
        tp_cum = np.cumsum(matches_sorted).astype(np.float64)
        fp_cum = np.cumsum(~matches_sorted).astype(np.float64)
        recall = tp_cum / max(1.0, float(total_gt))
        precision = tp_cum / np.maximum(1.0, tp_cum + fp_cum)
        recall = np.concatenate([[0.0], recall, [1.0]])
        precision = np.concatenate([[0.0], precision, [0.0]])
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
        results[thr] = ap
    return results


def precision_recall_f1_one_to_one(
    predictions: dict[str, dict[int, list[Any]]],
    gt_all: dict[str, dict[int, list[Any]]],
    iou_threshold: float = 0.5,
) -> tuple[float, float, float, int, int]:
    """Tính Precision, Recall, F1 với matching một-một (Hungarian algorithm).

    Trả về (precision, recall, f1, total_gt, total_pred).
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    for seq, seq_gt in gt_all.items():
        seq_pred = predictions.get(seq, {})
        for frame_number, gts in seq_gt.items():
            raw_preds = seq_pred.get(frame_number, [])
            preds, _ = _normalize_preds(raw_preds)
            total_gt += len(gts)
            if not gts or not preds:
                total_fp += len(preds)
                continue
            ious = np.array(
                [[iou(pred, gt) for gt in gts] for pred in preds], dtype=np.float64
            )
            match_cost = np.where(ious >= iou_threshold, -1.0 - ious * 1e-6, 0.0)
            pred_indices, gt_indices = linear_sum_assignment(match_cost)
            matched = sum(
                float(ious[p, g]) >= iou_threshold
                for p, g in zip(pred_indices, gt_indices)
            )
            total_tp += int(matched)
            total_fp += len(preds) - int(matched)
    total_fn = total_gt - total_tp
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_gt)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return precision, recall, f1, total_gt, total_tp + total_fp


def mota_motp(
    gt_all: dict[str, dict[int, list[Any]]],
    pred_all: dict[str, dict[int, list[Any]]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Tính MOTA và MOTP (per-frame matching, không có GT track ID nên IDSW=0).

    Trả về dict chứa mota, motp và các bộ đếm chi tiết.
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_iou_sum = 0.0
    total_matches = 0
    for seq, seq_gt in gt_all.items():
        seq_pred = pred_all.get(seq, {})
        for frame_number, gts in seq_gt.items():
            raw_preds = seq_pred.get(frame_number, [])
            preds, _ = _normalize_preds(raw_preds)
            total_gt += len(gts)
            if not gts or not preds:
                total_fp += len(preds)
                continue
            ious = np.array(
                [[iou(pred, gt) for gt in gts] for pred in preds], dtype=np.float64
            )
            match_cost = np.where(ious >= iou_threshold, -1.0 - ious * 1e-6, 0.0)
            pred_indices, gt_indices = linear_sum_assignment(match_cost)
            matched_mask = [
                float(ious[p, g]) >= iou_threshold
                for p, g in zip(pred_indices, gt_indices)
            ]
            matched_count = int(sum(matched_mask))
            total_tp += matched_count
            total_iou_sum += float(
                ious[pred_indices[matched_mask], gt_indices[matched_mask]].sum()
            ) if matched_count else 0.0
            total_matches += matched_count
            total_fp += len(preds) - matched_count
    total_fn = total_gt - total_tp
    mota = 1.0 - (total_fn + total_fp) / max(1, total_gt)
    motp = total_iou_sum / max(1, total_matches)
    return {
        "mota": float(mota),
        "motp": float(motp),
        "num_frag": 0,
        "num_switches": 0,
        "num_false_positives": total_fp,
        "num_misses": total_fn,
        "num_detections": total_tp,
    }


def _build_gt_tracks(
    gt_all: dict[str, dict[int, list[list[float]]]],
    iou_threshold: float = 0.5,
) -> dict[tuple[str, int, int], int]:
    """Xây dựng pseudo-GT tracks qua các frame liên tiếp.

    Vì VisDrone VID ground-truth không có track ID, chúng ta liên kết GT box giữa
    các frame liên tiếp (IoU >= iou_threshold) để gán mỗi GT box một `gt_track_id`
    duy nhất theo thời gian. Đây là cách ước lượng identity ground-truth chuẩn
    cho việc tính IDF1/HOTA/IDSW/Frag khi không có GT track ID.

    Trả về dict: (seq, frame_number, gt_idx) -> gt_track_id
    """
    from collections import defaultdict

    gt_to_track: dict[tuple[str, int, int], int] = {}
    next_gt_track_id = 0

    for seq, seq_gt in gt_all.items():
        # Lấy danh sách frame có GT, sắp xếp theo thứ tự
        sorted_frames = sorted(seq_gt.keys())
        # prev_gt_idx -> prev_box, prev_gt_track_id
        prev_frame_gt_info: list[tuple[int, list[float], int]] = []  # (gt_idx, box, gt_track_id)

        for frame_number in sorted_frames:
            gts = seq_gt[frame_number]
            curr_frame_gt_info: list[tuple[int, list[float], int]] = []

            # Danh sách prev GT track chưa được match ở frame này
            available_prev = list(prev_frame_gt_info)

            for g_idx, gt_box in enumerate(gts):
                best_prev_idx = -1
                best_iou = iou_threshold
                for pi, (prev_g_idx, prev_box, prev_track_id) in enumerate(available_prev):
                    i = iou(gt_box, prev_box)
                    if i >= best_iou:
                        best_iou = i
                        best_prev_idx = pi

                if best_prev_idx >= 0:
                    # Liên kết với GT track đã có
                    _, _, gt_track_id = available_prev.pop(best_prev_idx)
                else:
                    # Tạo GT track mới
                    gt_track_id = next_gt_track_id
                    next_gt_track_id += 1

                gt_to_track[(seq, frame_number, g_idx)] = gt_track_id
                curr_frame_gt_info.append((g_idx, gt_box, gt_track_id))

            prev_frame_gt_info = curr_frame_gt_info

    return gt_to_track


def compute_idf1_hota(
    gt_all: dict[str, dict[int, list[list[float]]]],
    pred_all: dict[str, dict[int, list[list[float]]]],
    pred_ids_all: dict[str, dict[int, list[int]]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Tính IDF1, HOTA, IDSW, Frag từ các matching per-frame kèm theo track ID.

    Vì ground-truth VisDrone VID không có track ID, chúng ta xây dựng pseudo-GT
    tracks bằng cách liên kết GT box giữa các frame liên tiếp (IoU >= 0.5), sau
    đó so sánh predicted track ID với pseudo-GT track ID.

    Trả về dict chứa mota, motp, idf1, hota, idsw, frag và các bộ đếm.
    """
    from collections import Counter, defaultdict

    # Xây dựng pseudo-GT tracks
    gt_to_track = _build_gt_tracks(gt_all, iou_threshold)

    total_gt = 0
    total_det = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_sum = 0.0

    # Danh sách các bản ghi match: (seq, frame, track_id, gt_track_id, iou)
    match_records: list[tuple[str, int, int, int, float]] = []

    for seq, seq_gt in gt_all.items():
        seq_pred = pred_all.get(seq, {})
        seq_ids = pred_ids_all.get(seq, {})
        for frame_number, gts in seq_gt.items():
            raw_preds = seq_pred.get(frame_number, [])
            raw_ids = seq_ids.get(frame_number, [])
            preds, _ = _normalize_preds(raw_preds)

            total_gt += len(gts)
            total_det += len(preds)

            if not gts or not preds:
                continue

            ious = np.array(
                [[iou(pred, gt) for gt in gts] for pred in preds], dtype=np.float64
            )
            match_cost = np.where(ious >= iou_threshold, -1.0 - ious * 1e-6, 0.0)
            pred_indices, gt_indices = linear_sum_assignment(match_cost)

            for p_idx, g_idx in zip(pred_indices, gt_indices):
                if float(ious[p_idx, g_idx]) >= iou_threshold:
                    track_id = raw_ids[p_idx] if p_idx < len(raw_ids) else -1
                    gt_track_id = gt_to_track.get((seq, frame_number, g_idx), -1)
                    iou_val = float(ious[p_idx, g_idx])
                    match_records.append((seq, frame_number, track_id, gt_track_id, iou_val))

    total_tp = len(match_records)
    total_fp = total_det - total_tp
    total_fn = total_gt - total_tp
    iou_sum = sum(r[4] for r in match_records)

    mota = 1.0 - (total_fn + total_fp) / max(1, total_gt)
    motp = iou_sum / max(1, total_tp)

    # ===== IDF1 =====
    # idtp: predicted track ID khớp với pseudo-GT track ID
    # idfp: predicted track ID khác pseudo-GT track ID (hoặc detection không match GT)
    # idfn: GT không được match với track nào
    idtp = 0
    idfp = 0
    for seq, frame_number, track_id, gt_track_id, iou_val in match_records:
        if track_id == gt_track_id:
            idtp += 1
        else:
            idfp += 1
    idfp += total_det - total_tp  # detections không khớp GT nào
    idfn = total_fn  # GT không được khớp với track nào

    idf1 = 2 * idtp / max(1, 2 * idtp + idfp + idfn)

    # ===== HOTA = sqrt(DetA * AssA) =====
    detA = idtp / max(1, idtp + idfp + idfn)
    assA = idtp / max(1, idtp + (total_tp - idtp) + (total_tp - idtp))
    hota = math.sqrt(detA * assA) if detA > 0 and assA > 0 else 0.0

    # ===== IDSW và Frag =====
    # IDSW: khi cùng một predicted track ID thay đổi pseudo-GT track ID giữa
    #       hai frame liên tiếp.
    # Frag: khi một predicted track ID bị ngắt quãng (không có detection trong
    #       frame liên tiếp sau khi đã xuất hiện trước đó).
    idsw = 0
    frag = 0

    # Gom các match theo predicted track_id, sắp xếp theo frame
    track_records: dict[int, list[tuple]] = defaultdict(list)
    for rec in match_records:
        track_records[rec[2]].append(rec)

    for track_id, records in track_records.items():
        records.sort(key=lambda r: (r[0], r[1]))
        prev_gt_track_id: int | None = None
        prev_frame: int | None = None
        prev_seq: str | None = None
        for seq, frame_number, tid, gt_track_id, iou_val in records:
            if prev_gt_track_id is not None and prev_seq == seq:
                if frame_number == prev_frame + 1:
                    # Frame liên tiếp: GT track ID thay đổi -> ID switch
                    if gt_track_id != prev_gt_track_id:
                        idsw += 1
                else:
                    # Có khoảng trắng giữa các frame -> fragmentation
                    frag += 1
            prev_gt_track_id = gt_track_id
            prev_frame = frame_number
            prev_seq = seq

    return {
        "mota": float(mota),
        "motp": float(motp),
        "idf1": float(idf1),
        "hota": float(hota),
        "idsw": float(idsw),
        "frag": float(frag),
        "num_gt": float(total_gt),
        "num_detections": float(total_det),
        "num_tp": float(total_tp),
        "num_fp": float(total_fp),
        "num_fn": float(total_fn),
    }


def expected_calibration_error(
    y_true: list[int], y_prob: list[float], n_bins: int = 10
) -> float:
    """Tính Expected Calibration Error (ECE)."""
    if len(y_true) == 0:
        return 0.0
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob_arr > bin_lower) & (y_prob_arr <= bin_upper)
        prop_in_bin = float(in_bin.mean())
        if prop_in_bin > 0:
            accuracy_in_bin = float(y_true_arr[in_bin].mean())
            avg_confidence_in_bin = float(y_prob_arr[in_bin].mean())
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def iou_matrix(
    boxes_a: list[list[float]] | np.ndarray,
    boxes_b: list[list[float]] | np.ndarray,
) -> np.ndarray:
    """Ma trận IoU (len(a), len(b)) cho hai tập box [x1, y1, x2, y2].

    Tương đương gọi iou() cho từng cặp nhưng tính vectorized, dùng cho các
    vòng lặp O(GT x Pred) ở runtime (coverage, containment gate, AP).
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    inter_w = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]),
        0.0,
        None,
    )
    inter_h = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]),
        0.0,
        None,
    )
    inter = inter_w * inter_h
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    out = np.zeros_like(inter, dtype=np.float64)
    np.divide(inter, union, out=out, where=union > 0)
    return out


def containment_matrix(
    gt_boxes: list[list[float]] | np.ndarray,
    protect_boxes: list[list[float]] | np.ndarray,
) -> np.ndarray:
    """Ma trận containment (len(gt), len(protect)) giống containment() từng cặp.

    containment(i, j) = diện tích giao / diện tích gt_box i.
    """
    g = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    p = np.asarray(protect_boxes, dtype=np.float64).reshape(-1, 4)
    if g.size == 0 or p.size == 0:
        return np.zeros((len(g), len(p)), dtype=np.float64)

    inter_w = np.clip(
        np.minimum(g[:, None, 2], p[None, :, 2]) - np.maximum(g[:, None, 0], p[None, :, 0]),
        0.0,
        None,
    )
    inter_h = np.clip(
        np.minimum(g[:, None, 3], p[None, :, 3]) - np.maximum(g[:, None, 1], p[None, :, 1]),
        0.0,
        None,
    )
    inter = inter_w * inter_h
    area_gt = np.clip(g[:, 2] - g[:, 0], 0.0, None) * np.clip(g[:, 3] - g[:, 1], 0.0, None)
    out = np.zeros_like(inter, dtype=np.float64)
    np.divide(inter, area_gt[:, None], out=out, where=area_gt[:, None] > 0)
    return out


# =============================================================================
# Metric mức frame: privacy/utility và chất lượng cổng phát hành
# =============================================================================
# `records` là list dict do step5 sinh ra, mỗi phần tử tương ứng 1 frame:
#   sequence, frame_number, block, risk, risk_level, n_gt, gt_ids,
#   gt_contained (list bool), unsafe, blur_area_px, frame_area_px,
#   gt_covered_area_px, psnr_bg, psnr_prot
# Trong đó `unsafe = True` nghĩa là NẾU phát hành frame này thì còn ít nhất
# một GT box chưa được che >= 90% diện tích.


def _safe_div(num: float, den: float) -> float:
    """Chia an toàn: trả NaN khi mẫu bằng 0 (không ghi 0 giả)."""
    if den is None or den == 0:
        return float("nan")
    return float(num) / float(den)


def _finite_values(values: list[float]) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def roc_auc_score(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray) -> float:
    """ROC-AUC bằng thống kê hạng (Mann-Whitney U), có xử lý điểm trùng.

    Không phụ thuộc sklearn; trùng score được gán hạng trung bình nên kết quả
    khớp `sklearn.metrics.roc_auc_score`. Trả NaN nếu chỉ có một lớp.
    """
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.size == 0 or y.size != s.size:
        return float("nan")
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(s, method="average")
    auc = (float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_score(
    y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray
) -> float:
    """PR-AUC (average precision) kiểu sklearn, gộp các score trùng nhau.

    Gộp theo nhóm score giống nhau để isotonic calibration (nhiều giá trị
    trùng) không làm AP phụ thuộc thứ tự sắp xếp.
    """
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.size == 0 or y.size != s.size:
        return float("nan")
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    # Biên của từng nhóm score bằng nhau.
    change = np.flatnonzero(np.diff(s_sorted) != 0)
    ends = np.concatenate([change, [s_sorted.size - 1]]) if s_sorted.size else np.array([], dtype=int)
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    start = 0
    for end in ends:
        block = y_sorted[start : end + 1]
        block_pos = int(block.sum())
        tp += block_pos
        fp += int(block.size) - block_pos
        precision = _safe_div(tp, tp + fp)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        start = int(end) + 1
    return float(ap)


def brier_score(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray) -> float:
    """Brier score = trung bình (prob - label)^2; thấp hơn là tốt hơn."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.size == 0 or y.size != p.size:
        return float("nan")
    keep = np.isfinite(p)
    if not keep.any():
        return float("nan")
    return float(np.mean((p[keep] - y[keep]) ** 2))


def gate_confusion_metrics(
    y_unsafe: list[bool] | np.ndarray, y_block: list[bool] | np.ndarray
) -> dict[str, float]:
    """Confusion matrix của OutputGate so với nhãn "nếu phát hành thì hở GT".

    Positive = frame nguy hiểm (unsafe). Dự đoán positive = frame bị chặn.
    - `false_block_rate`: frame an toàn bị chặn (mất utility).
    - `gate_leak_rate`: frame nguy hiểm được phát hành (rò rỉ coverage).
    """
    y = np.asarray(y_unsafe, dtype=bool).ravel()
    b = np.asarray(y_block, dtype=bool).ravel()
    if y.size == 0 or y.size != b.size:
        return {}
    tp = int(np.logical_and(y, b).sum())
    fp = int(np.logical_and(~y, b).sum())
    tn = int(np.logical_and(~y, ~b).sum())
    fn = int(np.logical_and(y, ~b).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1_den = precision + recall
    f1 = (
        2.0 * precision * recall / f1_den
        if np.isfinite(precision) and np.isfinite(recall) and f1_den > 0
        else float("nan")
    )
    tpr = recall
    tnr = _safe_div(tn, tn + fp)
    return {
        "gate_tp": tp,
        "gate_fp": fp,
        "gate_tn": tn,
        "gate_fn": fn,
        "gate_precision": precision,
        "gate_recall": recall,
        "gate_f1": float(f1),
        "gate_balanced_accuracy": float(0.5 * (tpr + tnr))
        if np.isfinite(tpr) and np.isfinite(tnr)
        else float("nan"),
        "false_block_rate": _safe_div(fp, fp + tn),
        "gate_leak_rate": _safe_div(fn, fn + tp),
    }


def privacy_utility_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Metric privacy/utility mức frame và mức danh tính (GT `target_id`).

    Quy ước:
    - Frame bị gate chặn được coi là **được bảo vệ hoàn toàn** (ảnh đen), nên
      được tính là protected cho TTP/identity leak, nhưng KHÔNG được cộng vào
      coverage IoU (giữ nguyên định nghĩa coverage cũ).
    - `ulfr` lấy mẫu số là mọi frame có GT (kể cả frame bị chặn) nên là chỉ số
      coverage-only, đọc cùng `block_rate`.
    """
    if not records:
        return {}

    frames_with_gt = [r for r in records if int(r.get("n_gt", 0)) > 0]
    released = [r for r in records if not bool(r.get("block"))]
    released_with_gt = [r for r in released if int(r.get("n_gt", 0)) > 0]
    unsafe_released = [r for r in released_with_gt if bool(r.get("unsafe"))]
    unsafe_gt_boxes = sum(
        sum(1 for c in r.get("gt_contained", []) if not c) for r in released
    )

    # TTP và identity leak theo target_id của VisDrone.
    first_seen: dict[tuple[str, int], int] = {}
    first_protected: dict[tuple[str, int], int] = {}
    for r in sorted(records, key=lambda x: (str(x.get("sequence")), int(x.get("frame_number", 0)))):
        seq = str(r.get("sequence"))
        frame_no = int(r.get("frame_number", 0))
        for gid, contained in zip(r.get("gt_ids", []), r.get("gt_contained", [])):
            key = (seq, int(gid))
            first_seen.setdefault(key, frame_no)
            protected = bool(r.get("block")) or bool(contained)
            if protected:
                first_protected.setdefault(key, frame_no)
    n_ids = len(first_seen)
    ttp_values = [
        float(first_protected[key] - first_seen[key])
        for key in first_protected
        if key in first_seen
    ]
    n_leaked = n_ids - len(first_protected)

    blur_ratios = []
    blur_area_sum = 0
    gt_covered_sum = 0
    for r in released:
        area = int(r.get("frame_area_px", 0))
        blur_px = int(r.get("blur_area_px", 0))
        if area > 0:
            blur_ratios.append(blur_px / float(area))
        blur_area_sum += blur_px
        gt_covered_sum += int(r.get("gt_covered_area_px", 0))

    psnr_bg = _finite_values([r.get("psnr_bg", float("nan")) for r in released])
    psnr_prot = _finite_values([r.get("psnr_prot", float("nan")) for r in released])

    return {
        "ulfr": _safe_div(len(unsafe_released), len(frames_with_gt)),
        "unsafe_release_rate": _safe_div(len(unsafe_released), len(released_with_gt)),
        "unsafe_frames": float(len(unsafe_released)),
        "frames_with_gt": float(len(frames_with_gt)),
        "released_frames_with_gt": float(len(released_with_gt)),
        "unsafe_released_gt_boxes": float(unsafe_gt_boxes),
        "leak_exposure_person_frames": float(unsafe_gt_boxes),
        "n_gt_identities": float(n_ids),
        "n_leaked_identities": float(n_leaked),
        "identity_leak_rate": _safe_div(n_leaked, n_ids),
        "ttp_p50_frames": float(np.percentile(ttp_values, 50)) if ttp_values else float("nan"),
        "ttp_p95_frames": float(np.percentile(ttp_values, 95)) if ttp_values else float("nan"),
        "ttp_censored_rate": _safe_div(n_leaked, n_ids),
        "blur_area_ratio": float(np.mean(blur_ratios)) if blur_ratios else float("nan"),
        "over_protection_ratio": (
            1.0 - gt_covered_sum / float(blur_area_sum) if blur_area_sum > 0 else float("nan")
        ),
        "psnr_bg_db": float(np.mean(psnr_bg)) if psnr_bg else float("nan"),
        "psnr_protected_db": float(np.mean(psnr_prot)) if psnr_prot else float("nan"),
    }


def gate_quality_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Chất lượng quyết định gate + calibration của frame risk.

    Nhãn dùng ở đây là `unsafe` (hở GT nếu phát hành), KHÔNG phải nhãn
    re-ID trong `risk_dataset.csv`; hai nguồn nhãn này không được trộn.
    """
    if not records:
        return {}
    y = [bool(r.get("unsafe")) for r in records]
    b = [bool(r.get("block")) for r in records]
    risk = [float(r.get("risk", float("nan"))) for r in records]
    result = gate_confusion_metrics(y, b)
    y_int = [1 if v else 0 for v in y]

    risk_auc = roc_auc_score(y_int, risk)
    risk_ap = average_precision_score(y_int, risk)
    risk_brier = brier_score(y_int, risk)
    risk_ece = expected_calibration_error(y_int, [float(v) for v in risk])
    risk_finite = [v for v in risk if np.isfinite(v)]

    levels = ["low", "medium", "high", "critical"]
    n = float(len(records))
    level_counts = {lvl: 0 for lvl in levels}
    for r in records:
        lvl = str(r.get("risk_level", "low"))
        if lvl in level_counts:
            level_counts[lvl] += 1

    result.update(
        {
            "risk_roc_auc": risk_auc,
            "risk_pr_auc": risk_ap,
            "risk_brier": risk_brier,
            "risk_ece": risk_ece,
            "risk_score_p50": float(np.percentile(risk_finite, 50))
            if risk_finite
            else float("nan"),
            "risk_score_p95": float(np.percentile(risk_finite, 95))
            if risk_finite
            else float("nan"),
        }
    )
    for lvl in levels:
        result[f"risk_level_{lvl}_share"] = level_counts[lvl] / n
    return result

