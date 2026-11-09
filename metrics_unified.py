"""Các hàm chỉ số đánh giá dùng chung cho hệ thống (detection, tracking, risk).

Chỉ sử dụng các chỉ số được công nhận trong literature:
- Detection: IoU, Containment, AP (COCO/VOC style), Precision/Recall/F1
- Tracking: MOTA, MOTP, IDF1, HOTA, IDSW, Frag
- Risk calibration: Expected Calibration Error (ECE)
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
                ious = np.array(
                    [[iou(pred, gt) for gt in gts] for pred in preds], dtype=np.float64
                )
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
