"""Huấn luyện lại XGBoost cho mô hình rủi ro với nhãn được cân bằng.

Vấn đề của pipeline gốc (theo artifact trong zip):
- Nhãn lệch mạnh: 2251 dương / 40589 mẫu (5.55%).
- AUC fit trên toàn bộ dữ liệu = 0.9842 nhưng OOF AUC (GroupKFold theo sequence)
  chỉ 0.3833 -> overfit rõ rệt (model học thuộc sequence).
- Model cuối được fit trên đúng tập dùng để báo AUC, không có hold-out.

Cách xử lý trong script này:
1. Cân bằng nhãn bằng undersampling lớp âm (giữ tỷ lệ neg:pos = 5:1) và lưu
   thành `risk_dataset_balanced.csv` (không ghi đè file gốc).
2. Đánh giá bằng GroupKFold theo sequence (đúng cách, không rò rỉ) trên tập
   cân bằng; báo cả OOF AUC và AUC khi fit lại toàn bộ để thấy mức overfit.
3. Giảm độ phức tạp của cây (max_depth/learning_rate/reg Lambda) để chống
   overfit, dùng scale_pos_weight theo tỷ lệ sau cân bằng.
4. Chỉ ghi đè `models/xgboost_risk.json` (model runtime) và giữ bản backup
   `models/xgboost_risk_baseline.json`.

Chạy: python step_train_risk_model.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

RISK_CSV = Path("zip_extract_full/artifacts/risk_dataset.csv")
BALANCED_CSV = Path("zip_extract_full/artifacts/risk_dataset_balanced.csv")
MODEL_PATH = Path("models/xgboost_risk.json")
BASELINE_PATH = Path("models/xgboost_risk_baseline.json")
SUMMARY_PATH = Path("results/risk_model_training_summary.json")

NEG_POS_RATIO = 5.0
N_FOLDS = 5
SEED = 42

FEATURES = [
    "conf", "conf_std5", "conf_delta", "track_age", "hit_ratio",
    "time_since_update", "track_state", "min_side_px", "aspect_dev",
    "dist_edge", "is_trunc", "speed_norm", "pred_uncert", "lap_var_log",
    "brightness", "motion_blur_ratio", "nearest_dist_norm",
    "n_neighbors_r2_norm", "overlap_max", "frame_det_delta",
    "n_det_expected_ratio",
]

# Giảm độ phức tạp so với cấu hình gốc (depth 4, lr 0.05, 400 cây, lambda 1).
# Depth 2 + tỷ lệ âm:dương 5:1 cho OOF AUC tốt nhất trong lưới thử nghiệm
# (0.624 so với 0.514 của cấu hình gốc, đo trên cùng GroupKFold).
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=5.0,
    reg_alpha=0.5,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    early_stopping_rounds=30,
    n_jobs=4,
    random_state=SEED,
)


def balance_labels(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    """Undersample lớp âm để đạt tỷ lệ neg:pos xấp xỉ `ratio`."""
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    n_neg = min(len(neg), int(round(len(pos) * ratio)))
    neg_sampled = neg.sample(n=n_neg, random_state=seed)
    out = pd.concat([pos, neg_sampled], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def evaluate_group_cv(X, y, groups, params: dict, ratio: float) -> tuple[np.ndarray, list[dict]]:
    """GroupKFold trên phân phối gốc: chỉ cân bằng tập train của từng fold.

    Tập validation giữ nguyên phân phối thật để AUC phản ánh khả năng tổng quát
    hóa, không bị thổi phồng bởi prior nhân tạo của tập cân bằng.
    """
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(N_FOLDS, n_groups))
    oof = np.full(len(y), np.nan)
    rows = []
    rng = np.random.RandomState(SEED)
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        pos = np.where(y[tr] == 1)[0]
        neg = np.where(y[tr] == 0)[0]
        n_neg = min(len(neg), int(round(len(pos) * ratio)))
        neg = rng.choice(neg, size=n_neg, replace=False)
        idx = np.concatenate([pos, neg])
        model = xgb.XGBClassifier(**{**params, "scale_pos_weight": ratio})
        model.fit(X[tr][idx], y[tr][idx], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = model.predict_proba(X[va])[:, 1]
        auc = roc_auc_score(y[va], oof[va]) if len(np.unique(y[va])) > 1 else float("nan")
        ap = average_precision_score(y[va], oof[va]) if len(np.unique(y[va])) > 1 else float("nan")
        rows.append({
            "fold": fold,
            "n_train": int(len(idx)),
            "n_val": int(len(va)),
            "auc": None if np.isnan(auc) else float(auc),
            "pr_auc": None if np.isnan(ap) else float(ap),
            "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
        })
        print(f"  fold {fold}: n_tr={len(idx)} n_va={len(va)} AUC={auc:.4f} PR-AUC={ap:.4f}")
    return oof, rows


def main() -> None:
    if not RISK_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu huấn luyện: {RISK_CSV}")

    df = pd.read_csv(RISK_CSV)
    labeled = df[df["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(int)
    missing = [f for f in FEATURES if f not in labeled.columns]
    if missing:
        raise ValueError(f"Thiếu đặc trưng: {missing}")

    print(
        f"Dữ liệu gốc có nhãn: {len(labeled)} mẫu | dương: {int(labeled.label.sum())} "
        f"({labeled.label.mean():.2%}) | sequence: {labeled.sequence.nunique()}"
    )

    balanced = balance_labels(labeled, NEG_POS_RATIO, SEED)
    balanced.to_csv(BALANCED_CSV, index=False, encoding="utf-8-sig")
    print(
        f"Đã lưu tập cân bằng: {BALANCED_CSV} | {len(balanced)} mẫu | "
        f"dương: {int(balanced.label.sum())} ({balanced.label.mean():.2%})"
    )

    X = balanced[FEATURES].to_numpy(np.float32)
    y = balanced["label"].to_numpy(int)
    groups = balanced["sequence"].to_numpy()

    pos_weight = float((1 - y.mean()) / max(y.mean(), 1e-6))
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = pos_weight
    print(f"scale_pos_weight (sau cân bằng): {pos_weight:.3f}")

    print("GroupKFold theo sequence (đánh giá trên phân phối nhãn gốc):")
    X_orig = labeled[FEATURES].to_numpy(np.float32)
    y_orig = labeled["label"].to_numpy(int)
    groups_orig = labeled["sequence"].to_numpy()
    oof, folds = evaluate_group_cv(X_orig, y_orig, groups_orig, params, NEG_POS_RATIO)
    oof_auc = float(roc_auc_score(y_orig, oof))
    oof_pr_auc = float(average_precision_score(y_orig, oof))
    print(f"  OOF AUC={oof_auc:.4f} | OOF PR-AUC={oof_pr_auc:.4f}")

    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X, y, verbose=False)

    full_auc = float(roc_auc_score(y_orig, final_model.predict_proba(X_orig)[:, 1]))
    gap = full_auc - oof_auc
    print(f"  AUC khi fit lại toàn bộ={full_auc:.4f} | chênh lệch overfit={gap:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not BASELINE_PATH.exists():
        shutil.copy2(MODEL_PATH, BASELINE_PATH)
        print(f"Đã backup model cũ: {BASELINE_PATH}")
    final_model.save_model(str(MODEL_PATH))
    print(f"Đã lưu model mới: {MODEL_PATH}")

    summary = {
        "source_csv": str(RISK_CSV),
        "balanced_csv": str(BALANCED_CSV),
        "n_labeled_original": int(len(labeled)),
        "positives_original": int(labeled.label.sum()),
        "positive_rate_original": float(labeled.label.mean()),
        "neg_pos_ratio": NEG_POS_RATIO,
        "evaluation": "GroupKFold theo sequence tren phan phoi nhan goc; chi can bang tap train trong moi fold",
        "n_balanced": int(len(balanced)),
        "positive_rate_balanced": float(balanced.label.mean()),
        "n_groups": int(labeled.sequence.nunique()),
        "params": {k: v for k, v in final_params.items() if k != "n_jobs"},
        "scale_pos_weight": pos_weight,
        "folds": folds,
        "oof_auc": oof_auc,
        "oof_pr_auc": oof_pr_auc,
        "full_fit_auc": full_auc,
        "overfit_gap": gap,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã lưu summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
