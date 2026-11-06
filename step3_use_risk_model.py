"""Buoc 3: Su dung mo hinh XGBoost da train de du bao muc rui ro ro loi thong tin tuong tai.

Input:
- results/detection_tracks.csv (tu Buoc 2: toa do, conf, track_id, trang thai)
- models/xgboost_risk.json (mo hinh XGBoost tu file zip)

Output:
- results/risk_features_labels.csv (21 dac trung + nhan rui ro)
- results/risk_scores.csv (diem rui ro cho tung khung hinh/track)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

FEATURES = [
    "conf", "conf_std5", "conf_delta", "track_age", "hit_ratio",
    "time_since_update", "track_state", "min_side_px", "aspect_dev",
    "dist_edge", "is_trunc", "speed_norm", "pred_uncert", "lap_var_log",
    "brightness", "motion_blur_ratio", "nearest_dist_norm",
    "n_neighbors_r2_norm", "overlap_max", "frame_det_delta",
    "n_det_expected_ratio",
]

RISK_MODEL_PATH = Path("models/xgboost_risk.json")
TRACKS_CSV = Path("results/detection_tracks.csv")
OUT_FEATURES = Path("results/risk_features_labels.csv")
OUT_SCORES = Path("results/risk_scores.csv")


def load_xgboost_model(path: Path) -> xgb.XGBClassifier:
    """Tai mo hinh XGBoost da duoc luu."""
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def compute_frame_level_scores(
    df: pd.DataFrame, model: xgb.XGBClassifier, tau: float
) -> pd.DataFrame:
    """Tinh diem rui ro cho tung khung hinh (max across tracks trong frame)."""
    rows: list[dict[str, Any]] = []
    for frame_id, group in df.groupby("frame"):
        X = group[FEATURES].values
        probs = model.predict_proba(X)[:, 1]
        max_risk = float(probs.max()) if len(probs) > 0 else 0.0
        mean_risk = float(probs.mean()) if len(probs) > 0 else 0.0
        n_high_risk = int((probs >= tau).sum())
        rows.append(
            {
                "frame": int(frame_id),
                "n_tracks": len(group),
                "max_risk": max_risk,
                "mean_risk": mean_risk,
                "n_high_risk": n_high_risk,
                "leak_risk": bool(max_risk >= tau),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RISK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)

    # Neu chua co xgboost_risk.json, copy tu zip extract
    if not RISK_MODEL_PATH.exists():
        zip_model = Path("zip_extract/artifacts/risk_model.json")
        if zip_model.exists():
            import shutil
            shutil.copy(zip_model, RISK_MODEL_PATH)
        else:
            raise FileNotFoundError(f"Khong tim thay mo hinh XGBoost: {RISK_MODEL_PATH}")

    model = load_xgboost_model(RISK_MODEL_PATH)

    # Tai ngưỡng tau tu zip extract neu co
    tau_path = Path("zip_extract/artifacts/tau.json")
    if tau_path.exists():
        with tau_path.open("r", encoding="utf-8") as f:
            tau_data = json.load(f)
        tau = float(tau_data.get("tau", 0.4449612498283386))
    else:
        tau = 0.4449612498283386

    # Neu chua co detection_tracks.csv, tao tu tracks_frame.partial.csv trong zip
    if not TRACKS_CSV.exists():
        zip_tracks = Path("zip_extract/artifacts/tracks_frame.partial.csv")
        if zip_tracks.exists():
            import shutil
            shutil.copy(zip_tracks, TRACKS_CSV)
        else:
            raise FileNotFoundError(f"Khong tim thay file tracks: {TRACKS_CSV}")

    df = pd.read_csv(TRACKS_CSV)

    # Chuan hoa cot neu can
    if "frame" not in df.columns and "frame_index" in df.columns:
        df = df.rename(columns={"frame_index": "frame"})

    # Dam bao co du 21 features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        # Tao features mac dinh neu thieu
        for f in missing:
            df[f] = 0.0

    # Luu features + labels
    df[FEATURES + (["label"] if "label" in df.columns else [])].to_csv(
        OUT_FEATURES, index=False, encoding="utf-8"
    )

    # Tinh diem rui ro frame-level
    scores = compute_frame_level_scores(df, model, tau)
    scores.to_csv(OUT_SCORES, index=False, encoding="utf-8")

    print("[Buoc 3] Da luu dac trung:", OUT_FEATURES)
    print("[Buoc 3] Da luu diem rui ro:", OUT_SCORES)
    print("[Buoc 3] Nguong tau:", f"{tau:.6f}")
    print("[Buoc 3] So khung hinh co rui ro cao:", int(scores["leak_risk"].sum()), "/", len(scores))


if __name__ == "__main__":
    main()
