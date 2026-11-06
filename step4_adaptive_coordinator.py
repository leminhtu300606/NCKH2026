"""Buoc 4: Bo dieu phoi thich ung - Dieu chinh tan suat phat hien dua tren muc rui ro.

Input:
- results/risk_scores.csv (diem rui ro tu Buoc 3)
- config/coordinator_rules.json (quy tac dieu phoi)

Output:
- config/coordinator_rules.json (quy tac da cap nhat duong tu du lieu)
- results/coordinator_decisions.csv (nhat ly quyet dieu phoi)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RISK_SCORES_CSV = Path("results/risk_scores.csv")
COORDINATOR_RULES = Path("config/coordinator_rules.json")
COORDINATOR_DECISIONS = Path("results/coordinator_decisions.csv")

# Quy tac mac duong (duoc su dung neu khong co du lieu risk_scores.csv)
DEFAULT_RULES = {
    "risk_low": 0.25,
    "risk_high": 0.50,
    "risk_critical": 0.75,
    "detect_interval_low": 5,
    "detect_interval_medium": 3,
    "detect_interval_high": 1,
    "detect_interval_critical": 1,
    "roi_expand_low": 0.10,
    "roi_expand_medium": 0.20,
    "roi_expand_high": 0.30,
    "roi_expand_critical": 0.50,
    "output_gate_block_threshold": 0.75,
}


def load_rules(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return dict(DEFAULT_RULES)


def save_rules(path: Path, rules: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)


def auto_tune_thresholds(risk_values: pd.Series, base_rules: dict[str, Any]) -> dict[str, Any]:
    """Tu dong chinh sua ngucha nguy co theo phan vi cua phan phoi thuc te.

    - risk_low:   phan vi 33 (bat dau muc trung binh)
    - risk_high:  phan vi 66 (bat dau muc cao)
    - risk_critical: phan vi 90 (bat dau muc nguy co nghiem trong)
    - output_gate_block_threshold: phan vi 95 (nguong chan dau ra)
    """
    rules = dict(base_rules)
    if len(risk_values) == 0:
        return rules

    r = risk_values.dropna()
    if len(r) == 0:
        return rules

    q33 = float(r.quantile(0.33))
    q66 = float(r.quantile(0.66))
    q90 = float(r.quantile(0.90))
    q95 = float(r.quantile(0.95))

    # Dieu chinh cac ngucha neu co du lieu du (it nhat 10 mau)
    if len(r) >= 10:
        rules["risk_low"] = q33 if q33 > 0 else base_rules["risk_low"]
        rules["risk_high"] = q66 if q66 > rules["risk_low"] else base_rules["risk_high"]
        rules["risk_critical"] = q90 if q90 > rules["risk_high"] else base_rules["risk_critical"]
        rules["output_gate_block_threshold"] = q95 if q95 > rules["risk_critical"] else base_rules["output_gate_block_threshold"]

    return rules


def decide_detection_interval(risk: float, rules: dict[str, Any]) -> int:
    """Tra ve khoang cach frame giua cac lan phat hien dua tren muc rui ro."""
    if risk >= rules["risk_critical"]:
        return rules["detect_interval_critical"]
    if risk >= rules["risk_high"]:
        return rules["detect_interval_high"]
    if risk >= rules["risk_low"]:
        return rules["detect_interval_medium"]
    return rules["detect_interval_low"]


def decide_roi_padding(risk: float, rules: dict[str, Any]) -> float:
    """Tra ve padding cho vung bao ve dua tren muc rui ro."""
    if risk >= rules["risk_critical"]:
        return rules["roi_expand_critical"]
    if risk >= rules["risk_high"]:
        return rules["roi_expand_high"]
    if risk >= rules["risk_low"]:
        return rules["roi_expand_low"]
    return rules["roi_expand_low"]


def should_block_output(risk: float, rules: dict[str, Any]) -> bool:
    """Chan dau ra neu rui ro cao qua nguong an toan."""
    return risk >= rules["output_gate_block_threshold"]


def classify_risk_level(risk: float, rules: dict[str, Any]) -> str:
    """Phan loai muc rui ro: low / medium / high / critical."""
    if risk >= rules["risk_critical"]:
        return "critical"
    if risk >= rules["risk_high"]:
        return "high"
    if risk >= rules["risk_low"]:
        return "medium"
    return "low"


def main() -> None:
    COORDINATOR_RULES.parent.mkdir(parents=True, exist_ok=True)
    COORDINATOR_DECISIONS.parent.mkdir(parents=True, exist_ok=True)

    base_rules = load_rules(COORDINATOR_RULES)

    # Tu dong dieu chinh ngucha neu co du lieu risk_scores.csv
    if RISK_SCORES_CSV.exists():
        df_scores = pd.read_csv(RISK_SCORES_CSV)
        if "max_risk" in df_scores.columns and len(df_scores) > 0:
            rules = auto_tune_thresholds(df_scores["max_risk"], base_rules)
            print(f"[Buoc 4] Da tu dong chinh sua ngucha tu {len(df_scores)} khung hinh:")
            print(f"         risk_low={rules['risk_low']:.4f}, risk_high={rules['risk_high']:.4f}, "
                  f"risk_critical={rules['risk_critical']:.4f}, "
                  f"block_threshold={rules['output_gate_block_threshold']:.4f}")
        else:
            rules = base_rules
            print("[Buoc 4] risk_scores.csv khong co cot max_risk, dung quy tac mac duong.")
    else:
        rules = base_rules
        print("[Buoc 4] Khong tim thay risk_scores.csv, dung quy tac mac duong.")

    save_rules(COORDINATOR_RULES, rules)

    if not RISK_SCORES_CSV.exists():
        print("[Buoc 4] Bo qua tao quyet dinh (khong co risk_scores.csv).")
        return

    df = pd.read_csv(RISK_SCORES_CSV)

    decisions: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        risk = float(row["max_risk"])
        frame = int(row["frame"])
        interval = decide_detection_interval(risk, rules)
        padding = decide_roi_padding(risk, rules)
        block = should_block_output(risk, rules)
        level = classify_risk_level(risk, rules)

        decisions.append(
            {
                "frame": frame,
                "max_risk": risk,
                "detect_interval": interval,
                "roi_padding": padding,
                "block_output": bool(block),
                "risk_level": level,
            }
        )

    decisions_df = pd.DataFrame(decisions)
    decisions_df.to_csv(COORDINATOR_DECISIONS, index=False, encoding="utf-8")

    n_blocked = int(decisions_df["block_output"].sum())
    n_total = len(decisions_df)
    print(f"[Buoc 4] Da luu quy tac dieu phoi: {COORDINATOR_RULES}")
    print(f"[Buoc 4] Da luu quyet dinh dieu phoi: {COORDINATOR_DECISIONS}")
    print(f"[Buoc 4] So khung hinh chan dau ra: {n_blocked} / {n_total}")
    print(f"[Buoc 4] Phan bo muc rui ro: {decisions_df['risk_level'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()