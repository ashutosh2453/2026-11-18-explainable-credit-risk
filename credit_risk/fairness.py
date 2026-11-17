"""Group fairness audit on held-out data, using protected attributes the model never sees."""

from __future__ import annotations

import numpy as np
import pandas as pd

FOUR_FIFTHS = 0.8


def protected_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    sex = df["personal_status"].astype(str).str.split().str[0].map({"male": "male", "female": "female"})
    age = np.where(df["age"] < 25, "under 25", "25 and over")
    return {
        "sex": sex.fillna("unknown"),
        "age_band": pd.Series(age, index=df.index),
        "foreign_worker": df["foreign_worker"].astype(str).map({"yes": "foreign worker", "no": "local worker"}),
    }


def audit(df: pd.DataFrame, y_bad: np.ndarray, approved: np.ndarray, min_group: int = 10) -> dict:
    """Approval rate (demographic parity), and approval rate among creditworthy applicants
    (equal opportunity) per group, with the four-fifths rule as the disparate impact flag."""
    report = {}
    for attr, groups in protected_groups(df).items():
        rows = []
        for g in sorted(groups.unique()):
            mask = (groups == g).to_numpy()
            good = mask & (y_bad == 0)
            rows.append({
                "group": g,
                "n": int(mask.sum()),
                "approval_rate": round(float(approved[mask].mean()), 4) if mask.any() else None,
                "approval_rate_if_creditworthy": round(float(approved[good].mean()), 4) if good.any() else None,
                "actual_bad_rate": round(float(y_bad[mask].mean()), 4) if mask.any() else None,
            })
        usable = [r for r in rows if r["n"] >= min_group and r["approval_rate"] is not None]
        di = eo = None
        if len(usable) >= 2:
            rates = [r["approval_rate"] for r in usable]
            di = round(min(rates) / max(rates), 4) if max(rates) > 0 else None
            tprs = [r["approval_rate_if_creditworthy"] for r in usable if r["approval_rate_if_creditworthy"] is not None]
            eo = round(max(tprs) - min(tprs), 4) if len(tprs) >= 2 else None
        report[attr] = {
            "groups": rows,
            "disparate_impact_ratio": di,
            "equal_opportunity_difference": eo,
            "passes_four_fifths_rule": None if di is None else di >= FOUR_FIFTHS,
        }
    return report
