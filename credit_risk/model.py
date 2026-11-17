"""Model training: candidate comparison, probability calibration and a cost-sensitive threshold."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, category_values, dataset_fingerprint, target

# Statlog cost matrix: approving a bad borrower costs 5x more than rejecting a good one.
COST_FALSE_APPROVAL = 5.0
COST_FALSE_REJECTION = 1.0


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num, NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        verbose_feature_names_out=False,
    )


def candidates(seed: int, n_estimators: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([("prep", make_preprocessor(True)),
                                         ("clf", LogisticRegression(max_iter=2000, C=0.5))]),
        "gradient_boosting": Pipeline([("prep", make_preprocessor(False)),
                                       ("clf", GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.05,
                                                                          max_depth=3, subsample=0.8, random_state=seed))]),
    }


def expected_cost(y_true: np.ndarray, p_bad: np.ndarray, threshold: float) -> float:
    """Average misclassification cost per applicant when rejecting everyone with P(bad) >= threshold."""
    reject = p_bad >= threshold
    false_approval = np.sum((~reject) & (y_true == 1))
    false_rejection = np.sum(reject & (y_true == 0))
    return float((COST_FALSE_APPROVAL * false_approval + COST_FALSE_REJECTION * false_rejection) / len(y_true))


def best_threshold(y_true: np.ndarray, p_bad: np.ndarray) -> float:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = [expected_cost(y_true, p_bad, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


@dataclass
class TrainedModel:
    name: str
    calibrated: CalibratedClassifierCV
    explain_pipeline: Pipeline  # uncalibrated pipeline fitted on the full training split (for SHAP)
    threshold: float
    categories: dict[str, list[str]]
    metrics: dict
    cv_auc: dict[str, float]
    background: pd.DataFrame
    test_frame: pd.DataFrame = field(repr=False)
    test_y: np.ndarray = field(repr=False)
    test_p_bad: np.ndarray = field(repr=False)
    trained_at: str = ""
    data_fingerprint: str = ""

    def predict_bad(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrated.predict_proba(X[FEATURES])[:, 1]


def train(df: pd.DataFrame, seed: int = 42, n_estimators: int = 300) -> TrainedModel:
    X, y = df[FEATURES], target(df).to_numpy()
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(X, y, np.arange(len(df)), test_size=0.25,
                                                              stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    cands = candidates(seed, n_estimators)
    cv_auc = {name: float(cross_val_score(p, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()) for name, p in cands.items()}
    name = max(cv_auc, key=cv_auc.get)

    # Calibrate with internal CV so the threshold is chosen on honest out-of-fold probabilities.
    calibrated = CalibratedClassifierCV(cands[name], method="sigmoid", cv=5).fit(X_tr, y_tr)
    explain_pipeline = candidates(seed, n_estimators)[name].fit(X_tr, y_tr)

    oof = np.zeros(len(y_tr))
    for fit_idx, val_idx in cv.split(X_tr, y_tr):
        m = CalibratedClassifierCV(candidates(seed, n_estimators)[name], method="sigmoid", cv=3)
        m.fit(X_tr.iloc[fit_idx], y_tr[fit_idx])
        oof[val_idx] = m.predict_proba(X_tr.iloc[val_idx])[:, 1]
    threshold = best_threshold(y_tr, oof)

    p_te = calibrated.predict_proba(X_te)[:, 1]
    reject = p_te >= threshold
    tn, fp, fn, tp = confusion_matrix(y_te, reject.astype(int), labels=[0, 1]).ravel()
    metrics = {
        "model": name,
        "test_size": int(len(y_te)),
        "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
        "brier": round(float(brier_score_loss(y_te, p_te)), 4),
        "threshold": threshold,
        "expected_cost": round(expected_cost(y_te, p_te, threshold), 4),
        "cost_reject_all": round(expected_cost(y_te, p_te, 0.0), 4),
        "cost_approve_all": round(expected_cost(y_te, p_te, 1.01), 4),
        "approval_rate": round(float(1 - reject.mean()), 4),
        "confusion": {"approved_good": int(tn), "rejected_good": int(fp), "approved_bad": int(fn), "rejected_bad": int(tp)},
        "bad_rate_among_approved": round(float(fn / max(tn + fn, 1)), 4),
    }
    return TrainedModel(
        name=name, calibrated=calibrated, explain_pipeline=explain_pipeline, threshold=threshold,
        categories=category_values(df), metrics=metrics, cv_auc={k: round(v, 4) for k, v in cv_auc.items()},
        background=X_tr.sample(min(100, len(X_tr)), random_state=seed),
        test_frame=df.iloc[idx_te].reset_index(drop=True), test_y=y_te, test_p_bad=p_te,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_fingerprint=dataset_fingerprint(),
    )
