"""SHAP explanations mapped back from one-hot columns to the applicant's original features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import shap

from .data import CATEGORICAL, FEATURES, LABELS, NUMERIC


@dataclass
class Contribution:
    feature: str
    label: str
    value: object
    shap: float  # contribution to the log-odds of default (positive = riskier)

    def to_dict(self) -> dict:
        return asdict(self)


class Explainer:
    """Exact Tree SHAP for gradient boosting, Linear SHAP for logistic regression.

    Explanations are in log-odds of default for the uncalibrated model. The deployed
    probability is a monotone (sigmoid) calibration of the same scores, so the
    direction and ranking of every reason is preserved.
    """

    def __init__(self, pipeline, background: pd.DataFrame) -> None:
        self.prep = pipeline.named_steps["prep"]
        self.clf = pipeline.named_steps["clf"]
        self.columns = list(self.prep.get_feature_names_out())
        bg = self.prep.transform(background[FEATURES])
        if hasattr(self.clf, "estimators_"):
            self._explainer = shap.TreeExplainer(self.clf)
        else:
            self._explainer = shap.LinearExplainer(self.clf, bg)
        # Map each encoded column back to the original feature it came from.
        self._owner = []
        for col in self.columns:
            owner = col if col in NUMERIC else next(c for c in CATEGORICAL if col.startswith(c + "_"))
            self._owner.append(owner)

    def _grouped(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        Z = self.prep.transform(X[FEATURES])
        values = np.asarray(self._explainer.shap_values(Z))
        if values.ndim == 3:  # some versions return one array per class
            values = values[..., 1]
        base = float(np.ravel(self._explainer.expected_value)[-1])
        grouped = np.zeros((len(X), len(FEATURES)))
        idx = {f: i for i, f in enumerate(FEATURES)}
        for j, owner in enumerate(self._owner):
            grouped[:, idx[owner]] += values[:, j]
        return grouped, base

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.decision_function(self.prep.transform(X[FEATURES]))

    def explain(self, applicant: pd.DataFrame) -> tuple[float, list[Contribution]]:
        grouped, base = self._grouped(applicant)
        row = applicant.iloc[0]
        contribs = [Contribution(f, LABELS[f], row[f].item() if hasattr(row[f], "item") else row[f], round(float(v), 4))
                    for f, v in zip(FEATURES, grouped[0])]
        contribs.sort(key=lambda c: abs(c.shap), reverse=True)
        return base, contribs

    def global_importance(self, X: pd.DataFrame) -> list[dict]:
        grouped, _ = self._grouped(X)
        mean_abs = np.abs(grouped).mean(axis=0)
        order = np.argsort(-mean_abs)
        return [{"feature": FEATURES[i], "label": LABELS[FEATURES[i]], "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in order]


def reason_codes(contribs: list[Contribution], limit: int = 4) -> list[str]:
    """Adverse-action style reasons: the factors that pushed the applicant's risk up the most."""
    reasons = []
    for c in contribs:
        if c.shap <= 0:
            continue
        reasons.append(f"{c.label}: {c.value}")
        if len(reasons) == limit:
            break
    return reasons
