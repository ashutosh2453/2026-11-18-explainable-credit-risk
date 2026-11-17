"""Scoring service: loads the trained artifact and produces decisions with explanations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from . import __version__
from .data import FEATURES, LABELS, NUMERIC, NUMERIC_BOUNDS, load_dataset, validate_applicant
from .explain import Explainer, reason_codes
from .fairness import audit
from .model import TrainedModel, train
from .recourse import find_recourse

ARTIFACT = Path(os.getenv("CREDIT_ARTIFACT", Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"))


@dataclass
class Bundle:
    model: TrainedModel
    explainer: Explainer
    fairness: dict
    importance: list[dict]
    version: str


def build_bundle(seed: int = 42, n_estimators: int = 300) -> Bundle:
    model = train(load_dataset(), seed=seed, n_estimators=n_estimators)
    explainer = Explainer(model.explain_pipeline, model.background)
    approved = model.test_p_bad < model.threshold
    return Bundle(model, explainer, audit(model.test_frame, model.test_y, approved),
                  explainer.global_importance(model.test_frame), __version__)


def save_bundle(bundle: Bundle, path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_or_train(path: Path = ARTIFACT) -> Bundle:
    if path.exists():
        bundle = joblib.load(path)
        if getattr(bundle, "version", None) == __version__:
            return bundle
    bundle = build_bundle()
    save_bundle(bundle, path)
    return bundle


class Scorer:
    def __init__(self, bundle: Bundle) -> None:
        self.b = bundle

    def schema(self) -> dict:
        m = self.b.model
        return {
            "numeric": {f: {"label": LABELS[f], "min": NUMERIC_BOUNDS[f][0], "max": NUMERIC_BOUNDS[f][1]} for f in NUMERIC},
            "categorical": {f: {"label": LABELS[f], "values": v} for f, v in m.categories.items()},
            "order": FEATURES,
            "example": m.test_frame[FEATURES].iloc[0].to_dict(),
        }

    def score(self, record: dict, with_recourse: bool = True) -> dict:
        m = self.b.model
        x = validate_applicant(record, m.categories)
        p_bad = float(m.predict_bad(x)[0])
        approved = p_bad < m.threshold
        base, contribs = self.b.explainer.explain(x)
        result = {
            "decision": "approve" if approved else "refer/decline",
            "probability_of_default": round(p_bad, 4),
            "threshold": m.threshold,
            "explanation": {
                "units": "log-odds of default (positive = riskier)",
                "base_value": round(base, 4),
                "model_score": round(float(self.b.explainer.margin(x)[0]), 4),
                "contributions": [c.to_dict() for c in contribs],
            },
            "reason_codes": [] if approved else reason_codes(contribs),
            "recourse": [],
        }
        if not approved and with_recourse:
            result["recourse"] = [o.to_dict() for o in find_recourse(x, m.predict_bad, m.threshold, m.categories)]
        return result

    def model_info(self) -> dict:
        m = self.b.model
        return {"version": self.b.version, "trained_at": m.trained_at, "data_fingerprint": m.data_fingerprint,
                "cv_auc": m.cv_auc, "metrics": m.metrics, "fairness": self.b.fairness,
                "global_importance": self.b.importance}


def approval_rate(bundle: Bundle) -> float:
    return float(np.mean(bundle.model.test_p_bad < bundle.model.threshold))
