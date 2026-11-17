import warnings

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk.api import create_app
from credit_risk.data import FEATURES, PROTECTED, ApplicantError, load_dataset, validate_applicant
from credit_risk.explain import reason_codes
from credit_risk.fairness import audit
from credit_risk.model import best_threshold, expected_cost
from credit_risk.recourse import CHECKING_ORDER, SAVINGS_ORDER, find_recourse
from credit_risk.service import Scorer, build_bundle

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")
def bundle():
    return build_bundle(seed=7, n_estimators=120)


@pytest.fixture(scope="session")
def scorer(bundle):
    return Scorer(bundle)


def record(row: pd.Series) -> dict:
    return {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in FEATURES}


@pytest.fixture(scope="session")
def examples(bundle):
    m = bundle.model
    order = np.argsort(m.test_p_bad)
    approved = record(m.test_frame.iloc[order[0]])
    rejected_idx = [i for i in order if m.test_p_bad[i] >= m.threshold]
    borderline = record(m.test_frame.iloc[rejected_idx[0]])
    worst = record(m.test_frame.iloc[order[-1]])
    return {"approved": approved, "borderline": borderline, "worst": worst}


# ---------------------------------------------------------------- data validation

def test_dataset_schema():
    df = load_dataset()
    assert df.shape == (1000, 21)
    assert (df["class"] == "bad").mean() == pytest.approx(0.3)


def test_validate_applicant_rejects_bad_records(bundle, examples):
    cats = bundle.model.categories
    assert list(validate_applicant(examples["approved"], cats).columns) == FEATURES
    for patch, msg in [({"duration": "long"}, "must be a number"), ({"duration": True}, "must be a number"),
                       ({"credit_amount": 5}, "between 100"), ({"purpose": "yacht"}, "must be one of"),
                       ({"age": 30}, "protected attributes are not accepted")]:
        with pytest.raises(ApplicantError, match=msg):
            validate_applicant({**examples["approved"], **patch}, cats)
    bad = dict(examples["approved"])
    del bad["housing"]
    with pytest.raises(ApplicantError, match="housing"):
        validate_applicant(bad, cats)


# ---------------------------------------------------------------- model

def test_cost_and_threshold_logic():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.3, 0.2, 0.1])
    assert expected_cost(y, p, 0.5) == pytest.approx(5 / 5)   # one false approval
    assert expected_cost(y, p, 0.35) == pytest.approx(0.0)
    assert 0.31 <= best_threshold(y, p) <= 0.4


def test_model_quality_beats_baselines(bundle):
    m = bundle.model.metrics
    assert m["roc_auc"] > 0.72
    assert m["brier"] < 0.21                       # better than predicting the base rate
    assert m["expected_cost"] < min(m["cost_reject_all"], m["cost_approve_all"])
    assert sum(m["confusion"].values()) == m["test_size"] == 250


def test_protected_attributes_are_not_model_inputs(bundle):
    used = set(bundle.model.explain_pipeline.named_steps["prep"].feature_names_in_)
    assert used == set(FEATURES) and not used & set(PROTECTED)


# ---------------------------------------------------------------- explanations

def test_shap_contributions_are_additive(bundle):
    ex = bundle.explainer
    X = bundle.model.test_frame[FEATURES].iloc[:20]
    for i in range(len(X)):
        base, contribs = ex.explain(X.iloc[[i]])
        assert {c.feature for c in contribs} == set(FEATURES)
        assert base + sum(c.shap for c in contribs) == pytest.approx(ex.margin(X.iloc[[i]])[0], abs=1e-3)


def test_global_importance_ranks_checking_status_high(bundle):
    top = [g["feature"] for g in bundle.importance[:4]]
    assert "checking_status" in top
    assert bundle.importance == sorted(bundle.importance, key=lambda g: -g["mean_abs_shap"])


def test_reason_codes_only_cite_risk_increasing_factors(bundle, examples):
    _, contribs = bundle.explainer.explain(validate_applicant(examples["worst"], bundle.model.categories))
    codes = reason_codes(contribs)
    risky = [c for c in contribs if c.shap > 0][:4]
    assert codes == [f"{c.label}: {c.value}" for c in risky] and 1 <= len(codes) <= 4


# ---------------------------------------------------------------- recourse

IMMUTABLE = {"credit_history", "purpose", "employment", "property_magnitude", "housing", "job", "own_telephone",
             "residence_since", "num_dependents"}


def test_recourse_flips_decision_with_actionable_minimal_changes(bundle, examples):
    m = bundle.model
    x = validate_applicant(examples["borderline"], m.categories)
    options = find_recourse(x, m.predict_bad, m.threshold, m.categories)
    assert options, "a borderline rejection should have at least one recourse option"
    feature_sets = [frozenset(c["feature"] for c in o.changes) for o in options]
    for o, feats in zip(options, feature_sets):
        assert not feats & IMMUTABLE
        assert o.new_probability < m.threshold
        changed = x.copy()
        for c in o.changes:
            changed.at[changed.index[0], c["feature"]] = c["new_value"]
            orig = x.iloc[0][c["feature"]]
            if c["feature"] == "savings_status":
                assert SAVINGS_ORDER.index(c["new_value"]) > SAVINGS_ORDER.index(orig)
            if c["feature"] == "checking_status":
                assert CHECKING_ORDER.index(c["new_value"]) > CHECKING_ORDER.index(orig)
            if c["feature"] in ("credit_amount", "duration", "installment_commitment", "existing_credits"):
                assert c["new_value"] < orig
        assert m.predict_bad(changed)[0] == pytest.approx(o.new_probability, abs=1e-4)
    for a in feature_sets:  # no option is a superset of another
        assert not any(b < a for b in feature_sets)
    assert [o.total_effort for o in options] == sorted(o.total_effort for o in options)


def test_no_recourse_needed_for_approved_applicant(bundle, examples):
    m = bundle.model
    assert find_recourse(validate_applicant(examples["approved"], m.categories), m.predict_bad, m.threshold, m.categories) == []


# ---------------------------------------------------------------- fairness

def test_fairness_audit_metrics():
    df = pd.DataFrame({
        "personal_status": ["male single"] * 10 + ["female div/dep/mar"] * 10 + ["male single"] * 3,
        "age": [30] * 20 + [22] * 3,
        "foreign_worker": ["yes"] * 23,
    })
    y = np.array([0] * 23)
    approved = np.array([1] * 10 + [1] * 6 + [0] * 4 + [1] * 3, dtype=bool)
    r = audit(df, y, approved)
    assert r["sex"]["disparate_impact_ratio"] == pytest.approx(0.6 / 1.0)
    assert r["sex"]["passes_four_fifths_rule"] is False
    assert r["sex"]["equal_opportunity_difference"] == pytest.approx(0.4)
    assert r["age_band"]["disparate_impact_ratio"] is None  # under-25 group has only 3 members
    assert r["foreign_worker"]["passes_four_fifths_rule"] is None


def test_real_audit_covers_all_protected_attributes(bundle):
    assert set(bundle.fairness) == {"sex", "age_band", "foreign_worker"}
    assert sum(g["n"] for g in bundle.fairness["sex"]["groups"]) == 250


# ---------------------------------------------------------------- API

@pytest.fixture(scope="session")
def client(bundle):
    with TestClient(create_app(bundle)) as c:
        yield c


def test_api_schema_and_ui(client):
    s = client.get("/schema").json()
    assert s["order"] == FEATURES and "age" not in s["order"]
    assert "Credit Risk" in client.get("/").text
    assert client.get("/health").json()["status"] == "ok"


def test_api_scores_approval_and_decline(client, examples):
    ok = client.post("/score", json=examples["approved"]).json()
    assert ok["decision"] == "approve" and ok["reason_codes"] == [] and ok["recourse"] == []
    bad = client.post("/score", json=examples["borderline"]).json()
    assert bad["decision"] == "refer/decline" and bad["reason_codes"] and bad["recourse"]
    assert bad["probability_of_default"] >= bad["threshold"]
    e = bad["explanation"]
    assert e["base_value"] + sum(c["shap"] for c in e["contributions"]) == pytest.approx(e["model_score"], abs=1e-2)
    fast = client.post("/score?recourse=false", json=examples["borderline"]).json()
    assert fast["recourse"] == []


def test_api_validation_errors(client, examples):
    r = client.post("/score", json={**examples["approved"], "purpose": "yacht"})
    assert r.status_code == 422 and any("purpose" in d for d in r.json()["detail"])
    r = client.post("/score", json={**examples["approved"], "personal_status": "male single"})
    assert r.status_code == 422
    assert client.post("/score", content="not json", headers={"content-type": "application/json"}).status_code == 422


def test_api_model_info(client):
    m = client.get("/model").json()
    assert m["metrics"]["roc_auc"] > 0.72 and m["global_importance"] and m["fairness"]
