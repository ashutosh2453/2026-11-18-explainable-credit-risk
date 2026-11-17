"""Dataset loading, schema and applicant validation for the German Credit (Statlog) data."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

DATA_PATH = Path(os.getenv("CREDIT_DATA", Path(__file__).resolve().parent.parent / "data" / "german_credit.csv"))
TARGET = "class"  # "good" / "bad"; the model predicts P(bad)

NUMERIC = ["duration", "credit_amount", "installment_commitment", "residence_since", "existing_credits", "num_dependents"]
CATEGORICAL = ["checking_status", "credit_history", "purpose", "savings_status", "employment", "other_parties",
               "property_magnitude", "other_payment_plans", "housing", "job", "own_telephone"]

# Protected / sensitive attributes are deliberately NOT model inputs. They are only used to audit
# outcomes for disparate impact (personal_status encodes sex in this dataset).
PROTECTED = ["personal_status", "age", "foreign_worker"]
FEATURES = NUMERIC + CATEGORICAL

LABELS = {
    "checking_status": "Checking account balance (DM)",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount (DM)",
    "savings_status": "Savings balance (DM)",
    "employment": "Years in current employment",
    "installment_commitment": "Instalment as % of income (1-4)",
    "other_parties": "Co-applicant / guarantor",
    "residence_since": "Years at current residence",
    "property_magnitude": "Most valuable property",
    "other_payment_plans": "Other instalment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits at this bank",
    "job": "Job type",
    "num_dependents": "Number of dependents",
    "own_telephone": "Registered telephone",
}

NUMERIC_BOUNDS = {
    "duration": (1, 120), "credit_amount": (100, 100_000), "installment_commitment": (1, 4),
    "residence_since": (1, 4), "existing_credits": (1, 10), "num_dependents": (0, 10),
}


class ApplicantError(ValueError):
    """An applicant record failed validation."""


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + PROTECTED + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("dataset contains missing values")
    return df


def dataset_fingerprint(path: Path = DATA_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def target(df: pd.DataFrame) -> pd.Series:
    return (df[TARGET] == "bad").astype(int)


def category_values(df: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAL}


def validate_applicant(record: dict, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Validate one applicant dict and return it as a single-row DataFrame in model column order."""
    errors = []
    row = {}
    for f in NUMERIC:
        v = record.get(f)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{f}: must be a number")
            continue
        lo, hi = NUMERIC_BOUNDS[f]
        if not lo <= v <= hi:
            errors.append(f"{f}: must be between {lo} and {hi}")
        row[f] = v
    for f in CATEGORICAL:
        v = record.get(f)
        if v not in categories[f]:
            errors.append(f"{f}: must be one of {categories[f]}")
        row[f] = v
    unknown = set(record) - set(FEATURES)
    if unknown:
        errors.append(f"unknown fields: {sorted(unknown)} (protected attributes are not accepted)")
    if errors:
        raise ApplicantError("; ".join(errors))
    return pd.DataFrame([row], columns=FEATURES)
