"""Train the model, save the artifact and write a model card: ``python -m credit_risk.train``."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from .model import COST_FALSE_APPROVAL, COST_FALSE_REJECTION
from .service import ARTIFACT, Bundle, build_bundle, save_bundle

CARD = Path(__file__).resolve().parent.parent / "MODEL_CARD.md"


def model_card(b: Bundle) -> str:
    m = b.model.metrics
    c = m["confusion"]
    lines = [
        "# Model card: explainable credit-risk scorer", "",
        f"*Generated {b.model.trained_at} · version {b.version} · data fingerprint `{b.model.data_fingerprint}`*", "",
        "## Intended use", "",
        "Educational and portfolio demonstration of explainable, fairness-audited credit scoring on the public German "
        "Credit (Statlog) dataset. **Not for real lending decisions**: the data is from 1994, small (1,000 applicants) "
        "and not representative of any current population.", "",
        "## Data", "",
        "German Credit (Statlog), UCI / OpenML `credit-g` (CC BY 4.0). 1,000 applicants, 30% labelled *bad*. "
        "Stratified 75/25 train/test split.", "",
        "Protected attributes (`personal_status` which encodes sex, `age`, `foreign_worker`) are **excluded from the model "
        "inputs** and used only for the fairness audit below.", "",
        "## Model", "",
        f"- Candidates (5-fold CV ROC-AUC on train): " + ", ".join(f"{k} {v}" for k, v in b.model.cv_auc.items()),
        f"- Selected: **{m['model']}**, sigmoid-calibrated (5-fold `CalibratedClassifierCV`)",
        f"- Decision threshold: reject when P(default) >= **{m['threshold']}**, chosen on out-of-fold predictions to minimise "
        f"expected cost with the Statlog cost matrix (false approval = {COST_FALSE_APPROVAL:g}, false rejection = {COST_FALSE_REJECTION:g})", "",
        "## Test-set performance", "",
        "| Metric | Value |", "|---|---|",
        f"| ROC-AUC | {m['roc_auc']} |", f"| Brier score | {m['brier']} |",
        f"| Expected cost per applicant | {m['expected_cost']} (reject-all baseline {m['cost_reject_all']}, approve-all {m['cost_approve_all']}) |",
        f"| Approval rate | {m['approval_rate']:.1%} |", f"| Default rate among approved | {m['bad_rate_among_approved']:.1%} |",
        f"| Confusion | approved good {c['approved_good']}, rejected good {c['rejected_good']}, approved bad {c['approved_bad']}, rejected bad {c['rejected_bad']} |",
        "",
        "## Most influential features (mean |SHAP|, test set)", "",
        "| Feature | Mean abs SHAP (log-odds) |", "|---|---|",
        *[f"| {g['label']} | {g['mean_abs_shap']} |" for g in b.importance[:8]], "",
        "## Fairness audit (test set)", "",
        "Approval rate = demographic parity; *approval rate if creditworthy* = equal opportunity. "
        "Disparate impact ratio below 0.8 fails the four-fifths rule. Groups with fewer than 10 members are reported but not compared.", "",
    ]
    for attr, r in b.fairness.items():
        verdict = {True: "passes", False: "**fails**", None: "not assessable (group too small)"}[r["passes_four_fifths_rule"]]
        lines += [f"**{attr}**: disparate impact ratio {r['disparate_impact_ratio']}, equal opportunity difference "
                  f"{r['equal_opportunity_difference']}, four-fifths rule {verdict}", "",
                  "| Group | n | Approval rate | Approval rate if creditworthy | Actual default rate |", "|---|---|---|---|---|",
                  *[f"| {g['group']} | {g['n']} | {g['approval_rate']} | {g['approval_rate_if_creditworthy']} | {g['actual_bad_rate']} |" for g in r["groups"]], ""]
    lines += [
        "## Limitations", "",
        "- Excluding protected attributes does not remove proxy effects; monitor the audit above on real outcome data.",
        "- The cost-optimal threshold is strict (high cost of false approvals), so the approval rate is low by design.",
        "- SHAP values explain the uncalibrated model's log-odds; calibration is monotone, so directions and rankings are preserved.",
        "- Recourse suggestions are model-based what-ifs, not guarantees, and only consider actionable features.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the credit-risk model and write the model card")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ARTIFACT)
    p.add_argument("--card", type=Path, default=CARD)
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    bundle = build_bundle(seed=args.seed)
    save_bundle(bundle, args.out)
    args.card.write_text(model_card(bundle), encoding="utf-8", newline="\n")
    m = bundle.model.metrics
    print(f"{m['model']}: ROC-AUC {m['roc_auc']}, threshold {m['threshold']}, approval {m['approval_rate']:.1%}")
    print(f"artifact -> {args.out}\nmodel card -> {args.card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
