# Model card: explainable credit-risk scorer

*Generated 2026-11-17T19:01:09+00:00 · version 0.1.0 · data fingerprint `9888fb06e7bf4296`*

## Intended use

Educational and portfolio demonstration of explainable, fairness-audited credit scoring on the public German Credit (Statlog) dataset. **Not for real lending decisions**: the data is from 1994, small (1,000 applicants) and not representative of any current population.

## Data

German Credit (Statlog), UCI / OpenML `credit-g` (CC BY 4.0). 1,000 applicants, 30% labelled *bad*. Stratified 75/25 train/test split.

Protected attributes (`personal_status` which encodes sex, `age`, `foreign_worker`) are **excluded from the model inputs** and used only for the fairness audit below.

## Model

- Candidates (5-fold CV ROC-AUC on train): logistic_regression 0.7636, gradient_boosting 0.7709
- Selected: **gradient_boosting**, sigmoid-calibrated (5-fold `CalibratedClassifierCV`)
- Decision threshold: reject when P(default) >= **0.19**, chosen on out-of-fold predictions to minimise expected cost with the Statlog cost matrix (false approval = 5, false rejection = 1)

## Test-set performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.8142 |
| Brier score | 0.1526 |
| Expected cost per applicant | 0.556 (reject-all baseline 0.7, approve-all 1.5) |
| Approval rate | 40.8% |
| Default rate among approved | 10.8% |
| Confusion | approved good 91, rejected good 84, approved bad 11, rejected bad 64 |

## Most influential features (mean |SHAP|, test set)

| Feature | Mean abs SHAP (log-odds) |
|---|---|
| Checking account balance (DM) | 0.7436 |
| Loan duration (months) | 0.4448 |
| Credit amount (DM) | 0.4078 |
| Savings balance (DM) | 0.3429 |
| Credit history | 0.3053 |
| Years in current employment | 0.239 |
| Loan purpose | 0.2343 |
| Other instalment plans | 0.2108 |

## Fairness audit (test set)

Approval rate = demographic parity; *approval rate if creditworthy* = equal opportunity. Disparate impact ratio below 0.8 fails the four-fifths rule. Groups with fewer than 10 members are reported but not compared.

**sex**: disparate impact ratio 0.8345, equal opportunity difference 0.0421, four-fifths rule passes

| Group | n | Approval rate | Approval rate if creditworthy | Actual default rate |
|---|---|---|---|---|
| female | 78 | 0.359 | 0.4902 | 0.3462 |
| male | 172 | 0.4302 | 0.5323 | 0.2791 |

**age_band**: disparate impact ratio 0.8417, equal opportunity difference 0.1021, four-fifths rule passes

| Group | n | Approval rate | Approval rate if creditworthy | Actual default rate |
|---|---|---|---|---|
| 25 and over | 202 | 0.4208 | 0.5066 | 0.2475 |
| under 25 | 48 | 0.3542 | 0.6087 | 0.5208 |

**foreign_worker**: disparate impact ratio None, equal opportunity difference None, four-fifths rule not assessable (group too small)

| Group | n | Approval rate | Approval rate if creditworthy | Actual default rate |
|---|---|---|---|---|
| foreign worker | 242 | 0.3967 | 0.506 | 0.3058 |
| local worker | 8 | 0.75 | 0.8571 | 0.125 |

## Limitations

- Excluding protected attributes does not remove proxy effects; monitor the audit above on real outcome data.
- The cost-optimal threshold is strict (high cost of false approvals), so the approval rate is low by design.
- SHAP values explain the uncalibrated model's log-odds; calibration is monotone, so directions and rankings are preserved.
- Recourse suggestions are model-based what-ifs, not guarantees, and only consider actionable features.
