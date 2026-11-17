"""Actionable recourse: the cheapest realistic changes that would flip a rejection into an approval.

Only features an applicant can actually change are considered (never age, sex, nationality,
employment history or credit history), each move goes in a plausible direction only
(more savings, a smaller or shorter loan, ...), and every move has an effort cost. A beam
search over combinations of up to three moves returns the lowest-effort distinct options.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

CHECKING_ORDER = ["<0", "0<=X<200", ">=200"]
SAVINGS_ORDER = ["<100", "100<=X<500", "500<=X<1000", ">=1000"]


@dataclass(frozen=True)
class Move:
    feature: str
    new_value: object
    description: str
    effort: float


@dataclass
class RecourseOption:
    changes: list[dict]
    total_effort: float
    new_probability: float

    def to_dict(self) -> dict:
        return asdict(self)


def _upgrades(order: list[str], current: str, feature: str, noun: str, base_effort: float) -> list[Move]:
    if current not in order:
        return []
    i = order.index(current)
    return [Move(feature, v, f"Raise {noun} to {v} DM", base_effort * (k + 1)) for k, v in enumerate(order[i + 1:])]


def candidate_moves(row: pd.Series, categories: dict[str, list[str]]) -> list[Move]:
    moves: list[Move] = []
    moves += _upgrades(CHECKING_ORDER, row["checking_status"], "checking_status", "checking balance", 2.0)
    moves += _upgrades(SAVINGS_ORDER, row["savings_status"], "savings_status", "savings", 2.5)
    amount = int(row["credit_amount"])
    for pct in (10, 25, 40):
        new = int(round(amount * (1 - pct / 100), -1))
        if new >= 250:
            moves.append(Move("credit_amount", new, f"Borrow {new} DM instead of {amount} DM (-{pct}%)", pct / 10))
    months = int(row["duration"])
    for cut in (6, 12, 24):
        if months - cut >= 6:
            moves.append(Move("duration", months - cut, f"Shorten the loan to {months - cut} months", cut / 6))
    if row["installment_commitment"] > 1:
        moves.append(Move("installment_commitment", int(row["installment_commitment"]) - 1,
                          "Lower the instalment-to-income ratio by one band", 2.0))
    if row["existing_credits"] > 1:
        moves.append(Move("existing_credits", int(row["existing_credits"]) - 1, "Pay off one existing credit", 3.0))
    if row["other_payment_plans"] != "none" and "none" in categories["other_payment_plans"]:
        moves.append(Move("other_payment_plans", "none", "Close other instalment plans", 3.0))
    if row["other_parties"] == "none" and "guarantor" in categories["other_parties"]:
        moves.append(Move("other_parties", "guarantor", "Add a guarantor", 3.5))
    return moves


def find_recourse(applicant: pd.DataFrame, predict_bad, threshold: float, categories: dict[str, list[str]],
                  max_changes: int = 3, beam_width: int = 25, max_options: int = 3) -> list[RecourseOption]:
    row = applicant.iloc[0]
    moves = candidate_moves(row, categories)
    if predict_bad(applicant)[0] < threshold or not moves:
        return []

    def apply(combo: tuple[Move, ...]) -> pd.DataFrame:
        x = applicant.copy()
        for m in combo:
            x.at[x.index[0], m.feature] = m.new_value
        return x

    beam: list[tuple[Move, ...]] = [()]
    found: list[RecourseOption] = []
    seen_features: set[frozenset] = set()
    for _ in range(max_changes):
        expanded = []
        for combo in beam:
            used = {m.feature for m in combo}
            for m in moves:
                if m.feature not in used:
                    expanded.append(tuple(sorted(combo + (m,), key=lambda mv: mv.feature)))
        expanded = list(dict.fromkeys(expanded))
        if not expanded:
            break
        batch = pd.concat([apply(c) for c in expanded], ignore_index=True)
        probs = predict_bad(batch)
        scored = sorted(zip(expanded, probs), key=lambda t: (t[1] >= threshold, sum(m.effort for m in t[0]), t[1]))
        next_beam = []
        for combo, p in scored:
            feats = frozenset(m.feature for m in combo)
            if p < threshold:
                # Keep only minimal options: skip supersets of an option already found.
                if not any(prev <= feats for prev in seen_features):
                    seen_features.add(feats)
                    found.append(RecourseOption(
                        [{"feature": m.feature, "new_value": m.new_value, "action": m.description} for m in combo],
                        round(sum(m.effort for m in combo), 2), round(float(p), 4)))
            else:
                next_beam.append(combo)
        beam = next_beam[:beam_width]
    found.sort(key=lambda o: (o.total_effort, o.new_probability))
    return found[:max_options]
