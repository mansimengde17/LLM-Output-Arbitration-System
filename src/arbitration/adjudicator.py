"""Adjudicator: weighs critic evidence, resolves conflicts, renders the
final verdict."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .critics import Critique, Issue


class ConfirmedIssue(BaseModel):
    dimension: str
    quote: str
    problem: str
    severity: int
    reasoning: str


class DismissedFlag(BaseModel):
    dimension: str
    problem: str
    reasoning: str


class Verdict(BaseModel):
    overall_score: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0.0, le=1.0)
    confirmed_issues: list[ConfirmedIssue] = []
    dismissed_flags: list[DismissedFlag] = []
    summary: str = ""


def clean_pass(critiques: list[Critique]) -> Verdict:
    """Short-circuit verdict when every critic reports clean."""
    mean_score = sum(c.score for c in critiques if not c.failed) / max(
        1, sum(not c.failed for c in critiques))
    return Verdict(
        overall_score=min(10, round(mean_score * 2)),
        confidence=0.95,
        summary="All critics independently reported the output clean;"
                " adjudication short-circuited to a high-confidence pass.")


def _corroborated(issue: Issue, critique: Critique,
                  others: list[Critique]) -> bool:
    """An issue is corroborated if another critic scored its dimension-adjacent
    quality low or flagged overlapping text."""
    for other in others:
        if other.dimension == critique.dimension or other.failed:
            continue
        if other.score <= 3:
            return True
        for other_issue in other.issues:
            overlap = set(issue.quote.lower().split()) & \
                set(other_issue.quote.lower().split())
            if len(overlap) >= 4:
                return True
    return False


def adjudicate(question: str, output: str, critiques: list[Critique],
               disagreements: list[dict]) -> Verdict:
    """Evidence-based resolution.

    Severity-3 findings stand on their own evidence (the quote either
    supports the problem or it does not). Severity-1 lone findings that no
    other critic corroborates are dismissed with reasoning, mirroring how a
    live LLM adjudicator is prompted to overrule weak flags.
    """
    confirmed: list[ConfirmedIssue] = []
    dismissed: list[DismissedFlag] = []
    active = [c for c in critiques if not c.failed]

    for critique in active:
        others = [c for c in active if c is not critique]
        for issue in critique.issues:
            if issue.severity >= 3:
                confirmed.append(ConfirmedIssue(
                    dimension=critique.dimension, quote=issue.quote,
                    problem=issue.problem, severity=issue.severity,
                    reasoning="critical finding backed by direct evidence"
                              " in the quoted text"))
            elif issue.severity == 2 or _corroborated(issue, critique, others):
                confirmed.append(ConfirmedIssue(
                    dimension=critique.dimension, quote=issue.quote,
                    problem=issue.problem, severity=issue.severity,
                    reasoning="major finding" if issue.severity == 2
                              else "minor finding corroborated by another"
                                   " critic's assessment"))
            else:
                dismissed.append(DismissedFlag(
                    dimension=critique.dimension, problem=issue.problem,
                    reasoning="lone minor flag with no corroboration from"
                              " other critics; overruled"))

    penalty = sum({1: 1, 2: 2, 3: 3}[i.severity] for i in confirmed)
    overall = max(1, 10 - penalty)
    agreement = 1.0 - 0.12 * len(disagreements)
    mean_self_confidence = (sum(c.self_confidence for c in active)
                            / max(1, len(active)))
    confidence = round(max(0.3, min(0.95, agreement * mean_self_confidence
                                    + (0.05 if not dismissed else 0.0))), 2)

    by_dim = {}
    for issue in confirmed:
        by_dim.setdefault(issue.dimension, 0)
        by_dim[issue.dimension] += 1
    summary_bits = [f"{count} {dim} issue(s)" for dim, count in by_dim.items()]
    summary = (f"Confirmed {', '.join(summary_bits)}."
               if confirmed else "No issues survived adjudication.")
    if dismissed:
        summary += f" Dismissed {len(dismissed)} uncorroborated flag(s)."
    if disagreements:
        summary += (f" Critics disagreed on {len(disagreements)} point(s),"
                    " which lowered confidence.")

    return Verdict(overall_score=overall, confidence=confidence,
                   confirmed_issues=confirmed, dismissed_flags=dismissed,
                   summary=summary)
