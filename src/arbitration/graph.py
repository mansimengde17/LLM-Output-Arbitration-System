"""Orchestration graph: fan-out to critics, collect, detect disagreement,
adjudicate, synthesize the verdict."""

from __future__ import annotations

import asyncio
import uuid

from pydantic import BaseModel

from .adjudicator import Verdict, adjudicate, clean_pass
from .critics import Critique, dispatch_critics


class Disagreement(BaseModel):
    kind: str          # severity_gap | lone_finding | score_spread
    description: str
    dimensions: list[str]


def detect_disagreements(critiques: list[Critique]) -> list[Disagreement]:
    disagreements: list[Disagreement] = []
    active = [c for c in critiques if not c.failed]

    scores = {c.dimension: c.score for c in active}
    if scores and max(scores.values()) - min(scores.values()) >= 3:
        low = min(scores, key=scores.get)
        high = max(scores, key=scores.get)
        disagreements.append(Disagreement(
            kind="score_spread",
            description=f"{low} critic scored {scores[low]} while {high}"
                        f" critic scored {scores[high]}",
            dimensions=[low, high]))

    with_issues = [c for c in active if c.issues]
    without = [c for c in active if not c.issues]
    if len(with_issues) == 1 and len(without) >= 1:
        lone = with_issues[0]
        disagreements.append(Disagreement(
            kind="lone_finding",
            description=f"only the {lone.dimension} critic found issues"
                        f" ({len(lone.issues)}); others reported clean",
            dimensions=[lone.dimension]))

    severities = {c.dimension: max((i.severity for i in c.issues), default=0)
                  for c in active}
    if severities and max(severities.values()) - min(severities.values()) >= 2 \
            and min(severities.values()) > 0:
        disagreements.append(Disagreement(
            kind="severity_gap",
            description="critics disagree on how severe the issues are"
                        f" ({severities})",
            dimensions=list(severities)))
    return disagreements


class Arbitration(BaseModel):
    arbitration_id: str
    question: str
    output: str
    critiques: list[Critique]
    disagreements: list[Disagreement]
    verdict: Verdict
    short_circuited: bool


async def arbitrate(output: str, question: str = "") -> Arbitration:
    critiques = await dispatch_critics(question, output)
    disagreements = detect_disagreements(critiques)

    active = [c for c in critiques if not c.failed]
    all_clean = active and all(not c.issues and c.score >= 4 for c in active)
    if all_clean:
        verdict = clean_pass(critiques)
        short_circuited = True
    else:
        verdict = adjudicate(question, output, critiques,
                             [d.model_dump() for d in disagreements])
        short_circuited = False

    if any(c.failed for c in critiques):
        failed_dims = [c.dimension for c in critiques if c.failed]
        verdict.confidence = round(verdict.confidence * 0.7, 2)
        verdict.summary += (f" Note: the {', '.join(failed_dims)} critic(s)"
                            " failed; this dimension has lower confidence.")

    return Arbitration(
        arbitration_id=f"arb-{uuid.uuid4().hex[:8]}",
        question=question, output=output, critiques=critiques,
        disagreements=disagreements, verdict=verdict,
        short_circuited=short_circuited)


def arbitrate_sync(output: str, question: str = "") -> Arbitration:
    return asyncio.run(arbitrate(output, question))
