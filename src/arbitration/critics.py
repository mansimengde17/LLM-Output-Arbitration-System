"""Critic agents.

Each critic evaluates one dimension and returns the same structured
critique. Offline evaluators are deterministic; live mode assigns each
critic a different provider on purpose so they do not share blind spots.
"""

from __future__ import annotations

import asyncio
import re
from typing import Literal

from pydantic import BaseModel, Field


class Issue(BaseModel):
    quote: str
    problem: str
    severity: int = Field(ge=1, le=3)   # 1 minor, 2 major, 3 critical


class Critique(BaseModel):
    dimension: Literal["accuracy", "logic", "completeness"]
    model_slot: str
    score: int = Field(ge=1, le=5)
    issues: list[Issue] = []
    self_confidence: float = Field(ge=0.0, le=1.0)
    failed: bool = False


# Small fact database for the offline accuracy critic. Each entry is
# (pattern that indicates the claim, regex that must hold for it to be true).
KNOWN_FACTS = [
    ("boiling point of water", r"100\s?(degrees|°)?\s?c|212\s?(degrees|°)?\s?f"),
    ("speed of light", r"299,?792,?458|3\s?[x*]\s?10\^?8"),
    ("capital of australia", r"canberra"),
    ("python released", r"1991"),
    ("great wall.*visible from space", r"not|cannot|myth|false"),
    ("humans use.*percent of their brain", r"not|myth|false|all of"),
]

CONTRADICTION_PATTERNS = [
    (r"\balways\b", r"\bexcept\b|\bsometimes\b|\bnot always\b"),
    (r"\bimpossible\b", r"\bpossible\b"),
    (r"\bincreased\b", r"\bdecreased\b"),
]

HEDGE_WORDS = ("probably", "might", "possibly", "i think", "arguably")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class AccuracyCritic:
    dimension = "accuracy"
    model_slot = "gpt-4o"

    async def evaluate(self, question: str, output: str) -> Critique:
        issues: list[Issue] = []
        lowered = output.lower()
        for topic, truth_pattern in KNOWN_FACTS:
            if re.search(topic, lowered) and not re.search(truth_pattern, lowered):
                sentence = next((s for s in _sentences(output)
                                 if re.search(topic, s.lower())), output[:80])
                issues.append(Issue(
                    quote=sentence[:160],
                    problem=f"claim about '{topic}' contradicts established"
                            " reference facts",
                    severity=3))
        for a, b in CONTRADICTION_PATTERNS:
            if re.search(a, lowered) and re.search(b, lowered):
                issues.append(Issue(
                    quote=output[:120],
                    problem=f"internally inconsistent: contains both"
                            f" '{a.strip(chr(92) + 'b')}' and a contradicting"
                            " qualifier",
                    severity=2))
        score = max(1, 5 - 2 * sum(i.severity == 3 for i in issues)
                    - sum(i.severity == 2 for i in issues))
        return Critique(dimension="accuracy", model_slot=self.model_slot,
                        score=score, issues=issues, self_confidence=0.85)


class LogicCritic:
    dimension = "logic"
    model_slot = "claude-sonnet"

    async def evaluate(self, question: str, output: str) -> Critique:
        issues: list[Issue] = []
        lowered = output.lower()
        # Conclusion words without any supporting connective.
        has_conclusion = bool(re.search(r"\btherefore\b|\bthus\b|\bso\b|"
                                        r"\bproves\b|\bmust be\b", lowered))
        has_support = bool(re.search(r"\bbecause\b|\bsince\b|\bgiven\b|"
                                     r"\bevidence\b|\bdata\b", lowered))
        if has_conclusion and not has_support:
            issues.append(Issue(
                quote=next((s for s in _sentences(output)
                            if re.search(r"therefore|thus|proves|must be",
                                         s.lower())), output[:80])[:160],
                problem="conclusion asserted without stated premises",
                severity=2))
        # Classic affirming-the-consequent shape: "if A then B; B; therefore A"
        if re.search(r"if .+ then .+", lowered) and "therefore" in lowered:
            if re.search(r"(observed|we see|happened|is true).*therefore",
                         lowered, re.S):
                issues.append(Issue(
                    quote=output[:160],
                    problem="argument has the shape of affirming the"
                            " consequent (if A then B; B; therefore A)",
                    severity=3))
        if re.search(r"\beveryone knows\b|\bobviously\b|\bclearly\b", lowered):
            issues.append(Issue(
                quote=next((s for s in _sentences(output)
                            if re.search(r"everyone knows|obviously|clearly",
                                         s.lower())), output[:80])[:160],
                problem="appeal to obviousness in place of an argument",
                severity=1))
        score = max(1, 5 - 2 * sum(i.severity == 3 for i in issues)
                    - sum(i.severity == 2 for i in issues)
                    - sum(i.severity == 1 for i in issues) // 2)
        return Critique(dimension="logic", model_slot=self.model_slot,
                        score=score, issues=issues, self_confidence=0.8)


class CompletenessCritic:
    dimension = "completeness"
    model_slot = "llama-3-8b-local"

    async def evaluate(self, question: str, output: str) -> Critique:
        issues: list[Issue] = []
        # Split the question into requested parts and check coverage.
        parts = re.split(r"\band\b|;|\?\s+", question.lower())
        parts = [p.strip(" ?.") for p in parts if len(p.strip()) > 12]
        lowered = output.lower()
        for part in parts:
            keywords = [w for w in re.findall(r"[a-z]{5,}", part)
                        if w not in ("please", "explain", "describe",
                                     "compare", "should", "would", "could")]
            if not keywords:
                continue
            covered = sum(w in lowered for w in keywords) / len(keywords)
            if covered < 0.34:
                issues.append(Issue(
                    quote=part[:160],
                    problem="this part of the question is not addressed"
                            " in the response",
                    severity=2))
        if len(output.split()) < 25 and len(parts) > 1:
            issues.append(Issue(
                quote=output[:120],
                problem="response is too brief for a multi-part question",
                severity=1))
        hedges = sum(lowered.count(h) for h in HEDGE_WORDS)
        score = max(1, 5 - sum(i.severity == 2 for i in issues)
                    - sum(i.severity == 1 for i in issues) // 2)
        return Critique(dimension="completeness", model_slot=self.model_slot,
                        score=score, issues=issues,
                        self_confidence=max(0.5, 0.8 - 0.05 * hedges))


ALL_CRITICS = (AccuracyCritic(), LogicCritic(), CompletenessCritic())


async def dispatch_critics(question: str, output: str,
                           critics=ALL_CRITICS) -> list[Critique]:
    """Run all critics in parallel with graceful degradation."""
    async def safe(critic):
        try:
            return await critic.evaluate(question, output)
        except Exception:
            return Critique(dimension=critic.dimension,
                            model_slot=critic.model_slot, score=3,
                            issues=[], self_confidence=0.0, failed=True)
    return list(await asyncio.gather(*(safe(c) for c in critics)))
