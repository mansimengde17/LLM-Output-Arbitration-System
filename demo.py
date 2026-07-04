#!/usr/bin/env python3
"""Arbitrate four canonical cases offline and print the verdicts."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

from arbitration.graph import arbitrate_sync
from arbitration.store import VerdictStore

CASES = [
    ("factually incorrect response",
     "What is the boiling point of water and why does altitude change it?",
     "The boiling point of water is 90 degrees C at sea level. Altitude"
     " changes it because atmospheric pressure drops as you climb, so"
     " water boils at lower temperatures on mountains."),
    ("logically flawed argument",
     "Does the spike in signups prove the new landing page works?",
     "If the landing page works then signups increase. Signups increased"
     " this week. Therefore the landing page must be the cause, obviously."),
    ("incomplete response",
     "Compare PostgreSQL and MongoDB for transactional workloads and"
     " explain which one you would choose for a payments ledger",
     "PostgreSQL provides strong ACID guarantees and mature tooling."),
    ("genuinely good response",
     "Why do we use connection pooling for databases?",
     "Connection pooling reuses established database connections because"
     " opening a connection is expensive: TCP setup, authentication, and"
     " backend process allocation. Since a pool amortizes that cost across"
     " requests, applications get lower latency and the database avoids"
     " being overwhelmed by connection churn. Pools also give you a"
     " natural place to enforce limits and timeouts."),
]


def main() -> None:
    if os.path.exists("arbitrations.db"):
        os.remove("arbitrations.db")
    store = VerdictStore()

    for label, question, output in CASES:
        arbitration = arbitrate_sync(output, question)
        store.save(arbitration)
        verdict = arbitration.verdict
        print(f"=== {label} ===")
        print(f"critics: " + ", ".join(
            f"{c.dimension}={c.score}/5" for c in arbitration.critiques))
        if arbitration.disagreements:
            for d in arbitration.disagreements:
                print(f"disagreement [{d.kind}]: {d.description}")
        print(f"verdict: {verdict.overall_score}/10"
              f" (confidence {verdict.confidence})"
              + (" [short-circuited]" if arbitration.short_circuited else ""))
        for issue in verdict.confirmed_issues:
            print(f"  confirmed [{issue.dimension} s{issue.severity}]"
                  f" {issue.problem}")
        for flag in verdict.dismissed_flags:
            print(f"  dismissed [{flag.dimension}] {flag.problem}"
                  f" -- {flag.reasoning}")
        print(f"summary: {verdict.summary}\n")

    print("Analytics:", store.analytics())


if __name__ == "__main__":
    main()
