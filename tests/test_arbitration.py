import sys
import unittest

sys.path.insert(0, "src")

from arbitration.critics import Critique, Issue
from arbitration.graph import arbitrate_sync, detect_disagreements


class ArbitrationTests(unittest.TestCase):
    def test_factual_error_confirmed(self):
        arb = arbitrate_sync(
            "The boiling point of water is 90 degrees C at sea level.",
            "What is the boiling point of water?")
        self.assertLess(arb.verdict.overall_score, 8)
        self.assertTrue(any(i.dimension == "accuracy"
                            for i in arb.verdict.confirmed_issues))

    def test_clean_output_short_circuits(self):
        arb = arbitrate_sync(
            "Connection pooling reuses connections because establishing"
            " them is expensive, since TCP setup and authentication add"
            " latency; the evidence from production benchmarks shows"
            " pooled workloads sustain higher throughput.",
            "Why use connection pooling?")
        self.assertTrue(arb.short_circuited)
        self.assertGreaterEqual(arb.verdict.confidence, 0.9)

    def test_incomplete_answer_flagged(self):
        arb = arbitrate_sync(
            "PostgreSQL provides strong ACID guarantees.",
            "Compare PostgreSQL and MongoDB for transactional workloads and"
            " explain which one you would choose for a payments ledger")
        self.assertTrue(any(i.dimension == "completeness"
                            for i in arb.verdict.confirmed_issues))

    def test_disagreement_detection_score_spread(self):
        critiques = [
            Critique(dimension="accuracy", model_slot="a", score=5,
                     issues=[], self_confidence=0.9),
            Critique(dimension="logic", model_slot="b", score=2,
                     issues=[Issue(quote="q", problem="p", severity=2)],
                     self_confidence=0.8),
            Critique(dimension="completeness", model_slot="c", score=5,
                     issues=[], self_confidence=0.8),
        ]
        kinds = {d.kind for d in detect_disagreements(critiques)}
        self.assertIn("score_spread", kinds)
        self.assertIn("lone_finding", kinds)


if __name__ == "__main__":
    unittest.main()
