import unittest
from src.risk_engine_v2 import TwoAxisRiskEngine
from src.review_queue import ReviewQueueManager


class RiskEngineV2Tests(unittest.TestCase):
    def setUp(self):
        self.engine = TwoAxisRiskEngine()
        self.queue = ReviewQueueManager()

    def test_treatment_cessation_triggers_hard_override(self):
        claim = "Stop taking insulin, diabetes can be cured naturally with bitter gourd."
        res = self.engine.evaluate_risk(claim, verdict_confidence=0.95)

        self.assertTrue(res.has_hard_override)
        self.assertTrue(res.needs_human_review)
        self.assertGreaterEqual(res.harm_axis, 0.90)

    def test_low_harm_high_confidence_is_low_risk(self):
        claim = "Warm water improves digestion."
        res = self.engine.evaluate_risk(claim, verdict_confidence=0.90)

        self.assertFalse(res.has_hard_override)
        self.assertFalse(res.needs_human_review)
        self.assertLess(res.harm_axis, 0.50)

    def test_review_queue_submission_and_decision(self):
        item = self.queue.submit_for_review(
            claim_text="Turmeric cures cancer completely.",
            risk_assessment={"harm_axis": 0.85, "needs_human_review": True}
        )
        self.assertEqual(item.status, "pending")

        updated = self.queue.apply_reviewer_decision(
            request_id=item.request_id,
            action="overridden",
            adjusted_verdict="FALSE",
            note="Confirmed false medical claim."
        )
        self.assertEqual(updated.status, "overridden")
        self.assertEqual(updated.adjusted_verdict, "FALSE")


if __name__ == "__main__":
    unittest.main()
