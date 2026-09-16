import unittest
from evaluation.eval_failure_attribution import FailureAttributionEvaluator


class FailureAttributionTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = FailureAttributionEvaluator()

    def test_routing_failure_detection(self):
        trace = {"route": "fact_check"}
        res = self.evaluator.diagnose_trace(
            predicted_verdict="SUPPORTED",
            expected_verdict="NOT_A_FACT_CHECK",
            graph_trace=trace
        )
        self.assertFalse(res.is_successful)
        self.assertEqual(res.primary_failure_stage, "ROUTING_FAILURE")

    def test_retrieval_failure_detection(self):
        trace = {
            "route": "fact_check",
            "query_lanes": {"support": "q1", "contradiction": "q2", "guideline": "q3"},
            "candidate_passages": []
        }
        res = self.evaluator.diagnose_trace(
            predicted_verdict="INSUFFICIENT_EVIDENCE",
            expected_verdict="SUPPORTED",
            graph_trace=trace
        )
        self.assertFalse(res.is_successful)
        self.assertEqual(res.primary_failure_stage, "RETRIEVAL_FAILURE")


if __name__ == "__main__":
    unittest.main()
