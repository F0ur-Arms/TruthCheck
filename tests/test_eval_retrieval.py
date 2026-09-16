import unittest
from evaluation.eval_retrieval import RetrievalEvaluator


class RetrievalEvalTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RetrievalEvaluator()

    def test_retrieval_evaluation_harness(self):
        corpus = [
            "Warm water improves digestion by breaking down food faster.",
            "Curd at night is safe and does not cause cold.",
            "Plastic rice is a viral myth."
        ]
        eval_items = [
            {
                "claim": "Warm water digestion",
                "relevant_passages": ["Warm water improves digestion by breaking down food faster."]
            }
        ]

        report = self.evaluator.evaluate_dataset(eval_items, corpus)

        self.assertEqual(len(report.lane_metrics), 5)
        self.assertGreaterEqual(report.overall_mrr, 0.5)
        self.assertGreaterEqual(report.reranking_ndcg_after, 0.5)


if __name__ == "__main__":
    unittest.main()
