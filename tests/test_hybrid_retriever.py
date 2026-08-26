import unittest
from src.hybrid_retriever import MultiLaneQueryGenerator, HybridRetriever, Reranker


class HybridRetrieverTests(unittest.TestCase):
    def test_multi_lane_query_generator(self):
        lanes = MultiLaneQueryGenerator.generate_query_lanes("Whey protein causes kidney damage", "whey protein", "kidney damage")
        self.assertIn("support", lanes)
        self.assertIn("contradiction", lanes)
        self.assertIn("guideline", lanes)
        self.assertIn("population", lanes)
        self.assertIn("india_context", lanes)
        self.assertIn("does not cause", lanes["contradiction"])
        self.assertIn("ICMR", lanes["india_context"])

    def test_hybrid_retrieval_combines_bm25_and_dense(self):
        passages = [
            "Warm water helps break down food faster and improves digestion.",
            "Kidney damage can be caused by severe chronic illness.",
            "Whey protein supplementation is safe for healthy adults.",
            "ICMR NIN dietary guidelines recommend balanced protein intake."
        ]

        def dummy_dense_fn(query, top_k=2):
            return [passages[0], passages[3]]

        retriever = HybridRetriever(passages=passages, dense_retriever_fn=dummy_dense_fn)
        lanes = {"support": "warm water digestion", "guideline": "ICMR protein guidelines"}
        results = retriever.retrieve_hybrid(lanes, top_per_lane=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertIn("passage", results[0])
        self.assertIn("rrf_score", results[0])

    def test_reranker_scores_candidates(self):
        reranker = Reranker(model_name="dummy-nonexistent-model")  # Will use fallback gracefully
        candidates = [
            {"passage": "Warm water promotes hydration and aids digestion.", "rrf_score": 0.05},
            {"passage": "Space rockets rely on rocket fuel.", "rrf_score": 0.04}
        ]

        reranked = reranker.rerank("Warm water digestion", candidates, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertIn("rerank_score", reranked[0])
        self.assertGreater(reranked[0]["rerank_score"], reranked[1]["rerank_score"])


if __name__ == "__main__":
    unittest.main()
