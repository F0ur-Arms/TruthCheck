import unittest

from src.evidence_engine import EvidenceEngine


class _Retriever:
    def retrieve_hybrid(self, lanes, top_per_lane=5):
        return [{"passage": f"p{i}", "passage_index": i} for i in range(3)]


class _Reranker:
    def rerank(self, claim, candidates, top_k=5):
        for c, s in zip(candidates, (0.9, 0.61, 0.2)):
            c["rerank_score"] = s
        return candidates[:top_k]


def _engine(min_relevance):
    return EvidenceEngine(kb_manager=None, hybrid_retriever=_Retriever(), reranker=_Reranker(),
                          nli_judge=None, min_relevance=min_relevance)


class EvidenceEngineGateTests(unittest.TestCase):
    def test_gate_drops_off_topic_passages(self):
        kept = _engine(0.6).retrieve("claim", {"support": "claim"})
        self.assertEqual([p["passage"] for p in kept], ["p0", "p1"])

    def test_no_gate_keeps_everything(self):
        self.assertEqual(len(_engine(None).retrieve("claim", {"support": "claim"})), 3)

    def test_nothing_relevant_means_no_passages(self):
        self.assertEqual(_engine(0.95).retrieve("claim", {"support": "claim"}), [])


if __name__ == "__main__":
    unittest.main()
