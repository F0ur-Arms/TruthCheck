import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.evidence_engine import EvidenceEngine
from src.source_tier import source_url


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

    def test_excluded_sources_are_never_used_as_evidence(self):
        kb = MagicMock()
        kb.source_for_index.side_effect = lambda i: {0: "testing.txt"}.get(i, "who__x.txt")
        engine = EvidenceEngine(kb, _Retriever(), _Reranker(), None, excluded_sources=["testing.txt"])
        self.assertEqual([p["passage"] for p in engine.retrieve("c", {"support": "c"})], ["p1", "p2"])

    def test_source_url_comes_from_the_file_header(self):
        with tempfile.TemporaryDirectory() as kb:
            Path(kb, "who__x.txt").write_text("SOURCE: WHO\nTITLE: X\nURL: https://www.who.int/x\n---\nbody")
            Path(kb, "bad.txt").write_text("URL: javascript:alert(1)\nbody")
            Path(kb, "plain.txt").write_text("no header here")
            source_url.cache_clear()
            with patch("src.source_tier.KB_PATH", kb):
                self.assertEqual(source_url("who__x.txt"), "https://www.who.int/x")
                self.assertIsNone(source_url("bad.txt"))
                self.assertIsNone(source_url("plain.txt"))
                self.assertIsNone(source_url("missing.txt"))
            source_url.cache_clear()


if __name__ == "__main__":
    unittest.main()
