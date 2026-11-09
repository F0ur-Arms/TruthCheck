import os
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

    def test_wider_pool_is_opt_in(self):
        with patch.dict(os.environ, {"TRUTHCHECK_RETRIEVAL_POOL": "50", "TRUTHCHECK_RERANK_KEEP": "2"}):
            retriever = MagicMock()
            retriever.retrieve_hybrid.return_value = [{"passage": f"p{i}"} for i in range(3)]
            engine = EvidenceEngine(None, retriever, _Reranker(), None)
            self.assertEqual(len(engine.retrieve("c", {"support": "c"})), 2)
            self.assertEqual(retriever.retrieve_hybrid.call_args.kwargs["top_per_lane"], 50)

    def test_excluded_sources_are_never_used_as_evidence(self):
        kb = MagicMock()
        kb.source_for_index.side_effect = lambda i: {0: "testing.txt"}.get(i, "who__x.txt")
        engine = EvidenceEngine(kb, _Retriever(), _Reranker(), None, excluded_sources=["testing.txt"])
        self.assertEqual([p["passage"] for p in engine.retrieve("c", {"support": "c"})], ["p1", "p2"])

    def test_excluded_prefix_turns_off_a_whole_batch(self):
        kb = MagicMock()
        files = {0: "pubmed_scifact_1.txt", 1: "pubmed_scifact_2.txt", 2: "pubmed_x.txt"}
        kb.source_for_index.side_effect = files.get
        engine = EvidenceEngine(kb, _Retriever(), _Reranker(), None, excluded_sources=["pubmed_scifact_*"])
        self.assertEqual([p["passage"] for p in engine.retrieve("c", {"support": "c"})], ["p2"])

    def test_plain_query_lanes_are_opt_in(self):
        engine = EvidenceEngine(None, _Retriever(), _Reranker(), None)
        default = engine.make_lanes("Laetrile cures cancer", "laetrile", "cancer")
        self.assertIn("population", default)
        with patch.dict(os.environ, {"TRUTHCHECK_QUERY_LANES": "v2"}):
            lanes = engine.make_lanes("Does laetrile cure cancer?", "laetrile", "cancer")
        self.assertEqual(lanes["support"], "Does laetrile cure cancer?")
        self.assertEqual(lanes["subject"], "laetrile")
        self.assertEqual(lanes["subject_outcome"], "laetrile cancer")
        self.assertEqual(lanes["keywords"], "laetrile cure cancer")
        self.assertFalse(any("CKD" in q or "does not cause" in q for q in lanes.values()))

    def test_plain_lanes_skip_a_subject_that_is_the_whole_claim(self):
        engine = EvidenceEngine(None, _Retriever(), _Reranker(), None)
        with patch.dict(os.environ, {"TRUTHCHECK_QUERY_LANES": "v2"}):
            lanes = engine.make_lanes("Turmeric helps the joints", "Turmeric helps the joints", None)
            no_stopwords = engine.make_lanes("Turmeric helps", None, None)
        self.assertEqual(lanes, {"support": "Turmeric helps the joints", "keywords": "Turmeric helps joints"})
        self.assertEqual(no_stopwords, {"support": "Turmeric helps"})


class _Judge:
    """NEI for any single passage; the combined text (passages joined by a blank line) gets `combined`."""

    def __init__(self, combined="SUPPORTS", single="NEI"):
        self.combined, self.single, self.calls = combined, single, 0

    def verify(self, claim, text):
        self.calls += 1
        label = self.combined if "\n\n" in text else self.single
        return {"verdict": label, "supports_probability": 0.8 if label == "SUPPORTS" else 0.1,
                "refutes_probability": 0.8 if label == "REFUTES" else 0.1}


def _scoring_engine(judge):
    kb = MagicMock()
    kb.source_for_index.side_effect = lambda i: f"who__{i}.txt"
    scorer = MagicMock()
    scorer.score_passage.return_value = MagicMock(overall_quality_score=0.5)
    return EvidenceEngine(kb, None, None, judge, evidence_scorer=scorer)


_RERANKED = [{"passage": f"part {i}", "passage_index": i, "rerank_score": s} for i, s in enumerate((0.9, 0.7, 0.3))]


class CombinedJudgmentTests(unittest.TestCase):
    def test_off_by_default(self):
        judge = _Judge()
        scored, _, _ = _scoring_engine(judge).score("claim", _RERANKED)
        self.assertEqual(len(scored), 3)
        self.assertEqual(judge.calls, 3)

    def test_all_nei_passages_are_judged_together(self):
        with patch.dict(os.environ, {"TRUTHCHECK_NLI_COMBINE": "2"}):
            scored, probs, _ = _scoring_engine(_Judge()).score("claim", _RERANKED)
        self.assertEqual(len(scored), 4)
        extra = scored[-1]
        self.assertEqual(extra["nli_verdict"], "SUPPORTS")
        self.assertEqual(extra["combined_from"], ["who__0.txt", "who__1.txt"])
        self.assertEqual(extra["source_file"], "who__0.txt")
        self.assertEqual(len(probs), 4)

    def test_inconclusive_combination_adds_nothing(self):
        with patch.dict(os.environ, {"TRUTHCHECK_NLI_COMBINE": "3"}):
            scored, _, _ = _scoring_engine(_Judge(combined="NEI")).score("claim", _RERANKED)
        self.assertEqual(len(scored), 3)

    def test_not_used_when_a_passage_already_decides(self):
        judge = _Judge(single="REFUTES")
        with patch.dict(os.environ, {"TRUTHCHECK_NLI_COMBINE": "3"}):
            scored, _, _ = _scoring_engine(judge).score("claim", _RERANKED)
        self.assertEqual(len(scored), 3)
        self.assertEqual(judge.calls, 3)


class SourceUrlTests(unittest.TestCase):
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
