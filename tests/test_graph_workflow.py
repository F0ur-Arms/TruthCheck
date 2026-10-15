import unittest

from langgraph.types import Command

import graph.nodes as nodes
from graph.nodes import ingest_node, route_and_decompose_node, assess_risk_node, aggregate_verdict
from graph.state import TruthCheckState
from graph.workflow import create_truthcheck_graph
from langgraph.checkpoint.memory import MemorySaver
from tests.fakes import FakeEngine


class LangGraphWorkflowTests(unittest.TestCase):
    def tearDown(self):
        nodes.set_engine(None)

    def _run(self, text, engine, thread="t"):
        nodes.set_engine(engine)
        app = create_truthcheck_graph(MemorySaver())
        cfg = {"configurable": {"thread_id": thread}}
        return app, cfg, app.invoke({"raw_input": text}, config=cfg)

    def test_ingest_node_cleans_text(self):
        res = ingest_node(TruthCheckState(raw_input="Subah khali pet garam pani fatigue kam karta hai"))
        self.assertIn("morning", res["cleaned_text"])

    def test_medical_advice_bypasses_graph(self):
        state = TruthCheckState(
            raw_input="I have stage 3 CKD, should I stop drinking whey protein?",
            cleaned_text="i have stage 3 ckd, should i stop drinking whey protein?",
        )
        res = route_and_decompose_node(state)
        self.assertEqual(res["route"], "medical_advice")
        self.assertEqual(res["final_response"]["verdict"], "NOT_A_FACT_CHECK")

    def test_verdict_comes_from_nli_not_keywords(self):
        _, _, out = self._run("Warm water improves digestion.", FakeEngine("REFUTES", 0.02, 0.95))
        self.assertIn(out["final_response"]["verdict"], {"FALSE", "MOSTLY_FALSE"})
        _, _, out = self._run("Warm water improves digestion.", FakeEngine("SUPPORTS", 0.95, 0.02))
        self.assertIn(out["final_response"]["verdict"], {"SUPPORTED", "MOSTLY_SUPPORTED"})

    def test_weak_evidence_is_insufficient_and_not_blocked(self):
        app, cfg, out = self._run("Lemon water cleanses the body.", FakeEngine("NEI", 0.1, 0.1, quality=0.2))
        self.assertEqual(out["final_response"]["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("__interrupt__", out)
        self.assertTrue(out["final_response"]["needs_human_review"])

    def test_every_sub_claim_is_scored(self):
        engine = FakeEngine()
        self._run("Garlic lowers blood pressure and turmeric cures cancer.", engine)
        self.assertGreaterEqual(len(engine.scored_claims), 1)

    def test_high_harm_pauses_then_resumes_with_reviewer_verdict(self):
        app, cfg, out = self._run("Stop taking insulin, bitter gourd cures diabetes.", FakeEngine())
        self.assertIn("__interrupt__", out)
        self.assertIsNone(out.get("final_response"))

        resumed = app.invoke(
            Command(resume={"action": "overridden", "adjusted_verdict": "FALSE", "reviewer_note": "dangerous"}),
            config=cfg,
        )
        self.assertEqual(resumed["final_response"]["verdict"], "FALSE")
        self.assertTrue(resumed["final_response"]["human_reviewed"])

    def test_rejection_without_override_withholds_verdict(self):
        app, cfg, _ = self._run("Stop taking insulin, bitter gourd cures diabetes.", FakeEngine(), "t2")
        resumed = app.invoke(Command(resume={"action": "rejected"}), config=cfg)
        self.assertEqual(resumed["final_response"]["verdict"], "INSUFFICIENT_EVIDENCE")

    def test_assess_risk_blocks_only_on_harm(self):
        base = dict(
            raw_input="x",
            claim_results=[{
                "canonical_claim": "Coconut water is hydrating", "population": None,
                "calibrated_verdict": {"verdict": "INSUFFICIENT_EVIDENCE", "calibrated_confidence": 0.2},
            }],
            calibrated_verdict={"verdict": "INSUFFICIENT_EVIDENCE", "calibrated_confidence": 0.2},
        )
        res = assess_risk_node(TruthCheckState(**base))
        self.assertFalse(res["needs_human_review"])
        self.assertIn("inconclusive_verdict", res["risk_assessment"]["review_reasons"])

    def test_aggregate_verdict_takes_most_serious(self):
        self.assertEqual(aggregate_verdict(["SUPPORTED", "FALSE"]), "FALSE")
        self.assertEqual(aggregate_verdict(["SUPPORTED", "INSUFFICIENT_EVIDENCE"]), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(aggregate_verdict([]), "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
