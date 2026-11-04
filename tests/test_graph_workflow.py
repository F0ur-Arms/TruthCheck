import os
import unittest
from unittest.mock import patch

from langgraph.types import Command

import graph.nodes as nodes
from graph.nodes import ingest_node, route_and_decompose_node, assess_risk_node, aggregate_verdict
from graph.state import TruthCheckState
from graph.workflow import create_truthcheck_graph
from langgraph.checkpoint.memory import MemorySaver
from tests.fakes import FakeEngine


REVIEW_ON = {"TRUTHCHECK_HUMAN_REVIEW": "on"}


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

    @patch.dict(os.environ, REVIEW_ON)
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

    @patch.dict(os.environ, REVIEW_ON)
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

    def _assess(self, text, verdict, llm_harm, raw="x"):
        return assess_risk_node(TruthCheckState(
            raw_input=raw,
            claim_results=[{
                "canonical_claim": text, "population": None, "high_harm": llm_harm,
                "calibrated_verdict": {"verdict": verdict, "calibrated_confidence": 0.9},
            }],
            calibrated_verdict={"verdict": verdict, "calibrated_confidence": 0.9},
        ))

    def test_nothing_waits_on_a_reviewer_by_default(self):
        _, _, out = self._run("Stop taking insulin, bitter gourd cures diabetes.", FakeEngine())
        self.assertNotIn("__interrupt__", out)
        self.assertEqual(out["final_response"]["verdict"], "WITHHELD")
        self.assertEqual(out["final_response"]["evidence"], [])
        self.assertEqual(out["final_response"]["claims"], [])
        self.assertIn("cannot confirm", out["final_response"]["safety_notice"])

    def test_harmful_claim_the_system_supports_is_withheld(self):
        for verdict in ("SUPPORTED", "SUPPORTED_WITH_CAVEATS", "MIXED_EVIDENCE"):
            res = self._assess("Drinking bleach cures COVID-19.", verdict, True)
            self.assertTrue(res["risk_assessment"]["withhold"])
            self.assertFalse(res["needs_human_review"])

    @patch.dict(os.environ, REVIEW_ON)
    def test_reviewer_mode_holds_instead_of_withholding(self):
        res = self._assess("Drinking bleach cures COVID-19.", "SUPPORTED", True)
        self.assertTrue(res["needs_human_review"])
        self.assertFalse(res["risk_assessment"]["withhold"])

    def test_harmful_claim_not_endorsed_is_answered_and_flagged(self):
        for verdict in ("FALSE", "INSUFFICIENT_EVIDENCE"):
            res = self._assess("Drinking bleach cures COVID-19.", verdict, True)
            self.assertFalse(res["risk_assessment"]["withhold"])
            self.assertTrue(res["risk_assessment"]["flag_for_review"])
            self.assertIn("high_harm", res["risk_assessment"]["review_reasons"])

    def test_llm_clears_a_descriptive_claim_the_keywords_would_hold(self):
        res = self._assess("Cancer is a disease.", "SUPPORTED", False)
        self.assertFalse(res["risk_assessment"]["withhold"])
        self.assertNotIn("high_harm", res["risk_assessment"]["review_reasons"])

    def test_keyword_flag_stops_any_verdict_when_llm_gave_no_judgement(self):
        self.assertTrue(self._assess("Cancer is a disease.", "FALSE", None)["risk_assessment"]["withhold"])
        self.assertFalse(self._assess("Cancer is a disease.", "INSUFFICIENT_EVIDENCE", None)["risk_assessment"]["withhold"])

    def test_cessation_language_is_never_endorsed_and_always_gets_a_notice(self):
        refuted = self._assess("Stop taking insulin.", "FALSE", False)["risk_assessment"]
        self.assertFalse(refuted["withhold"])
        self.assertTrue(refuted["high_harm"])
        self.assertTrue(self._assess("Stop taking insulin.", "SUPPORTED", False)["risk_assessment"]["withhold"])
        # The English sub-claim loses Hinglish wording, so the original text is checked too.
        res = self._assess("Bitter gourd helps diabetes.", "SUPPORTED", False, raw="sugar ki dawa band kar do")
        self.assertTrue(res["risk_assessment"]["withhold"])
        self.assertIn("treatment_cessation_override", res["risk_assessment"]["review_reasons"])

    def test_withheld_answer_never_carries_the_endorsement(self):
        res = nodes.generate_response_node(TruthCheckState(
            raw_input="x",
            claim_results=[{"canonical_claim": "x", "scored_passages": [{"passage": "p", "rerank_score": 1.0}],
                            "calibrated_verdict": {"verdict": "SUPPORTED", "calibrated_confidence": 0.9}}],
            calibrated_verdict={"verdict": "SUPPORTED", "calibrated_confidence": 0.9},
            risk_assessment={"high_harm": True, "withhold": True},
        ))["final_response"]
        self.assertEqual((res["verdict"], res["evidence"], res["claims"]), ("WITHHELD", [], []))
        self.assertNotIn("SUPPORTED", res["explanation"])

    def test_safety_notice_only_on_unendorsed_harmful_claims(self):
        def respond(verdict, high_harm):
            return nodes.generate_response_node(TruthCheckState(
                raw_input="x", claim_results=[],
                calibrated_verdict={"verdict": verdict, "calibrated_confidence": 0.9},
                risk_assessment={"high_harm": high_harm},
            ))["final_response"]["safety_notice"]
        self.assertIn("goes against", respond("FALSE", True))
        self.assertIn("could not verify", respond("INSUFFICIENT_EVIDENCE", True))
        self.assertIsNone(respond("SUPPORTED", True))
        self.assertIsNone(respond("FALSE", False))

    def test_aggregated_verdict_survives_into_the_final_response(self):
        nodes.set_engine(FakeEngine())
        out = nodes.evidence_and_calibration_node(TruthCheckState(
            raw_input="x",
            sub_claims=[{"claim_id": "c1", "canonical_claim": "a"}, {"claim_id": "c2", "canonical_claim": "b"}],
            claim_evidence=[{"claim_id": "c1", "reranked": [{"passage": "p", "passage_index": 0, "rerank_score": 1.0}]},
                            {"claim_id": "c2", "reranked": []}],
        ))
        self.assertEqual(out["calibrated_verdict"]["verdict"], "SUPPORTED_WITH_CAVEATS")
        self.assertEqual(len(out["claim_results"]), 2)

    def test_aggregate_verdict_rules(self):
        self.assertEqual(aggregate_verdict([]), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(aggregate_verdict(["SUPPORTED", "SUPPORTED"]), "SUPPORTED")
        self.assertEqual(aggregate_verdict(["FALSE", "INSUFFICIENT_EVIDENCE"]), "FALSE")
        self.assertEqual(aggregate_verdict(["MOSTLY_FALSE", "FALSE"]), "FALSE")
        self.assertEqual(aggregate_verdict(["SUPPORTED", "FALSE"]), "MIXED_EVIDENCE")
        self.assertEqual(aggregate_verdict(["SUPPORTED", "INSUFFICIENT_EVIDENCE"]), "SUPPORTED_WITH_CAVEATS")
        self.assertEqual(
            aggregate_verdict(["SUPPORTED", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"]),
            "INSUFFICIENT_EVIDENCE",
        )


if __name__ == "__main__":
    unittest.main()
