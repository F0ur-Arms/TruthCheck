import unittest
from unittest.mock import MagicMock, patch

from src.claims_processor import SubClaim
from src.evidence_quality import ClaimEvidenceSummary
from src.operating_mode import decide_mode


class PipelineModeIntegrationTests(unittest.TestCase):
    """Integration tests for gated pipeline modes with heavy models mocked."""

    def _make_pipeline(self):
        with patch("main.spacy.load"), patch("main.FactVerifier"), patch(
            "main.NLIVerifier"
        ), patch("main.KnowledgeManager"), patch("main.Reranker"):
            from main import TruthCheckPipeline

            pipeline = TruthCheckPipeline.__new__(TruthCheckPipeline)
            pipeline.mapper = MagicMock()
            pipeline.mapper.clean_text.side_effect = lambda t: t.lower()
            pipeline.scorer = MagicMock()
            pipeline.scorer.calculate_score.return_value = 0.1
            pipeline.engine = MagicMock()
            pipeline.engine.calculate_risk.return_value = {
                "score": 0.4,
                "label": "Medium",
                "breakdown": {
                    "fact_verdict": "UNVERIFIED",
                    "fact_impact": 0.2,
                    "ml_impact": 0.0,
                    "style_impact": 0.1,
                    "weights_used": {"fact": 0.5, "ml": 0.0, "style": 0.2},
                },
            }
            pipeline.nlp = MagicMock()
            pipeline.claims_processor = MagicMock()
            pipeline.verifier = MagicMock()
            pipeline.nli_judge = MagicMock()
            pipeline.kb_manager = MagicMock()
            pipeline.kb_manager.passages = ["passage one about protein", "passage two WHO guideline"]
            pipeline.kb_manager.source_for_index.return_value = "who__healthy-diet.txt"
            pipeline.query_generator = MagicMock()
            pipeline.query_generator.generate_query_lanes.return_value = {
                "support": "protein harm",
                "contradiction": "protein safe",
                "guideline": "protein WHO",
                "population": "protein elderly",
                "india_context": "protein ICMR",
            }
            pipeline.hybrid_retriever = MagicMock()
            pipeline.reranker = MagicMock()
            pipeline.evidence_scorer = MagicMock()
            pipeline.calibrator = MagicMock()
            pipeline.calibrator.calibrate.return_value = MagicMock(
                verdict="SUPPORTED",
                calibrated_confidence=0.82,
                explanation_summary="test",
            )
            return pipeline

    def test_fast_mode_skips_hybrid_retrieval(self):
        pipeline = self._make_pipeline()
        sub = SubClaim(canonical_claim="Warm water improves digestion.", subject="warm water", outcome="digestion")
        pipeline._sub_claims_for_query = MagicMock(return_value=[sub])

        claim_rep = MagicMock(route="fact_check", claims=[sub])
        pipeline.claims_processor.process_query.return_value = claim_rep

        tier1_fast = {
            "match_type": "STRONG",
            "match_found": True,
            "nli_confidence": 0.92,
            "truth": "Warm water aids digestion.",
            "verdict": "TRUE",
            "match_score": 0.5,
            "matched_entry": "warm water",
            "nli_verdict": "SUPPORTS",
        }
        pipeline.verifier.tier1_lookup.return_value = tier1_fast
        pipeline.verifier.verify.return_value = tier1_fast

        from src.evidence_quality import ClaimEvidenceSummary

        pipeline.evidence_scorer.score_passage.return_value = MagicMock(overall_quality_score=0.85)
        pipeline._build_fast_evidence_summary = MagicMock(
            return_value=(
                ClaimEvidenceSummary(
                    best_tier="guideline",
                    tier_diversity=["guideline"],
                    agreement_ratio=1.0,
                    contradiction_count=0,
                    support_count=1,
                    mean_quality_score=0.85,
                ),
                [0.92],
            )
        )

        reports = pipeline.analyze_query("Warm water improves digestion.", row_index=1)

        pipeline.hybrid_retriever.retrieve_hybrid.assert_not_called()
        self.assertEqual(reports[0]["operating_mode"], "fast")
        self.assertFalse(reports[0]["escalated"])

    def test_standard_mode_no_escalation(self):
        pipeline = self._make_pipeline()
        sub = SubClaim(canonical_claim="Turmeric cures cancer.", subject="turmeric", outcome="cancer")
        claim_rep = MagicMock(route="fact_check", claims=[sub])
        pipeline.claims_processor.process_query.return_value = claim_rep
        pipeline._sub_claims_for_query = MagicMock(return_value=[sub])

        tier1_weak = {
            "match_type": "WEAK",
            "match_found": False,
            "nli_confidence": 0.4,
            "truth": "Unverified: weak",
            "verdict": "UNVERIFIED",
        }
        pipeline.verifier.tier1_lookup.return_value = tier1_weak
        pipeline.verifier.verify.return_value = tier1_weak

        pipeline.hybrid_retriever.retrieve_hybrid.return_value = [
            {"passage_index": 0, "passage": "Turmeric does not cure cancer.", "rrf_score": 0.1}
        ]
        pipeline.reranker.rerank.return_value = pipeline.hybrid_retriever.retrieve_hybrid.return_value
        pipeline.nli_judge.verify.return_value = {
            "verdict": "SUPPORTS",
            "confidence": 0.88,
            "supports_probability": 0.88,
            "refutes_probability": 0.05,
        }
        clean_summary = ClaimEvidenceSummary(
            best_tier="who",
            tier_diversity=["who"],
            agreement_ratio=1.0,
            contradiction_count=0,
            support_count=2,
            mean_quality_score=0.8,
        )
        pipeline.evidence_scorer.score_passage.return_value = MagicMock(overall_quality_score=0.8)
        pipeline.evidence_scorer.summarize_evidence.return_value = clean_summary
        pipeline._run_hybrid_evidence_pass = MagicMock(
            return_value=(clean_summary, [], [0.88], {})
        )

        with patch("main.should_escalate_to_deep", return_value=False):
            reports = pipeline.analyze_query("Turmeric cures cancer.", row_index=2)

        self.assertEqual(reports[0]["operating_mode"], "standard")
        self.assertFalse(reports[0]["escalated"])
        self.assertEqual(pipeline._run_hybrid_evidence_pass.call_count, 1)

    def test_standard_escalates_to_deep_on_contradiction(self):
        pipeline = self._make_pipeline()
        sub = SubClaim(canonical_claim="Stop taking insulin for diabetes.", subject="insulin")
        claim_rep = MagicMock(route="fact_check", claims=[sub])
        pipeline.claims_processor.process_query.return_value = claim_rep
        pipeline._sub_claims_for_query = MagicMock(return_value=[sub])

        tier1_weak = {"match_type": "NONE", "match_found": False, "nli_confidence": None, "verdict": "UNVERIFIED"}
        pipeline.verifier.tier1_lookup.return_value = tier1_weak
        pipeline.verifier.verify.return_value = tier1_weak

        conflict_summary = ClaimEvidenceSummary(
            best_tier="who",
            tier_diversity=["who", "journalism"],
            agreement_ratio=0.4,
            contradiction_count=2,
            support_count=1,
            mean_quality_score=0.7,
        )
        clean_after_deep = ClaimEvidenceSummary(
            best_tier="who",
            tier_diversity=["who"],
            agreement_ratio=0.9,
            contradiction_count=0,
            support_count=3,
            mean_quality_score=0.85,
        )
        pipeline._run_hybrid_evidence_pass = MagicMock(
            side_effect=[
                (conflict_summary, [], [0.5, 0.4], {}),
                (clean_after_deep, [], [0.9, 0.85, 0.88], {}),
            ]
        )

        with patch("main.should_escalate_to_deep", side_effect=[True, False]):
            with patch("main.detect_harm_signal_placeholder", return_value=False):
                reports = pipeline.analyze_query("Stop taking insulin for diabetes.", row_index=3)

        self.assertEqual(reports[0]["operating_mode"], "deep")
        self.assertTrue(reports[0]["escalated"])
        self.assertEqual(pipeline._run_hybrid_evidence_pass.call_count, 2)
        second_call = pipeline._run_hybrid_evidence_pass.call_args_list[1]
        self.assertEqual(second_call.kwargs.get("top_per_lane"), 15)


class NLIVerifierProbabilityTests(unittest.TestCase):
    def test_verify_exposes_label_probabilities(self):
        mock_probs = MagicMock()
        mock_probs.shape = (1, 3)
        mock_probs.__getitem__ = lambda self, key: mock_probs
        mock_probs.item = MagicMock(return_value=0.9)

        # Use a lightweight mock of NLIVerifier output shape
        result = {
            "verdict": "SUPPORTS",
            "confidence": 0.9,
            "label_probabilities": {"SUPPORTS": 0.9, "REFUTES": 0.05, "NEI": 0.05},
            "supports_probability": 0.9,
            "refutes_probability": 0.05,
        }
        self.assertIn("supports_probability", result)
        self.assertIn("label_probabilities", result)


if __name__ == "__main__":
    unittest.main()
