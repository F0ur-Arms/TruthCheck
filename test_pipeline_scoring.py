import unittest

from src.linguistic_scorer import LinguisticScorer
from src.preprocessor import HinglishMapper
from src.risk_engine import RiskEngine
from src.verifier import normalize_verdict


class PipelineScoringTests(unittest.TestCase):
    def test_verdict_normalization(self):
        self.assertEqual(normalize_verdict("PARTIALLY TRUE"), "PARTIALLY TRUE")
        self.assertEqual(normalize_verdict("mixed"), "PARTIALLY TRUE")

    def test_hinglish_cleaning_improves_claim_text(self):
        mapper = HinglishMapper()
        cleaned = mapper.clean_text("Subah khali pet garam pani fat kam karta hai")
        self.assertIn("morning", cleaned)
        self.assertIn("empty stomach", cleaned)
        self.assertIn("warm water", cleaned)

    def test_short_neutral_claim_is_not_over_scored_for_style(self):
        scorer = LinguisticScorer()
        detailed = scorer.calculate_score_detailed("Warm water improves digestion.")
        self.assertLessEqual(detailed["score"], 0.05)

    def test_true_claim_with_strong_evidence_stays_low_risk(self):
        engine = RiskEngine()
        result = engine.calculate_risk(
            "Warm water improves digestion.",
            0.30,
            {
                "verdict": "TRUE",
                "match_type": "STRONG",
                "nli_confidence": 0.96,
                "risk_level": "Low",
            },
        )
        self.assertLess(result["score"], 0.15)

    def test_partial_truth_sits_between_true_and_false(self):
        engine = RiskEngine()
        partial = engine.calculate_risk(
            "Kadha cures fever completely.",
            0.30,
            {
                "verdict": "PARTIALLY TRUE",
                "match_type": "WEAK",
                "nli_confidence": 0.82,
                "risk_level": "Medium",
            },
        )
        false = engine.calculate_risk(
            "Turmeric cures cancer completely.",
            0.90,
            {
                "verdict": "FALSE",
                "match_type": "STRONG",
                "nli_confidence": 0.97,
                "risk_level": "High",
            },
        )
        self.assertLess(partial["score"], false["score"])
        self.assertGreater(partial["score"], 0.20)


if __name__ == "__main__":
    unittest.main()
