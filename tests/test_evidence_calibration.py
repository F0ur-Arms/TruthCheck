import unittest
from src.evidence_quality import EvidenceQualityScorer, ClaimEvidenceSummary
from src.calibration import ConfidenceCalibrator


class EvidenceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.scorer = EvidenceQualityScorer()
        self.calibrator = ConfidenceCalibrator()

    def test_evidence_quality_scoring_tier_weights(self):
        who_passage = "WHO guideline on healthy dietary protein intake for adults."
        toi_passage = "Times of India lifestyle report on superfoods."

        score_who = self.scorer.score_passage(who_passage, "protein", "digestion")
        score_toi = self.scorer.score_passage(toi_passage, "protein", "digestion")

        self.assertGreater(score_who.source_tier_score, score_toi.source_tier_score)
        self.assertGreater(score_who.overall_quality_score, score_toi.overall_quality_score)

    def test_file_tier_decides_over_words_in_the_text(self):
        text = "People who smoke have a higher risk of heart disease."
        self.assertEqual(self.scorer.score_passage(text, source_tier="journalism").source_tier_score, 0.35)
        self.assertEqual(self.scorer.score_passage(text, source_tier="cdc").source_tier_score, 1.0)
        # Without a file, the pronoun "who" is not the WHO.
        self.assertEqual(self.scorer.score_passage(text).source_tier_score, 0.50)

    def test_new_source_prefixes_have_tiers(self):
        from src.source_tier import tier_from_filename
        for name, tier in (("cdc_covid-about.txt", "cdc"), ("mohfw_tb.txt", "mohfw"),
                           ("nhm_immunisation.txt", "nhm"), ("medline_aspirin.txt", "medline")):
            self.assertEqual(tier_from_filename(name), tier)

    def test_text_tier_needs_the_source_named(self):
        from src.source_tier import tier_from_passage_text
        self.assertEqual(tier_from_passage_text("People who smoke are at risk."), "journalism")
        self.assertEqual(tier_from_passage_text("The WHO recommends two doses."), "who")

    def test_calibrator_population_split_produces_supported_with_caveats(self):
        summary = ClaimEvidenceSummary(
            best_tier="guideline",
            tier_diversity=["who", "icmr"],
            agreement_ratio=0.85,
            contradiction_count=1,
            support_count=3,
            mean_quality_score=0.85
        )

        result = self.calibrator.calibrate(
            evidence_summary=summary,
            top_nli_probabilities=[0.92, 0.88, 0.84],
            has_population_split=True
        )

        self.assertEqual(result.verdict, "SUPPORTED_WITH_CAVEATS")
        self.assertGreaterEqual(result.calibrated_confidence, 0.70)

    def test_low_confidence_produces_insufficient_evidence(self):
        summary = ClaimEvidenceSummary(
            best_tier="journalism",
            tier_diversity=["journalism"],
            agreement_ratio=0.50,
            contradiction_count=1,
            support_count=1,
            mean_quality_score=0.30
        )

        result = self.calibrator.calibrate(
            evidence_summary=summary,
            top_nli_probabilities=[0.45],
            has_population_split=False
        )

        self.assertEqual(result.verdict, "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
