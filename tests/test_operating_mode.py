import unittest

from src.evidence_quality import ClaimEvidenceSummary
from src.operating_mode import (
    AGREEMENT_ESCALATION_THRESHOLD,
    decide_mode,
    detect_harm_signal_placeholder,
    escalation_reason,
    should_escalate_to_deep,
)
from src.verifiernew import NLI_MIN_CONFIDENCE


class DecideModeTests(unittest.TestCase):
    def test_strong_match_high_nli_is_fast(self):
        tier1 = {
            "match_type": "STRONG",
            "match_found": True,
            "nli_confidence": NLI_MIN_CONFIDENCE + 0.1,
        }
        self.assertEqual(decide_mode(tier1), "fast")

    def test_strong_match_low_nli_is_standard(self):
        tier1 = {
            "match_type": "STRONG",
            "match_found": True,
            "nli_confidence": NLI_MIN_CONFIDENCE - 0.01,
        }
        self.assertEqual(decide_mode(tier1), "standard")

    def test_weak_match_is_standard(self):
        tier1 = {
            "match_type": "WEAK",
            "match_found": True,
            "nli_confidence": 0.95,
        }
        self.assertEqual(decide_mode(tier1), "standard")

    def test_no_match_is_standard(self):
        tier1 = {"match_type": "NONE", "match_found": False, "nli_confidence": None}
        self.assertEqual(decide_mode(tier1), "standard")


class ShouldEscalateTests(unittest.TestCase):
    def test_contradiction_escalates(self):
        summary = ClaimEvidenceSummary(
            best_tier="guideline",
            tier_diversity=["who"],
            agreement_ratio=0.5,
            contradiction_count=+1,
            support_count=1,
            mean_quality_score=0.8,
        )
        self.assertTrue(should_escalate_to_deep(summary, harm_signal=False))

    def test_harm_signal_escalates(self):
        summary = ClaimEvidenceSummary(
            best_tier="guideline",
            tier_diversity=["who"],
            agreement_ratio=1.0,
            contradiction_count=0,
            support_count=2,
            mean_quality_score=0.8,
        )
        self.assertTrue(should_escalate_to_deep(summary, harm_signal=True))

    def test_low_agreement_with_directional_evidence_escalates(self):
        summary = ClaimEvidenceSummary(
            best_tier="study",
            tier_diversity=["pubmed"],
            agreement_ratio=AGREEMENT_ESCALATION_THRESHOLD - 0.05,
            contradiction_count=2,
            support_count=3,
            mean_quality_score=0.7,
        )
        self.assertTrue(should_escalate_to_deep(summary, harm_signal=False))

    def test_low_confidence_clean_agreement_does_not_escalate(self):
        summary = ClaimEvidenceSummary(
            best_tier="journalism",
            tier_diversity=["journalism"],
            agreement_ratio=1.0,
            contradiction_count=0,
            support_count=0,
            mean_quality_score=0.3,
        )
        self.assertFalse(should_escalate_to_deep(summary, harm_signal=False))

    def test_all_nei_no_contradiction_does_not_escalate(self):
        summary = ClaimEvidenceSummary(
            best_tier="none",
            tier_diversity=[],
            agreement_ratio=0.5,
            contradiction_count=0,
            support_count=0,
            mean_quality_score=0.4,
        )
        self.assertFalse(should_escalate_to_deep(summary, harm_signal=False))


class HarmSignalPlaceholderTests(unittest.TestCase):
    def test_detects_medication_cessation(self):
        text = "I have diabetes, should I stop taking my medication?"
        self.assertTrue(detect_harm_signal_placeholder(text))

    def test_benign_claim_no_harm_signal(self):
        text = "Warm water improves digestion."
        self.assertFalse(detect_harm_signal_placeholder(text))


class EscalationReasonTests(unittest.TestCase):
    def test_reason_for_harm_signal(self):
        summary = ClaimEvidenceSummary(
            best_tier="none",
            tier_diversity=[],
            agreement_ratio=1.0,
            contradiction_count=0,
            support_count=0,
            mean_quality_score=0.5,
        )
        self.assertEqual(escalation_reason(summary, harm_signal=True), "harm_signal_placeholder")


if __name__ == "__main__":
    unittest.main()
