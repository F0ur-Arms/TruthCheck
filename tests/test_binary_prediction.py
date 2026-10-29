import unittest

from main import get_binary_prediction


class BinaryPredictionMappingTests(unittest.TestCase):
    def test_supported_maps_to_real(self):
        reports = [{"verdict": "SUPPORTED", "legacy_risk_score": 0.9}]
        self.assertEqual(get_binary_prediction(reports), 0)

    def test_false_maps_to_fake(self):
        reports = [{"verdict": "FALSE", "legacy_risk_score": 0.1}]
        self.assertEqual(get_binary_prediction(reports), 1)

    def test_supported_with_caveats_maps_to_real(self):
        self.assertEqual(get_binary_prediction([{"verdict": "SUPPORTED_WITH_CAVEATS"}]), 0)

    def test_unmapped_verdicts_abstain(self):
        for verdict in ("INSUFFICIENT_EVIDENCE", "MIXED_EVIDENCE", "NOT_A_FACT_CHECK"):
            reports = [{"verdict": verdict, "legacy_risk_score": 0.8}]
            self.assertIsNone(get_binary_prediction(reports))

    def test_error_or_empty_abstains(self):
        self.assertIsNone(get_binary_prediction([]))
        self.assertIsNone(get_binary_prediction([{"error": "boom"}]))

    def test_legacy_fallback_is_opt_in(self):
        reports = [{"verdict": "INSUFFICIENT_EVIDENCE", "legacy_risk_score": 0.8}]
        self.assertEqual(get_binary_prediction(reports, risk_threshold=0.5), 1)
        reports[0]["legacy_risk_score"] = 0.2
        self.assertEqual(get_binary_prediction(reports, risk_threshold=0.5), 0)


if __name__ == "__main__":
    unittest.main()
