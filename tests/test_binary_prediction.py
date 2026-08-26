import unittest

from main import get_binary_prediction


class BinaryPredictionMappingTests(unittest.TestCase):
    def test_supported_maps_to_real(self):
        reports = [{"verdict": "SUPPORTED", "legacy_risk_score": 0.9}]
        self.assertEqual(get_binary_prediction(reports), 0)

    def test_false_maps_to_fake(self):
        reports = [{"verdict": "FALSE", "legacy_risk_score": 0.1}]
        self.assertEqual(get_binary_prediction(reports), 1)

    def test_insufficient_evidence_falls_back_to_legacy_risk(self):
        reports = [{"verdict": "INSUFFICIENT_EVIDENCE", "legacy_risk_score": 0.8}]
        self.assertEqual(get_binary_prediction(reports), 1)
        reports[0]["legacy_risk_score"] = 0.2
        self.assertEqual(get_binary_prediction(reports), 0)


if __name__ == "__main__":
    unittest.main()
