import unittest
from src.claims_processor import ClaimsProcessor


class ClaimsProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = ClaimsProcessor()

    def test_personal_medical_advice_routing_heuristic(self):
        query = "I have kidney disease, should I stop drinking whey protein?"
        result = self.processor.process_query(query, query.lower())
        self.assertEqual(result.route, "medical_advice")
        self.assertIsNotNone(result.safety_response)
        self.assertIn("Medical Safety Advice Notice", result.safety_response)

    def test_fact_check_routing_heuristic(self):
        query = "Warm water improves digestion."
        result = self.processor.process_query(query, query.lower())
        self.assertEqual(result.route, "fact_check")
        self.assertIsNone(result.safety_response)
        self.assertGreaterEqual(len(result.claims), 1)

    def test_compound_claim_heuristic_decomposition(self):
        query = "Turmeric cures cancer and warm water improves digestion."
        result = self.processor.process_query(query, query.lower())
        self.assertEqual(result.route, "fact_check")
        self.assertEqual(len(result.claims), 2)
        self.assertIn("turmeric cures cancer", result.claims[0].canonical_claim)
        self.assertIn("warm water improves digestion", result.claims[1].canonical_claim)


if __name__ == "__main__":
    unittest.main()
