import unittest
from fastapi.testclient import TestClient
from app.api.main import app


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")

    def test_fact_check_api_endpoint(self):
        res = self.client.post(
            "/api/v1/fact-check",
            json={"text": "Warm water improves digestion.", "mode": "standard"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("request_id", data)
        self.assertIn("verdict", data)
        self.assertIn("trace_id", data)

        # Retrieve trace
        trace_id = data["trace_id"]
        trace_res = self.client.get(f"/api/v1/trace/{trace_id}")
        self.assertEqual(trace_res.status_code, 200)
        self.assertEqual(trace_res.json()["trace_id"], trace_id)

    def test_medical_advice_routing_via_api(self):
        res = self.client.post(
            "/api/v1/fact-check",
            json={"text": "I have stage 3 CKD, should I stop drinking whey protein?"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["verdict"], "NOT_A_FACT_CHECK")
        self.assertIn("Medical Safety Advice Notice", data["explanation"])


if __name__ == "__main__":
    unittest.main()
