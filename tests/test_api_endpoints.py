import unittest
from fastapi.testclient import TestClient

import graph.nodes as nodes
from app.api.main import app
from tests.fakes import FakeEngine


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        nodes.set_engine(FakeEngine())
        self.client = TestClient(app)

    def tearDown(self):
        nodes.set_engine(None)

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
        self.assertEqual(data["status"], "completed")
        self.assertIn("request_id", data)
        self.assertIn("verdict", data)
        self.assertTrue(data["evidence"])

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

    def test_high_harm_claim_is_held_until_reviewed(self):
        res = self.client.post(
            "/api/v1/fact-check",
            json={"text": "Stop taking insulin, bitter gourd cures diabetes."}
        )
        data = res.json()
        self.assertEqual(data["status"], "pending_review")
        self.assertIsNone(data["verdict"])
        rid = data["request_id"]
        self.assertIn(rid, [i["request_id"] for i in self.client.get("/api/v1/review/pending").json()])

        done = self.client.post(
            f"/api/v1/review/{rid}",
            json={"action": "overridden", "adjusted_verdict": "FALSE", "note": "unsafe"}
        )
        self.assertEqual(done.status_code, 200)
        self.assertEqual(done.json()["result"]["verdict"], "FALSE")
        self.assertNotIn(rid, [i["request_id"] for i in self.client.get("/api/v1/review/pending").json()])

    def test_unknown_review_id_is_404(self):
        res = self.client.post("/api/v1/review/does-not-exist", json={"action": "approved"})
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
