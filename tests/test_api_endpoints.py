import json
import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

import graph.nodes as nodes
from app.api.main import app
from tests.fakes import FakeEngine


ADMIN = {"X-Admin-Token": "test-token"}


@patch.dict(os.environ, {"TRUTHCHECK_ADMIN_TOKEN": "test-token"})
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
        trace_res = self.client.get(f"/api/v1/trace/{trace_id}", headers=ADMIN)
        self.assertEqual(trace_res.status_code, 200)
        self.assertEqual(trace_res.json()["trace_id"], trace_id)

    def test_trace_and_review_need_the_admin_token(self):
        for headers in ({}, {"X-Admin-Token": "wrong"}):
            self.assertEqual(self.client.get("/api/v1/trace/x", headers=headers).status_code, 403)
            self.assertEqual(self.client.get("/api/v1/review/pending", headers=headers).status_code, 403)
            self.assertEqual(self.client.post("/api/v1/review/x", json={"action": "approved"}, headers=headers).status_code, 403)
        with patch.dict(os.environ):
            del os.environ["TRUTHCHECK_ADMIN_TOKEN"]
            self.assertEqual(self.client.get("/api/v1/review/pending", headers=ADMIN).status_code, 403)

    def test_stream_reports_stages_then_the_same_result(self):
        res = self.client.post("/api/v1/fact-check/stream", json={"text": "Warm water improves digestion."})
        self.assertEqual(res.status_code, 200)
        events = [json.loads(line) for line in res.text.splitlines()]
        stages = [e["stage"] for e in events if "stage" in e]
        self.assertIn("hybrid_retrieve_and_rerank", stages)
        self.assertEqual(stages[-1], "generate_response")
        self.assertEqual(events[-1]["result"]["status"], "completed")
        self.assertTrue(events[-1]["result"]["evidence"])

    def test_notice_follows_the_language_of_the_claim(self):
        self.assertIn("डॉक्टर", nodes._safety_notice("WITHHELD", nodes._notice_language("दवा बंद करो", None)))
        self.assertIn("doctor se", nodes._safety_notice("FALSE", nodes._notice_language("dawa band karo", {"language": "Hinglish"})))
        self.assertIn("talk to a doctor", nodes._safety_notice("FALSE", nodes._notice_language("Stop insulin", {"language": "english"})))

    def test_medical_advice_routing_via_api(self):
        res = self.client.post(
            "/api/v1/fact-check",
            json={"text": "I have stage 3 CKD, should I stop drinking whey protein?"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["verdict"], "NOT_A_FACT_CHECK")
        self.assertIn("Medical Safety Advice Notice", data["explanation"])

    def test_high_harm_claim_is_answered_without_endorsing_it(self):
        data = self.client.post(
            "/api/v1/fact-check",
            json={"text": "Stop taking insulin, bitter gourd cures diabetes."}
        ).json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["verdict"], "WITHHELD")
        self.assertEqual(data["evidence"], [])
        self.assertIn("talk to a doctor", data["safety_notice"])

    @patch.dict(os.environ, {"TRUTHCHECK_HUMAN_REVIEW": "on"})
    def test_high_harm_claim_is_held_until_reviewed(self):
        res = self.client.post(
            "/api/v1/fact-check",
            json={"text": "Stop taking insulin, bitter gourd cures diabetes."}
        )
        data = res.json()
        self.assertEqual(data["status"], "pending_review")
        self.assertIsNone(data["verdict"])
        self.assertNotIn("proposed_verdict", data)
        rid = data["request_id"]
        self.assertIn(rid, [i["request_id"] for i in self.client.get("/api/v1/review/pending", headers=ADMIN).json()])

        done = self.client.post(
            f"/api/v1/review/{rid}",
            json={"action": "overridden", "adjusted_verdict": "FALSE", "note": "unsafe"},
            headers=ADMIN,
        )
        self.assertEqual(done.status_code, 200)
        self.assertEqual(done.json()["result"]["verdict"], "FALSE")
        self.assertNotIn(rid, [i["request_id"] for i in self.client.get("/api/v1/review/pending", headers=ADMIN).json()])

    def test_unknown_review_id_is_404(self):
        res = self.client.post("/api/v1/review/does-not-exist", json={"action": "approved"}, headers=ADMIN)
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
