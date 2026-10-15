import json
import unittest
from unittest import mock

from src.RAGpipeline.llm_nli import FallbackNLI, LLMNLIJudge, LABEL_PRECISION


def _reply(content):
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


class LLMNLITests(unittest.TestCase):
    def setUp(self):
        self.judge = LLMNLIJudge(base_url="http://x/v1", api_key="k", model="m")

    def test_label_and_probabilities(self):
        with mock.patch("requests.post", return_value=_reply(json.dumps({"label": "refutes"}))):
            res = self.judge.verify("claim", "passage")
        self.assertEqual(res["verdict"], "REFUTES")
        self.assertEqual(res["refutes_probability"], LABEL_PRECISION["REFUTES"])
        self.assertAlmostEqual(sum(res["label_probabilities"].values()), 1.0)

    def test_bad_label_raises(self):
        with mock.patch("requests.post", return_value=_reply(json.dumps({"label": "MAYBE"}))):
            with self.assertRaises(ValueError):
                self.judge.verify("c", "p")

    def test_fallback_used_when_llm_is_down(self):
        class Local:
            def verify(self, c, e):
                return {"verdict": "NEI"}

        fb = FallbackNLI(self.judge, Local)
        with mock.patch("requests.post", side_effect=ConnectionError("down")):
            self.assertEqual(fb.verify("c", "p")["verdict"], "NEI")
        self.assertEqual(fb.fallback_calls, 1)

    def test_fallback_not_built_when_llm_works(self):
        def boom():
            raise AssertionError("fallback must not be built")

        fb = FallbackNLI(self.judge, boom)
        with mock.patch("requests.post", return_value=_reply(json.dumps({"label": "SUPPORTS"}))):
            self.assertEqual(fb.verify("c", "p")["verdict"], "SUPPORTS")


if __name__ == "__main__":
    unittest.main()
