import unittest

from src.binary_scoring import score_with_abstain


class ScoreWithAbstainTests(unittest.TestCase):
    def test_abstain_is_counted_wrong_in_strict_and_excluded_from_answered(self):
        s = score_with_abstain([0, 1, 1, 0], [0, 1, None, None])
        self.assertEqual(s["abstained"], 2)
        self.assertAlmostEqual(s["coverage"], 0.5)
        self.assertAlmostEqual(s["strict_accuracy"], 0.5)
        self.assertAlmostEqual(s["answered_accuracy"], 1.0)

    def test_all_abstain_has_no_answered_metrics(self):
        s = score_with_abstain([0, 1], [None, None])
        self.assertEqual(s["strict_accuracy"], 0.0)
        self.assertNotIn("answered_accuracy", s)


if __name__ == "__main__":
    unittest.main()
