import unittest

from evaluation.generate_evaluation_report import generate_section56_report

KW = dict(change_summary="c", architecture_why="w", files_changed=["a.py"], behavior_desc="b", tests_summary="t")


class EvaluationReportTests(unittest.TestCase):
    def test_refuses_without_metrics(self):
        with self.assertRaises(ValueError):
            generate_section56_report(metrics={}, **KW)

    def test_unmeasured_values_are_not_invented(self):
        rep = generate_section56_report(metrics={"accuracy": 0.7, "f1_score": 0.7}, **KW)
        self.assertIn("not measured", rep)
        for fake in ("0.8800", "0.0800", "Safety Override Recall: 100.0%"):
            self.assertNotIn(fake, rep)

    def test_baseline_read_from_log(self):
        rep = generate_section56_report(metrics={"accuracy": 0.7, "f1_score": 0.7}, **KW)
        self.assertIn("0.6831", rep)


if __name__ == "__main__":
    unittest.main()
