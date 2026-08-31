import unittest
from src.calibration_eval import EmpiricalCalibrationEvaluator


class CalibrationEvalTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = EmpiricalCalibrationEvaluator(num_bins=5)

    def test_perfect_calibration(self):
        y_true = [1, 1, 0, 0]
        y_prob = [0.95, 0.85, 0.15, 0.05]
        report = self.evaluator.evaluate(y_true, y_prob)

        self.assertLess(report.brier_score, 0.05)
        self.assertLessEqual(report.ece, 0.10)
        self.assertEqual(report.num_samples, 4)

    def test_poor_calibration_ece(self):
        # Overconfident predictions
        y_true = [0, 0, 0, 0]
        y_prob = [0.90, 0.95, 0.88, 0.92]
        report = self.evaluator.evaluate(y_true, y_prob)

        self.assertGreater(report.brier_score, 0.70)
        self.assertGreater(report.ece, 0.80)


if __name__ == "__main__":
    unittest.main()
