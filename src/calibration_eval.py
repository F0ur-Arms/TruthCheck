"""Empirical Calibration Metrics Evaluator.

Computes:
1. Brier Score (Mean Squared Error between predicted probability and true binary label).
2. Expected Calibration Error (ECE) across M probability bins.
3. Reliability Diagram curve data (confidence vs accuracy per bin).
4. Confidence Bucket Accuracy report.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field


class CalibrationMetricsReport(BaseModel):
    brier_score: float
    ece: float
    num_samples: int
    num_bins: int
    reliability_diagram: List[Dict[str, Any]]
    confidence_bucket_accuracy: Dict[str, float]


class EmpiricalCalibrationEvaluator:
    """Evaluates empirical calibration quality of confidence probability scores."""

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def evaluate(self, y_true: List[int], y_prob: List[float]) -> CalibrationMetricsReport:
        if len(y_true) != len(y_prob):
            raise ValueError("Length of y_true and y_prob must match.")
        
        n = len(y_true)
        if n == 0:
            return CalibrationMetricsReport(
                brier_score=0.0,
                ece=0.0,
                num_samples=0,
                num_bins=self.num_bins,
                reliability_diagram=[],
                confidence_bucket_accuracy={}
            )

        # 1. Brier Score
        brier = sum((p - y) ** 2 for p, y in zip(y_prob, y_true)) / n

        # 2. ECE & Reliability Diagram
        bins: List[List[Tuple[float, int]]] = [[] for _ in range(self.num_bins)]
        for p, y in zip(y_prob, y_true):
            # Bin index
            bin_idx = min(int(p * self.num_bins), self.num_bins - 1)
            bins[bin_idx].append((p, y))

        ece = 0.0
        reliability_diagram = []

        for i, bin_items in enumerate(bins):
            bin_lower = i / self.num_bins
            bin_upper = (i + 1) / self.num_bins

            if not bin_items:
                reliability_diagram.append({
                    "bin_index": i,
                    "range": f"[{bin_lower:.1f}, {bin_upper:.1f})",
                    "count": 0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0
                })
                continue

            count = len(bin_items)
            mean_conf = sum(p for p, _ in bin_items) / count
            accuracy = sum(y for _, y in bin_items) / count
            gap = abs(accuracy - mean_conf)

            ece += (count / n) * gap

            reliability_diagram.append({
                "bin_index": i,
                "range": f"[{bin_lower:.1f}, {bin_upper:.1f})",
                "count": count,
                "mean_confidence": round(mean_conf, 4),
                "accuracy": round(accuracy, 4),
                "gap": round(gap, 4)
            })

        # 3. Confidence Bucket Accuracy
        buckets = {
            "very_high (>90%)": [y for p, y in zip(y_prob, y_true) if p >= 0.90],
            "high (80-90%)": [y for p, y in zip(y_prob, y_true) if 0.80 <= p < 0.90],
            "moderate (70-80%)": [y for p, y in zip(y_prob, y_true) if 0.70 <= p < 0.80],
            "low (<70%)": [y for p, y in zip(y_prob, y_true) if p < 0.70]
        }

        bucket_acc = {}
        for b_name, b_values in buckets.items():
            if b_values:
                bucket_acc[b_name] = round(sum(b_values) / len(b_values), 4)
            else:
                bucket_acc[b_name] = 0.0

        return CalibrationMetricsReport(
            brier_score=round(brier, 4),
            ece=round(ece, 4),
            num_samples=n,
            num_bins=self.num_bins,
            reliability_diagram=reliability_diagram,
            confidence_bucket_accuracy=bucket_acc
        )
