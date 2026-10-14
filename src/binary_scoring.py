"""Binary scoring that keeps abstentions visible instead of hiding them in a fallback."""

from __future__ import annotations

from typing import Dict, List, Optional

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def score_with_abstain(y_true: List[int], y_pred: List[Optional[int]]) -> Dict[str, float]:
    """Score predictions where None means the system abstained.

    strict_*: an abstention counts as wrong (mapped to the opposite class).
    answered_*: computed only on claims the system committed to.
    """
    n = len(y_true)
    answered = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    strict_pred = [1 - t if p is None else p for t, p in zip(y_true, y_pred)]

    out: Dict[str, float] = {
        "n": n,
        "abstained": n - len(answered),
        "coverage": len(answered) / n if n else 0.0,
        "strict_accuracy": accuracy_score(y_true, strict_pred) if n else 0.0,
        "strict_f1_weighted": f1_score(y_true, strict_pred, average="weighted", zero_division=0) if n else 0.0,
    }
    if answered:
        at, ap = zip(*answered)
        out.update(
            answered_accuracy=accuracy_score(at, ap),
            answered_precision_weighted=precision_score(at, ap, average="weighted", zero_division=0),
            answered_recall_weighted=recall_score(at, ap, average="weighted", zero_division=0),
            answered_f1_weighted=f1_score(at, ap, average="weighted", zero_division=0),
        )
    return out
