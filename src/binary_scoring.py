"""Binary scoring that keeps abstentions visible instead of hiding them in a fallback."""

from __future__ import annotations

from typing import Dict, List, Optional

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


_FAKE_VERDICTS = frozenset({"FALSE", "MISLEADING", "MOSTLY_FALSE"})
_REAL_VERDICTS = frozenset({"SUPPORTED", "SUPPORTED_WITH_CAVEATS", "MOSTLY_SUPPORTED"})


def get_binary_prediction(reports, risk_threshold=None):
    """
    0 = Real/Safe
    1 = Fake/Misinformation
    None = abstain (error, or a verdict with no binary meaning such as
    INSUFFICIENT_EVIDENCE / MIXED_EVIDENCE / NOT_A_FACT_CHECK).

    Pass risk_threshold to restore the legacy behaviour of deciding abstentions
    from the old risk score; the default reports them honestly instead.
    """
    if not reports or "error" in reports[0]:
        return None

    for r in reports:
        verdict = r.get("verdict", "")
        if verdict in _FAKE_VERDICTS or verdict == "FALSE":
            return 1
        if verdict in _REAL_VERDICTS or verdict == "TRUE":
            return 0

    if risk_threshold is None:
        return None
    max_risk = max(r.get("legacy_risk_score", r.get("risk_score", 0)) for r in reports)
    return 1 if max_risk > risk_threshold else 0


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
