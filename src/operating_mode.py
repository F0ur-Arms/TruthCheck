"""Operating-mode gate for the TruthCheck pipeline.

Decides whether a claim stays on the cheap Tier-1 fast path or proceeds to
hybrid retrieval (standard, with optional one-shot escalation to deep).
"""

from __future__ import annotations

from typing import Any, Dict

from src.evidence_quality import ClaimEvidenceSummary
from src.verifiernew import NLI_MIN_CONFIDENCE

# Agreement below this threshold indicates genuine directional conflict
# (both supports and refutes present). Aligns with calibration.py MIXED_EVIDENCE
# boundary at agreement_ratio < 0.70.
AGREEMENT_ESCALATION_THRESHOLD = 0.70

# PLACEHOLDER — Part 10.2 two-axis risk model is out of scope for this PR.
# Simple keyword gate for medication/treatment cessation language only.
_HARM_KEYWORDS = (
    "stop taking",
    "stop your medication",
    "discontinue",
    "quit medication",
    "stop treatment",
    "cease medication",
    "don't take",
    "avoid medication",
    "stop the medicine",
    "stop my medication",
    "should i stop",
)


def decide_mode(tier1_result: dict) -> str:
    """Return ``fast`` when Tier-1 KB match is STRONG with sufficient NLI confidence."""
    if (
        tier1_result.get("match_type") == "STRONG"
        and tier1_result.get("match_found", False)
        and tier1_result.get("nli_confidence", 0.0) >= NLI_MIN_CONFIDENCE
    ):
        return "fast"
    return "standard"


def detect_harm_signal_placeholder(text: str) -> bool:
    """PLACEHOLDER harm hook — keyword check for cessation/treatment-stop language."""
    lowered = text.lower()
    return any(kw in lowered for kw in _HARM_KEYWORDS)


def should_escalate_to_deep(
    evidence_summary: ClaimEvidenceSummary,
    harm_signal: bool,
) -> bool:
    """Return True when evidence conflicts, contradicts, or harm placeholder fires."""
    if harm_signal:
        return True

    if evidence_summary.contradiction_count > 0:
        return True

    directional_total = evidence_summary.support_count + evidence_summary.contradiction_count
    if (
        directional_total > 0
        and evidence_summary.agreement_ratio < AGREEMENT_ESCALATION_THRESHOLD
    ):
        return True

    return False


def escalation_reason(
    evidence_summary: ClaimEvidenceSummary,
    harm_signal: bool,
) -> str:
    """Human-readable reason string for deep-mode escalation logging."""
    if harm_signal:
        return "harm_signal_placeholder"
    if evidence_summary.contradiction_count > 0:
        return f"contradiction_count={evidence_summary.contradiction_count}"
    if evidence_summary.agreement_ratio < AGREEMENT_ESCALATION_THRESHOLD:
        return f"agreement_ratio={evidence_summary.agreement_ratio}"
    return "unknown"
