"""Phase 9: Two-Axis Risk Engine & Harm Assessment.

Decouples Verdict Confidence (Axis 1) from Potential Harm (Axis 2):
Axis 1: Verdict Confidence (from Phase 5 calibration model).
Axis 2: Potential Harm Score derived from:
  - Medical actionability (does it change real-world behavior?)
  - Severity of worst plausible outcome
  - Population vulnerability (pregnancy, pediatric, CKD, chronic disease)
  - Behavioral proximity (direct instruction vs general assertion)

Also enforces Hard Safety Overrides for treatment/medication cessation language.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


CESSATION_PATTERNS = [
    r"\bstop taking\b", r"\bdiscontinue\b", r"\bdon'?t take\b",
    r"\breplace insulin\b", r"\breplace chemotherapy\b", r"\breplace medication\b",
    r"\breplace treatment\b", r"\bdon'?t go to (the )?hospital\b",
    r"\bcure cancer naturally\b", r"\bstop your meds\b",
    r"\bdawa band\b", r"\binsulin mat lo\b"
]

VULNERABLE_POPULATIONS = [
    "pregnant", "pregnancy", "pediatric", "child", "infant", "ckd",
    "kidney disease", "cancer", "diabetes", "heart disease", "elderly"
]


class RiskAssessment(BaseModel):
    confidence_axis: float
    harm_axis: float
    intervention_priority: float
    actionability_score: float
    severity_score: float
    vulnerability_score: float
    proximity_score: float
    has_hard_override: bool
    needs_human_review: bool
    risk_level: str


class TwoAxisRiskEngine:
    """Computes decoupled verdict confidence and potential harm scores."""

    def __init__(self, confidence_threshold: float = 0.60, harm_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self.harm_threshold = harm_threshold

    def evaluate_risk(
        self,
        claim_text: str,
        verdict_confidence: float,
        target_population: Optional[str] = None
    ) -> RiskAssessment:
        text_lower = claim_text.lower()

        # 1. Hard Override Check (Medication / Treatment Cessation)
        has_override = any(re.search(pat, text_lower) for pat in CESSATION_PATTERNS)

        # 2. Compute Harm Axis components
        actionability = self._compute_actionability(text_lower)
        severity = self._compute_severity(text_lower)
        vulnerability = self._compute_vulnerability(text_lower, target_population)
        proximity = self._compute_proximity(text_lower)

        # Composite Harm Score
        harm_score = round(
            (0.35 * severity) +
            (0.30 * actionability) +
            (0.20 * vulnerability) +
            (0.15 * proximity),
            4
        )

        if has_override:
            harm_score = max(harm_score, 0.95)

        # Priority decision surface: Either low confidence OR high harm escalates priority
        priority = round(max(1.0 - verdict_confidence, harm_score), 4)

        needs_review = (
            has_override or
            verdict_confidence < self.confidence_threshold or
            harm_score >= self.harm_threshold
        )

        if harm_score >= 0.75:
            risk_lvl = "CRITICAL / HIGH HARM"
        elif harm_score >= 0.45:
            risk_lvl = "MEDIUM HARM"
        else:
            risk_lvl = "LOW HARM"

        return RiskAssessment(
            confidence_axis=round(verdict_confidence, 4),
            harm_axis=harm_score,
            intervention_priority=priority,
            actionability_score=actionability,
            severity_score=severity,
            vulnerability_score=vulnerability,
            proximity_score=proximity,
            has_hard_override=has_override,
            needs_human_review=needs_review,
            risk_level=risk_lvl
        )

    def _compute_actionability(self, text: str) -> float:
        action_keywords = ["stop", "avoid", "cure", "prevent", "drink", "eat", "take", "treat", "substitute"]
        hits = sum(1 for kw in action_keywords if kw in text)
        return min(1.0, 0.3 + (hits * 0.25))

    def _compute_severity(self, text: str) -> float:
        severe_terms = ["cancer", "stroke", "heart attack", "kidney failure", "death", "fatal", "blindness", "aids", "hiv"]
        if any(term in text for term in severe_terms):
            return 0.95
        moderate_terms = ["diabetes", "bp", "blood pressure", "infection", "ulcer", "liver"]
        if any(term in text for term in moderate_terms):
            return 0.65
        return 0.25

    def _compute_vulnerability(self, text: str, population: Optional[str]) -> float:
        combined = (text + " " + (population or "")).lower()
        if any(pop in combined for pop in VULNERABLE_POPULATIONS):
            return 0.90
        return 0.40

    def _compute_proximity(self, text: str) -> float:
        instruction_starts = ["stop", "do not", "never", "always", "start", "use this"]
        if any(text.startswith(start) for start in instruction_starts):
            return 0.90
        return 0.40
