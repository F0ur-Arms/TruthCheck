"""Phase 5: Evidence Quality Scoring & Domain-Conditional Temporal Weighting.

Implements:
1. Source Tier weighting (Government/WHO/ICMR Guidelines > Meta-analysis/RCT > Journalism).
2. Domain-conditional recency decay (Fast-moving domains vs Slow-moving physiological domains).
3. Specificity & Population Match scoring.
4. Passage-level evidence quality score & claim-level evidence summary.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


SOURCE_TIERS = {
    "who": 1.0,
    "icmr": 1.0,
    "nih": 1.0,
    "fssai": 1.0,
    "pubmed": 0.85,
    "guideline": 0.95,
    "cleveland": 0.80,
    "mayo": 0.80,
    "nhs": 0.80,
    "thip": 0.65,
    "vishvas": 0.65,
    "toi": 0.40,
    "indianexpress": 0.40,
    "journalism": 0.35,
}

FAST_MOVING_KEYWORDS = {"vaccine", "covid", "virus", "infection", "outbreak", "variant", "drug", "fda"}


class PassageQualityScore(BaseModel):
    passage_id: str
    source_tier_score: float
    recency_score: float
    specificity_score: float
    population_score: float
    overall_quality_score: float


class ClaimEvidenceSummary(BaseModel):
    best_tier: str
    tier_diversity: List[str]
    agreement_ratio: float
    contradiction_count: int
    support_count: int
    mean_quality_score: float


class EvidenceQualityScorer:
    """Computes continuous evidence quality scores for retrieved passages."""

    def __init__(self, current_year: int = 2026):
        self.current_year = current_year

    def score_passage(
        self,
        passage_text: str,
        claim_subject: Optional[str] = None,
        claim_outcome: Optional[str] = None,
        target_population: Optional[str] = None,
        publication_year: Optional[int] = None
    ) -> PassageQualityScore:
        # 1. Source Tier
        source_score = self._compute_source_tier(passage_text)

        # 2. Recency Score
        recency_score = self._compute_recency(passage_text, publication_year)

        # 3. Specificity Match
        spec_score = self._compute_specificity(passage_text, claim_subject, claim_outcome)

        # 4. Population Match
        pop_score = self._compute_population_match(passage_text, target_population)

        # Weighted combination
        overall = round(
            (0.35 * source_score) +
            (0.20 * recency_score) +
            (0.25 * spec_score) +
            (0.20 * pop_score),
            4
        )

        return PassageQualityScore(
            passage_id=passage_text[:50],
            source_tier_score=source_score,
            recency_score=recency_score,
            specificity_score=spec_score,
            population_score=pop_score,
            overall_quality_score=overall
        )

    def summarize_evidence(self, scored_passages: List[Dict[str, Any]]) -> ClaimEvidenceSummary:
        if not scored_passages:
            return ClaimEvidenceSummary(
                best_tier="none",
                tier_diversity=[],
                agreement_ratio=0.0,
                contradiction_count=0,
                support_count=0,
                mean_quality_score=0.0
            )

        tiers = [p.get("source_tier", "journalism") for p in scored_passages]
        nli_verdicts = [p.get("nli_verdict", "NEI") for p in scored_passages]

        supports = sum(1 for v in nli_verdicts if v == "SUPPORTS")
        refutes = sum(1 for v in nli_verdicts if v == "REFUTES")
        total_directional = supports + refutes
        agreement = (max(supports, refutes) / total_directional) if total_directional > 0 else 0.5

        best = "guideline" if any(t in {"who", "icmr", "nih", "guideline"} for t in tiers) else ("study" if "pubmed" in tiers else "journalism")
        mean_q = sum(p.get("quality_score", 0.5) for p in scored_passages) / len(scored_passages)

        return ClaimEvidenceSummary(
            best_tier=best,
            tier_diversity=[str(t) for t in set(tiers)],
            agreement_ratio=round(agreement, 4),
            contradiction_count=refutes,
            support_count=supports,
            mean_quality_score=round(mean_q, 4)
        )

    def _compute_source_tier(self, text: str) -> float:
        text_lower = text.lower()
        for source, weight in SOURCE_TIERS.items():
            if source in text_lower:
                return weight
        return 0.50

    def _compute_recency(self, text: str, year: Optional[int]) -> float:
        if year is None:
            # Extract year from text if present
            match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
            year = int(match.group(1)) if match else 2020

        age = max(0, self.current_year - year)
        is_fast_moving = any(kw in text.lower() for kw in FAST_MOVING_KEYWORDS)

        if is_fast_moving:
            # Heavy decay for fast-moving domains
            return max(0.2, 1.0 - (age * 0.15))
        else:
            # Gentle decay for slow-moving physiology/nutrition domains
            return max(0.5, 1.0 - (age * 0.03))

    def _compute_specificity(self, text: str, subject: Optional[str], outcome: Optional[str]) -> float:
        if not subject and not outcome:
            return 0.70

        text_lower = text.lower()
        sub_hit = (subject.lower() in text_lower) if subject else True
        out_hit = (outcome.lower() in text_lower) if outcome else True

        if sub_hit and out_hit:
            return 1.0
        elif sub_hit or out_hit:
            return 0.65
        return 0.30

    def _compute_population_match(self, text: str, target_population: Optional[str]) -> float:
        if not target_population:
            return 0.75  # General population assumption

        text_lower = text.lower()
        target_lower = target_population.lower()

        if target_lower in text_lower:
            return 1.0
        elif any(w in text_lower for w in target_lower.split()):
            return 0.65
        return 0.35
