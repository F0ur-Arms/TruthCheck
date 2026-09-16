"""Typed state representation for TruthCheck v2 LangGraph workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TruthCheckState(BaseModel):
    raw_input: str
    cleaned_text: str = ""
    route: str = Field(default="fact_check")
    canonical_claim: Optional[Dict[str, Any]] = None
    query_lanes: Dict[str, str] = Field(default_factory=dict)
    candidate_passages: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_passages: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_summary: Optional[Dict[str, Any]] = None
    calibrated_verdict: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    needs_human_review: bool = False
    human_reviewer_decision: Optional[Dict[str, Any]] = None
    final_response: Optional[Dict[str, Any]] = None
