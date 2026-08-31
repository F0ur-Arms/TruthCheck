"""Stage-Isolated Failure Attribution Evaluator (Section 40).

Classifies pipeline execution failures into 16 explicit failure stages:
INPUT_FAILURE, LANGUAGE_FAILURE, NORMALIZATION_FAILURE, ROUTING_FAILURE,
DECOMPOSITION_FAILURE, CONCEPT_NORMALIZATION_FAILURE, QUERY_PLANNING_FAILURE,
RETRIEVAL_FAILURE, RERANKING_FAILURE, EVIDENCE_QUALITY_FAILURE, NLI_FAILURE,
POPULATION_REASONING_FAILURE, TEMPORAL_REASONING_FAILURE, CALIBRATION_FAILURE,
RISK_FAILURE, RESPONSE_GENERATION_FAILURE.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class FailureAttributionResult(BaseModel):
    is_successful: bool
    primary_failure_stage: Optional[str] = None
    failure_description: Optional[str] = None
    stage_diagnostics: Dict[str, Any] = Field(default_factory=dict)


class FailureAttributionEvaluator:
    """Diagnoses and attributes end-to-end pipeline failure causes."""

    VALID_FAILURE_STAGES = {
        "INPUT_FAILURE", "LANGUAGE_FAILURE", "NORMALIZATION_FAILURE", "ROUTING_FAILURE",
        "DECOMPOSITION_FAILURE", "CONCEPT_NORMALIZATION_FAILURE", "QUERY_PLANNING_FAILURE",
        "RETRIEVAL_FAILURE", "RERANKING_FAILURE", "EVIDENCE_QUALITY_FAILURE", "NLI_FAILURE",
        "POPULATION_REASONING_FAILURE", "TEMPORAL_REASONING_FAILURE", "CALIBRATION_FAILURE",
        "RISK_FAILURE", "RESPONSE_GENERATION_FAILURE"
    }

    def diagnose_trace(
        self,
        predicted_verdict: str,
        expected_verdict: str,
        graph_trace: Dict[str, Any]
    ) -> FailureAttributionResult:
        if predicted_verdict == expected_verdict:
            return FailureAttributionResult(is_successful=True)

        route = graph_trace.get("route", "fact_check")
        query_lanes = graph_trace.get("query_lanes", {})
        candidates = graph_trace.get("candidate_passages", [])
        reranked = graph_trace.get("reranked_passages", [])
        evidence_summary = graph_trace.get("evidence_summary", {})
        risk = graph_trace.get("risk_assessment", {})

        # Stage 1: Routing Check
        if expected_verdict == "NOT_A_FACT_CHECK" and route != "medical_advice":
            return FailureAttributionResult(
                is_successful=False,
                primary_failure_stage="ROUTING_FAILURE",
                failure_description="Personal medical advice query leaked past routing gate into fact-check path."
            )

        # Stage 2: Query Planning Check
        if not query_lanes or len(query_lanes) < 3:
            return FailureAttributionResult(
                is_successful=False,
                primary_failure_stage="QUERY_PLANNING_FAILURE",
                failure_description="Multi-lane adversarial query generator failed to generate required query lanes."
            )

        # Stage 3: Retrieval Check
        if not candidates:
            return FailureAttributionResult(
                is_successful=False,
                primary_failure_stage="RETRIEVAL_FAILURE",
                failure_description="Hybrid retriever returned zero candidate passages."
            )

        # Stage 4: Reranking Check
        if not reranked:
            return FailureAttributionResult(
                is_successful=False,
                primary_failure_stage="RERANKING_FAILURE",
                failure_description="Cross-Encoder reranker produced empty candidate list."
            )

        # Stage 5: Evidence Quality Check
        mean_q = evidence_summary.get("mean_quality_score", 0.0)
        if mean_q < 0.20 and expected_verdict in {"SUPPORTED", "FALSE"}:
            return FailureAttributionResult(
                is_successful=False,
                primary_failure_stage="EVIDENCE_QUALITY_FAILURE",
                failure_description="Evidence quality scorer assigned low quality to valid ground truth evidence."
            )

        # Stage 6: Calibration Check
        if predicted_verdict in {"INSUFFICIENT_EVIDENCE", "MIXED_EVIDENCE"} and expected_verdict in {"SUPPORTED", "FALSE"}:
            return FailureAttributionResult(
                is_successful=False,
                primary_failure_stage="CALIBRATION_FAILURE",
                failure_description="Confidence calibrator produced underconfident logit for sufficient evidence."
            )

        # Default NLI / Aggregation failure fallback
        return FailureAttributionResult(
            is_successful=False,
            primary_failure_stage="NLI_FAILURE",
            failure_description=f"Specialist NLI or aggregation error: predicted {predicted_verdict} vs expected {expected_verdict}."
        )
