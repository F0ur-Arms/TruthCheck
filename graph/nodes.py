"""Standalone, unit-testable node functions for the LangGraph workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

from graph.state import TruthCheckState
from src.preprocessor import HinglishMapper
from src.claims_processor import ClaimsProcessor, SubClaim
from src.calibration import ConfidenceCalibrator
from src.risk_engine_v2 import TwoAxisRiskEngine


mapper = HinglishMapper()
claims_processor = ClaimsProcessor()
calibrator = ConfidenceCalibrator()
risk_engine = TwoAxisRiskEngine()

_engine = None

# Most serious first. A compound claim takes the most serious sub-claim verdict, so one
# unsupported part stops the whole input from being reported as SUPPORTED.
VERDICT_SEVERITY = [
    "FALSE", "MOSTLY_FALSE", "MISLEADING", "MIXED_EVIDENCE", "INSUFFICIENT_EVIDENCE",
    "SUPPORTED_WITH_CAVEATS", "MOSTLY_SUPPORTED", "SUPPORTED",
]


def get_engine():
    global _engine
    if _engine is None:
        from src.evidence_engine import EvidenceEngine
        _engine = EvidenceEngine.from_defaults()
    return _engine


def set_engine(engine) -> None:
    global _engine
    _engine = engine


def aggregate_verdict(verdicts: List[str]) -> str:
    if not verdicts:
        return "INSUFFICIENT_EVIDENCE"
    return min(verdicts, key=lambda v: VERDICT_SEVERITY.index(v) if v in VERDICT_SEVERITY else len(VERDICT_SEVERITY))


def ingest_node(state: TruthCheckState) -> Dict[str, Any]:
    cleaned = mapper.clean_text(state.raw_input)
    return {"cleaned_text": cleaned}


def route_and_decompose_node(state: TruthCheckState) -> Dict[str, Any]:
    claim_rep = claims_processor.process_query(state.raw_input, state.cleaned_text)

    if claim_rep.route == "medical_advice":
        return {
            "route": "medical_advice",
            "canonical_claim": claim_rep.model_dump(),
            "final_response": {
                "input": state.raw_input,
                "verdict": "NOT_A_FACT_CHECK",
                "route": "medical_advice",
                "safety_response": claim_rep.safety_response,
                "explanation": claim_rep.safety_response,
                "source": "Medical Safety Advice Gate",
                "risk_score": 1.0,
                "risk_level": "High (Medical Advice Request)"
            }
        }

    sub_claims = claim_rep.claims or [SubClaim(canonical_claim=state.cleaned_text or state.raw_input)]
    return {
        "route": "fact_check",
        "canonical_claim": claim_rep.model_dump(),
        "sub_claims": [c.model_dump() for c in sub_claims],
    }


def hybrid_retrieve_and_rerank_node(state: TruthCheckState) -> Dict[str, Any]:
    if state.route == "medical_advice":
        return {}

    engine = get_engine()
    claim_evidence = []
    flat: List[Dict[str, Any]] = []
    for sc in state.sub_claims:
        lanes = engine.make_lanes(sc["canonical_claim"], sc.get("subject"), sc.get("outcome"))
        reranked = engine.retrieve(sc["canonical_claim"], lanes, top_per_lane=5)
        claim_evidence.append({"claim_id": sc.get("claim_id"), "lanes": lanes, "reranked": reranked})
        flat.extend(reranked)
    return {"claim_evidence": claim_evidence, "reranked_passages": flat}


def evidence_and_calibration_node(state: TruthCheckState) -> Dict[str, Any]:
    if state.route == "medical_advice":
        return {}

    engine = get_engine()
    claim_results = []
    for sc, ev in zip(state.sub_claims, state.claim_evidence):
        scored, nli_probs, summary = engine.score(
            sc["canonical_claim"], ev["reranked"],
            subject=sc.get("subject"), outcome=sc.get("outcome"), population=sc.get("population"),
        )
        has_population_split = bool(
            sc.get("population") and sc.get("population_confidence") not in ("unspecified", "", None)
        )
        calibrated = calibrator.calibrate(
            evidence_summary=summary,
            top_nli_probabilities=nli_probs,
            has_population_split=has_population_split,
        )
        claim_results.append({
            "claim_id": sc.get("claim_id"),
            "canonical_claim": sc["canonical_claim"],
            "population": sc.get("population"),
            "scored_passages": scored,
            "evidence_summary": summary.model_dump(),
            "calibrated_verdict": calibrated.model_dump(),
        })

    verdicts = [r["calibrated_verdict"]["verdict"] for r in claim_results]
    overall = aggregate_verdict(verdicts)
    driver = next((r for r in claim_results if r["calibrated_verdict"]["verdict"] == overall), None)
    return {
        "claim_results": claim_results,
        "evidence_summary": driver["evidence_summary"] if driver else None,
        "calibrated_verdict": driver["calibrated_verdict"] if driver else None,
    }


def assess_risk_node(state: TruthCheckState) -> Dict[str, Any]:
    if state.route == "medical_advice":
        return {"needs_human_review": False}

    per_claim = []
    for r in state.claim_results:
        ra = risk_engine.evaluate_risk(
            r["canonical_claim"],
            r["calibrated_verdict"]["calibrated_confidence"],
            target_population=r.get("population"),
        )
        per_claim.append(ra.model_dump())

    if not per_claim:
        return {"risk_assessment": None, "needs_human_review": False}

    worst = max(per_claim, key=lambda x: x["harm_axis"])
    verdict = (state.calibrated_verdict or {}).get("verdict", "INSUFFICIENT_EVIDENCE")
    # Only high harm or a medication-cessation override blocks the response on a reviewer.
    # Low confidence alone is answered as INSUFFICIENT_EVIDENCE and flagged, not blocked.
    blocking = any(
        c["has_hard_override"] or c["harm_axis"] >= risk_engine.harm_threshold for c in per_claim
    )
    low_confidence = any(c["confidence_axis"] < risk_engine.confidence_threshold for c in per_claim)
    reasons = []
    if any(c["has_hard_override"] for c in per_claim):
        reasons.append("treatment_cessation_override")
    if worst["harm_axis"] >= risk_engine.harm_threshold:
        reasons.append("high_harm")
    if low_confidence:
        reasons.append("low_confidence")
    if any(r["calibrated_verdict"]["verdict"] in {"INSUFFICIENT_EVIDENCE", "MIXED_EVIDENCE"}
           for r in state.claim_results):
        reasons.append("inconclusive_verdict")

    return {
        "risk_assessment": {**worst, "verdict": verdict, "review_reasons": reasons,
                            "flag_for_review": bool(reasons), "per_claim": per_claim},
        "needs_human_review": blocking,
    }


def human_review_node(state: TruthCheckState) -> Dict[str, Any]:
    if state.human_reviewer_decision:
        return {"needs_human_review": False}

    decision = interrupt({
        "claim": state.raw_input,
        "proposed_verdict": (state.calibrated_verdict or {}).get("verdict"),
        "confidence": (state.calibrated_verdict or {}).get("calibrated_confidence"),
        "review_reasons": (state.risk_assessment or {}).get("review_reasons", []),
        "evidence": [p.get("passage") for p in state.reranked_passages[:3]],
    })
    return {"human_reviewer_decision": decision, "needs_human_review": False}


def generate_response_node(state: TruthCheckState) -> Dict[str, Any]:
    if state.final_response:
        return {}

    verdict = state.calibrated_verdict.get("verdict") if state.calibrated_verdict else "INSUFFICIENT_EVIDENCE"
    conf = state.calibrated_verdict.get("calibrated_confidence", 0.0) if state.calibrated_verdict else 0.0

    if state.human_reviewer_decision:
        decision = state.human_reviewer_decision
        if decision.get("adjusted_verdict"):
            verdict = decision["adjusted_verdict"]
        elif decision.get("action") == "rejected":
            verdict = "INSUFFICIENT_EVIDENCE"

    evidence_details = sorted(
        (sp for r in state.claim_results for sp in r["scored_passages"]),
        key=lambda sp: sp.get("rerank_score") or 0.0,
        reverse=True,
    )[:3]
    evidence_passages = [sp["passage"] for sp in evidence_details if sp.get("passage")]
    explanation = state.calibrated_verdict.get("explanation_summary") if state.calibrated_verdict else "Evidence inconclusive."

    # If local LLM API is configured, synthesize evidence-grounded summary
    from src.llm_fallback import ConfiguredLLMVerifier
    llm_verifier = ConfiguredLLMVerifier()
    if llm_verifier.configured and evidence_passages:
        try:
            llm_res = llm_verifier.verify(
                f"Claim: {state.raw_input}\nVerdict: {verdict}\nEvidence: {' | '.join(evidence_passages)}"
            )
            if llm_res and llm_res.get("explanation"):
                explanation = llm_res["explanation"]
        except Exception:
            pass

    res = {
        "input": state.raw_input,
        "verdict": verdict,
        "confidence": conf,
        "explanation": explanation,
        "evidence": evidence_passages,
        "evidence_details": [
            {k: sp.get(k) for k in ("passage", "source_file", "source_tier", "nli_verdict", "rerank_score")}
            for sp in evidence_details
        ],
        "claims": [
            {"claim": r["canonical_claim"], "verdict": r["calibrated_verdict"]["verdict"],
             "confidence": r["calibrated_verdict"]["calibrated_confidence"]}
            for r in state.claim_results
        ],
        "risk": state.risk_assessment,
        "human_reviewed": bool(state.human_reviewer_decision),
        "needs_human_review": bool((state.risk_assessment or {}).get("flag_for_review"))
        and not state.human_reviewer_decision,
        "source": "TruthCheck v2 LangGraph Engine"
    }
    return {"final_response": res}
