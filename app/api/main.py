"""Phase 10: Production FastAPI Web Application for TruthCheck v2."""

from __future__ import annotations

import uuid
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph.workflow import graph_app
from src.review_queue import global_review_queue
from src.risk_engine_v2 import TwoAxisRiskEngine

app = FastAPI(
    title="TruthCheck v2 Healthcare Fact-Verification API",
    version="2.0.0",
    description="Medical evidence verification engine with multi-lane hybrid retrieval and calibrated risk scoring."
)

risk_engine = TwoAxisRiskEngine()
# Simple in-memory audit trace repository
trace_repository: Dict[str, Dict[str, Any]] = {}


class FactCheckRequest(BaseModel):
    text: str
    input_type: str = Field(default="text")
    language: str = Field(default="auto")
    mode: str = Field(default="standard")  # "fast", "standard", "deep"


class ReviewDecisionRequest(BaseModel):
    action: str  # "approved", "rejected", "overridden"
    adjusted_verdict: Optional[str] = None
    note: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "TruthCheck v2 Engine"}


@app.post("/api/v1/fact-check")
def fact_check(req: FactCheckRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    request_id = str(uuid.uuid4())
    trace_id = f"trace-{request_id}"
    config = {"configurable": {"thread_id": request_id}}

    initial_state = {"raw_input": req.text}
    graph_result = graph_app.invoke(initial_state, config=config)

    final_resp = graph_result.get("final_response") or {}
    needs_review = graph_result.get("needs_human_review", False)
    risk_info = graph_result.get("risk_assessment") or {}

    # Store reproducibility trace
    trace_repository[trace_id] = {
        "request_id": request_id,
        "trace_id": trace_id,
        "mode": req.mode,
        "raw_input": req.text,
        "graph_state": graph_result
    }

    if needs_review:
        global_review_queue.submit_for_review(
            claim_text=req.text,
            risk_assessment=risk_info,
            calibrated_verdict=graph_result.get("calibrated_verdict")
        )

    return {
        "request_id": request_id,
        "verdict": final_resp.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "confidence": final_resp.get("confidence", 0.0),
        "explanation": final_resp.get("explanation", ""),
        "evidence": final_resp.get("evidence", []),
        "needs_human_review": needs_review,
        "trace_id": trace_id
    }


@app.get("/api/v1/trace/{trace_id}")
def get_trace(trace_id: str):
    trace = trace_repository.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace ID not found.")
    return trace


@app.get("/api/v1/review/pending")
def list_pending_reviews():
    return global_review_queue.get_pending_reviews()


@app.post("/api/v1/review/{request_id}")
def submit_review_decision(request_id: str, decision: ReviewDecisionRequest):
    item = global_review_queue.apply_reviewer_decision(
        request_id=request_id,
        action=decision.action,
        adjusted_verdict=decision.adjusted_verdict,
        note=decision.note
    )
    if not item:
        raise HTTPException(status_code=404, detail="Review request ID not found.")
    return {"status": "success", "updated_item": item}
