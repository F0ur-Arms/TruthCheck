"""Phase 10: Production FastAPI Web Application for TruthCheck v2."""

from __future__ import annotations

import hmac
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, List
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langgraph.types import Command

from graph.workflow import graph_app
from src.review_queue import global_review_queue
from src.storage import DocumentStore
from src.risk_engine_v2 import TwoAxisRiskEngine

app = FastAPI(
    title="TruthCheck v2 Healthcare Fact-Verification API",
    version="2.0.0",
    description="Medical evidence verification engine with multi-lane hybrid retrieval and calibrated risk scoring."
)

risk_engine = TwoAxisRiskEngine()
# Simple in-memory audit trace repository
trace_repository = DocumentStore("traces")


class FactCheckRequest(BaseModel):
    text: str
    input_type: str = Field(default="text")
    language: str = Field(default="auto")
    mode: str = Field(default="standard")  # "fast", "standard", "deep"


class ReviewDecisionRequest(BaseModel):
    action: str  # "approved", "rejected", "overridden"
    adjusted_verdict: Optional[str] = None
    note: Optional[str] = None


def require_admin(x_admin_token: Optional[str] = Header(default=None)) -> None:
    """Traces and the review queue hold what users typed: closed unless TRUTHCHECK_ADMIN_TOKEN is set and sent."""
    expected = os.environ.get("TRUTHCHECK_ADMIN_TOKEN")
    if not expected or not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=403, detail="Admin token required.")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "TruthCheck v2 Engine"}


@app.post("/api/v1/fact-check")
def fact_check(req: FactCheckRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    request_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": request_id}}
    graph_result = graph_app.invoke({"raw_input": req.text}, config=config)
    return _finish(req, request_id, config, graph_result)


@app.post("/api/v1/fact-check/stream")
def fact_check_stream(req: FactCheckRequest):
    """Same check as /fact-check, as JSON lines: {"stage": node} as each step ends, then {"result": response}."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    def events() -> Iterator[str]:
        request_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": request_id}}
        try:
            for update in graph_app.stream({"raw_input": req.text}, config=config, stream_mode="updates"):
                for node in update:
                    if node != "__interrupt__":
                        yield json.dumps({"stage": node}) + "\n"
            snapshot = graph_app.get_state(config)
            graph_result = dict(snapshot.values)
            if snapshot.next:
                graph_result["__interrupt__"] = True
            yield json.dumps({"result": _finish(req, request_id, config, graph_result)}) + "\n"
        except Exception:
            yield json.dumps({"error": "The check failed on the server. Please try again."}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


def _finish(req: FactCheckRequest, request_id: str, config: Dict[str, Any], graph_result: Dict[str, Any]) -> Dict[str, Any]:
    trace_id = f"trace-{request_id}"
    trace_repository.put(trace_id, {
        "request_id": request_id,
        "trace_id": trace_id,
        "mode": req.mode,
        "raw_input": req.text,
        "graph_state": graph_result
    })

    if graph_result.get("__interrupt__"):
        state = graph_app.get_state(config).values
        risk_info = state.get("risk_assessment") or {}
        global_review_queue.submit_for_review(
            claim_text=req.text,
            risk_assessment=risk_info,
            calibrated_verdict=state.get("calibrated_verdict"),
            request_id=request_id,
        )
        proposed = state.get("calibrated_verdict") or {}
        return {
            "request_id": request_id,
            "status": "pending_review",
            "verdict": None,
            "confidence": proposed.get("calibrated_confidence", 0.0),
            "explanation": "High-harm claim held for human review before a verdict is released.",
            "evidence": [],
            "needs_human_review": True,
            "trace_id": trace_id
        }

    return _response_from_state(request_id, trace_id, graph_result)


def _response_from_state(request_id: str, trace_id: str, graph_result: Dict[str, Any]) -> Dict[str, Any]:
    final_resp = graph_result.get("final_response") or {}
    risk_info = graph_result.get("risk_assessment") or {}
    flagged = bool(final_resp.get("needs_human_review"))
    if flagged and not final_resp.get("human_reviewed"):
        global_review_queue.submit_for_review(
            claim_text=graph_result.get("raw_input", ""),
            risk_assessment=risk_info,
            calibrated_verdict=graph_result.get("calibrated_verdict"),
            request_id=request_id,
        )
    return {
        "request_id": request_id,
        "status": "completed",
        "verdict": final_resp.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "confidence": final_resp.get("confidence", 0.0),
        "explanation": final_resp.get("explanation", ""),
        "evidence": final_resp.get("evidence", []),
        "evidence_details": final_resp.get("evidence_details", []),
        "claims": final_resp.get("claims", []),
        "safety_notice": final_resp.get("safety_notice"),
        "needs_human_review": flagged,
        "trace_id": trace_id
    }


@app.get("/api/v1/trace/{trace_id}", dependencies=[Depends(require_admin)])
def get_trace(trace_id: str):
    trace = trace_repository.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace ID not found.")
    return trace


@app.get("/api/v1/review/pending", dependencies=[Depends(require_admin)])
def list_pending_reviews():
    return global_review_queue.get_pending_reviews()


@app.post("/api/v1/review/{request_id}", dependencies=[Depends(require_admin)])
def submit_review_decision(request_id: str, decision: ReviewDecisionRequest):
    config = {"configurable": {"thread_id": request_id}}
    snapshot = graph_app.get_state(config)
    item = global_review_queue.get_review_item(request_id)
    if item is None and not snapshot.next:
        raise HTTPException(status_code=404, detail="Review request ID not found.")

    decision_payload = {
        "action": decision.action,
        "adjusted_verdict": decision.adjusted_verdict,
        "reviewer_note": decision.note,
    }
    updated = global_review_queue.apply_reviewer_decision(
        request_id=request_id,
        action=decision.action,
        adjusted_verdict=decision.adjusted_verdict,
        note=decision.note
    )
    result = None
    if snapshot.next:
        graph_result = graph_app.invoke(Command(resume=decision_payload), config=config)
        trace_id = f"trace-{request_id}"
        result = _response_from_state(request_id, trace_id, graph_result)
        trace = trace_repository.get(trace_id)
        if trace:
            trace["graph_state"] = graph_result
            trace_repository.put(trace_id, trace)
    return {"status": "success", "updated_item": updated, "result": result}


# Serve the static web UI (index.html, styles.css, app.js) from the same
# origin as the API, so the browser client can call /api/v1/... with a
# relative path and no CORS configuration is needed. Mounted last so it
# never shadows an explicit API route above.
_web_dir = Path(__file__).resolve().parent.parent.parent / "web"
if _web_dir.is_dir():
    app.mount("/", StaticFiles(directory=_web_dir, html=True), name="web")
