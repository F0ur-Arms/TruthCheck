"""Human Review Queue Manager for persisting paused LangGraph executions."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from src.storage import DocumentStore


class ReviewItem(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    claim_text: str
    risk_assessment: Dict[str, Any]
    calibrated_verdict: Optional[Dict[str, Any]] = None
    status: str = Field(default="pending")  # "pending", "approved", "rejected", "overridden"
    reviewer_note: Optional[str] = None
    adjusted_verdict: Optional[str] = None


class ReviewQueueManager:
    """Thread-safe review queue manager."""

    def __init__(self, store: Optional[DocumentStore] = None):
        self._store = store or DocumentStore("review_items")

    def submit_for_review(
        self,
        claim_text: str,
        risk_assessment: Dict[str, Any],
        calibrated_verdict: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> ReviewItem:
        item = ReviewItem(
            **({"request_id": request_id} if request_id else {}),
            claim_text=claim_text,
            risk_assessment=risk_assessment,
            calibrated_verdict=calibrated_verdict
        )
        self._store.put(item.request_id, item.model_dump())
        return item

    def get_pending_reviews(self) -> List[ReviewItem]:
        return [item for item in (ReviewItem(**d) for d in self._store.all()) if item.status == "pending"]

    def get_review_item(self, request_id: str) -> Optional[ReviewItem]:
        doc = self._store.get(request_id)
        return ReviewItem(**doc) if doc else None

    def apply_reviewer_decision(
        self,
        request_id: str,
        action: str,
        adjusted_verdict: Optional[str] = None,
        note: Optional[str] = None
    ) -> Optional[ReviewItem]:
        item = self.get_review_item(request_id)
        if not item:
            return None

        item.status = action
        item.adjusted_verdict = adjusted_verdict
        item.reviewer_note = note
        self._store.put(item.request_id, item.model_dump())
        return item


global_review_queue = ReviewQueueManager()
