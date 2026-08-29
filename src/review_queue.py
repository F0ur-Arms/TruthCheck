"""Human Review Queue Manager for persisting paused LangGraph executions."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


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

    def __init__(self):
        self._queue: Dict[str, ReviewItem] = {}

    def submit_for_review(
        self,
        claim_text: str,
        risk_assessment: Dict[str, Any],
        calibrated_verdict: Optional[Dict[str, Any]] = None
    ) -> ReviewItem:
        item = ReviewItem(
            claim_text=claim_text,
            risk_assessment=risk_assessment,
            calibrated_verdict=calibrated_verdict
        )
        self._queue[item.request_id] = item
        return item

    def get_pending_reviews(self) -> List[ReviewItem]:
        return [item for item in self._queue.values() if item.status == "pending"]

    def get_review_item(self, request_id: str) -> Optional[ReviewItem]:
        return self._queue.get(request_id)

    def apply_reviewer_decision(
        self,
        request_id: str,
        action: str,
        adjusted_verdict: Optional[str] = None,
        note: Optional[str] = None
    ) -> Optional[ReviewItem]:
        item = self._queue.get(request_id)
        if not item:
            return None

        item.status = action
        item.adjusted_verdict = adjusted_verdict
        item.reviewer_note = note
        return item


global_review_queue = ReviewQueueManager()
