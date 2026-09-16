"""Data contracts deliberately kept independent of the main application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenTag:
    """One source token and its language/script classification."""

    token: str
    start: int
    end: int
    label: str
    confidence: float


@dataclass(frozen=True)
class TransformationValidation:
    """Safety result for a derived representation."""

    accepted: bool
    confidence: float
    warnings: tuple[str, ...] = ()


@dataclass
class MultilingualClaim:
    """Source claim plus safe derived representations for retrieval.

    ``original_text`` must always be used for UI, audit logs, and user-facing
    explanations.  ``english_gloss`` is a derived retrieval aid only.
    """

    original_text: str
    script: str
    source_language: str
    token_tags: list[TokenTag]
    protected_entities: list[str] = field(default_factory=list)
    canonical_indic_text: str | None = None
    english_gloss: str | None = None
    transformation_confidence: float = 0.0
    linguistic_score: dict[str, object] | None = None
    warnings: list[str] = field(default_factory=list)
    validation: TransformationValidation | None = None

    @property
    def retrieval_queries(self) -> list[str]:
        """Ordered, deduplicated queries an integrator may pass to retrieval."""
        values = (self.english_gloss, self.canonical_indic_text, self.original_text)
        return list(dict.fromkeys(value for value in values if value and value.strip()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
