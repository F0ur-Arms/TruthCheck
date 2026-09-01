"""Embedding-based semantic preservation with structural integrity checks."""

from __future__ import annotations

import re

from .types import TransformationValidation

NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:[a-zA-Z%]+)?\b")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


class SemanticPreserver:
    """Uses LaBSE cross-lingual embeddings, not keyword heuristics."""

    def __init__(self, model_name: str = "sentence-transformers/LaBSE", *, min_similarity: float = 0.72,
                 local_files_only: bool = False) -> None:
        self.model_name, self.min_similarity, self.local_files_only, self._model = model_name, min_similarity, local_files_only, None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install sentence-transformers for semantic preservation.") from exc
        self._model = SentenceTransformer(self.model_name, local_files_only=self.local_files_only)

    @staticmethod
    def _structural_values(text: str) -> set[str]:
        return ({match.group(0).casefold().strip() for match in NUMBER_PATTERN.finditer(text)} |
                {match.group(0).casefold() for match in URL_PATTERN.finditer(text)})

    def validate(self, source: str, translated: str) -> TransformationValidation:
        if not translated or not translated.strip():
            return TransformationValidation(False, 0.0, ("Translation produced no text.",))
        missing = [value for value in self._structural_values(source) if value not in translated.casefold()]
        self._load()
        vectors = self._model.encode([source, translated], normalize_embeddings=True)
        similarity = float(vectors[0] @ vectors[1])
        warnings = [f"Protected value missing from gloss: {value}" for value in missing]
        if similarity < self.min_similarity:
            warnings.append(f"Low cross-lingual semantic similarity: {similarity:.3f}")
        return TransformationValidation(not warnings, max(0.0, similarity - 0.2 * len(missing)), tuple(warnings))
