"""Medical entity extraction using token-level language tags (no external NER model).

Leverages MuRIL Hinglish LID confidence scores to identify medical/English terms
that should be preserved during translation and transformation.
"""

from __future__ import annotations

import re
from typing import Sequence

from .types import TokenTag


class MedicalEntityGuard:
    """Extract medical entities from token tags without requiring a separate NER model.
    
    Strategy:
    1. High-confidence English words (en tag, confidence > 0.95) → likely medical/drug names
    2. All number_unit tokens (dosages like "500mg")
    3. Capitalized English words (proper nouns, drug names, conditions)
    4. Medical keywords: disease, drug, symptom patterns
    """

    # Common medical keywords/suffixes to enhance pattern matching
    MEDICAL_KEYWORDS = {
        "mg", "ml", "gm", "iu", "mcg",  # dosage units
        "diabetes", "cancer", "fever", "infection", "allergy", "asthma",
        "vitamin", "supplement", "medicine", "drug", "tablet", "injection",
        "blood", "heart", "kidney", "liver", "lung", "brain", "bone",
        "infection", "disease", "syndrome", "treatment", "therapy",
    }

    def __init__(self) -> None:
        """No external model needed; uses existing token tags."""
        pass

    def extract_from_tags(self, tags: Sequence[TokenTag]) -> set[str]:
        """Extract medical entities from token tags."""
        entities: set[str] = set()

        for tag in tags:
            token = tag.token.strip()
            if not token or len(token) < 2:
                continue

            # 1. High-confidence English words (likely drug/medical names)
            if tag.label == "en" and tag.confidence > 0.95:
                entities.add(token)
                continue

            # 2. All number/dosage tokens (500mg, 10ml, etc.)
            if tag.label == "number_unit":
                entities.add(token)
                continue

            # 3. Capitalized English tokens (proper nouns: drug names, diseases)
            if tag.label == "en" and tag.confidence > 0.90 and token[0].isupper():
                entities.add(token)
                continue

            # 4. Medical keyword patterns (case-insensitive)
            if any(keyword in token.lower() for keyword in self.MEDICAL_KEYWORDS):
                entities.add(token)
                continue

        return entities

    def protected_entities(self, text: str, tags: Sequence[TokenTag] | None = None) -> set[str]:
        """Extract protected medical entities from text and/or token tags.
        
        Args:
            text: Original claim text (for fallback pattern matching)
            tags: Token tags from MuRIL LID (preferred source)
        
        Returns:
            Set of medical entity strings to preserve during transformation
        """
        entities: set[str] = set()

        # If tags are provided, use them (high-confidence source)
        if tags:
            entities.update(self.extract_from_tags(tags))

        # Fallback: pattern-based extraction from raw text
        # Capitalized words, medical terms, dosage patterns
        for match in re.finditer(r"\b[A-Z][a-z]+\b", text):  # Capitalized words
            entities.add(match.group(0))

        for match in re.finditer(r"\d+\s*(?:mg|ml|gm|iu|mcg|%)", text, re.IGNORECASE):  # Dosages
            entities.add(match.group(0))

        return entities
