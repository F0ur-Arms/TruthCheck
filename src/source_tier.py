"""Lightweight medical_kb filename → evidence source tier lookup."""

from __future__ import annotations

import re
from typing import Optional

from src.evidence_quality import SOURCE_TIERS

# Ordered prefix rules — first match wins.
_FILENAME_TIER_RULES = (
    (re.compile(r"^who", re.I), "who"),
    (re.compile(r"^icmr", re.I), "icmr"),
    (re.compile(r"^nin", re.I), "nin"),
    (re.compile(r"^nih", re.I), "nih"),
    (re.compile(r"^fssai", re.I), "fssai"),
    (re.compile(r"^pubmed", re.I), "pubmed"),
    (re.compile(r"^vishvas", re.I), "vishvas"),
    (re.compile(r"^thip", re.I), "thip"),
    (re.compile(r"^mayo", re.I), "mayo"),
    (re.compile(r"^cleveland", re.I), "cleveland"),
    (re.compile(r"^nhs", re.I), "nhs"),
    (re.compile(r"^healthline", re.I), "guideline"),
    (re.compile(r"^toi", re.I), "toi"),
    (re.compile(r"^indianexpress", re.I), "indianexpress"),
    (re.compile(r"^foodfacts", re.I), "guideline"),
)


def tier_from_filename(filename: Optional[str]) -> str:
    """Map a medical_kb source filename to an evidence tier label."""
    if not filename:
        return "journalism"

    stem = filename.lower().removesuffix(".txt")
    for pattern, tier in _FILENAME_TIER_RULES:
        if pattern.search(stem):
            return tier
    return "journalism"


def tier_from_passage_text(text: str) -> str:
    """Fallback tier label inferred from passage body when filename is unknown."""
    text_lower = text.lower()
    for source in SOURCE_TIERS:
        if source in text_lower:
            return source
    return "journalism"
