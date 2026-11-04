"""Lightweight medical_kb filename → evidence source tier lookup."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from config import KB_PATH

from src.evidence_quality import SOURCE_TIERS

# Ordered prefix rules — first match wins.
_FILENAME_TIER_RULES = (
    (re.compile(r"^who", re.I), "who"),
    (re.compile(r"^icmr", re.I), "icmr"),
    (re.compile(r"^nin", re.I), "nin"),
    (re.compile(r"^nih", re.I), "nih"),
    (re.compile(r"^fssai", re.I), "fssai"),
    (re.compile(r"^cdc", re.I), "cdc"),
    (re.compile(r"^mohfw", re.I), "mohfw"),
    (re.compile(r"^nhm", re.I), "nhm"),
    (re.compile(r"^medline", re.I), "medline"),
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


@lru_cache(maxsize=4096)
def source_url(filename: Optional[str]) -> Optional[str]:
    """The URL in a knowledge-base file's header, when it has one."""
    if not filename:
        return None
    try:
        with open(Path(KB_PATH) / Path(filename).name, encoding="utf-8", errors="ignore") as f:
            for line in (next(f, "") for _ in range(5)):
                if line.startswith("URL:"):
                    url = line[4:].strip()
                    return url if url.startswith(("http://", "https://")) else None
    except OSError:
        pass
    return None


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
    for source in SOURCE_TIERS:
        # Whole words only; WHO must be capitalised so the pronoun does not match.
        pattern = r"\bWHO\b" if source == "who" else rf"\b{source}\b"
        if re.search(pattern, text, 0 if source == "who" else re.I):
            return source
    return "journalism"
