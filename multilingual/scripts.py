"""Unicode script detection and source-preserving tokenisation."""

from __future__ import annotations

import string
from collections import Counter
from typing import Iterable


SCRIPT_RANGES = {
    "Devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Bengali": ((0x0980, 0x09FF),),
    "Gujarati": ((0x0A80, 0x0AFF),),
    "Gurmukhi": ((0x0A00, 0x0A7F),),
    "Kannada": ((0x0C80, 0x0CFF),),
    "Malayalam": ((0x0D00, 0x0D7F),),
    "Odia": ((0x0B00, 0x0B7F),),
    "Tamil": ((0x0B80, 0x0BFF),),
    "Telugu": ((0x0C00, 0x0C7F),),
    "Arabic": ((0x0600, 0x06FF),),
}

SCRIPT_TO_INDIC_TRANS_CODE = {
    "Devanagari": "hin_Deva",
    "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr",
    "Gurmukhi": "pan_Guru",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Odia": "ory_Orya",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Arabic": "urd_Arab",
    "Latin": "eng_Latn",
}

# Python's ``\w`` does not include the combining vowel signs and viramas used
# in Indic scripts.  A regex based on it therefore turned e.g. ``हल्दी`` into
# several bogus tokens.  Keep each run of non-space, non-punctuation Unicode
# characters together instead, and emit punctuation as its own source token.
_PUNCTUATION = frozenset(string.punctuation) | {"।", "॥"}


def char_script(character: str) -> str:
    codepoint = ord(character)
    for script, ranges in SCRIPT_RANGES.items():
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    if character.isascii() and character.isalpha():
        return "Latin"
    return "Common"


def detect_script(text: str) -> str:
    """Return the dominant meaningful script, or ``Mixed``/``Unknown``."""
    scripts = [char_script(character) for character in text]
    counts = Counter(script for script in scripts if script not in {"Common", "Unknown"})
    if not counts:
        return "Unknown"
    if len(counts) > 1:
        return "Mixed"
    return counts.most_common(1)[0][0]


def iter_tokens(text: str) -> Iterable[tuple[str, int, int]]:
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        start = index
        if text[index] in _PUNCTUATION:
            index += 1
        else:
            while (index < len(text) and not text[index].isspace()
                   and text[index] not in _PUNCTUATION):
                index += 1
        yield text[start:index], start, index


def language_code_for_script(script: str) -> str:
    return SCRIPT_TO_INDIC_TRANS_CODE.get(script, "eng_Latn")
