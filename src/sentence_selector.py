"""Keep only the evidence sentences most relevant to a claim before NLI.

NLI judges trained on short pairs do badly on abstract-length passages; scoring each
sentence against the claim and keeping the top k (in original order) shortens the premise.
"""

from __future__ import annotations

import re
from typing import Callable, List, Sequence

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SPLIT.split(text.strip()) if s.strip()]


def select_sentences(claim: str, text: str, score_pairs: Callable[[Sequence[Sequence[str]]], Sequence[float]],
                     k: int = 2) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= k:
        return text
    scores = score_pairs([[claim, s] for s in sentences])
    keep = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:k])
    return " ".join(sentences[i] for i in keep)
