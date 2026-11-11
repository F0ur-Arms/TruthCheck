"""Split a knowledge-base file into passages.

lines (default): every line of 20+ characters is a passage. Bullets, table rows and headings become
passages of their own; in round 13, 40% of the passages the judge read were under 8 words
("- Reported number of cases: 1"), which take judging slots and almost never settle a claim.

window (TRUTHCHECK_KB_CHUNKING=window): consecutive lines are grouped into passages of about
TARGET words, a heading starts a new passage and stays with the text under it, a very long line is
split at sentence ends, and the page title is put in front of each passage so a passage read on its
own still says what it is about.
"""

from __future__ import annotations

import re
from typing import List, Optional

TARGET_WORDS = 90
MAX_WORDS = 170
MIN_WORDS = 25
_HEADER = ("SOURCE:", "TITLE:", "URL:")
_BULLET = re.compile(r"^[\s\-–•▪◦*]+")


def line_passages(lines: List[str]) -> List[str]:
    return [
        line for line in (l.strip() for l in lines)
        if len(line) >= 20
        and not line.isupper()
        and not line.startswith(_HEADER)
        and "─" not in line
    ]


def _words(text: str) -> int:
    return len(text.split())


def _is_heading(line: str) -> bool:
    if line.isupper() and _words(line) <= 12:
        return True
    if line.endswith("?") and _words(line) <= 20:
        return True
    return _words(line) <= 10 and not re.search(r"[.!:;,)\]]$", line) and not _BULLET.match(line)


def _split_long(line: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", line)
    out, cur = [], []
    for s in sentences:
        if cur and _words(" ".join(cur + [s])) > MAX_WORDS:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    # Text without sentence punctuation (tables flattened into one line) is cut by word count.
    pieces = []
    for piece in out:
        words = piece.split()
        pieces += [" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS)]
    return pieces


def window_passages(lines: List[str], title: Optional[str] = None) -> List[str]:
    title = (title or "").strip()
    body = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(_HEADER) or "─" in line:
            continue
        line = _BULLET.sub("", line).strip()
        if line:
            body.extend(_split_long(line) if _words(line) > MAX_WORDS else [line])

    chunks: List[List[str]] = []
    cur: List[str] = []

    def flush():
        if cur:
            chunks.append(list(cur))
            cur.clear()

    for line in body:
        n = sum(_words(x) for x in cur)
        if _is_heading(line):
            # A heading opens a new passage unless the current one is still too short to stand alone.
            if n >= MIN_WORDS:
                flush()
            cur.append(line)
            continue
        if cur and n + _words(line) > MAX_WORDS and n >= MIN_WORDS:
            flush()
        cur.append(line)
        if sum(_words(x) for x in cur) >= TARGET_WORDS:
            flush()
    if cur:
        # A short tail joins the previous passage instead of standing alone.
        if chunks and sum(_words(x) for x in cur) < MIN_WORDS and \
                sum(_words(x) for x in chunks[-1]) + sum(_words(x) for x in cur) <= MAX_WORDS + MIN_WORDS:
            chunks[-1].extend(cur)
        else:
            chunks.append(list(cur))

    out = []
    for chunk in chunks:
        text = " ".join(chunk)
        if _words(text) < 5 or len(text) < 20:
            continue
        if title and not text.lower().startswith(title.lower()):
            text = f"{title}: {text}"
        out.append(text)
    return out


def file_title(lines: List[str]) -> Optional[str]:
    for line in lines[:5]:
        if line.startswith("TITLE:"):
            return line[len("TITLE:"):].strip()
    return None
