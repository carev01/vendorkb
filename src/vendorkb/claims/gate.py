"""The evidence gate: a claim is kept only when its quote is verbatim in the chunk.

Both strings are normalized (lowercase, whitespace collapsed, curly quotes and dashes unified).
A normalized substring is accepted as "exact"; otherwise ``rapidfuzz.fuzz.partial_ratio`` at
or above ``FUZZY_THRESHOLD`` is accepted as "fuzzy:<score>"; anything else is rejected as
"no-match:<score>", and an empty quote as "empty".
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 90.0

_TRANSLATE = str.maketrans({
    **dict.fromkeys("‘’‚‛′`´", "'"),
    **dict.fromkeys("“”„‟″«»", '"'),
    **dict.fromkeys("‐‑‒–—―−﹘﹣－", "-"),
})
_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    return _WS.sub(" ", s.translate(_TRANSLATE).lower()).strip()


def check(quote: str, chunk_text: str) -> tuple[bool, str]:
    q = normalize(quote)
    if not q:
        return False, "empty"
    t = normalize(chunk_text)
    if q in t:
        return True, "exact"
    # partial_ratio aligns the shorter string inside the longer: a quote longer than the
    # chunk would be scored by the chunk inside the quote, so it never fuzzy-matches.
    score = float(fuzz.partial_ratio(q, t)) if len(q) <= len(t) else float(fuzz.ratio(q, t))
    if len(q) <= len(t) and score >= FUZZY_THRESHOLD:
        return True, f"fuzzy:{score:.1f}"
    return False, f"no-match:{score:.1f}"
