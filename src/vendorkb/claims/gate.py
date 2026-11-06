"""The evidence gate: a claim is kept only when its quote is verbatim in the chunk.

Both strings are normalized (lowercase, whitespace collapsed, curly quotes and dashes unified,
zero-width characters and soft hyphens dropped, "…" spelled "..."). Then, in order:

1. a normalized substring is accepted as "exact";
2. a substring once every non-alphanumeric character is a space (markdown, table pipes,
   punctuation) is accepted as "exact-nopunct";
3. ``rapidfuzz.fuzz.partial_ratio`` >= ``FUZZY_THRESHOLD`` is accepted as "fuzzy:<score>"
   only if every quote word of 3+ characters appears in the aligned chunk span, so an
   inserted "not" or a swapped platform name fails however high the score.

Anything else is "no-match:<score>" (with the missing words, if the score passed). A quote
under ``MIN_WORDS`` words *and* under ``MIN_CHARS`` characters is "too-short": a lone table
cell such as "Yes" ties nothing to a capability. ``match`` also returns the matched chunk span
as it stands in the original text: that, not the model's quote, is the stored evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 90.0
MIN_WORDS = 3
MIN_CHARS = 15
MIN_WORD_LEN = 3

_TRANSLATE = str.maketrans({
    **dict.fromkeys("‘’‚‛′`´", "'"),
    **dict.fromkeys("“”„‟″«»", '"'),
    **dict.fromkeys("‐‑‒–—―−﹘﹣－", "-"),
    **dict.fromkeys("​‌‍⁠﻿­", None),
    "…": "...",
})
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str
    span: str | None = None  # the matched chunk text, verbatim; None when rejected


def _mapped(s: str, *, nopunct: bool = False) -> tuple[str, list[int]]:
    """Normalize ``s``; ``idx[i]`` is the index in ``s`` of normalized character ``i``."""
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(s):
        for c in ch.translate(_TRANSLATE).lower():
            if c.isspace() or (nopunct and not c.isalnum()):
                if out and out[-1] != " ":
                    out.append(" ")
                    idx.append(i)
            else:
                out.append(c)
                idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def normalize(s: str) -> str:
    return _mapped(s)[0]


def _span(original: str, idx: list[int], start: int, end: int) -> str:
    return original[idx[start]:idx[end - 1] + 1]


def _find(q: str, t: str, idx: list[int], original: str) -> str | None:
    """The original text of ``q``'s first occurrence in ``t``, matched on whole-string
    boundaries of ``q`` (a leading space in ``t`` is never part of the span)."""
    at = t.find(q)
    return None if at < 0 else _span(original, idx, at, at + len(q))


def match(quote: str, chunk_text: str) -> GateResult:
    q, _ = _mapped(quote)
    if not q:
        return GateResult(False, "empty")
    if len(q.split()) < MIN_WORDS and len(q) < MIN_CHARS:
        return GateResult(False, "too-short")
    t, idx = _mapped(chunk_text)
    if (span := _find(q, t, idx, chunk_text)) is not None:
        return GateResult(True, "exact", span)
    qp = _mapped(quote, nopunct=True)[0].strip()
    tp, idxp = _mapped(chunk_text, nopunct=True)
    if qp and tp.startswith(" "):
        tp, idxp = tp[1:], idxp[1:]
    if qp and (span := _find(qp, tp, idxp, chunk_text)) is not None:
        return GateResult(True, "exact-nopunct", span)
    if len(q) > len(t):
        # partial_ratio would align the chunk inside the quote: never a fuzzy match
        return GateResult(False, f"no-match:{float(fuzz.ratio(q, t)):.1f}")
    al = fuzz.partial_ratio_alignment(q, t)
    score = float(al.score) if al is not None else 0.0
    if al is None or score < FUZZY_THRESHOLD:
        return GateResult(False, f"no-match:{score:.1f}")
    start, end = al.dest_start, al.dest_end
    while start > 0 and t[start - 1].isalnum():  # widen to whole words
        start -= 1
    while end < len(t) and t[end].isalnum():
        end += 1
    while start < end and t[start] == " ":
        start += 1
    while end > start and t[end - 1] == " ":
        end -= 1
    span_words = set(_WORD.findall(t[start:end]))
    missing = sorted({w for w in _WORD.findall(q) if len(w) >= MIN_WORD_LEN} - span_words)
    if missing:
        return GateResult(False, f"no-match:{score:.1f} missing:{','.join(missing)}")
    return GateResult(True, f"fuzzy:{score:.1f}", _span(chunk_text, idx, start, end))


def check(quote: str, chunk_text: str) -> tuple[bool, str]:
    r = match(quote, chunk_text)
    return r.ok, r.reason
