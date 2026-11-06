from __future__ import annotations

import logging
import re

from openai import AsyncOpenAI

from vendorkb.modes import ENUM_PER_VENDOR, ENUM_PER_VENDOR_MULTI, compare_facets, enumerate_vendors
from vendorkb.scope import ScopeIndex, find_facets
from vendorkb.search import Hit, SearchResult, search_docs
from vendorkb.shape import WORKLOAD_VENDORS, detect_shape

log = logging.getLogger(__name__)

# up to ~40 vendor lines over 60 chunks, with reasoning on
ENUMERATION_MAX_TOKENS = 8000
REFUSAL = "I don't have enough information to answer that from the available sources."

SYSTEM_RULES = (
    "You answer questions about backup-industry vendor documentation.\n"
    "Rules:\n"
    "- Answer only from the numbered sources in the user message; use no outside knowledge.\n"
    "- Cite with [n] markers after each claim, where n is the source number. "
    "Several sources may be cited as [1][2].\n"
    "- Never write URLs or links; only [n] markers identify sources.\n"
    "- If the sources do not answer the question, reply with exactly this sentence and "
    f"nothing else: {REFUSAL}"
)

ENUMERATION_RULES = (
    "\n- This question asks which vendors do something. Answer with one line per vendor, "
    "exactly in the form `Vendor (Product, ...): how it is done [n]`.\n"
    "- Begin each line with one of the exact vendor names listed after \"Vendors:\" in the "
    "user message; never an alias or a product name.\n"
    "- If you use the product group, use only the product names listed for that vendor "
    "after \"Vendors:\".\n"
    "- Each line describes only its own vendor and never mentions other vendors.\n"
    "- Write no line for a vendor whose sources do not state the capability.\n"
    "- Each line states what its vendor does, never what it does not do or what is not "
    "documented.\n"
    "- Add nothing after the lines."
)

_GROUP = re.compile(r"([ ]?)\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def build_prompt(question: str, hits: list[Hit]) -> str:
    """User message: numbered sources then the question. Rules live in SYSTEM_RULES."""
    blocks = []
    for n, h in enumerate(hits, start=1):
        trail = " > ".join(h.section_path)
        blocks.append(f"[{n}] {h.title} | {trail}\n{h.text}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n"


def build_enumeration_prompt(question: str, hits: list[Hit], vendors: list[str]) -> str:
    items = []
    for v in vendors:
        products = list(dict.fromkeys(h.product for h in hits if h.vendor == v and h.product))
        items.append(f"{v} ({', '.join(products)})" if products else v)
    return build_prompt(question, hits) + "Vendors: " + "; ".join(items) + "\n"


def closing_line(names: list[str]) -> str:
    if not names:
        return ""
    return "No supporting passage found for: " + ", ".join(sorted(names)) + "."


_BULLET = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s+")
_EMPHASIS = re.compile(r"^(?:\*\*|__|\*)")
_PRODUCTS = re.compile(r"^ ?\(((?:[^()]|\([^()]*\))*)\)")  # one level of nesting


def _parse_line(line: str, vendors: list[str]) -> tuple[str, str | None, str] | None:
    """(vendor, products, rest) when the line is ``Vendor [(Product, ...)]: rest`` with the
    vendor an exact name from ``vendors`` (longest match first), allowing a leading bullet or
    number and markdown emphasis around the name; else None."""
    s = _BULLET.sub("", line.strip())
    s = _EMPHASIS.sub("", s)
    for v in sorted(vendors, key=len, reverse=True):
        if not s.startswith(v):
            continue
        rest = _EMPHASIS.sub("", s[len(v):])
        m = _PRODUCTS.match(rest)
        products = m.group(1).strip() if m else None
        if m:
            rest = _EMPHASIS.sub("", rest[m.end():])
        if rest.startswith(":"):
            return v, products or None, _EMPHASIS.sub("", rest[1:]).strip()
    return None


def _canonical(vendor: str, products: str | None, rest: str) -> str:
    return f"{vendor}{f' ({products})' if products else ''}: {rest}"


def _names(name: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE) is not None


def _names_other(vendor: str, products: str | None, rest: str, corpus: list[str]) -> bool:
    text = f"{products or ''} {rest}"
    return any(_names(v, text) for v in corpus if v != vendor)


def _check_lines(text: str, hits: list[Hit], vendors: list[str],
                 corpus: list[str]) -> tuple[str, list[str], int, int]:
    kept: list[str] = []
    listed: list[str] = []
    dropped = 0
    other = 0

    for line in text.splitlines():
        if line.strip() == "":
            continue
        parsed = _parse_line(line, vendors)
        if parsed is None:
            dropped += 1
            continue
        vendor, products, rest = parsed
        own = 0
        foreign: set[str] = set()  # vendors of the markers stripped from this line

        def repl(m: re.Match[str]) -> str:
            nonlocal own
            nums = [int(x) for x in re.split(r"\s*,\s*", m.group(2))]
            keep = [n for n in nums if not (1 <= n <= len(hits)) or hits[n - 1].vendor == vendor]
            foreign.update(hits[n - 1].vendor for n in nums if n not in keep)
            own += sum(1 for n in keep if 1 <= n <= len(hits))
            if keep:
                return f"{m.group(1)}[{', '.join(str(n) for n in keep)}]"
            # a fully stripped marker takes its preceding space with it (as in
            # resolve_markers), unless another marker follows: "X [2][1]" -> "X [1]"
            return m.group(1) if m.string.startswith("[", m.end()) else ""

        rest = re.sub(r" {2,}", " ", _GROUP.sub(repl, rest)).strip()
        # no own citation; or the line describes the vendor whose citation was stripped (a
        # workload vendor is exempt: its own chunks are co-cited for the workload it names)
        if own == 0 or any(_names(v, f"{products or ''} {rest}")
                           for v in foreign - WORKLOAD_VENDORS):
            dropped += 1
            continue
        if _names_other(vendor, products, rest, corpus):
            other += 1
        if products is not None:
            known = {(h.product or "").lower() for h in hits if h.vendor == vendor}
            if not all(p.strip().lower() in known for p in products.split(",")):
                products = None
        kept.append(_canonical(vendor, products, rest))
        if vendor not in listed:
            listed.append(vendor)
    return "\n".join(kept), listed, dropped, other


def check_enumeration_lines(text: str, hits: list[Hit],
                            vendors: list[str]) -> tuple[str, list[str], int]:
    """Keep a line only when it leads with exactly one of ``vendors`` and one of its [n]
    markers resolves to a hit of that vendor; markers resolving to another vendor's hit are
    removed. Kept lines are rewritten as ``Vendor (Product, ...): text [n]``; the product
    group stays only when every product is the ``product`` of a hit of that vendor.
    Returns (kept_text, vendors_listed, dropped)."""
    kept, listed, dropped, _ = _check_lines(text, hits, vendors, [])
    return kept, listed, dropped


def count_lines_naming_other_vendors(text: str, corpus_vendors: list[str]) -> int:
    """Lines (``Vendor (...): text``) whose product group or text names another corpus
    vendor, whole word, any case. Counted, not dropped: a workload can share a vendor's name."""
    count = 0
    for line in text.splitlines():
        parsed = _parse_line(line, corpus_vendors)
        if parsed is not None and _names_other(*parsed, corpus_vendors):
            count += 1
    return count


def resolve_markers(raw: str, hits: list[Hit]) -> tuple[str, list[dict], int]:
    valid = len(hits)
    order: list[int] = []
    unknown = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal unknown
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(2))]
        keep = []
        for n in nums:
            if 1 <= n <= valid:
                keep.append(n)
                if n not in order:
                    order.append(n)
            else:
                unknown += 1
        # A fully removed marker takes ONE preceding space with it; nothing else is touched.
        return f"{m.group(1)}[{', '.join(str(n) for n in keep)}]" if keep else ""

    text = _GROUP.sub(repl, raw)
    citations = []
    for n in order:
        h = hits[n - 1]
        citations.append({"marker": n, "chunk_id": h.chunk_id, "doc_id": h.doc_id,
                          "url": h.url, "title": h.title, "section_path": h.section_path,
                          # the evidence as the model saw it: judging never depends on a
                          # later DB state (a re-ingest may replace the chunk)
                          "text": h.text})
    return text, citations, unknown


def _is_refusal(text: str) -> bool:
    return text.strip().startswith(REFUSAL)


async def _call_model(client: AsyncOpenAI, settings, user: str, system: str = SYSTEM_RULES,
                      max_tokens: int = 4000
                      ) -> tuple[str | None, dict, str | None, str | None]:
    """One model call with a single retry. Returns (content, usage_totals, error,
    finish_reason)."""
    totals: dict = {"prompt_tokens": 0, "completion_tokens": 0, "cost": None}
    error: str | None = None
    for attempt in (1, 2):
        try:
            resp = await client.chat.completions.create(
                model=settings.judge_model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,
                max_tokens=max_tokens,
                extra_body={"reasoning": {"effort": settings.answer_reasoning_effort},
                            "provider": {"sort": "throughput", "allow_fallbacks": True},
                            "usage": {"include": True}},
            )
        except Exception as e:  # noqa: BLE001 - recorded, retried once, then reported
            error = f"{type(e).__name__}: {e}"
            log.warning("vendorkb answer model call failed (attempt %d): %s", attempt, error)
            continue
        u = getattr(resp, "usage", None)
        if u is not None:
            totals["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            totals["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            cost = getattr(u, "cost", None)
            if cost is not None:
                totals["cost"] = (totals["cost"] or 0) + cost
        choices = getattr(resp, "choices", None)
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if content is None or content.strip() == "":
            error = "empty reply"
            log.warning("vendorkb answer model returned an empty reply (attempt %d)", attempt)
            continue
        return content, totals, None, getattr(choice, "finish_reason", None)
    return None, totals, error, None


async def answer(conn, question: str, *, settings, embedder, reranker,
                 scope_index: ScopeIndex, client: AsyncOpenAI) -> dict:
    shape = detect_shape(question, scope_index)
    # a single-shaped question keeps its named scope even with "vendors" wording
    # (detect_scope drops it there): enumerations never reach the scoped search
    vendors, products = find_facets(question, scope_index)
    relaxed = False
    facets_empty: list[str] = []
    checked: list[str] = []
    kept: list[str] = []
    system = SYSTEM_RULES
    if shape.kind == "enumeration":
        checked = list(scope_index.vendors)
        er = await enumerate_vendors(conn, question, embedder=embedder, reranker=reranker,
                                     per_vendor=ENUM_PER_VENDOR,
                                     per_vendor_multi=ENUM_PER_VENDOR_MULTI,
                                     vendors=checked)
        result = SearchResult(er.hits, er.rerank_ok)
        kept = er.kept
        system = SYSTEM_RULES + ENUMERATION_RULES
        vendors, products = [], []
    elif shape.kind == "comparison":
        cr = await compare_facets(conn, question, embedder=embedder, reranker=reranker,
                                  facets=shape.facets)
        result = SearchResult(cr.hits, cr.rerank_ok)
        facets_empty = cr.facets_empty
        vendors = [n for k, n in shape.facets if k == "vendor"]
        products = [n for k, n in shape.facets if k == "product"]
    else:
        scoped = bool(vendors or products)
        result = await search_docs(conn, question, embedder=embedder, reranker=reranker,
                                   vendor=vendors or None, product=products or None)
        if scoped and not result.hits:
            relaxed = True
            result = await search_docs(conn, question, embedder=embedder, reranker=reranker)
    out: dict = {"query": question, "answer": None, "citations": [], "unknown_markers": 0,
                 "uncited": False, "shape": shape.kind,
                 "scope": {"vendors": vendors, "products": products, "relaxed": relaxed},
                 "rerank_ok": result.rerank_ok, "retrieved": len(result.hits),
                 "usage": None, "error": None}
    if shape.kind == "comparison":
        out["facets_empty"] = facets_empty
    if shape.kind == "enumeration":
        out.update(vendors_checked=checked, vendors_listed=[], lines_dropped=0, lines_naming_other_vendors=0)

    def closing() -> str:
        if shape.kind == "enumeration":
            return closing_line([v for v in checked if v not in out["vendors_listed"]])
        return closing_line(facets_empty)

    def finish(body: str) -> dict:
        tail = closing()
        out["answer"] = f"{body}\n{tail}" if tail else body
        return out

    if not result.hits:
        return finish(REFUSAL)
    prompt = (build_enumeration_prompt(question, result.hits, kept)
              if shape.kind == "enumeration" else build_prompt(question, result.hits))
    content, usage, error, finish_reason = await _call_model(
        client, settings, prompt, system,
        max_tokens=ENUMERATION_MAX_TOKENS if shape.kind == "enumeration" else 4000)
    out["usage"] = usage
    if shape.kind == "enumeration":
        # a cut-off reply loses its trailing vendor lines, which the closing line then lists
        out["truncated"] = finish_reason == "length"
        if out["truncated"]:
            log.warning("vendorkb enumeration answer truncated at %d tokens",
                        ENUMERATION_MAX_TOKENS)
    if content is None:
        out["error"] = error
        return out
    if shape.kind != "enumeration" and _is_refusal(content):
        return finish(content.strip())
    if shape.kind == "enumeration":
        # never short-circuit on the refusal sentence: it is just a non-vendor line here
        content, listed, dropped, other = _check_lines(content, result.hits, kept, checked)
        out.update(vendors_listed=listed, lines_dropped=dropped,
                   lines_naming_other_vendors=other)
        if content == "":
            return finish(REFUSAL)
    text, citations, unknown = resolve_markers(content, result.hits)
    text = text.strip()
    if text == "":
        out["unknown_markers"] = unknown
        out["error"] = "empty reply"
        return out
    out.update(citations=citations, unknown_markers=unknown, uncited=not citations)
    return finish(text)
