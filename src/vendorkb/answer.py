from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from vendorkb.modes import ENUM_PER_VENDOR, ENUM_PER_VENDOR_MULTI, compare_facets, enumerate_vendors
from vendorkb.scope import ScopeIndex, find_facets, product_surfaces
from vendorkb.search import Hit, SearchResult, search_docs
from vendorkb.shape import WORKLOAD_VENDORS, Facet, Shape, detect_shape, named_facets

if TYPE_CHECKING:
    import psycopg

    from vendorkb.clients import Embedder, Reranker
    from vendorkb.taxonomy.load import Topic
    from vendorkb.topics import Grid, TopicHits

log = logging.getLogger(__name__)

# up to ~40 vendor lines over 60 chunks, with reasoning on
# The model's reasoning uses most of it (5.8k-8k tokens for a 20-30 line answer, 2026-11-08).
ENUMERATION_MAX_TOKENS = 16000
# A topic question with more sides than this keeps the plain comparison path: the per-aspect
# searches and rerank scale with the sides (4 sides x 8 aspects).
TOPIC_MAX_SIDES = 4
# The summary above a topic grid: a short call, its reasoning included.
GRID_SUMMARY_MAX_TOKENS = 2000
GRID_SUMMARY_RULES = (
    "Summarise this coverage table of backup vendors in at most four sentences, using only the "
    "counts and names given; no citations, no claims beyond the table. A vendor not listed may "
    "still offer a capability: never state or imply that a vendor lacks one, and never write "
    "\"only\".")
# A summary saying this is dropped (the grid stands alone): the table records facts found, and
# a vendor with none may still offer the capability.
_ABSENCE = re.compile(r"\b(only|no vendors?|none of|lacks?|lacking|does not offer|do not offer"
                      r"|doesn['\u2019]t|don['\u2019]t|not supported|unsupported)\b", re.IGNORECASE)
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


def topic_rules(topic: Topic, *, products_named: bool) -> str:
    """Rules for a broad question on ``topic``: one part of the answer per aspect, in order."""
    head = (f"\n- The question asks about {topic.name.lower()} in general. Organise the answer by "
            f"these aspects, in this order: {'; '.join(a.name for a in topic.aspects)}.\n")
    if products_named:
        return head + ("- For each product in the question, state what its sources show for each "
                       "aspect, with citations, and name the product each statement is about (a "
                       "vendor may have several); for an aspect its sources do not cover, write "
                       "\"the retrieved sources do not cover this\", never that the product "
                       "lacks it.")
    return head + ("- The question names no product: for each aspect, describe what the sources "
                   "show, with citations, naming the vendor and product each statement comes "
                   "from; do not present the vendors named as a complete list.")


# `vendorkb sync` runs in a 1 Gi pod and imports this module through the CLI, and the topics code
# pulls in the taxonomy stack (numpy, scikit-learn): these import it on first use.
async def load_topics(conn: psycopg.AsyncConnection) -> list[Topic]:
    from vendorkb.topics import load_topics as load
    return await load(conn)


def detect_topic(question: str, topics: list[Topic],
                 exclude: Iterable[str] = ()) -> Topic | None:
    from vendorkb.topics import detect_topic as detect
    return detect(question, topics, exclude)


def is_broad(question: str, topic: Topic) -> bool:
    from vendorkb.topics import is_broad as broad
    return broad(question, topic)


def topic_applies(question: str, shape: Shape, topic: Topic, named: list[Facet]) -> bool:
    """Whether a question that triggers ``topic`` gets the topic's answer. A "which vendors"
    question gets the grid only with no facet and nothing but broad words (``is_broad``); a
    single or comparison question keeps today's path when it names a workload vendor beside
    another facet ("How does Veeam protect Microsoft 365 against ransomware?"). ``named`` are
    the facets as the question names them (``shape.named_facets``), before ``detect_shape``
    drops a workload vendor from the sides of a weak comparison."""
    if shape.kind == "enumeration":
        return not shape.facets and is_broad(question, topic)
    workload = [f for f in named if f[0] == "vendor" and f[1] in WORKLOAD_VENDORS]
    return not (workload and len(workload) < len(named))


async def topic_hits(conn: psycopg.AsyncConnection, question: str, topic: Topic, *,
                     facets: list[Facet], embedder: Embedder, reranker: Reranker | None,
                     budget: int | None = None, per_aspect: int | None = None,
                     own: int | None = None) -> TopicHits:
    """``topics.topic_hits``; a tunable left None keeps that function's own default."""
    from vendorkb.topics import topic_hits as hits
    tunables = {k: v for k, v in (("budget", budget), ("per_aspect", per_aspect), ("own", own))
                if v is not None}
    return await hits(conn, question, topic, facets=facets, embedder=embedder,
                      reranker=reranker, **tunables)


async def topic_grid(conn: psycopg.AsyncConnection, topic: Topic, *, vendors: list[str],
                     embedder: Embedder, reranker: Reranker | None,
                     catch_up: bool | None = None) -> Grid:
    """``topics.topic_grid``; ``catch_up`` left None keeps that function's own default."""
    from vendorkb.topics import topic_grid as grid
    extra = {} if catch_up is None else {"catch_up": catch_up}
    return await grid(conn, topic, vendors=vendors, embedder=embedder, reranker=reranker,
                      **extra)


def render_grid(grid: Grid, topic: Topic) -> tuple[str, list[Hit]]:
    from vendorkb.topics import render_grid as render
    return render(grid, topic)


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


def _facets_without_hits(facets: list[Facet], hits: list[Hit]) -> list[str]:
    """Names of the facets none of ``hits`` belongs to."""
    return [name for kind, name in facets
            if not any((h.vendor if kind == "vendor" else h.product) == name for h in hits)]


def closing_line(names: list[str]) -> str:
    if not names:
        return ""
    return "No supporting passage found for: " + ", ".join(sorted(names)) + "."


def unconfirmed_line(names: list[str]) -> str:
    """Vendors with retrieved passages that no checked answer line used."""
    if not names:
        return ""
    return "Passages found but not judged to show it: " + ", ".join(sorted(names)) + "."


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


def citation(n: int, hit: Hit) -> dict:
    """The citation of marker ``n``: the chunk and its page."""
    return {"marker": n, "chunk_id": hit.chunk_id, "doc_id": hit.doc_id,
            "url": hit.url, "title": hit.title, "section_path": hit.section_path,
            # the evidence as the model saw it: judging never depends on a
            # later DB state (a re-ingest may replace the chunk)
            "text": hit.text}


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
    return text, [citation(n, hits[n - 1]) for n in order], unknown


def _renumber(text: str, sub: list[Hit], hits: list[Hit]) -> str:
    """[n] markers into ``sub`` rewritten as markers into ``hits`` (``sub`` is a subset of
    ``hits``); a marker out of ``sub``'s range is dropped."""
    position = {id(h): i for i, h in enumerate(hits, 1)}

    def repl(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(2))]
        mapped = [str(position[id(sub[n - 1])]) for n in nums if 1 <= n <= len(sub)]
        return f"{m.group(1)}[{', '.join(mapped)}]" if mapped else ""
    return _GROUP.sub(repl, text)


async def _second_pass(client: AsyncOpenAI, settings, question: str, hits: list[Hit],
                       kept: list[str], checked: list[str], listed: list[str]
                       ) -> tuple[str, list[str], int, int, dict | None]:
    """One more call over the kept vendors the first answer left out, with only their
    passages: in a prompt of up to 60 passages for 30-odd vendors the model skips vendors whose
    passages state the capability (32 of 419 expected vendors on the labelled questions,
    2026-11-08). Lines are checked against those passages, then their markers renumbered
    into ``hits``. Returns (lines, vendors listed, lines dropped, lines naming other vendors,
    usage); a failed call adds nothing."""
    omitted = [v for v in kept if v not in listed and any(h.vendor == v for h in hits)]
    if not omitted:
        return "", [], 0, 0, None
    sub = [h for h in hits if h.vendor in omitted]
    content, usage, error, _ = await _call_model(
        client, settings, build_enumeration_prompt(question, sub, omitted),
        SYSTEM_RULES + ENUMERATION_RULES, max_tokens=ENUMERATION_MAX_TOKENS)
    if content is None:
        log.warning("vendorkb enumeration second pass failed: %s", error)
        return "", [], 0, 0, usage
    text, added, dropped, other = _check_lines(content, sub, omitted, checked)
    return _renumber(text, sub, hits), added, dropped, other, usage


def _add_usage(a: dict | None, b: dict | None) -> dict | None:
    if a is None or b is None:
        return a or b
    cost = None if a["cost"] is None and b["cost"] is None else (a["cost"] or 0) + (b["cost"] or 0)
    return {"prompt_tokens": a["prompt_tokens"] + b["prompt_tokens"],
            "completion_tokens": a["completion_tokens"] + b["completion_tokens"], "cost": cost}


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


def _summary_prompt(grid: Grid, topic: Topic) -> str:
    """One line per aspect: how many vendors have a capability fact for it, and which. A page
    not yet analysed is no fact (the summary must not state what nothing verified). Worded as
    facts found, never as vendors lacking the aspect."""
    lines = []
    for aspect in topic.aspects:
        names = sorted(v for (v, a), cell in grid.cells.items()
                       if a == aspect.id and cell.mark != "recent")
        lines.append(f"{aspect.name}: facts found for {len(names)} vendors: {', '.join(names)}"
                     if names else f"{aspect.name}: no facts found")
    return "\n".join(lines)


async def _answer_grid(conn, question: str, topic: Topic, *, settings, embedder, reranker,
                       scope_index: ScopeIndex, client: AsyncOpenAI) -> dict:
    """A "which vendors" question on ``topic``: the grid of every vendor with documents by the
    topic's aspects, built from the capability table in code (``topic_grid``), under a short
    model summary of its counts. The model writes no cell and no citation: ``[n]`` is the
    n-th cell of the table. A failed summary leaves the grid alone."""
    checked = list(scope_index.vendors)
    grid = await topic_grid(conn, topic, vendors=checked, embedder=embedder, reranker=reranker)
    table, hits = render_grid(grid, topic)
    out: dict = {
        "query": question, "answer": table,
        "citations": [citation(n, h) for n, h in enumerate(hits, start=1)],
        "unknown_markers": 0, "uncited": False, "shape": "enumeration", "topic": topic.id,
        "aspects": [a.name for a in topic.aspects],
        "scope": {"vendors": [], "products": [], "relaxed": False},
        "rerank_ok": grid.rerank_ok, "retrieved": len(hits),
        "reranked_docs": grid.reranked_docs, "catch_up_ok": grid.catch_up_ok,
        "gated_out": grid.gated_out,
        "truncated": False, "usage": None, "error": None, "summary_error": None,
        "vendors_checked": checked, "vendors_listed": list(grid.vendors), "lines_dropped": 0,
        "lines_naming_other_vendors": 0, "second_pass_listed": 0}
    if not hits:  # nothing to summarise
        return out
    content, usage, error, finish_reason = await _call_model(
        client, settings, _summary_prompt(grid, topic), GRID_SUMMARY_RULES,
        max_tokens=GRID_SUMMARY_MAX_TOKENS)
    out["usage"] = usage
    out["truncated"] = finish_reason == "length"
    if out["truncated"]:
        log.warning("vendorkb grid summary truncated at %d tokens", GRID_SUMMARY_MAX_TOKENS)
    if content is None:
        # `error` stays None: the grid is an answer, and mcp_server turns `error` into a ToolError
        log.warning("vendorkb grid summary failed: %s", error)
        out["summary_error"] = error
        return out
    stripped = 0

    def drop(m: re.Match[str]) -> str:
        nonlocal stripped
        stripped += len(re.split(r"\s*,\s*", m.group(2)))
        return ""

    # a marker in the summary would cite a cell of the table by accident; counted by number,
    # as resolve_markers counts the ones it drops
    summary = _GROUP.sub(drop, content).strip()
    out["unknown_markers"] = stripped
    if not summary:
        out["summary_error"] = "empty reply"
    elif _ABSENCE.search(summary):
        log.warning("vendorkb grid summary dropped: it states or implies absence")
        out["summary_error"] = "absence wording"
    else:
        out["answer"] = f"{summary}\n\n{table}"
    return out


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
    topic = None
    reranked_docs = 0  # the topic path's (TopicHits.reranked_docs)
    # a "which vendors" topic question gets the grid only with TOPIC_GRID on; off, it is any
    # other enumeration (no topic lookup at all)
    if (settings.topic_grid if shape.kind == "enumeration"
            else len(shape.facets) <= TOPIC_MAX_SIDES):
        # a product's name ("PowerProtect Cyber Recovery") never stands for the topic; the
        # offerings (taxonomy) are not in the scope index, so they are not excluded here
        topic = detect_topic(question, await load_topics(conn),
                             exclude=product_surfaces(scope_index))
        if topic is not None and not topic_applies(question, shape, topic,
                                                   named_facets(question, scope_index)):
            topic = None  # a qualified question: today's answer
    if shape.kind == "enumeration" and topic is not None:
        return await _answer_grid(conn, question, topic, settings=settings, embedder=embedder,
                                  reranker=reranker, scope_index=scope_index, client=client)
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
    elif topic is not None:
        # the sides are the question's facets (a vendor and its own product count once, see
        # detect_shape); none for an unscoped question, searched across the whole corpus
        th = await topic_hits(conn, question, topic, facets=shape.facets, embedder=embedder,
                              reranker=reranker)
        result = SearchResult(th.hits, th.rerank_ok)
        reranked_docs = th.reranked_docs
        system = SYSTEM_RULES + topic_rules(topic, products_named=bool(shape.facets))
        vendors = [n for k, n in shape.facets if k == "vendor"]
        products = [n for k, n in shape.facets if k == "product"]
        if len(shape.facets) >= 2:  # a single question may name several sides too
            facets_empty = _facets_without_hits(shape.facets, th.hits)
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
                 "topic": None if topic is None else topic.id,
                 "scope": {"vendors": vendors, "products": products, "relaxed": relaxed},
                 "rerank_ok": result.rerank_ok, "retrieved": len(result.hits),
                 "usage": None, "error": None}
    if topic is not None:
        out["aspects"] = [a.name for a in topic.aspects]
        out["reranked_docs"] = reranked_docs
    if shape.kind == "comparison" or (topic is not None and len(shape.facets) >= 2):
        out["facets_empty"] = facets_empty
    if shape.kind == "enumeration":
        out.update(vendors_checked=checked, vendors_listed=[], lines_dropped=0, lines_naming_other_vendors=0,
                   second_pass_listed=0)

    def closing() -> str:
        if shape.kind == "enumeration":
            unlisted = [v for v in checked if v not in out["vendors_listed"]]
            seen = {h.vendor for h in result.hits} & set(kept)
            lines = (unconfirmed_line([v for v in unlisted if v in seen]),
                     closing_line([v for v in unlisted if v not in seen]))
            return "\n".join(line for line in lines if line)
        return closing_line(facets_empty)

    def finish(body: str) -> dict:
        tail = closing()
        out["answer"] = f"{body}\n{tail}" if tail else body
        return out

    if not result.hits:
        return finish(REFUSAL)
    prompt = (build_enumeration_prompt(question, result.hits, kept)
              if shape.kind == "enumeration" else build_prompt(question, result.hits))
    wide = shape.kind == "enumeration" or topic is not None  # a long, structured answer
    content, usage, error, finish_reason = await _call_model(
        client, settings, prompt, system, max_tokens=ENUMERATION_MAX_TOKENS if wide else 4000)
    out["usage"] = usage
    if wide:
        # a cut-off reply loses its tail: an enumeration's trailing vendor lines, which the
        # closing line then lists, or a topic answer's last aspects
        out["truncated"] = finish_reason == "length"
        if out["truncated"]:
            log.warning("vendorkb %s answer truncated at %d tokens",
                        "enumeration" if shape.kind == "enumeration" else "topic",
                        ENUMERATION_MAX_TOKENS)
    if content is None:
        out["error"] = error
        return out
    if shape.kind != "enumeration" and _is_refusal(content):
        return finish(content.strip())
    if shape.kind == "enumeration":
        # never short-circuit on the refusal sentence: it is just a non-vendor line here
        content, listed, dropped, other = _check_lines(content, result.hits, kept, checked)
        extra, added, dropped2, other2, usage2 = await _second_pass(
            client, settings, question, result.hits, kept, checked, listed)
        if extra:
            content = f"{content}\n{extra}" if content else extra
        out.update(vendors_listed=listed + [v for v in added if v not in listed],
                   lines_dropped=dropped + dropped2, lines_naming_other_vendors=other + other2,
                   second_pass_listed=len(added), usage=_add_usage(usage, usage2))
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
