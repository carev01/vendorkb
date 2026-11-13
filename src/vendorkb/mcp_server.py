"""The MCP server (DESIGN.md §6): read-only tools over vendorkb for a downstream agent.

Every tool returns text for the model and the same result as structured content. Every fact and
passage carries its source URL and chunk id from the database; nothing here writes a URL.

- ``list_vendors``: vendors and products with documentation, document and fact counts.
- ``list_dimensions`` / ``list_targets``: browse the taxonomy; resolve the user's wording to ids.
  ``list_dimensions`` also shows the topics (broad subjects of several dimensions) a query names.
- ``query_capabilities``: the cross-vendor fact table (support, conditions, values), each fact
  with its evidence quote and source; by dimension, or by topic with the facts grouped by aspect.
- ``search_docs``: hybrid search over the documentation chunks, reranked.
- ``read_section``: a chunk with its neighbours or its whole section.
- ``answer_question``: vendorkb's cited answer (retrieval plus synthesis) in one call.
- ``gather_evidence``: that answer's evidence without the synthesis, for the agent to write from.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from vendorkb.mcp_backend import Backend
from vendorkb.search import Hit
from vendorkb.taxonomy.capabilities import CapFilter, CapResult, Fact
from vendorkb.taxonomy.load import Topic
from vendorkb.topics import match_topics

log = logging.getLogger(__name__)

MAX_K = 12  # agents ask for k=30 to "see everything"; 12 x 1,200 characters fits a tool result
MAX_LIMIT = 200
WIDE_SHOWN = 10  # product-wide facts shown at most: the workload-specific facts come first
QUOTE_SHOWN = 240  # quote characters per fact in the text (structured content has it whole)
# Per search hit in the text output (structured content has it whole): agent clients cap a tool
# result's size, and 8 x 2,500 characters overflowed them in the 2026-11-08 agent eval.
PASSAGE_CHARS = 1200
# Said of a fact, or of a read_section result, whose text is no longer on its page.
EARLIER_VERSION = " (from an earlier version of the page)"
# One part of a read_document reply (characters of chunk text). Whole chunks are packed in order;
# a chunk longer than this is a part by itself. A starting size, to be lowered to what Copilot
# Studio shows uncut (docs/deploy/copilot-studio.md).
PART_CHARS = 20_000
# The Sources section of a grid answer (a "which vendors" topic question), in characters: a grid
# of 239 cells listed one title and URL per cell came to 36k characters, past client limits.
GRID_SOURCES_CHARS = 8000
# Citations of a grid answer in the structured content (``citations_total`` counts them all).
GRID_CITATIONS_SHOWN = 60

INSTRUCTIONS = """\
vendorkb answers questions about backup and data-protection vendors' technical documentation,
precisely and with citations. Use only what these tools return.

Asking:
- Ask in plain words, one standalone question per call. Never pad a question with
  instructions, vendor lists, examples or formatting requests: every added word changes which
  passages are retrieved, and a padded "which vendors" question loses vendors. Pass the user's
  question word for word, or cleaned up (typos, a follow-up made complete), and keep any
  qualifier it has ("without agents", "natively", a version).
- gather_evidence is the default: it checks every vendor for a "which vendors" question, each
  side of a comparison, each aspect of a broad subject (ransomware protection), and returns
  numbered passages to write from. Read every part ("Part 1 of N") before writing.
- answer_question returns vendorkb's own written answer for the same evidence, checked in code
  (each line cites its own vendor's passage); relay it with its markers and Sources list.

Filling gaps:
- A vendor with no passage, or listed under "No supporting passage found for": search_docs
  with the vendor filter (exact names from list_vendors). A product or vendor missing from
  list_vendors has no documentation here.
- The text around a passage: read_section; a whole cited page: read_document (in parts).
- Structured facts across vendors (support, limits, editions): query_capabilities;
  query_capabilities(topic=...) groups a broad subject by aspect, and list_dimensions with the
  subject shows the topic's aspects. A vendor missing from its table does not lack the feature:
  check it with search_docs. When a filter matches nothing, list_dimensions / list_targets turn
  the wording into taxonomy ids.

Answering:
- Cite every factual statement: its [N] with a Sources list (title and link from the passage's
  source line), or the link itself. Never invent a link.
- Not found means not documented in vendorkb, never that the product lacks the feature. Show
  conflicting sources side by side. Keep vendor and product names as given."""

# The ``question`` of answer_question. An agent that appended "List all vendors documented,
# distinguishing supported versus partial support…" lowered every rerank score: a vendor fell
# under the enumeration floor (BACKLOG 39).
VERBATIM = ("The user's question, word for word. Add no instructions, detail, examples or "
            "formatting requests: every added word changes which passages are retrieved.")

_RO = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _page_head(h: Hit) -> str:
    """The first lines of a page's text: vendor, product, title and source."""
    return (f"{h.vendor} · {h.product or 'unknown product'} — {h.title}\n"
            f"source: {h.url or 'no url'}{EARLIER_VERSION if h.superseded else ''}")


def _parts(hits: list[Hit]) -> list[list[Hit]]:
    """Whole chunks, in order, packed into parts of at most PART_CHARS characters of text (a
    longer chunk alone)."""
    parts: list[list[Hit]] = [[]]
    size = 0
    for h in hits:
        if parts[-1] and size + len(h.text) > PART_CHARS:
            parts.append([])
            size = 0
        parts[-1].append(h)
        size += len(h.text)
    return parts


def _result(text: str, structured: dict[str, Any]) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)],
                          structured_content=structured)


# Section names of a query_capabilities result that agents pass back as a target
# ("product-wide"), normalized: they name no taxonomy target, so the filter is dropped with a note.
SECTIONS = frozenset({"product-wide", "product-level", "unresolved-target", "target-unmapped"})


def _section(target: str | None) -> str | None:
    """The result section ``target`` names, or None."""
    if target is None:
        return None
    key = "-".join(target.strip().lower().replace("_", " ").replace("-", " ").split())
    return key if key in SECTIONS else None


def _fail(tool: str, e: Exception) -> ToolError:
    log.exception("mcp: %s failed", tool)
    return ToolError(f"{tool} failed: {type(e).__name__}: {e}")


def _value(f: Fact) -> str:
    if f.value is None:
        return ""
    v = f"{f.value:g}" + (f" {f.unit}" if f.unit else "")
    return f"value {f.bound} {v}" if f.bound else f"value {v}"


def format_fact(n: int, f: Fact) -> str:
    where = " / ".join(x for x in (f.offering_id, f.edition_id) if x)
    target = f.target_id or (f'"{f.target_raw}" (not in the taxonomy)'
                             if f.scope == "target_unmapped" and f.target_raw else "product-wide")
    if f.variant_id:
        target += f" [{f.variant_id}]"
    if f.object_type_id:
        target += f" · {f.object_type_id}"
    quals = [q for q in (
        f"availability: {f.availability}" if f.availability not in (None, "unstated") else "",
        f"default: {f.default_state}" if f.default_state else "",
        _value(f),
        f"conditions: {f.conditions}" if f.conditions else "") if q]
    when = f"document fetched {f.doc_version[:10]}" if f.doc_version else "document date unknown"
    if not f.is_current and not f.evidence_superseded:  # a superseded row says why below
        when += "; a newer document says otherwise"
    lines = [f"[{n}] {f.vendor} · {f.product or 'unknown product'}"
             f"{f' ({where})' if where else ''} | {f.dimension_id} | {target} | "
             f"{f.support.upper()}"]
    if quals:
        lines.append("    " + "; ".join(quals))
    quote = " ".join(f.evidence_quote.split())
    if len(quote) > QUOTE_SHOWN:
        quote = quote[:QUOTE_SHOWN] + " […]"
    lines.append(f'    quote: "{quote}"')
    source = (f"    source: {f.title or 'untitled'} — {f.url or 'no url'}"
              f" (chunk {f.chunk_id}; {when})")
    if f.evidence_superseded:
        source += EARLIER_VERSION
    lines.append(source)
    return "\n".join(lines)


_FACT_GROUPS = ("facts", "product_level", "product_wide", "unresolved_target")


def format_capabilities(res: CapResult, limit: int,
                        start: int = 0) -> tuple[str, dict[str, Any]]:
    """The result as text and structured content; the facts are numbered from ``start`` + 1."""
    groups = [("facts", "Facts", res.facts),
              ("product_level", "Product-level facts (less specific than the offering/edition"
               " filter)", res.product_level),
              ("product_wide", "Product-wide facts (no specific workload)", res.product_wide),
              ("unresolved_target", "Facts about a workload the taxonomy does not list (its name"
               " as the documentation wrote it)", res.unresolved_target)]
    parts: list[str] = []
    structured: dict[str, Any] = {}
    n = start
    for key, title, facts in groups:
        shown = facts[:min(limit, WIDE_SHOWN) if key in ("product_wide", "unresolved_target")
                      else limit]
        total = max(res.totals.get(key, 0), len(facts))
        structured[key] = [dataclasses.asdict(f) for f in shown]
        structured[f"{key}_total"] = total
        if not facts:
            continue
        head = f"{title}: {total}" + (f" (first {len(shown)} shown)" if total > len(shown)
                                       else "")
        lines = [head]
        for f in shown:
            n += 1
            lines.append(format_fact(n, f))
        parts.append("\n".join(lines))
    structured["conflicts"] = [{"conflict_group": c.conflict_group, "size": c.size,
                                "claim_ids": [f.claim_id for f in c.facts]}
                               for c in res.conflicts]
    if res.conflicts:
        parts.append(f"{len(res.conflicts)} conflict group(s): the same vendor, product, "
                     "dimension and target with different support or values; report both "
                     "sides with their sources (the newer document is marked current).")
    structured["hint"] = res.hint
    if res.hint:
        parts.append(f"No facts. {res.hint}")
    return "\n\n".join(parts) or "No facts.", structured


def format_topic(topic: Topic) -> tuple[str, dict[str, Any]]:
    """A topic and its aspects, each with its dimension ids, as text and structured content."""
    lines = [f"Topic {topic.id} — {topic.name}:"]
    lines += [f"- {a.name}: {', '.join(a.dimensions)}" for a in topic.aspects]
    return "\n".join(lines), {"id": topic.id, "name": topic.name, "aspects": [
        {"id": a.id, "name": a.name, "dimensions": list(a.dimensions)} for a in topic.aspects]}


def format_topic_capabilities(topic: Topic, results: list[CapResult],
                              limit: int) -> tuple[str, dict[str, Any]]:
    """A topic's facts grouped by aspect: ``results`` holds one result per aspect, each cut at
    ``limit`` rows per group. An aspect with no facts keeps its heading."""
    parts = [f"Topic {topic.id} — {topic.name}: facts grouped by aspect, up to {limit} per "
             f"group in each aspect."]
    aspects: list[dict[str, Any]] = []
    n = 0
    for aspect, res in zip(topic.aspects, results, strict=True):
        text, structured = format_capabilities(res, limit, start=n)
        n += sum(len(structured[k]) for k in _FACT_GROUPS)
        parts.append(f"## {aspect.name}\n{text}")
        aspects.append({"id": aspect.id, "name": aspect.name,
                        "dimensions": list(aspect.dimensions), **structured})
    return "\n\n".join(parts), {"topic": topic.id, "name": topic.name, "aspect_limit": limit,
                                 "aspects": aspects}


def _passage(n: int, *, vendor: str, product: str | None, title: str | None,
             section_path: list[str], url: str | None, chunk_id: str, text: str,
             whole: bool = False) -> str:
    """One numbered passage of a text result, cut at ``PASSAGE_CHARS`` unless ``whole``."""
    if not whole and len(text) > PASSAGE_CHARS:
        text = text[:PASSAGE_CHARS] + f" […] (read_section {chunk_id} for the rest)"
    section = " › ".join(section_path)
    return (f"[{n}] {vendor} · {product or 'unknown product'} — {title}"
            f"{f' › {section}' if section else ''}\n    source: {url or 'no url'}"
            f" (chunk {chunk_id})\n{text}")


def format_hits(hits: list[Hit]) -> tuple[str, list[dict[str, Any]]]:
    parts, rows = [], []
    for n, h in enumerate(hits, 1):
        parts.append(_passage(n, vendor=h.vendor, product=h.product, title=h.title,
                              section_path=h.section_path, url=h.url, chunk_id=h.chunk_id,
                              text=h.text))
        rows.append({"chunk_id": h.chunk_id, "doc_id": h.doc_id, "vendor": h.vendor,
                     "product": h.product, "title": h.title, "section_path": h.section_path,
                     "url": h.url, "rerank_score": h.rerank_score, "text": h.text})
    return "\n\n".join(parts), rows


# The last lines of a gather_evidence reply: the answer model's rules, for the agent that
# writes in its place (BACKLOG 42).
EVIDENCE_RULES = (
    "Write the answer from these passages only, and cite each statement with the [N] of the "
    "passage that states it. State for a vendor only what its own passages show. A vendor or "
    "aspect with no passage is not documented in what was retrieved: say so, never that the "
    "product lacks it. Keep vendor and product names as given.")
NO_EVIDENCE = ("No passage was found. Say that the documentation vendorkb covers does not "
               "answer this; do not answer from general knowledge.")


def _read_as(out: dict[str, Any]) -> str:
    """How vendorkb read the question, for the first lines of a gather_evidence reply."""
    scope = out.get("scope") or {}
    sides = ", ".join([*(scope.get("vendors") or []), *(scope.get("products") or [])])
    topic = out.get("topic")
    by = f" on {topic}, by aspect ({'; '.join(out.get('aspects') or [])})" if topic else ""
    if out.get("shape") == "enumeration":
        checked = len(out.get("vendors_checked") or [])
        if out.get("table") is not None:
            return f"a question about every vendor on {topic} ({checked} checked): the facts found"
        return f"a question about every vendor ({checked} checked)"
    if out.get("shape") == "comparison":
        return f"a comparison of {sides}{by}"
    if sides:
        return f"a question about {sides}{by}"
    return f"a question naming no vendor or product{by}"


def _evidence_group(out: dict[str, Any], passage: dict[str, Any]) -> str | None:
    """The heading a passage goes under: its aspect for a topic, its vendor for a "which
    vendors" question or a comparison, none for a single question."""
    if out.get("topic"):
        return passage.get("aspect") or "General"
    if out.get("shape") in ("enumeration", "comparison"):
        return str(passage.get("vendor"))
    return None


def format_evidence(out: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    """A gather_evidence reply as parts of (text, the passages shown in it). Passages keep the
    numbers ``answer`` would give them, each passage whole (the answer model reads it whole: a
    chunk is at most about 1,200 tokens; cut at 1,200 characters, the agent wrote 30% fewer
    claims, 2026-11-10), grouped under headings (a group's passages together,
    groups in order of first appearance; for a topic, the kit's order after "General"), packed
    whole into parts of at most ``PART_CHARS`` characters (headings and the lines around them
    come on top; a longer passage is a part by itself). Every part starts with
    how the question was read; the last one ends with what was not found and the rules. A
    grid's table is one part with its sources, its cells' passages left to read_section."""
    passages = out.get("passages") or []
    head = f"Evidence for: {out.get('query')}\nRead as: {_read_as(out)}."
    not_found = out.get("not_found") or []
    if out.get("table") is not None:
        sources = format_sources({"citations": passages, "topic": out.get("topic"),
                                  "shape": "enumeration"})
        body = out["table"] + (f"\n\n{sources}" if sources else "")
        return [(f"{head} Part 1 of 1.\n\n{body}\n\n{EVIDENCE_RULES}",
                 passages[:GRID_CITATIONS_SHOWN])]
    if not passages:
        return [(f"{head} Part 1 of 1.\n\n{NO_EVIDENCE}", [])]
    order: dict[str | None, int] = {"General": -1} if out.get("topic") else {}
    for i, name in enumerate(out.get("aspects") or []):
        order.setdefault(name, i)
    for p in passages:
        order.setdefault(_evidence_group(out, p), len(order))
    ranked = sorted(passages, key=lambda p: order[_evidence_group(out, p)])  # stable
    shown = {p["marker"]: _passage(
        p["marker"], vendor=p["vendor"], product=p.get("product"), title=p.get("title"),
        section_path=p.get("section_path") or [], url=p.get("url"), chunk_id=p["chunk_id"],
        text=p["text"], whole=True) for p in ranked}
    parts: list[list[dict[str, Any]]] = [[]]
    size = 0
    for p in ranked:
        chars = len(shown[p["marker"]])
        if parts[-1] and size + chars > PART_CHARS:
            parts.append([])
            size = 0
        parts[-1].append(p)
        size += chars
    tail = []
    if not_found:
        tail.append(f"No supporting passage found for: {', '.join(not_found)}.")
    if out.get("topic"):
        covered = {p.get("aspect") for p in passages}
        if missing := [a for a in out.get("aspects") or [] if a not in covered]:
            tail.append(f"No passage was retrieved for: {', '.join(missing)}.")
    tail.append(EVIDENCE_RULES)
    texts = []
    for i, part in enumerate(parts, start=1):
        blocks = [f"{head} Part {i} of {len(parts)}."]
        group: str | None = None
        for p in part:
            if (name := _evidence_group(out, p)) != group and name is not None:
                blocks.append(f"## {name}")  # repeated where a group continues in a new part
            group = name
            blocks.append(shown[p["marker"]])
        blocks += tail if i == len(parts) else [
            f"Call gather_evidence again with part={i + 1} (same question) before you answer."]
        texts.append(("\n\n".join(blocks), part))
    return texts


def format_sources(out: dict[str, Any]) -> str:
    """The Sources section of an answer: ``[n] title — url (chunk id)`` per citation; for a grid
    answer (``topic`` set, shape enumeration), each page once with all its markers and its first
    cited chunk, ``[3, 17] url (chunk id)``, cut at ``GRID_SOURCES_CHARS`` with a pointer to the
    topic's facts. The chunk id is in the text so an agent can ``read_document`` a cited page.
    Empty with no citation."""
    citations = out.get("citations") or []
    if not citations:
        return ""
    if not (out.get("topic") and out.get("shape") == "enumeration"):
        return "Sources:\n" + "\n".join(
            f"[{c['marker']}] {c.get('title') or 'untitled'} — {c.get('url') or 'no url'}"
            f" (chunk {c.get('chunk_id')})" for c in citations)
    pages: dict[str, list[dict[str, Any]]] = {}
    for c in citations:
        pages.setdefault(c.get("url") or "no url", []).append(c)
    lines = [f"[{', '.join(str(c['marker']) for c in cs)}] {url} (chunk {cs[0].get('chunk_id')})"
             for url, cs in pages.items()]
    text = "Sources:\n" + "\n".join(lines)
    shown = len(lines)
    while len(text) > GRID_SOURCES_CHARS and shown > 0:
        shown -= 1
        text = "\n".join(["Sources:", *lines[:shown],
                          f"… {len(lines) - shown} more sources: query_capabilities("
                          f"topic=\"{out['topic']}\") lists the cited facts."])
    return text


def build_server(backend: Backend, label: str = "main") -> MCPServer:
    """The tools over ``backend``; ``label`` names the MCP key in the question logs."""
    server = MCPServer("vendorkb", instructions=INSTRUCTIONS)

    async def _topic_facts(term: str, base: CapFilter, limit: int) -> tuple[str, dict[str, Any]]:
        """The facts of the topic ``term`` names (an id, a name or a trigger): one query per
        aspect on one connection, each with ``base``'s filters on the aspect's dimensions and
        its share of ``limit``. A filter that names no taxonomy entity fails every aspect the
        same way, so its hint is returned once."""
        try:
            known = await backend.topics()
        except Exception as e:
            raise _fail("query_capabilities", e) from e
        if not (found := match_topics(term, known)):
            listing = ", ".join(f"{t.id} ({t.name})" for t in known) or "none defined"
            raise ToolError(f"unknown topic {term!r}; known topics: {listing}")
        topic = found[0]
        share = max(1, limit // max(1, len(topic.aspects)))
        try:
            results = await backend.capabilities_many(
                [dataclasses.replace(base, dimension_ids=a.dimensions) for a in topic.aspects],
                share)
        except Exception as e:
            raise _fail("query_capabilities", e) from e
        if failed := next((r for r in results if r.unresolved), None):
            text, structured = format_capabilities(failed, share)
            return text, {**structured, "topic": topic.id, "name": topic.name, "aspects": []}
        return format_topic_capabilities(topic, results, share)

    @server.tool(name="list_vendors", annotations=_RO, description=(
        "List the vendors and products that have documentation, with document counts and how "
        "many capability facts each vendor has. Use it for valid vendor/product filter names."))
    async def list_vendors() -> CallToolResult:
        try:
            vendors = await backend.vendors()
        except Exception as e:
            raise _fail("list_vendors", e) from e
        lines = [f"{v['vendor']} — {v['documents']} documents, {v['capability_facts']} "
                 "capability facts; products: " + ", ".join(
                     f"{p['product'] or 'unknown'} ({p['documents']})" for p in v["products"])
                 for v in vendors]
        return _result("\n".join(lines) or "No documentation.", {"vendors": vendors})

    @server.tool(name="list_dimensions", annotations=_RO, description=(
        "Browse the taxonomy's dimensions (comparable capability aspects such as immutability, "
        "retention or backup coverage). `query` matches an id, an alias or part of a name; "
        "`family` narrows to one family (coverage, retention, security, ...). A query that "
        "names a topic (a broad subject such as ransomware) also shows it with its aspects. "
        "Use the ids with query_capabilities(dimension=...)."))
    async def list_dimensions(query: str | None = None, family: str | None = None,
                              limit: int = 25) -> CallToolResult:
        try:
            rows = await backend.dimensions(query, family, max(1, min(limit, MAX_LIMIT)))
            found = match_topics(query, await backend.topics()) if query and query.strip() else []
        except Exception as e:
            raise _fail("list_dimensions", e) from e
        lines = [f"{d['id']} — {d['name']}: {d['definition']}"
                 + (f" (aliases: {', '.join(d['aliases'][:8])})" if d["aliases"] else "")
                 for d in rows]
        if not found:
            return _result("\n".join(lines) or "No dimension matches.", {"dimensions": rows})
        shown = [format_topic(t) for t in found]
        blocks = [text for text, _ in shown] + (["\n".join(lines)] if lines else [])
        return _result("\n\n".join(blocks),
                       {"dimensions": rows, "topics": [data for _, data in shown]})

    @server.tool(name="list_targets", annotations=_RO, description=(
        "Browse the taxonomy's targets (workloads, platforms, services and storage a backup "
        "product protects, such as Amazon RDS or Microsoft 365). `query` matches an id, an "
        "alias or part of a name. Use the ids with query_capabilities(target=...)."))
    async def list_targets(query: str | None = None, limit: int = 25) -> CallToolResult:
        try:
            rows = await backend.targets(query, max(1, min(limit, MAX_LIMIT)))
        except Exception as e:
            raise _fail("list_targets", e) from e
        lines = [f"{t['id']} — {t['name']} [class {t['class']}]"
                 + (f" (aliases: {', '.join(t['aliases'][:8])})" if t["aliases"] else "")
                 for t in rows]
        return _result("\n".join(lines) or "No target matches.", {"targets": rows})

    @server.tool(name="query_capabilities", annotations=_RO, description=(
        "Query the cross-vendor capability table: which vendors document a capability for a "
        "workload, with support (supported / partial / not_supported / unclear), conditions, "
        "values and an evidence quote and source for every fact. Filters take a taxonomy id or "
        "alias (dimension, target, offering, edition) or exact values (vendor, support, "
        "availability, value range); `vendor` takes one name or a list. Give at least one "
        "filter. `topic` (a broad subject such as ransomware protection; list_dimensions shows "
        "them) covers all its dimensions at once, with the facts grouped by aspect and `limit` "
        "shared among the aspects; it is exclusive with `dimension`. A vendor missing here is "
        "not evidence of absence: check it with search_docs."))
    async def query_capabilities(
            dimension: str | None = None, topic: str | None = None, target: str | None = None,
            vendor: str | list[str] | None = None,
            family: str | None = None, offering: str | None = None, edition: str | None = None,
            object_type: str | None = None,
            support: Literal["supported", "partial", "not_supported", "unclear"] | None = None,
            availability: str | None = None, value_min: float | None = None,
            value_max: float | None = None, limit: int = 20) -> CallToolResult:
        topic = (topic or "").strip() or None
        if topic is not None and dimension is not None:
            raise ToolError("query_capabilities takes a topic or a dimension, not both: a topic "
                            "stands for all the dimensions of its aspects")
        note = None
        if (section := _section(target)) is not None:
            note = (f"Note: {target!r} is a section of the results, not a taxonomy target, so "
                    f"the target filter was ignored; {section} facts are listed in their own "
                    "section.")
            target = None
        names = [vendor] if isinstance(vendor, str) else vendor or []
        vendors = tuple(v.strip() for v in names if v.strip()) or None
        f = CapFilter(vendor=vendors, offering=offering, edition=edition, dimension=dimension,
                      family=family, target=target, object_type=object_type, support=support,
                      availability=availability, value_min=value_min, value_max=value_max)
        limit = max(1, min(limit, MAX_LIMIT))
        if topic is not None:
            text, structured = await _topic_facts(topic, f, limit)
        else:
            if all(v is None for v in dataclasses.asdict(f).values()):
                raise ToolError((f"{note} " if note else "")
                                + "query_capabilities needs at least one filter (e.g. dimension "
                                "and target, or topic; see list_dimensions and list_targets)")
            try:
                res = await backend.capabilities(f, limit)
            except Exception as e:
                raise _fail("query_capabilities", e) from e
            text, structured = format_capabilities(res, limit)
        if note:
            text, structured = f"{note}\n{text}", {**structured, "note": note}
        return _result(text, structured)

    @server.tool(name="search_docs", annotations=_RO, description=(
        "Search the vendors' documentation (hybrid keyword and semantic search, reranked). "
        "Returns passages with their source URL and chunk id. `vendor` / `product` restrict "
        "the search (names from list_vendors). Use it for details the capability table does "
        "not hold, and to check vendors it does not list."))
    async def search_docs(query: str, vendor: list[str] | None = None,
                          product: list[str] | None = None, k: int = 5) -> CallToolResult:
        if not query.strip():
            raise ToolError("search_docs needs a query")
        try:
            res = await backend.search(query, vendor or None, product or None,
                                       max(1, min(k, MAX_K)))
        except Exception as e:
            raise _fail("search_docs", e) from e
        text, rows = format_hits(res.hits)
        return _result(text or "No passages found.", {"hits": rows, "reranked": res.rerank_ok})

    @server.tool(name="read_section", annotations=_RO, description=(
        "Read a passage returned by search_docs in context: `expand` \"0\" is the chunk alone, "
        "\"1\" adds the chunks before and after it, \"full\" returns its whole section."))
    async def read_section(chunk_id: str,
                           expand: Literal["0", "1", "full"] = "1") -> CallToolResult:
        try:
            hits = await backend.section(chunk_id, expand)
        except Exception as e:
            raise _fail("read_section", e) from e
        if not hits:
            raise ToolError(f"no chunk {chunk_id!r} (it may have been replaced by a newer sync;"
                            " search again)")
        text = _page_head(hits[0]) + "\n\n" + "\n\n".join(x.text for x in hits)
        chunks = [{**row, "superseded": x.superseded}
                  for row, x in zip(format_hits(hits)[1], hits, strict=True)]
        return _result(text, {"chunks": chunks})

    @server.tool(name="read_document", annotations=_RO, description=(
        "The full text of the page a passage comes from (any chunk_id from search_docs, "
        "read_section or answer_question's sources), in parts: start at part 1; the reply "
        "says how many parts there are. Use it when the user asks for the source document."))
    async def read_document(chunk_id: str, part: int = 1) -> CallToolResult:
        try:
            hits = await backend.section(chunk_id, "page")
        except Exception as e:
            raise _fail("read_document", e) from e
        if not hits:
            raise ToolError(f"no chunk {chunk_id!r} (it may have been replaced by a newer sync;"
                            " search again)")
        parts = _parts(hits)
        if not 1 <= part <= len(parts):
            raise ToolError(f"part {part} does not exist: this page has parts 1-{len(parts)}")
        chosen = parts[part - 1]
        text = (_page_head(hits[0]) + f"\npart {part} of {len(parts)}\n\n"
                + "\n\n".join(x.text for x in chosen))
        return _result(text, {"chunk_ids": [x.chunk_id for x in chosen], "part": part,
                              "parts": len(parts), "url": hits[0].url})

    @server.tool(name="answer_question", annotations=_RO, description=(
        "Answer a question from the documentation in one call: vendorkb retrieves, checks every "
        "vendor for 'which vendors' questions, compares named vendors, and writes an answer "
        "that cites its sources as [N] (the Sources list follows; always show it). It refuses "
        "when the documentation does not support an answer. Pass the user's question word for "
        "word. Slower than search_docs."))
    async def answer_question(
            question: Annotated[str, Field(description=VERBATIM)]) -> CallToolResult:
        if not question.strip():
            raise ToolError("answer_question needs a question")
        started = time.monotonic()
        try:
            out = await backend.answer(question)
        except Exception as e:
            raise _fail("answer_question", e) from e
        # the question as the agent sent it (agents rewrite the user's), on one line, and how
        # long the answer took (the public route cuts a request at 100 s)
        log.info("answer_question %s", json.dumps(
            {"key": label, "question": question, "shape": out.get("shape"),
             "topic": out.get("topic"), "retrieved": out.get("retrieved"), "citations": len(out.get("citations") or []),
             "seconds": round(time.monotonic() - started, 1)}, ensure_ascii=False))
        if out.get("error"):
            raise ToolError(f"answer_question failed: {out['error']}")
        sources = format_sources(out)
        text = (out.get("answer") or "No answer.") + (f"\n\n{sources}" if sources else "")
        structured = {k: out.get(k) for k in ("answer", "shape", "scope", "uncited")}
        citations = out.get("citations") or []
        grid = bool(out.get("topic")) and out.get("shape") == "enumeration"
        structured["citations"] = [{k: c.get(k) for k in ("marker", "chunk_id", "url", "title",
                                                          "section_path")}
                                   for c in (citations[:GRID_CITATIONS_SHOWN] if grid
                                             else citations)]
        if grid:
            structured["citations_total"] = len(citations)
        return _result(text, structured)

    @server.tool(name="gather_evidence", annotations=_RO, description=(
        "The evidence for a question, without a written answer: vendorkb reads the question as "
        "answer_question does (every vendor checked for 'which vendors' questions, each side "
        "of a comparison, each aspect of a broad subject) and returns the numbered passages it "
        "found, grouped by vendor or aspect, with what it found nothing for. Write the answer "
        "yourself from those passages only, citing each statement as [N]. Pass the user's "
        "question word for word. A long reply comes in parts: read every part first."))
    async def gather_evidence(question: Annotated[str, Field(description=VERBATIM)],
                              part: int = 1) -> CallToolResult:
        if not question.strip():
            raise ToolError("gather_evidence needs a question")
        started = time.monotonic()
        try:
            out = await backend.evidence(question)
        except Exception as e:
            raise _fail("gather_evidence", e) from e
        log.info("gather_evidence %s", json.dumps(
            {"key": label, "question": question, "shape": out.get("shape"),
             "topic": out.get("topic"), "retrieved": out.get("retrieved"), "part": part,
             "seconds": round(time.monotonic() - started, 1)}, ensure_ascii=False))
        if out.get("error"):
            raise ToolError(f"gather_evidence failed: {out['error']}")
        parts = format_evidence(out)
        if not 1 <= part <= len(parts):
            raise ToolError(f"part {part} does not exist: this evidence has parts 1-{len(parts)}")
        text, shown = parts[part - 1]
        structured = {k: out.get(k) for k in ("shape", "topic", "aspects", "scope", "retrieved",
                                              "not_found", "rerank_ok")}
        structured.update(part=part, parts=len(parts), passages=[
            {k: p.get(k) for k in ("marker", "chunk_id", "vendor", "product", "title", "url",
                                   "aspect")} for p in shown])
        return _result(text, structured)

    return server
