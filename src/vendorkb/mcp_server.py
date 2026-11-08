"""The MCP server (DESIGN.md §6): read-only tools over vendorkb for a downstream agent.

Every tool returns text for the model and the same result as structured content. Every fact and
passage carries its source URL and chunk id from the database; nothing here writes a URL.

- ``list_vendors``: vendors and products with documentation, document and fact counts.
- ``list_dimensions`` / ``list_targets``: browse the taxonomy; resolve the user's wording to ids.
- ``query_capabilities``: the cross-vendor fact table (support, conditions, values), each fact
  with its evidence quote and source.
- ``search_docs``: hybrid search over the documentation chunks, reranked.
- ``read_section``: a chunk with its neighbours or its whole section.
- ``answer_question``: vendorkb's cited answer (retrieval plus synthesis) in one call.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from vendorkb.mcp_backend import Backend
from vendorkb.search import Hit
from vendorkb.taxonomy.capabilities import CapFilter, CapResult, Fact

log = logging.getLogger(__name__)

MAX_K = 25
MAX_LIMIT = 200
PASSAGE_CHARS = 2500  # per search hit in the text output (structured content has it whole)

INSTRUCTIONS = """\
vendorkb answers questions about backup vendors' technical documentation, precisely and with
citations. Rules:
- Every factual statement in your answer must cite a source these tools returned (its URL, or
  title and URL). If nothing supports a statement, say you don't know. Never invent a URL.
- For cross-vendor questions ("which vendors support X for Y?"), call query_capabilities
  first, then check the vendors it does not list with search_docs: a vendor missing from the
  table does not mean the vendor lacks the feature. Separate "supported", "not supported" and
  "no evidence found".
- Use list_dimensions / list_targets to turn the user's wording into taxonomy ids when
  query_capabilities reports that a filter matched nothing.
- For one vendor's details (limits, versions, prerequisites), search_docs with the vendor
  filter, then read_section for the surrounding text.
- answer_question returns a complete cited answer in one call when you do not need to combine
  sources yourself."""

_RO = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _result(text: str, structured: dict[str, Any]) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)],
                          structured_content=structured)


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
    target = f.target_id or "product-wide"
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
    if not f.is_current:
        when += "; a newer document says otherwise"
    lines = [f"[{n}] {f.vendor} · {f.product or 'unknown product'}"
             f"{f' ({where})' if where else ''} | {f.dimension_id} | {target} | "
             f"{f.support.upper()}"]
    if quals:
        lines.append("    " + "; ".join(quals))
    lines.append(f'    quote: "{" ".join(f.evidence_quote.split())}"')
    lines.append(f"    source: {f.title or 'untitled'} — {f.url or 'no url'}"
                 f" (chunk {f.chunk_id}; {when})")
    return "\n".join(lines)


def format_capabilities(res: CapResult, limit: int) -> tuple[str, dict[str, Any]]:
    groups = [("facts", "Facts", res.facts),
              ("product_level", "Product-level facts (less specific than the offering/edition"
               " filter)", res.product_level),
              ("product_wide", "Product-wide facts (no specific workload)", res.product_wide)]
    parts: list[str] = []
    structured: dict[str, Any] = {}
    n = 0
    for key, title, facts in groups:
        shown = facts[:limit]
        structured[key] = [dataclasses.asdict(f) for f in shown]
        structured[f"{key}_total"] = len(facts)
        if not facts:
            continue
        head = f"{title}: {len(facts)}" + (f" (first {limit} shown)" if len(facts) > limit
                                            else "")
        lines = [head]
        for f in shown:
            n += 1
            lines.append(format_fact(n, f))
        parts.append("\n".join(lines))
    structured["conflicts"] = [{"conflict_group": c.conflict_group,
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


def format_hits(hits: list[Hit]) -> tuple[str, list[dict[str, Any]]]:
    parts, rows = [], []
    for n, h in enumerate(hits, 1):
        text = h.text if len(h.text) <= PASSAGE_CHARS else h.text[:PASSAGE_CHARS] + " […]"
        section = " › ".join(h.section_path)
        parts.append(f"[{n}] {h.vendor} · {h.product or 'unknown product'} — {h.title}"
                     f"{f' › {section}' if section else ''}\n    source: {h.url or 'no url'}"
                     f" (chunk {h.chunk_id})\n{text}")
        rows.append({"chunk_id": h.chunk_id, "doc_id": h.doc_id, "vendor": h.vendor,
                     "product": h.product, "title": h.title, "section_path": h.section_path,
                     "url": h.url, "rerank_score": h.rerank_score, "text": h.text})
    return "\n\n".join(parts), rows


def build_server(backend: Backend) -> MCPServer:
    server = MCPServer("vendorkb", instructions=INSTRUCTIONS)

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
        "`family` narrows to one family (coverage, retention, security, ...). Use the ids with "
        "query_capabilities(dimension=...)."))
    async def list_dimensions(query: str | None = None, family: str | None = None,
                              limit: int = 25) -> CallToolResult:
        try:
            rows = await backend.dimensions(query, family, max(1, min(limit, MAX_LIMIT)))
        except Exception as e:
            raise _fail("list_dimensions", e) from e
        lines = [f"{d['id']} — {d['name']}: {d['definition']}"
                 + (f" (aliases: {', '.join(d['aliases'][:8])})" if d["aliases"] else "")
                 for d in rows]
        return _result("\n".join(lines) or "No dimension matches.", {"dimensions": rows})

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
        "availability, value range). Give at least one filter. A vendor missing here is not "
        "evidence of absence: check it with search_docs."))
    async def query_capabilities(
            dimension: str | None = None, target: str | None = None, vendor: str | None = None,
            family: str | None = None, offering: str | None = None, edition: str | None = None,
            object_type: str | None = None,
            support: Literal["supported", "partial", "not_supported", "unclear"] | None = None,
            availability: str | None = None, value_min: float | None = None,
            value_max: float | None = None, limit: int = 40) -> CallToolResult:
        f = CapFilter(vendor=vendor, offering=offering, edition=edition, dimension=dimension,
                      family=family, target=target, object_type=object_type, support=support,
                      availability=availability, value_min=value_min, value_max=value_max)
        if all(v is None for v in dataclasses.asdict(f).values()):
            raise ToolError("query_capabilities needs at least one filter (e.g. dimension and "
                            "target; see list_dimensions and list_targets)")
        try:
            res = await backend.capabilities(f)
        except Exception as e:
            raise _fail("query_capabilities", e) from e
        text, structured = format_capabilities(res, max(1, min(limit, MAX_LIMIT)))
        return _result(text, structured)

    @server.tool(name="search_docs", annotations=_RO, description=(
        "Search the vendors' documentation (hybrid keyword and semantic search, reranked). "
        "Returns passages with their source URL and chunk id. `vendor` / `product` restrict "
        "the search (names from list_vendors). Use it for details the capability table does "
        "not hold, and to check vendors it does not list."))
    async def search_docs(query: str, vendor: list[str] | None = None,
                          product: list[str] | None = None, k: int = 8) -> CallToolResult:
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
        h = hits[0]
        text = (f"{h.vendor} · {h.product or 'unknown product'} — {h.title}\n"
                f"source: {h.url or 'no url'}\n\n" + "\n\n".join(x.text for x in hits))
        return _result(text, {"chunks": format_hits(hits)[1]})

    @server.tool(name="answer_question", annotations=_RO, description=(
        "Answer a question from the documentation in one call: vendorkb retrieves, checks every "
        "vendor for 'which vendors' questions, compares named vendors, and writes an answer "
        "that cites its sources as [N] (the Sources list follows; always show it). It refuses "
        "when the documentation does not support an answer. Slower than search_docs."))
    async def answer_question(question: str) -> CallToolResult:
        if not question.strip():
            raise ToolError("answer_question needs a question")
        try:
            out = await backend.answer(question)
        except Exception as e:
            raise _fail("answer_question", e) from e
        if out.get("error"):
            raise ToolError(f"answer_question failed: {out['error']}")
        sources = [f"[{c['marker']}] {c.get('title') or 'untitled'} — {c.get('url') or 'no url'}"
                   for c in out.get("citations") or []]
        text = (out.get("answer") or "No answer.") + (
            "\n\nSources:\n" + "\n".join(sources) if sources else "")
        structured = {k: out.get(k) for k in ("answer", "shape", "scope", "uncited")}
        structured["citations"] = [{k: c.get(k) for k in ("marker", "chunk_id", "url", "title",
                                                          "section_path")}
                                   for c in out.get("citations") or []]
        return _result(text, structured)

    return server
