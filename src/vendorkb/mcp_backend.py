"""What the MCP tools read (``mcp_server``): the corpus, the taxonomy and the capabilities table.

``Backend`` is the interface the tools call; ``DbBackend`` implements it on the vendorkb
database with the same functions the CLI uses (``search_docs``, ``read_section``,
``capabilities.query``, ``answer``, ``load_topics``). Each call opens and closes its own
connection and HTTP clients, so one slow or failed call never affects another.
"""
from __future__ import annotations

import re
import time
from typing import Any, Literal, Protocol

from openai import AsyncOpenAI

from vendorkb.answer import answer, evidence
from vendorkb.clients import Embedder, Reranker
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect
from vendorkb.scope import scope_index_from_db
from vendorkb.search import Hit, SearchResult, read_section, search_docs
from vendorkb.taxonomy import capabilities
from vendorkb.taxonomy.capabilities import CapFilter, CapResult
from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import Topic
from vendorkb.topics import load_topics

Expand = Literal["0", "1", "full", "page"]
EVIDENCE_TTL = 600.0  # seconds a question's evidence is kept for its later parts
EVIDENCE_KEPT = 32  # questions kept at most


class Backend(Protocol):
    async def vendors(self) -> list[dict[str, Any]]: ...
    async def dimensions(self, query: str | None, family: str | None,
                         limit: int) -> list[dict[str, Any]]: ...
    async def targets(self, query: str | None, limit: int) -> list[dict[str, Any]]: ...
    async def topics(self) -> list[Topic]: ...
    async def capabilities(self, f: capabilities.CapFilter,
                           limit: int) -> capabilities.CapResult: ...
    async def capabilities_many(self, filters: list[CapFilter],
                                limit: int) -> list[CapResult]: ...
    async def search(self, query: str, vendor: list[str] | None, product: list[str] | None,
                     k: int) -> SearchResult: ...
    async def section(self, chunk_id: str, expand: Expand) -> list[Hit]: ...
    async def answer(self, question: str) -> dict[str, Any]: ...
    async def evidence(self, question: str) -> dict[str, Any]: ...


# A taxonomy entry matches a query when the query is its id or an exact (normalized) alias
# (rank 0), a whole word or phrase in its id, name or an alias (1), or the start of a word there
# (2). A match inside a word (3: "rman" in "performance") counts only when nothing ranks higher.
def _rank(id_col: str, name_col: str) -> str:
    def hit(op: str, param: str) -> str:
        return (f"{id_col} {op} %({param})s OR {name_col} {op} %({param})s"
                f" OR bool_or(a.alias {op} %({param})s)")
    return (f"CASE WHEN %(q)s::text IS NULL THEN 0"
            f" WHEN {id_col} = %(q)s OR bool_or(a.alias = %(norm)s) THEN 0"
            f" WHEN {hit('~*', 'word')} THEN 1 WHEN {hit('~*', 'start')} THEN 2"
            f" WHEN {hit('ILIKE', 'like')} THEN 3 END")


_BEST = """
WHERE rank < 3 OR (rank = 3 AND NOT EXISTS (SELECT 1 FROM m WHERE rank < 3))
ORDER BY rank, id
LIMIT %(limit)s
"""
_DIMENSIONS = f"""
WITH m AS (
SELECT d.dimension_id AS id, d.family, d.name, d.definition, d.applies_to, d.value_unit,
       coalesce(array_agg(a.alias ORDER BY a.alias) FILTER (WHERE a.alias IS NOT NULL), '{{}}')
           AS aliases,
       {_rank("d.dimension_id", "d.name")} AS rank
FROM dimensions d
LEFT JOIN aliases a ON a.entity_type = 'dimension' AND a.entity_id = d.dimension_id
WHERE (%(family)s::text IS NULL OR d.family = %(family)s)
GROUP BY d.dimension_id
)
SELECT id, family, name, definition, applies_to, value_unit, aliases FROM m
{_BEST}"""
_TARGETS = f"""
WITH m AS (
SELECT t.target_id AS id, t.name, t.class_id, t.parent_id,
       coalesce(array_agg(a.alias ORDER BY a.alias) FILTER (WHERE a.alias IS NOT NULL), '{{}}')
           AS aliases,
       {_rank("t.target_id", "t.name")} AS rank
FROM targets t
LEFT JOIN aliases a ON a.entity_type = 'target' AND a.entity_id = t.target_id
GROUP BY t.target_id
)
SELECT id, name, class_id, parent_id, aliases FROM m
{_BEST}"""
_REGEX_SPECIAL = re.compile(r"([\\.^$*+?()\[\]{}|])")


def _query_params(query: str | None, limit: int) -> dict[str, Any]:
    q = query.strip() if query and query.strip() else None
    norm = normalize(q) if q else None
    literal = _REGEX_SPECIAL.sub(r"\\\1", norm) if norm else None
    return {"q": q, "norm": norm, "like": f"%{q}%" if q else None,
            "word": rf"\m{literal}\M" if literal else None,
            "start": rf"\m{literal}" if literal else None, "limit": limit}


class DbBackend:
    def __init__(self, settings: VendorkbSettings) -> None:
        self.settings = settings
        # gather_evidence is read in parts, one call each: the evidence of a question is kept
        # for EVIDENCE_TTL seconds so the parts are of one retrieval (and it is paid for once)
        self._evidence: dict[str, tuple[float, dict[str, Any]]] = {}

    async def vendors(self) -> list[dict[str, Any]]:
        conn = await connect(self.settings.vendorkb_dsn)
        try:
            cur = await conn.execute(
                "SELECT d.vendor, d.product, count(*) FROM documents d"
                " GROUP BY d.vendor, d.product ORDER BY d.vendor, d.product")
            docs = await cur.fetchall()
            cur = await conn.execute("SELECT vendor, count(*) FROM capabilities GROUP BY vendor")
            facts = {v: n for v, n in await cur.fetchall()}
        finally:
            await conn.close()
        out: dict[str, dict[str, Any]] = {}
        for vendor, product, n in docs:
            v = out.setdefault(vendor, {"vendor": vendor, "documents": 0,
                                        "capability_facts": facts.get(vendor, 0),
                                        "products": []})
            v["documents"] += n
            v["products"].append({"product": product, "documents": n})
        return list(out.values())

    async def dimensions(self, query: str | None, family: str | None,
                         limit: int) -> list[dict[str, Any]]:
        conn = await connect(self.settings.vendorkb_dsn)
        try:
            cur = await conn.execute(_DIMENSIONS, {**_query_params(query, limit),
                                                   "family": family})
            rows = await cur.fetchall()
        finally:
            await conn.close()
        return [{"id": r[0], "family": r[1], "name": r[2], "definition": " ".join(r[3].split()),
                 "applies_to": r[4], "value_unit": r[5], "aliases": list(r[6])} for r in rows]

    async def targets(self, query: str | None, limit: int) -> list[dict[str, Any]]:
        conn = await connect(self.settings.vendorkb_dsn)
        try:
            cur = await conn.execute(_TARGETS, _query_params(query, limit))
            rows = await cur.fetchall()
        finally:
            await conn.close()
        return [{"id": r[0], "name": r[1], "class": r[2], "parent": r[3], "aliases": list(r[4])}
                for r in rows]

    async def topics(self) -> list[Topic]:
        """Every topic of the loaded taxonomy."""
        conn = await connect(self.settings.vendorkb_dsn)
        try:
            return await load_topics(conn)
        finally:
            await conn.close()

    async def capabilities(self, f: capabilities.CapFilter,
                           limit: int) -> capabilities.CapResult:
        conn = await connect(self.settings.vendorkb_dsn)
        try:  # bounded in SQL: a broad filter matches hundreds of thousands of rows
            return await capabilities.query(conn, f, limit=limit)
        finally:
            await conn.close()

    async def capabilities_many(self, filters: list[CapFilter],
                                limit: int) -> list[CapResult]:
        """``capabilities`` for each filter, in order, on one connection. It stops after a
        result that reports unresolved filters, the rest being the same query on other
        dimensions: the list is then shorter than ``filters``."""
        conn = await connect(self.settings.vendorkb_dsn)
        try:
            results: list[CapResult] = []
            for f in filters:
                results.append(await capabilities.query(conn, f, limit=limit))
                if results[-1].unresolved:
                    break
            return results
        finally:
            await conn.close()

    async def search(self, query: str, vendor: list[str] | None, product: list[str] | None,
                     k: int) -> SearchResult:
        conn = await connect(self.settings.vendorkb_dsn)
        try:
            async with Embedder(self.settings) as embedder, Reranker(self.settings) as reranker:
                return await search_docs(conn, query, embedder=embedder, reranker=reranker,
                                         vendor=vendor, product=product, k=k)
        finally:
            await conn.close()

    async def section(self, chunk_id: str, expand: Expand) -> list[Hit]:
        conn = await connect(self.settings.vendorkb_dsn)
        try:
            return await read_section(conn, chunk_id, expand)
        finally:
            await conn.close()

    async def evidence(self, question: str) -> dict[str, Any]:
        now = time.monotonic()
        for key in [k for k, (t, _) in self._evidence.items() if now - t > EVIDENCE_TTL]:
            del self._evidence[key]
        if question in self._evidence:
            return self._evidence[question][1]
        conn = await connect(self.settings.vendorkb_dsn)
        try:
            scope_index = await scope_index_from_db(conn)
            async with Embedder(self.settings) as embedder, Reranker(self.settings) as reranker:
                out = await evidence(conn, question, settings=self.settings, embedder=embedder,
                                     reranker=reranker, scope_index=scope_index)
        finally:
            await conn.close()
        if not out.get("error"):
            while len(self._evidence) >= EVIDENCE_KEPT:
                del self._evidence[next(iter(self._evidence))]  # the oldest
            self._evidence[question] = (now, out)
        return out

    async def answer(self, question: str) -> dict[str, Any]:
        conn = await connect(self.settings.vendorkb_dsn)
        client = AsyncOpenAI(base_url=self.settings.judge_base_url,
                             api_key=self.settings.judge_api_key)
        try:
            scope_index = await scope_index_from_db(conn)
            async with Embedder(self.settings) as embedder, Reranker(self.settings) as reranker:
                return await answer(conn, question, settings=self.settings, embedder=embedder,
                                    reranker=reranker, scope_index=scope_index, client=client)
        finally:
            await client.close()
            await conn.close()


class HiddenVendorError(RuntimeError):
    """A hidden vendor's row reached a guarded backend: the database role did not hide it."""


class GuardedBackend:
    """``inner`` for a key that must never see ``hidden`` vendors (case-insensitive). The
    database role is the guarantee (row-level security, ``access``); this is the second line:
    lists drop a hidden vendor's rows, and a fact or evidence passage of one fails the call
    rather than show it."""

    def __init__(self, inner: Backend, hidden: list[str]) -> None:
        self.inner, self.hidden = inner, {v.lower() for v in hidden}

    def _shown(self, vendor: str | None) -> bool:
        return (vendor or "").lower() not in self.hidden

    def _check(self, vendors: list[str | None], where: str) -> None:
        if not all(map(self._shown, vendors)):
            raise HiddenVendorError(f"a hidden vendor's row reached {where}")

    async def vendors(self) -> list[dict[str, Any]]:
        return [v for v in await self.inner.vendors() if self._shown(v["vendor"])]

    async def dimensions(self, query: str | None, family: str | None,
                         limit: int) -> list[dict[str, Any]]:
        return await self.inner.dimensions(query, family, limit)

    async def targets(self, query: str | None, limit: int) -> list[dict[str, Any]]:
        return await self.inner.targets(query, limit)

    async def topics(self) -> list[Topic]:
        return await self.inner.topics()

    def _check_facts(self, res: CapResult) -> CapResult:
        facts = [*res.facts, *res.product_level, *res.product_wide, *res.unresolved_target,
                 *(f for c in res.conflicts for f in c.facts)]
        self._check([f.vendor for f in facts], "query_capabilities")
        return res

    async def capabilities(self, f: CapFilter, limit: int) -> CapResult:
        return self._check_facts(await self.inner.capabilities(f, limit))

    async def capabilities_many(self, filters: list[CapFilter],
                                limit: int) -> list[CapResult]:
        return [self._check_facts(r) for r in await self.inner.capabilities_many(filters, limit)]

    async def search(self, query: str, vendor: list[str] | None, product: list[str] | None,
                     k: int) -> SearchResult:
        res = await self.inner.search(query, vendor, product, k)
        return SearchResult([h for h in res.hits if self._shown(h.vendor)], res.rerank_ok)

    async def section(self, chunk_id: str, expand: Expand) -> list[Hit]:
        return [h for h in await self.inner.section(chunk_id, expand) if self._shown(h.vendor)]

    async def answer(self, question: str) -> dict[str, Any]:
        return await self.inner.answer(question)

    async def evidence(self, question: str) -> dict[str, Any]:
        out = await self.inner.evidence(question)
        self._check([p.get("vendor") for p in out.get("passages") or []], "gather_evidence")
        return out
