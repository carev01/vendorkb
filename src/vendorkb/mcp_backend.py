"""What the MCP tools read (``mcp_server``): the corpus, the taxonomy and the capabilities table.

``Backend`` is the interface the tools call; ``DbBackend`` implements it on the vendorkb
database with the same functions the CLI uses (``search_docs``, ``read_section``,
``capabilities.query``, ``answer``). Each call opens and closes its own connection and HTTP
clients, so one slow or failed call never affects another.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol

from openai import AsyncOpenAI

from vendorkb.answer import answer
from vendorkb.clients import Embedder, Reranker
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect
from vendorkb.scope import scope_index_from_db
from vendorkb.search import Hit, SearchResult, read_section, search_docs
from vendorkb.taxonomy import capabilities
from vendorkb.taxonomy.cluster import normalize

Expand = Literal["0", "1", "full"]


class Backend(Protocol):
    async def vendors(self) -> list[dict[str, Any]]: ...
    async def dimensions(self, query: str | None, family: str | None,
                         limit: int) -> list[dict[str, Any]]: ...
    async def targets(self, query: str | None, limit: int) -> list[dict[str, Any]]: ...
    async def capabilities(self, f: capabilities.CapFilter,
                           limit: int) -> capabilities.CapResult: ...
    async def search(self, query: str, vendor: list[str] | None, product: list[str] | None,
                     k: int) -> SearchResult: ...
    async def section(self, chunk_id: str, expand: Expand) -> list[Hit]: ...
    async def answer(self, question: str) -> dict[str, Any]: ...


# A taxonomy entry matches a query when the query is its id, an exact (normalized) alias, or a
# substring of its id, name or an alias; exact matches sort first.
_DIMENSIONS = """
SELECT d.dimension_id, d.family, d.name, d.definition, d.applies_to, d.value_unit,
       coalesce(array_agg(a.alias ORDER BY a.alias) FILTER (WHERE a.alias IS NOT NULL), '{}')
FROM dimensions d
LEFT JOIN aliases a ON a.entity_type = 'dimension' AND a.entity_id = d.dimension_id
WHERE (%(family)s::text IS NULL OR d.family = %(family)s)
GROUP BY d.dimension_id
HAVING %(q)s::text IS NULL OR d.dimension_id = %(q)s OR bool_or(a.alias = %(norm)s)
    OR d.dimension_id ILIKE %(like)s OR d.name ILIKE %(like)s OR bool_or(a.alias ILIKE %(like)s)
ORDER BY (d.dimension_id = %(q)s OR coalesce(bool_or(a.alias = %(norm)s), false)) DESC,
         d.dimension_id
LIMIT %(limit)s
"""
_TARGETS = """
SELECT t.target_id, t.name, t.class_id, t.parent_id,
       coalesce(array_agg(a.alias ORDER BY a.alias) FILTER (WHERE a.alias IS NOT NULL), '{}')
FROM targets t
LEFT JOIN aliases a ON a.entity_type = 'target' AND a.entity_id = t.target_id
GROUP BY t.target_id
HAVING %(q)s::text IS NULL OR t.target_id = %(q)s OR bool_or(a.alias = %(norm)s)
    OR t.target_id ILIKE %(like)s OR t.name ILIKE %(like)s OR bool_or(a.alias ILIKE %(like)s)
ORDER BY (t.target_id = %(q)s OR coalesce(bool_or(a.alias = %(norm)s), false)) DESC,
         t.target_id
LIMIT %(limit)s
"""


def _query_params(query: str | None, limit: int) -> dict[str, Any]:
    q = query.strip() if query and query.strip() else None
    return {"q": q, "norm": normalize(q) if q else None,
            "like": f"%{q}%" if q else None, "limit": limit}


class DbBackend:
    def __init__(self, settings: VendorkbSettings) -> None:
        self.settings = settings

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

    async def capabilities(self, f: capabilities.CapFilter,
                           limit: int) -> capabilities.CapResult:
        conn = await connect(self.settings.vendorkb_dsn)
        try:  # bounded in SQL: a broad filter matches hundreds of thousands of rows
            return await capabilities.query(conn, f, limit=limit)
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
