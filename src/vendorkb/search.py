from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import psycopg
from pgvector import Vector

from vendorkb.clients import Embedder, Reranker

FTS_LIMIT = 50
VEC_LIMIT = 50
RRF_K = 60
RERANK_CANDIDATES = 40

_COLS = ("c.chunk_id::text, c.doc_id::text, c.text, c.section_path, d.vendor, d.product, "
         "d.title, d.url")


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    section_path: list[str]
    vendor: str
    product: str | None
    title: str
    url: str | None
    rrf_rank: int
    rerank_score: float | None


@dataclass
class SearchResult:
    hits: list[Hit]
    rerank_ok: bool


def rrf(rankings: list[list[str]], *, k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion; ties broken by first appearance."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos)
    first = {item: n for n, item in enumerate(scores)}
    return sorted(scores, key=lambda i: (-scores[i], first[i]))


def _hit(row: tuple, rrf_rank: int = 0, rerank_score: float | None = None) -> Hit:
    return Hit(chunk_id=row[0], doc_id=row[1], text=row[2], section_path=list(row[3]),
               vendor=row[4], product=row[5], title=row[6] if row[6] is not None else "",
               url=row[7], rrf_rank=rrf_rank, rerank_score=rerank_score)


def _filters(vendor: list[str] | None, product: list[str] | None) -> tuple[str, list]:
    """Scope filter. Vendor and product are ORed, as answer-api scopes (a document matches
    when its vendor is named OR its product is named); a single filter applies alone."""
    if vendor and product:
        return (" AND (d.vendor = ANY(%s) OR d.product = ANY(%s))",
                [list(vendor), list(product)])
    if vendor:
        return " AND d.vendor = ANY(%s)", [list(vendor)]
    if product:
        return " AND d.product = ANY(%s)", [list(product)]
    return "", []


# websearch_to_tsquery ANDs every term, so a natural-language question matches almost
# nothing. Its terms are ORed instead (ts_rank_cd still ranks chunks matching more terms
# higher). A stopword-only question gives an empty tsquery: NULLIF turns it into NULL,
# which matches no chunk instead of raising.
_TSQUERY = ("NULLIF(replace(websearch_to_tsquery('english', %s)::text, ' & ', ' | '), '')"
            "::tsquery")


async def _vector_rows(conn: psycopg.AsyncConnection, sql: str, params: list) -> list[tuple]:
    """Vector query with HNSW iterative scan, so a WHERE filter cannot starve the result."""
    cur = await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = await cur.fetchone()
    ver = tuple(int(x) for x in (row[0] if row else "0").split(".")[:2])
    async with conn.transaction():
        if ver >= (0, 8):
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        else:
            await conn.execute("SET LOCAL hnsw.ef_search = 1000")
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


async def search_docs(conn: psycopg.AsyncConnection, query: str, *, embedder: Embedder,
                      reranker: Reranker | None, vendor: list[str] | None = None,
                      product: list[str] | None = None, k: int = 10,
                      use_rerank: bool = True) -> SearchResult:
    # An embedder failure propagates: the caller records it.
    qvec = (await embedder.embed([query]))[0]
    where, fparams = _filters(vendor, product)
    fts_sql = (f"SELECT {_COLS} FROM chunks c JOIN documents d USING (doc_id), "
               f"(SELECT {_TSQUERY} AS tsq) q "
               "WHERE c.tsv @@ q.tsq" + where +
               " ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.chunk_id LIMIT %s")
    # Inner ORDER BY distance alone keeps the HNSW index usable (adding a tiebreak column
    # there forces an exact scan); the outer sort makes the order of equal distances stable.
    vec_sql = (f"SELECT * FROM (SELECT {_COLS}, c.embedding <=> %s AS dist "
               f"FROM chunks c JOIN documents d USING (doc_id) WHERE true{where} "
               "ORDER BY c.embedding <=> %s LIMIT %s) t ORDER BY dist, chunk_id")
    cur = await conn.execute(fts_sql, [query, *fparams, FTS_LIMIT])
    fts_rows = await cur.fetchall()
    vec_rows = await _vector_rows(conn, vec_sql, [Vector(qvec), *fparams, Vector(qvec), VEC_LIMIT])

    rows = {r[0]: r for r in [*fts_rows, *vec_rows]}
    order = rrf([[r[0] for r in fts_rows], [r[0] for r in vec_rows]])
    rrf_pos = {cid: n for n, cid in enumerate(order, start=1)}
    candidates = order[:RERANK_CANDIDATES]

    if use_rerank and reranker is not None and candidates:
        ranked = await reranker.rerank(query, [rows[c][2] for c in candidates], k)
        if ranked:  # None (outage) and [] for non-empty candidates are both failures
            hits = [_hit(rows[candidates[i]], rrf_pos[candidates[i]], score)
                    for i, score in ranked[:k]]
            return SearchResult(hits=hits, rerank_ok=True)
    return SearchResult(hits=[_hit(rows[c], rrf_pos[c]) for c in candidates[:k]],
                        rerank_ok=False)


async def read_section(conn: psycopg.AsyncConnection, chunk_id: str,
                       expand: Literal["0", "1", "full"] = "0") -> list[Hit]:
    cur = await conn.execute(
        "SELECT doc_id, ordinal, section_path FROM chunks WHERE chunk_id = %s", (chunk_id,))
    anchor = await cur.fetchone()
    if anchor is None:
        return []
    doc_id, ordinal, path = anchor
    base = f"SELECT {_COLS} FROM chunks c JOIN documents d USING (doc_id) WHERE c.doc_id = %s"
    if expand == "0":
        sql, params = base + " AND c.chunk_id = %s", [doc_id, chunk_id]
    elif expand == "1":
        sql, params = (base + " AND c.ordinal BETWEEN %s AND %s",
                       [doc_id, ordinal - 1, ordinal + 1])
    else:
        sql, params = base + " AND c.section_path = %s", [doc_id, list(path)]
    cur = await conn.execute(sql + " ORDER BY c.ordinal", params)
    return [_hit(r) for r in await cur.fetchall()]
