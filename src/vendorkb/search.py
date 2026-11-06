from __future__ import annotations

import weakref
from dataclasses import dataclass, replace
from typing import Literal

import psycopg
from pgvector import Vector

from vendorkb.clients import Embedder, Reranker

FTS_LIMIT = 50
VEC_LIMIT = 50
RRF_K = 60
RERANK_CANDIDATES = 40
# RRF weight of the full-text ranking (the vector ranking weighs 1.0). Read at call time, so
# an eval can set ``search.FTS_WEIGHT`` to try down-weighting keyword search.
FTS_WEIGHT = 1.0
# Rows of the one corpus-wide vector query an enumeration makes over all its vendors; a vendor
# with fewer than its share there gets its own scoped query (vendor_candidates).
ENUM_VEC_POOL = 2000
# The live column, and the candidate column `vendorkb reembed` fills for an embedding A/B.
VECTOR_COLUMNS = ("embedding", "embedding_next")

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


def rrf(rankings: list[list[str]], *, k: int = RRF_K,
        weights: list[float] | None = None) -> list[str]:
    """Reciprocal rank fusion, each ranking's contribution multiplied by its weight (default
    1.0 each); ties broken by first appearance."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(weights)} weights for {len(rankings)} rankings")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for pos, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + pos)
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


_PGVECTOR_VERSION: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


async def _pgvector_version(conn: psycopg.AsyncConnection) -> tuple[int, ...]:
    """The pgvector version, read once per connection."""
    ver = _PGVECTOR_VERSION.get(conn)
    if ver is None:
        cur = await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = await cur.fetchone()
        ver = tuple(int(x) for x in (row[0] if row else "0").split(".")[:2])
        _PGVECTOR_VERSION[conn] = ver
    return ver


async def _vector_rows(conn: psycopg.AsyncConnection, sql: str, params: list) -> list[tuple]:
    """Vector query with HNSW iterative scan, so a WHERE filter cannot starve the result."""
    ver = await _pgvector_version(conn)
    async with conn.transaction():
        if ver >= (0, 8):
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        else:
            await conn.execute("SET LOCAL hnsw.ef_search = 1000")
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


def _vec_sql(vector_column: str, where: str) -> str:
    # Inner ORDER BY distance alone keeps the HNSW index usable (adding a tiebreak column
    # there forces an exact scan); the outer sort makes the order of equal distances stable.
    return (f"SELECT * FROM (SELECT {_COLS}, c.{vector_column} <=> %s AS dist "
            f"FROM chunks c JOIN documents d USING (doc_id) WHERE true{where} "
            f"ORDER BY c.{vector_column} <=> %s LIMIT %s) t ORDER BY dist, chunk_id")


async def hybrid_candidates(conn: psycopg.AsyncConnection, query: str, qvec: list[float], *,
                            vendor: list[str] | None = None,
                            product: list[str] | None = None,
                            limit: int = RERANK_CANDIDATES,
                            vector_column: str = "embedding") -> list[Hit]:
    """Full-text and vector candidates fused by RRF, in RRF order (``rrf_rank`` set, no
    rerank score). ``vector_column`` names the chunks column to search (an embedding A/B
    reads a candidate column); only names in ``VECTOR_COLUMNS`` are accepted."""
    if vector_column not in VECTOR_COLUMNS:
        raise ValueError(f"unknown vector column {vector_column!r}")
    where, fparams = _filters(vendor, product)
    fts_sql = (f"SELECT {_COLS} FROM chunks c JOIN documents d USING (doc_id), "
               f"(SELECT {_TSQUERY} AS tsq) q "
               "WHERE c.tsv @@ q.tsq" + where +
               " ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.chunk_id LIMIT %s")
    vec_sql = _vec_sql(vector_column, where)
    cur = await conn.execute(fts_sql, [query, *fparams, FTS_LIMIT])
    fts_rows = await cur.fetchall()
    vec_rows = await _vector_rows(conn, vec_sql, [Vector(qvec), *fparams, Vector(qvec), VEC_LIMIT])

    rows = {r[0]: r for r in [*fts_rows, *vec_rows]}
    order = rrf([[r[0] for r in fts_rows], [r[0] for r in vec_rows]],
                weights=[FTS_WEIGHT, 1.0])
    return [_hit(rows[c], n) for n, c in enumerate(order[:limit], start=1)]


async def vendor_candidates(conn: psycopg.AsyncConnection, query: str, qvec: list[float],
                            vendors: list[str], *, limit: int) -> dict[str, list[Hit]]:
    """``hybrid_candidates`` for every vendor at once, keyed by vendor (vendors with no
    candidates are absent), each list in its own RRF order cut to ``limit``.

    The statement count does not grow with the vendor count: ONE full-text query ranked once
    and cut per vendor, ONE vector query over all the vendors (``ENUM_VEC_POOL`` rows), then a
    scoped vector query only for each vendor short of ``limit`` rows in that pool (a small
    vendor, so a cheap scan). Per-vendor scoped searches cost ~4 s each on the full corpus,
    the OR'd full-text query being a parallel seq scan every time."""
    vendors = list(dict.fromkeys(vendors))
    if not vendors:
        return {}
    fts_sql = (f"SELECT * FROM (SELECT {_COLS}, row_number() OVER (PARTITION BY d.vendor "
               "ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.chunk_id) AS rn "
               f"FROM chunks c JOIN documents d USING (doc_id), (SELECT {_TSQUERY} AS tsq) q "
               "WHERE c.tsv @@ q.tsq AND d.vendor = ANY(%s)) t "
               "WHERE rn <= %s ORDER BY vendor, rn")
    cur = await conn.execute(fts_sql, [query, list(vendors), limit])
    fts: dict[str, list[tuple]] = {}
    for r in await cur.fetchall():
        fts.setdefault(r[4], []).append(r)

    vec: dict[str, list[tuple]] = {v: [] for v in vendors}
    where = " AND d.vendor = ANY(%s)"
    for r in await _vector_rows(conn, _vec_sql("embedding", where),
                                [Vector(qvec), list(vendors), Vector(qvec), ENUM_VEC_POOL]):
        if len(vec[r[4]]) < limit:
            vec[r[4]].append(r)
    for v in vendors:
        if len(vec[v]) < limit:
            vec[v] = await _vector_rows(conn, _vec_sql("embedding", where),
                                        [Vector(qvec), [v], Vector(qvec), limit])

    out: dict[str, list[Hit]] = {}
    for v in vendors:
        rows = {r[0]: r for r in [*fts.get(v, []), *vec[v]]}
        order = rrf([[r[0] for r in fts.get(v, [])], [r[0] for r in vec[v]]],
                    weights=[FTS_WEIGHT, 1.0])
        if order:
            out[v] = [_hit(rows[c], n) for n, c in enumerate(order[:limit], start=1)]
    return out


async def search_docs(conn: psycopg.AsyncConnection, query: str, *, embedder: Embedder,
                      reranker: Reranker | None, vendor: list[str] | None = None,
                      product: list[str] | None = None, k: int = 10,
                      use_rerank: bool = True,
                      vector_column: str = "embedding") -> SearchResult:
    """``vector_column`` names the chunks column to search (an embedding A/B reads a
    candidate column); only names in ``VECTOR_COLUMNS`` are accepted."""
    if vector_column not in VECTOR_COLUMNS:
        raise ValueError(f"unknown vector column {vector_column!r}")
    # An embedder failure propagates: the caller records it.
    qvec = (await embedder.embed([query], input_type="query"))[0]
    candidates = await hybrid_candidates(conn, query, qvec, vendor=vendor, product=product,
                                         vector_column=vector_column)

    if use_rerank and reranker is not None and candidates:
        ranked = await reranker.rerank(query, [h.text for h in candidates], k)
        if ranked:  # None (outage) and [] for non-empty candidates are both failures
            hits = [replace(candidates[i], rerank_score=score) for i, score in ranked[:k]]
            return SearchResult(hits=hits, rerank_ok=True)
    return SearchResult(hits=candidates[:k], rerank_ok=False)


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
