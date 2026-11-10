from __future__ import annotations

import logging
import weakref
from collections.abc import Sequence
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
FTS_WEIGHT = 0.25  # 2026-11-06: single recall@10 0.833 -> 0.933, multi 0.737 -> 0.895, pool recall kept
# Rows of the one corpus-wide vector query an enumeration makes over all its vendors; a vendor
# with fewer than its share there gets its own scoped query (vendor_candidates).
ENUM_VEC_POOL = 2000
# The live column, and the candidate column `vendorkb reembed` fills for an embedding A/B.
VECTOR_COLUMNS = ("embedding", "embedding_next")
# Unextracted chunks a grid's catch-up ranks at most, the most recently fetched pages first: an
# extraction backlog must not make the one catch-up statement sort without bound.
UNEXTRACTED_POOL = 2000

log = logging.getLogger(__name__)

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
    # an earlier version of its page (only ``read_section`` and the topic grid return one)
    superseded: bool = False


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


# A superseded chunk (an earlier version of a page, kept for its claims: store.write_document)
# is never retrieved. Every chunk query that ranks for relevance carries this condition.
_CURRENT = " AND c.superseded_at IS NULL"


def _filters(vendor: list[str] | None, product: list[str] | None) -> tuple[str, list]:
    """Chunk filter: current chunks only, within the scope. Vendor and product are ORed, as
    answer-api scopes (a document matches when its vendor is named OR its product is named); a
    single filter applies alone."""
    if vendor and product:
        return (_CURRENT + " AND (d.vendor = ANY(%s) OR d.product = ANY(%s))",
                [list(vendor), list(product)])
    if vendor:
        return _CURRENT + " AND d.vendor = ANY(%s)", [list(vendor)]
    if product:
        return _CURRENT + " AND d.product = ANY(%s)", [list(product)]
    return _CURRENT, []


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
    return (await _vector_rows_many(conn, [(sql, params)]))[0]


async def _vector_rows_many(conn: psycopg.AsyncConnection,
                            queries: list[tuple[str, list]]) -> list[list[tuple]]:
    """Several vector queries in one transaction under one HNSW iterative-scan setting."""
    ver = await _pgvector_version(conn)
    out = []
    async with conn.transaction():
        if ver >= (0, 8):
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        else:
            await conn.execute("SET LOCAL hnsw.ef_search = 1000")
        for sql, params in queries:
            cur = await conn.execute(sql, params)
            out.append(await cur.fetchall())
    return out


def _vec_sql(vector_column: str, where: str) -> str:
    # ``where`` must be ``_filters`` output: it carries the current-chunk condition.
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


async def facet_candidates(conn: psycopg.AsyncConnection, query: str, qvec: list[float],
                           facets: list[tuple[str, str]], *, limit: int,
                           vector_column: str = "embedding") -> list[list[Hit]]:
    """``hybrid_candidates(vendor=[name])`` (kind "vendor") or ``(product=[name])`` (kind
    "product") for every facet, in facet order ([] for a facet with no candidates).

    ONE full-text query ranked per facet (the OR'd full-text query is a seq scan, so it runs
    once, not once per facet), then one transaction holding one SET LOCAL and a scoped
    vector query per facet (an HNSW scan each, cheap next to the full-text scan)."""
    if vector_column not in VECTOR_COLUMNS:
        raise ValueError(f"unknown vector column {vector_column!r}")
    if not facets:
        return []
    kinds = [k for k, _ in facets]
    names = [n for _, n in facets]
    fts_sql = (f"SELECT * FROM (SELECT f.idx, {_COLS}, row_number() OVER (PARTITION BY f.idx "
               "ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.chunk_id) AS rn "
               f"FROM chunks c JOIN documents d USING (doc_id), (SELECT {_TSQUERY} AS tsq) q, "
               "unnest(%s::text[], %s::text[]) WITH ORDINALITY AS f(kind, name, idx) "
               f"WHERE c.tsv @@ q.tsq{_CURRENT} AND ((f.kind = 'vendor' AND d.vendor = f.name) "
               "OR (f.kind = 'product' AND d.product = f.name))) t "
               "WHERE rn <= %s ORDER BY idx, rn")
    cur = await conn.execute(fts_sql, [query, kinds, names, FTS_LIMIT])
    fts: list[list[tuple]] = [[] for _ in facets]
    for r in await cur.fetchall():
        fts[r[0] - 1].append(r[1:-1])
    queries = []
    for kind, name in facets:
        where, fparams = _filters([name] if kind == "vendor" else None,
                                  [name] if kind == "product" else None)
        queries.append((_vec_sql(vector_column, where),
                        [Vector(qvec), *fparams, Vector(qvec), VEC_LIMIT]))
    vecs = await _vector_rows_many(conn, queries)
    out = []
    for fts_rows, vec_rows in zip(fts, vecs):
        rows = {r[0]: r for r in [*fts_rows, *vec_rows]}
        order = rrf([[r[0] for r in fts_rows], [r[0] for r in vec_rows]],
                    weights=[FTS_WEIGHT, 1.0])
        out.append([_hit(rows[c], n) for n, c in enumerate(order[:limit], start=1)])
    return out


async def vector_candidates(conn: psycopg.AsyncConnection, qvecs: Sequence[list[float]],
                            sides: Sequence[tuple[str, str] | None], *, limit: int,
                            vector_column: str = "embedding") -> list[list[list[Hit]]]:
    """The ``limit`` current chunks nearest each query vector within each side, as ``out[i][j]``
    (vector ``i``, side ``j``), nearest first with ``rrf_rank`` the position (no rerank score).
    A side is a (kind, name) pair, kind "vendor" or "product", or None for the whole corpus.

    No full-text search: that is a seq scan whose cost grows with the corpus and with every
    query text (about 2 s each on the full corpus), where a scoped HNSW scan is cheap. One
    transaction holds one SET LOCAL and a scoped vector query per pair."""
    if vector_column not in VECTOR_COLUMNS:
        raise ValueError(f"unknown vector column {vector_column!r}")
    queries = []
    for qvec in qvecs:
        for side in sides:
            where, fparams = _filters([side[1]] if side and side[0] == "vendor" else None,
                                      [side[1]] if side and side[0] == "product" else None)
            queries.append((_vec_sql(vector_column, where),
                            [Vector(qvec), *fparams, Vector(qvec), limit]))
    rows = iter(await _vector_rows_many(conn, queries) if queries else [])
    return [[[_hit(r, n) for n, r in enumerate(next(rows), start=1)] for _ in sides]
            for _ in qvecs]


async def unextracted_candidates(conn: psycopg.AsyncConnection, queries: list[str],
                                 qvecs: list[list[float]], *, limit: int) -> list[list[Hit]]:
    """For each query, the ``limit`` best current chunks among those with no final extraction
    of the claims-v3 family ('done' or 'failed', by any model): pages synced since their last
    extraction, whose facts the capability table does not hold yet. In RRF order (``rrf_rank``
    set, no rerank score), ``out[i]`` for ``queries[i]`` and ``qvecs[i]``.

    ONE statement for every query. The unextracted chunks are materialised once, at most
    ``UNEXTRACTED_POOL`` of them, those of the most recently fetched pages (the anti-join is
    the main cost: 1.2 s for 622 chunks of the full corpus; the pool size is logged at DEBUG),
    and each query is ranked within that small set by vector distance (an exact sort: an HNSW
    scan filtered to so few rows starves) and by full-text rank (each tsquery parsed once, not
    per chunk), top ``VEC_LIMIT`` and ``FTS_LIMIT``; a corpus-wide full-text scan per query
    would cost ~2 s each. The fusion is ``hybrid_candidates``', in Python."""
    if not queries:
        return []
    # Imported here, not at the top: this module must stay free of the claims stack
    # (anthropic), as store.retire_superseded's import (tests/unit/test_vendorkb_cli_imports.py).
    from vendorkb.claims.extract import PROMPT_FAMILY
    asked = " UNION ALL ".join(
        f"SELECT {n} AS idx, %s::vector AS qvec, {_TSQUERY} AS tsq"
        for n in range(1, len(queries) + 1))
    sql = ("WITH pool AS MATERIALIZED (SELECT c.chunk_id, c.tsv, c.embedding FROM chunks c"
           " JOIN documents d ON d.doc_id = c.doc_id"
           f" WHERE true{_CURRENT} AND NOT EXISTS (SELECT 1 FROM extractions e"
           " WHERE e.chunk_id = c.chunk_id AND e.status IN ('done', 'failed')"
           " AND e.prompt_version LIKE %s) ORDER BY d.fetched_at DESC, c.chunk_id LIMIT %s),"
           f" qs AS MATERIALIZED ({asked}),"
           " ranked AS (SELECT 'v' AS kind, q.idx, p.chunk_id, row_number() OVER (PARTITION BY"
           " q.idx ORDER BY p.embedding <=> q.qvec, p.chunk_id::text) AS rn FROM qs q, pool p"
           " UNION ALL SELECT 'f', q.idx, p.chunk_id, row_number() OVER (PARTITION BY q.idx"
           " ORDER BY ts_rank_cd(p.tsv, q.tsq) DESC, p.chunk_id::text) FROM qs q, pool p"
           " WHERE p.tsv @@ q.tsq)"
           f" SELECT {_COLS}, jsonb_agg(jsonb_build_array(r.kind, r.idx, r.rn)),"
           " (SELECT count(*) FROM pool) FROM ranked r"
           " JOIN chunks c ON c.chunk_id = r.chunk_id JOIN documents d ON d.doc_id = c.doc_id"
           " WHERE (r.kind = 'v' AND r.rn <= %s) OR (r.kind = 'f' AND r.rn <= %s)"
           " GROUP BY c.chunk_id, d.doc_id")
    params: list = [PROMPT_FAMILY, UNEXTRACTED_POOL]
    for query, qvec in zip(queries, qvecs, strict=True):
        params += [Vector(qvec), query]
    cur = await conn.execute(sql, [*params, VEC_LIMIT, FTS_LIMIT])
    rows: dict[str, tuple] = {}
    ranks: dict[str, list[list[tuple[int, str]]]] = {"v": [[] for _ in queries],
                                                     "f": [[] for _ in queries]}
    pool = 0  # every pool chunk has a vector rank, so no row means an empty pool
    for *cols, found, pool in await cur.fetchall():
        rows[cols[0]] = tuple(cols)
        for kind, idx, rn in found:
            ranks[kind][idx - 1].append((rn, cols[0]))
    log.debug("vendorkb catch-up: a pool of %d unextracted chunks", pool)
    out = []
    for fts, vec in zip(ranks["f"], ranks["v"]):
        order = rrf([[c for _, c in sorted(fts)], [c for _, c in sorted(vec)]],
                    weights=[FTS_WEIGHT, 1.0])
        out.append([_hit(rows[c], n) for n, c in enumerate(order[:limit], start=1)])
    return out


# The enumeration's vector query: ids only, same shape as _vec_sql (text chunk_id as the
# distance tiebreak), but an EXACT order: the "+ 0" keeps the planner off the HNSW index. The
# pool asks for ENUM_VEC_POOL (2,000) neighbours, which the index reaches only by iterating
# (5-11 s, approximate; the exact sort over the corpus takes about 2 s, in parallel), and a
# small vendor's own query sorts just its chunks. The enumeration floor and depth were
# calibrated on the exact order (2026-11-10).
ENUM_VEC_SQL = ("SELECT chunk_id, vendor FROM (SELECT c.chunk_id::text AS chunk_id, d.vendor, "
                "c.embedding <=> %s AS dist FROM chunks c JOIN documents d USING (doc_id) "
                f"WHERE d.vendor = ANY(%s){_CURRENT} ORDER BY (c.embedding <=> %s) + 0 LIMIT %s) t "
                "ORDER BY dist, chunk_id")


async def vendor_candidates(conn: psycopg.AsyncConnection, query: str, qvec: list[float],
                            vendors: list[str], *, limit: int) -> dict[str, list[Hit]]:
    """``hybrid_candidates(vendor=[v], limit=limit)`` for every vendor at once, keyed by vendor
    (vendors with no candidates are absent): per vendor, RRF over its top ``FTS_LIMIT``
    full-text and top ``VEC_LIMIT`` vector rows, cut to ``limit``.

    The statement count does not grow with the vendor count: ONE full-text query ranked once
    and cut per vendor, ONE vector query over all the vendors (``ENUM_VEC_POOL`` rows, an exact
    sort: ``ENUM_VEC_SQL``), a
    scoped vector query only for each vendor with fewer than ``limit`` rows in that pool (a
    small vendor, so a cheap scan), then one fetch of the fused chunks' columns. The ranking
    queries carry ids only, so the 2000-row pool does not ship chunk text. Per-vendor scoped
    searches cost ~4 s each on the full corpus, the OR'd full-text query being a parallel seq
    scan every time.

    A vendor with ``limit`` to ``VEC_LIMIT - 1`` pooled rows fuses only those pooled rows,
    where the scoped search would read up to ``VEC_LIMIT``: its vector list can be shorter
    than the scoped one, a deliberate trade that keeps big vendors on the pool."""
    vendors = list(dict.fromkeys(vendors))
    if not vendors:
        return {}
    fts_sql = ("SELECT chunk_id::text, vendor FROM (SELECT c.chunk_id, d.vendor, "
               "row_number() OVER (PARTITION BY d.vendor "
               "ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.chunk_id) AS rn "
               f"FROM chunks c JOIN documents d USING (doc_id), (SELECT {_TSQUERY} AS tsq) q "
               f"WHERE c.tsv @@ q.tsq{_CURRENT} AND d.vendor = ANY(%s)) t "
               "WHERE rn <= %s ORDER BY vendor, rn")
    cur = await conn.execute(fts_sql, [query, vendors, FTS_LIMIT])
    fts: dict[str, list[str]] = {}
    for cid, v in await cur.fetchall():
        fts.setdefault(v, []).append(cid)

    vec: dict[str, list[str]] = {v: [] for v in vendors}
    for cid, v in await _vector_rows(conn, ENUM_VEC_SQL,
                                     [Vector(qvec), vendors, Vector(qvec), ENUM_VEC_POOL]):
        if len(vec[v]) < VEC_LIMIT:
            vec[v].append(cid)
    for v in vendors:
        if len(vec[v]) < limit:  # short in the pool: its own scoped query replaces its rows
            vec[v] = [cid for cid, _ in await _vector_rows(
                conn, ENUM_VEC_SQL, [Vector(qvec), [v], Vector(qvec), VEC_LIMIT])]

    orders = {v: rrf([fts.get(v, []), vec[v]], weights=[FTS_WEIGHT, 1.0])[:limit]
              for v in vendors}
    ids = [cid for order in orders.values() for cid in order]
    if not ids:
        return {}
    cur = await conn.execute(f"SELECT {_COLS} FROM chunks c JOIN documents d USING (doc_id) "
                             "WHERE c.chunk_id = ANY(%s::uuid[])", [ids])
    rows = {r[0]: r for r in await cur.fetchall()}
    return {v: [_hit(rows[c], n) for n, c in enumerate(order, start=1) if c in rows]
            for v, order in orders.items() if order}


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
                       expand: Literal["0", "1", "full", "page"] = "0") -> list[Hit]:
    """The chunk, its neighbours, its section or (``"page"``) the whole page, all of the anchor's
    own version of the page:
    the current chunks around a current anchor; around a superseded one, the chunks superseded
    in the same write (one ``write_document`` call sets one ``superseded_at``), never those of
    another earlier version; those hits carry ``superseded=True``."""
    cur = await conn.execute(
        "SELECT doc_id, ordinal, section_path, superseded_at FROM chunks WHERE chunk_id = %s",
        (chunk_id,))
    anchor = await cur.fetchone()
    if anchor is None:
        return []
    doc_id, ordinal, path, superseded_at = anchor
    # Two spellings, not one IS NOT DISTINCT FROM: the planner can prove neither partial index
    # (superseded_at IS NULL / IS NOT NULL) from that, and would scan the whole table.
    base = f"SELECT {_COLS} FROM chunks c JOIN documents d USING (doc_id) WHERE c.doc_id = %s"
    args: list = [doc_id]
    if superseded_at is None:
        base += " AND c.superseded_at IS NULL"
    else:
        base += " AND c.superseded_at = %s"
        args.append(superseded_at)
    if expand == "0":
        sql, params = base + " AND c.chunk_id = %s", [*args, chunk_id]
    elif expand == "1":
        sql, params = base + " AND c.ordinal BETWEEN %s AND %s", [*args, ordinal - 1, ordinal + 1]
    elif expand == "full":
        sql, params = base + " AND c.section_path = %s", [*args, list(path)]
    else:  # "page": every chunk of the anchor's version
        sql, params = base, args
    cur = await conn.execute(sql + " ORDER BY c.ordinal", params)
    hits = [_hit(r) for r in await cur.fetchall()]
    # every chunk returned is of the anchor's version: the text may be gone from the page
    return hits if superseded_at is None else [replace(h, superseded=True) for h in hits]
