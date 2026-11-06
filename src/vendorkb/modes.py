"""Cross-vendor retrieval: per-vendor enumeration and per-facet comparison, both on top of
``search.hybrid_candidates`` with one query embedding reused across the fan-out."""

from __future__ import annotations

from dataclasses import dataclass, replace

import psycopg

from vendorkb.clients import Embedder, Reranker
from vendorkb.search import RERANK_CANDIDATES, Hit, hybrid_candidates
from vendorkb.shape import Facet

# Placeholder until the eval calibrates it (plan Task 6): 0.0 keeps every vendor whose
# evidence the reranker scored.
ENUMERATION_MIN_SCORE = 0.0


@dataclass
class EnumerationResult:
    hits: list[Hit]
    kept: list[str]  # vendors whose evidence goes to the answer
    dropped: list[str]  # vendors whose best score fell below the floor
    empty: list[str]  # vendors with no candidates
    rerank_ok: bool


@dataclass
class ComparisonResult:
    hits: list[Hit]
    facets_empty: list[str]
    rerank_ok: bool


def _pick(candidates: list[Hit], per_vendor: int, per_vendor_multi: int) -> list[Hit]:
    """The best chunk per product first (RRF order), then the next best overall. A vendor
    with several products among its candidates keeps up to ``per_vendor_multi``."""
    products = list(dict.fromkeys(h.product for h in candidates))
    cap = per_vendor_multi if len(products) > 1 else per_vendor
    first = [next(h.chunk_id for h in candidates if h.product == p) for p in products][:cap]
    rest = [h.chunk_id for h in candidates if h.chunk_id not in first]
    chosen = set((first + rest)[:cap])
    return [h for h in candidates if h.chunk_id in chosen]


def _truncate(groups: list[list[Hit]], max_hits: int) -> list[list[Hit]]:
    """Drop chunks off the ends of the groups (each best-first) until ``max_hits`` remain:
    always from the largest group, the lowest-ranked one on a tie, so no vendor loses its
    last chunk while another still has more than one."""
    groups = [list(g) for g in groups]
    while sum(map(len, groups)) > max_hits:
        size = max(map(len, groups))
        if size <= 1:
            break
        victim = max(i for i, g in enumerate(groups) if len(g) == size)
        groups[victim].pop()
    return groups


async def enumerate_vendors(conn: psycopg.AsyncConnection, query: str, *, embedder: Embedder,
                            reranker: Reranker | None, vendors: list[str],
                            per_vendor: int = 3, per_vendor_multi: int = 5,
                            min_score: float = ENUMERATION_MIN_SCORE, max_hits: int = 60,
                            use_rerank: bool = True) -> EnumerationResult:
    """Search every vendor separately, rerank all their chunks in one call and keep the
    vendors whose best score reaches ``min_score``. On a reranker failure (or with
    ``use_rerank=False``, the free eval mode) every vendor with candidates is kept, in the
    order given, its chunks in RRF order, and ``rerank_ok`` is False."""
    # An embedder failure propagates: the caller records it.
    qvec = (await embedder.embed([query], input_type="query"))[0]
    picked: dict[str, list[Hit]] = {}
    empty: list[str] = []
    # Serial on the one connection; spec section 9.5 asks to measure before pooling.
    for v in vendors:
        cands = await hybrid_candidates(conn, query, qvec, vendor=[v],
                                        limit=per_vendor_multi * 2)
        if cands:
            picked[v] = _pick(cands, per_vendor, per_vendor_multi)
        else:
            empty.append(v)

    flat = [h for hs in picked.values() for h in hs]
    ranked = None
    if use_rerank and reranker is not None and flat:
        ranked = await reranker.rerank(query, [h.text for h in flat], len(flat))
    if not ranked:  # None (outage) and [] for non-empty input are both failures
        kept = list(picked)
        groups = _truncate([picked[v] for v in kept], max_hits)
        return EnumerationResult(hits=[h for g in groups for h in g], kept=kept, dropped=[],
                                 empty=empty, rerank_ok=False)

    scored: dict[str, list[Hit]] = {v: [] for v in picked}
    for i, score in ranked:
        hit = replace(flat[i], rerank_score=score)
        scored[hit.vendor].append(hit)
    best = {v: max((h.rerank_score for h in hs if h.rerank_score is not None),
                   default=float("-inf")) for v, hs in scored.items()}
    kept = sorted((v for v in picked if best[v] >= min_score), key=lambda v: -best[v])
    dropped = [v for v in picked if v not in kept]
    groups = _truncate([sorted(scored[v], key=lambda h: -(h.rerank_score or 0.0))
                        for v in kept], max_hits)
    return EnumerationResult(hits=[h for g in groups for h in g], kept=kept, dropped=dropped,
                             empty=empty, rerank_ok=True)


async def compare_facets(conn: psycopg.AsyncConnection, query: str, *, embedder: Embedder,
                         reranker: Reranker | None, facets: list[Facet], per_facet: int = 5,
                         use_rerank: bool = True) -> ComparisonResult:
    """Search each facet with an equal share of the candidate pool and rerank each facet
    separately, so every facet's quota survives the rerank. Hits interleave facet by facet;
    a facet with no candidates is reported in ``facets_empty``, never relaxed."""
    if not facets:
        return ComparisonResult(hits=[], facets_empty=[], rerank_ok=False)
    # An embedder failure propagates: the caller records it.
    qvec = (await embedder.embed([query], input_type="query"))[0]
    limit = max(1, RERANK_CANDIDATES // len(facets))
    per: list[list[Hit]] = []
    facets_empty: list[str] = []
    rerank_ok = use_rerank and reranker is not None
    for kind, name in facets:
        cands = await hybrid_candidates(conn, query, qvec, limit=limit,
                                        vendor=[name] if kind == "vendor" else None,
                                        product=[name] if kind == "product" else None)
        if not cands:
            facets_empty.append(name)
            continue
        ranked = None
        if use_rerank and reranker is not None:
            ranked = await reranker.rerank(query, [h.text for h in cands], per_facet)
        if ranked:
            per.append([replace(cands[i], rerank_score=s) for i, s in ranked[:per_facet]])
        else:  # None (outage) and [] for non-empty candidates are both failures
            rerank_ok = False
            per.append(cands[:per_facet])
    hits = [g[n] for n in range(max(map(len, per), default=0)) for g in per if n < len(g)]
    return ComparisonResult(hits=hits, facets_empty=facets_empty, rerank_ok=rerank_ok)
