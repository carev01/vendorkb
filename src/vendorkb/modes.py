"""Cross-vendor retrieval: per-vendor enumeration and per-facet comparison, both on top of
batched hybrid candidate steps (``search.vendor_candidates``, ``search.facet_candidates``)
with one query embedding reused across the fan-out."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace

import psycopg

from vendorkb.clients import Embedder, Reranker
from vendorkb.search import (
    _COLS, RERANK_CANDIDATES, Hit, _hit, embed_queries, facet_candidates, vendor_candidates,
)
from vendorkb.shape import Facet

# Chunks per vendor that answer() and the eval send to the rerank. 3/5 (the spec's default and
# this module's) cost ~115K rerank tokens per question on the 43-vendor corpus, over the spec's
# 100K trigger (§9.5), so callers shrink it to 2/3.
ENUM_PER_VENDOR = 2
ENUM_PER_VENDOR_MULTI = 3
# Candidates per vendor the reranker judges. The answer still gets each vendor's best 2/3 of
# them. At 2/3 judged (the vendor's first candidates), a vendor's best passage was often never
# scored and the vendors listed moved with the question's wording: Rubrik's Entra ID page,
# fourth for "Which vendors support Entra ID backup?", scores 0.74 (BACKLOG 43). Measured
# 2026-11-10 on the 23 labelled questions, 12 judged per vendor, shallower depths read from the
# same scores: at about 2.5 noise vendors a question, recall 0.956 (2/3 at floor 0.54), 0.971
# (6 at 0.56), 0.971 (8 at 0.56), 0.959 (12 at 0.58). About 170K rerank tokens a question, 70K
# before.
ENUM_RERANK_DEPTH = 6
ENUM_RERANK_BATCH = 100  # passages per rerank call; the calls of a question run together
# calibrated at ENUM_RERANK_DEPTH 6 (2026-11-10): the highest floor with calibration recall
# >= 0.95 (0.964; holdout 0.977). Was 0.54 at 2/3 judged (2026-11-06).
ENUMERATION_MIN_SCORE = 0.56


# Passages of a vendor's capability facts added to its candidates when the question names a
# taxonomy target or dimension by an exact alias (``fact_candidates``); 0 turns it off. A
# vendor's six candidates follow the question's wording, its fact passages do not: for three
# wordings of "which vendors back up Entra ID", the best fact passage of Commvault, Eon, Bacula,
# Rubrik and Datto scored over the floor every time, where Commvault's was a candidate for one
# wording in three and Bacula's never (BACKLOG 43, 2026-11-10).
ENUM_FACT_CHUNKS = 3
# Dimensions nearly every question names without meaning them ("backup products", "restore"):
# they pick no fact passages on their own; with a target they only order its passages.
GENERIC_DIMENSIONS = frozenset({"coverage.backup", "coverage.restore"})
MAX_ALIAS_WORDS = 6  # the longest phrase of a question looked up as an alias
_TOKEN = re.compile(r"[^\s()\[\]{},;:!?\"“”]+")


@dataclass
class EnumerationResult:
    hits: list[Hit]
    kept: list[str]  # vendors whose evidence goes to the answer
    dropped: list[str]  # vendors whose best score fell below the floor
    empty: list[str]  # vendors with no candidates
    rerank_ok: bool
    # each scored vendor's best rerank score, dropped vendors included ({} without a rerank)
    best_scores: dict[str, float] = field(default_factory=dict)
    # vendors whose candidates included passages of their capability facts (fact_candidates)
    fact_vendors: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    hits: list[Hit]
    facets_empty: list[str]
    rerank_ok: bool


def _pick(candidates: list[Hit], per_vendor: int, per_vendor_multi: int) -> list[Hit]:
    """The best chunk per product first (RRF order), then the next best overall, in that
    order, so trimming from the end loses a product's second chunk before another product's
    only one. A vendor with several products among its candidates keeps up to
    ``per_vendor_multi``."""
    products = list(dict.fromkeys(h.product for h in candidates))
    cap = per_vendor_multi if len(products) > 1 else per_vendor
    first = [next(h for h in candidates if h.product == p) for p in products][:cap]
    ids = {h.chunk_id for h in first}
    return (first + [h for h in candidates if h.chunk_id not in ids])[:cap]


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


def _phrases(question: str) -> dict[str, tuple[int, int]]:
    """Every phrase of up to ``MAX_ALIAS_WORDS`` words of ``question``, lowercased, with the
    word positions it spans (start, end); a phrase ending in a plural is there in the singular
    too ("immutable backups" also as "immutable backup"). Brackets and sentence punctuation
    separate words; a word keeps its inner hyphens, dots and slashes ("hyper-v")."""
    words = [w for w in (t.strip(".'’`*") for t in _TOKEN.findall(question.lower())) if w]
    out: dict[str, tuple[int, int]] = {}
    for i in range(len(words)):
        for j in range(i + 1, min(i + MAX_ALIAS_WORDS, len(words)) + 1):
            out.setdefault(" ".join(words[i:j]), (i, j))
            last = words[j - 1]
            if len(last) > 3 and last.endswith("s"):
                out.setdefault(" ".join([*words[i:j - 1], last[:-1]]), (i, j))
    return out


async def question_taxonomy(conn: psycopg.AsyncConnection,
                            question: str) -> tuple[list[str], list[str]]:
    """(target ids, dimension ids) the question names by an exact alias of the loaded taxonomy
    (names are aliases too). The longest phrase wins: a phrase inside a longer matched one
    names nothing ("azure" in "azure ad"). No embedding, no model: one indexed lookup."""
    phrases = _phrases(question)
    if not phrases:
        return [], []
    cur = await conn.execute(
        "SELECT alias, entity_type, entity_id FROM aliases"
        " WHERE entity_type IN ('target', 'dimension') AND alias = ANY(%s)", (list(phrases),))
    rows = sorted(await cur.fetchall(),
                  key=lambda r: (phrases[r[0]][0] - phrases[r[0]][1], phrases[r[0]][0], r[1], r[2]))
    taken: list[tuple[int, int]] = []
    targets: list[str] = []
    dimensions: list[str] = []
    for alias, kind, entity_id in rows:
        i, j = phrases[alias]
        if any(i < e and s < j and (s, e) != (i, j) for s, e in taken):
            continue
        taken.append((i, j))
        ids = targets if kind == "target" else dimensions
        if entity_id not in ids:
            ids.append(entity_id)
    return targets, dimensions


async def fact_candidates(conn: psycopg.AsyncConnection, question: str, vendors: list[str], *,
                          limit: int = ENUM_FACT_CHUNKS) -> dict[str, list[Hit]]:
    """For each vendor, up to ``limit`` current chunks behind its supported or partial
    capability facts on what the question names (``question_taxonomy``): the facts on a named
    target (or a target under it), those on a named dimension first; with no target, the facts
    on a named dimension that is not one of ``GENERIC_DIMENSIONS``. Chunks with a supported
    fact, a current one and more facts come first. {} when the question names neither, no
    taxonomy is loaded or ``limit`` is 0. One statement for every vendor.

    These are candidates, nothing more: the reranker scores them against the question like any
    other passage, and the same floor decides."""
    if limit <= 0 or not vendors:
        return {}
    targets, dimensions = await question_taxonomy(conn, question)
    if not targets and not [d for d in dimensions if d not in GENERIC_DIMENSIONS]:
        return {}
    cur = await conn.execute("SELECT value FROM taxonomy_meta WHERE key = 'taxonomy_version'")
    row = await cur.fetchone()
    if row is None:
        return {}
    scope = ("(cap.target_id = ANY(%(targets)s) OR cap.target_id LIKE ANY(%(under)s))"
             if targets else "cap.dimension_id = ANY(%(dimensions)s)")
    cur = await conn.execute(
        f"SELECT p.vendor, {_COLS} FROM (SELECT f.vendor, f.chunk_id, row_number() OVER"
        " (PARTITION BY f.vendor ORDER BY f.on_dim DESC, f.sup DESC, f.cur DESC, f.n DESC,"
        " f.chunk_id) AS rn FROM (SELECT cap.vendor, r.chunk_id,"
        " bool_or(cap.dimension_id = ANY(%(dimensions)s)) AS on_dim,"
        " bool_or(cap.support = 'supported') AS sup, bool_or(cap.is_current) AS cur,"
        " count(*) AS n FROM capabilities cap JOIN raw_claims r ON r.claim_id = cap.claim_id"
        " JOIN chunks ch ON ch.chunk_id = r.chunk_id"
        " WHERE cap.support IN ('supported', 'partial') AND cap.taxonomy_version = %(version)s"
        f" AND cap.vendor = ANY(%(vendors)s) AND ch.superseded_at IS NULL AND {scope}"
        " GROUP BY cap.vendor, r.chunk_id) f) p"
        " JOIN chunks c ON c.chunk_id = p.chunk_id JOIN documents d ON d.doc_id = c.doc_id"
        " WHERE p.rn <= %(limit)s ORDER BY p.vendor, p.rn",
        {"targets": targets, "under": [t + ".%" for t in targets], "dimensions": dimensions,
         "version": row[0], "vendors": vendors, "limit": limit})
    out: dict[str, list[Hit]] = {}
    for vendor, *cols in await cur.fetchall():
        out.setdefault(vendor, []).append(_hit(tuple(cols), len(out.get(vendor, [])) + 1))
    return out


async def _rerank_batched(reranker: Reranker, query: str,
                          texts: list[str]) -> list[tuple[int, float]] | None:
    """``reranker.rerank`` over ``texts`` in calls of ``ENUM_RERANK_BATCH``, run together;
    (index into ``texts``, score) pairs. A passage's score does not depend on the others in
    its call. None when any call fails or returns nothing: half the vendors judged would drop
    the other half's."""
    starts = range(0, len(texts), ENUM_RERANK_BATCH)
    replies = await asyncio.gather(*(
        reranker.rerank(query, texts[i:i + ENUM_RERANK_BATCH], len(texts[i:i + ENUM_RERANK_BATCH]))
        for i in starts))
    if not all(replies):
        return None
    return [(start + i, score) for start, reply in zip(starts, replies, strict=True)
            for i, score in reply or []]


async def enumerate_vendors(conn: psycopg.AsyncConnection, query: str, *, embedder: Embedder,
                            reranker: Reranker | None, vendors: list[str],
                            per_vendor: int = 3, per_vendor_multi: int = 5,
                            min_score: float = ENUMERATION_MIN_SCORE, max_hits: int = 60,
                            use_rerank: bool = True,
                            depth: int = ENUM_RERANK_DEPTH,
                            fact_chunks: int | None = None) -> EnumerationResult:
    """Search every vendor (one batched candidate step, ``search.vendor_candidates``), rerank
    each vendor's first ``depth`` candidates (and its best chunk per product beyond them, the
    ``_pick`` of its candidates), with up to ``fact_chunks`` passages of its capability facts
    on what the question names (``fact_candidates``; None reads ``ENUM_FACT_CHUNKS``), and
    keep the vendors whose best score reaches ``min_score``.
    A kept vendor's hits are its best ``per_vendor`` passages by score (``per_vendor_multi``,
    the best of each product first, for a vendor with several products among them). On a
    reranker failure (or with ``use_rerank=False``, the free eval mode) every vendor with
    candidates is kept with the ``_pick`` of its candidates, in the order given, and
    ``rerank_ok`` is False. A vendor the reranker left wholly unscored is kept too, after the
    scored survivors, and also clears ``rerank_ok``: it was never judged, so it is not
    ``dropped``."""
    vendors = list(dict.fromkeys(vendors))
    # An embedder failure propagates: the caller records it.
    qvec = (await embed_queries(conn, embedder, [query]))[0]
    cands = await vendor_candidates(conn, query, qvec, vendors,
                                    limit=max(depth, per_vendor_multi * 2))
    can_rerank = use_rerank and reranker is not None
    facts = await fact_candidates(
        conn, query, vendors, limit=ENUM_FACT_CHUNKS if fact_chunks is None else fact_chunks
    ) if can_rerank else {}  # unranked, a fact passage is no evidence: only the reranker's say
    picked = {v: _pick(cands.get(v, [])[:per_vendor_multi * 2], per_vendor, per_vendor_multi)
              for v in vendors if v in cands or v in facts}
    empty = [v for v in vendors if v not in picked]

    def judged(v: str) -> list[Hit]:
        own = cands.get(v, [])
        ids = {h.chunk_id for h in [*own[:depth], *picked[v]]}
        first = [h for h in own if h.chunk_id in ids]
        # a fact passage ranks after the vendor's own candidates until it is scored
        extra = [replace(h, rrf_rank=len(own) + n) for n, h in enumerate(
            (h for h in facts.get(v, []) if h.chunk_id not in ids), start=1)]
        return first + extra

    flat = [h for v in picked for h in judged(v)]
    if not flat:  # nothing needed reranking: not a reranker failure
        return EnumerationResult(hits=[], kept=[], dropped=[], empty=empty,
                                 rerank_ok=can_rerank)
    ranked = None
    if can_rerank and reranker is not None:
        ranked = await _rerank_batched(reranker, query, [h.text for h in flat])
    if not ranked:  # None (outage) and [] for non-empty input are both failures
        kept = [v for v in picked if picked[v]]  # not a vendor known by its facts alone
        groups = _truncate([picked[v] for v in kept], max_hits)
        return EnumerationResult(hits=[h for g in groups for h in g], kept=kept, dropped=[],
                                 empty=empty, rerank_ok=False)

    scores = dict(reversed(ranked))  # index -> score, the first entry winning a repeat
    scored: dict[str, list[Hit]] = {v: [] for v in picked}
    for i in dict.fromkeys(i for i, _ in ranked):
        scored[flat[i].vendor].append(replace(flat[i], rerank_score=scores[i]))
    unscored = [v for v in picked if not scored[v]]
    best = {v: max(h.rerank_score or 0.0 for h in hs) for v, hs in scored.items() if hs}
    survivors = sorted((v for v in best if best[v] >= min_score), key=lambda v: -best[v])
    dropped = [v for v in best if v not in survivors]

    def chosen(v: str) -> list[Hit]:
        by_score = sorted(scored[v], key=lambda h: (-(h.rerank_score or 0.0), h.rrf_rank))
        return sorted(_pick(by_score, per_vendor, per_vendor_multi),
                      key=lambda h: -(h.rerank_score or 0.0))

    groups = _truncate([chosen(v) for v in survivors] + [picked[v] for v in unscored], max_hits)
    return EnumerationResult(hits=[h for g in groups for h in g], kept=survivors + unscored,
                             dropped=dropped, empty=empty, rerank_ok=not unscored,
                             best_scores=best, fact_vendors=[v for v in picked if v in facts])


def _distinct(hits: list[Hit], n: int) -> list[Hit]:
    """The first ``n`` hits whose text (whitespace-collapsed) has not come before: a page
    published in two guides comes back twice with one score and took two of a side's five
    slots (cv-12, 2026-11-08)."""
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        key = " ".join(h.text.split())
        if key not in seen:
            seen.add(key)
            out.append(h)
            if len(out) == n:
                break
    return out


async def compare_facets(conn: psycopg.AsyncConnection, query: str, *, embedder: Embedder,
                         reranker: Reranker | None, facets: list[Facet], per_facet: int = 5,
                         use_rerank: bool = True) -> ComparisonResult:
    """Search each facet with an equal share of the candidate pool (one batched candidate
    step, ``search.facet_candidates``) and rerank each facet separately, the reranks running
    concurrently, so every facet's quota survives the rerank. A passage whose text repeats one
    already kept for the facet is skipped (``_distinct``). Hits interleave facet by facet;
    a facet with no candidates is reported in ``facets_empty``, never relaxed."""
    if not facets:
        return ComparisonResult(hits=[], facets_empty=[], rerank_ok=False)
    # An embedder failure propagates: the caller records it.
    qvec = (await embed_queries(conn, embedder, [query]))[0]
    limit = max(1, RERANK_CANDIDATES // len(facets))
    cands = await facet_candidates(conn, query, qvec, list(facets), limit=limit)
    facets_empty = [name for (_, name), c in zip(facets, cands) if not c]
    found = [c for c in cands if c]
    rerank_ok = use_rerank and reranker is not None

    async def rerank(c: list[Hit]) -> list[tuple[int, float]] | None:
        if not use_rerank or reranker is None:
            return None
        return await reranker.rerank(query, [h.text for h in c], len(c))

    per: list[list[Hit]] = []
    for c, ranked in zip(found, await asyncio.gather(*(rerank(c) for c in found))):
        if ranked:
            per.append(_distinct([replace(c[i], rerank_score=s) for i, s in ranked], per_facet))
        else:  # None (outage) and [] for non-empty candidates are both failures
            rerank_ok = False
            per.append(_distinct(c, per_facet))
    hits = [g[n] for n in range(max(map(len, per), default=0)) for g in per if n < len(g)]
    return ComparisonResult(hits=hits, facets_empty=facets_empty, rerank_ok=rerank_ok)
