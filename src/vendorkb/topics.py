"""Topics: the taxonomy's broad subjects (a fixed kit of aspects), and spotting one in a question.

A topic fires only on a broad question: one of its triggers is in the question and the question
names none of its aspects (no aspect cue). Triggers and cues are compared with the question after
the same normalising: lowercase, hyphens and dashes as spaces, whitespace collapsed. A trigger
matches as a whole phrase, with an optional plural suffix; a cue matches at a word start, so a
stem such as ``immutab`` covers every form of the word. A phrase that normalises to nothing is
skipped.

``topic_hits`` retrieves a topic question's evidence: the question's own passages for each side
(a named vendor or product, or the whole corpus when none is named), then a small quota of
passages for each aspect of the topic and each side, so one aspect's pages cannot crowd out
another's. Every rerank of the question, the own passages' included, counts against one document
budget.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, replace
from functools import lru_cache

import psycopg

from vendorkb.clients import Embedder, Reranker
from vendorkb.modes import _distinct
from vendorkb.search import (
    _COLS, Hit, _hit, facet_candidates, hybrid_candidates, vector_candidates)
from vendorkb.shape import Facet
from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import Aspect, Topic, current_version

# Documents one topic question sends to the reranker in all, its own passages' reranks included:
# the hard cap on the main cost (rerank tokens).
TOPIC_RERANK_DOCS = 240
# Rerank calls of one topic question in flight at once (the rest wait their turn).
TOPIC_RERANKS_IN_FLIGHT = 4

_HYPHENS = re.compile("[-\u2010\u2011\u2013\u2014]")  # hyphen-minus, hyphens, en/em dash


def _norm(text: str) -> str:
    return normalize(_HYPHENS.sub(" ", text))


@lru_cache(maxsize=512)
def _trigger(phrase: str) -> re.Pattern[str] | None:
    """None for a phrase that normalises to nothing: it must never match everything."""
    text = _norm(phrase)
    return re.compile(rf"(?<!\w){re.escape(text)}(?:s|es)?(?!\w)") if text else None


@lru_cache(maxsize=512)
def _cue(phrase: str) -> re.Pattern[str] | None:
    text = _norm(phrase)
    return re.compile(rf"(?<!\w){re.escape(text)}") if text else None


def _found(pattern: re.Pattern[str] | None, text: str) -> bool:
    return pattern is not None and pattern.search(text) is not None


def detect_topic(question: str, topics: list[Topic]) -> Topic | None:
    """The first of ``topics`` whose trigger the question holds and none of whose aspects it
    names, else None (the question is handled as it is without topics)."""
    text = _norm(question)
    for topic in topics:
        if not any(_found(_trigger(t), text) for t in topic.triggers):
            continue
        if not any(_found(_cue(c), text) for a in topic.aspects for c in a.cues):
            return topic
    return None


async def load_topics(conn: psycopg.AsyncConnection) -> list[Topic]:
    """Every topic of the loaded taxonomy, by topic id, its aspects in YAML order."""
    cur = await conn.execute(  # one statement: a reload cannot split a topic from its aspects
        "SELECT t.topic_id, t.name, t.triggers, a.aspect_id, a.name, a.query, a.cues,"
        " a.dimensions FROM topics t JOIN topic_aspects a USING (topic_id)"
        ' ORDER BY t.topic_id COLLATE "C", a.ordinal')
    heads: dict[str, tuple[str, tuple[str, ...]]] = {}
    aspects: dict[str, list[Aspect]] = {}
    for topic_id, name, triggers, aspect_id, aspect_name, query, cues, dimensions in (
            await cur.fetchall()):
        heads[topic_id] = (name, tuple(triggers))
        aspects.setdefault(topic_id, []).append(
            Aspect(aspect_id, aspect_name, query, tuple(cues), tuple(dimensions)))
    return [Topic(topic_id, name, triggers, tuple(aspects[topic_id]))
            for topic_id, (name, triggers) in heads.items()]


@dataclass
class TopicHits:
    hits: list[Hit]
    aspect_of: dict[str, str]  # chunk_id -> aspect id, for the passages kept for an aspect
    rerank_ok: bool
    reranked_docs: int  # documents the reranker scored, over every call that succeeded


async def fact_chunks(conn: psycopg.AsyncConnection, aspect: Aspect, facet: Facet,
                      version: str, limit: int) -> list[Hit]:
    """The current chunks behind the facet's supported or partial facts on the aspect's
    dimensions (``capabilities`` rows of taxonomy ``version``), best first: supported over
    partial, then current over superseded facts, then the newest page. A chunk is listed once
    however many facts it carries; a chunk that is an earlier version of its page never is."""
    if limit <= 0:
        return []
    kind, name = facet
    column = "cap.vendor" if kind == "vendor" else "cap.product"
    cur = await conn.execute(
        f"SELECT {_COLS} FROM (SELECT r.chunk_id, bool_or(cap.support = 'supported') AS sup,"
        " bool_or(cap.is_current) AS cur, max(cap.doc_version) AS ver"
        " FROM capabilities cap JOIN raw_claims r USING (claim_id)"
        " WHERE cap.dimension_id = ANY(%s) AND cap.support IN ('supported', 'partial')"
        f" AND cap.taxonomy_version = %s AND {column} = %s GROUP BY r.chunk_id) f"
        " JOIN chunks c ON c.chunk_id = f.chunk_id JOIN documents d ON d.doc_id = c.doc_id"
        " WHERE c.superseded_at IS NULL"
        " ORDER BY f.sup DESC, f.cur DESC, f.ver DESC, c.chunk_id LIMIT %s",
        [list(aspect.dimensions), version, name, limit])
    return [_hit(row, n) for n, row in enumerate(await cur.fetchall(), start=1)]


def _key(text: str) -> str:
    return " ".join(text.split())


async def _rerank(reranker: Reranker | None, query: str,
                  candidates: list[Hit]) -> tuple[list[Hit], bool]:
    """The candidates best first with their rerank scores, and True; or, when relevance could
    not be scored (no reranker, an outage), the candidates as they came, in RRF order and
    unscored, and False. No candidates is an empty list and True: nothing needed scoring."""
    if not candidates:
        return [], True
    if reranker is not None:
        ranked = await reranker.rerank(query, [h.text for h in candidates], len(candidates))
        if ranked:  # None (outage) and [] for non-empty candidates are both failures
            return [replace(candidates[i], rerank_score=s) for i, s in ranked], True
    return candidates, False


async def topic_hits(conn: psycopg.AsyncConnection, question: str, topic: Topic, *,
                     facets: list[Facet], embedder: Embedder, reranker: Reranker | None,
                     budget: int = TOPIC_RERANK_DOCS, per_aspect: int = 2,
                     own: int = 3) -> TopicHits:
    """Evidence for a broad question on ``topic``: for each side (``facets``, or the whole
    corpus when there are none) the question's own best ``own`` passages, then for each aspect
    and side the best ``per_aspect`` passages for the aspect's query.

    The question and the aspect queries are embedded in one call, as written (no side name).
    The question's own candidates come from hybrid search (``own * 3`` per side): one full-text
    statement, ``facet_candidates`` for two or more facets. An aspect's candidates come from
    vector search alone (``search.vector_candidates``, one transaction for every aspect and
    side): the full-text scan costs about 2 s per query text on the full corpus, so a topic
    question pays for one, not for one per aspect and side. Only the rerank query names the
    side, as ``"<side name>: <aspect query>"``.

    Every rerank counts against ``budget`` documents: the own passages' candidates first, the
    rest shared out equally over the (side, aspect) lists, at least 2 each (so a small budget
    can be exceeded). A list's candidates are the side's fact chunks (``fact_chunks``, up to
    half the list, for a side only) and the aspect's nearest chunks, in that order. The lists
    are reranked concurrently, at most ``TOPIC_RERANKS_IN_FLIGHT`` calls at a time, none
    retried: one whose rerank fails keeps its order (``rrf_rank``, the list position),
    unscored, and clears ``rerank_ok``; ``reranked_docs`` counts the calls that succeeded.

    Hits are the own passages (side order), then the aspect passages in aspect order (side order
    within an aspect), a text repeated anywhere kept once at its first place. An aspect passes
    over passages an earlier aspect took (so a page that covers two aspects does not leave the
    second short); an own passage an aspect also chose keeps its place and is labelled in
    ``aspect_of``. An embedder failure propagates: the caller records it."""
    sides: list[Facet | None] = list(facets) or [None]
    vecs = await embedder.embed([question, *(a.query for a in topic.aspects)],
                                input_type="query")
    version = await current_version(conn)
    if len(facets) >= 2:
        own_lists = await facet_candidates(conn, question, vecs[0], list(facets), limit=own * 3)
    else:
        side = sides[0]
        own_lists = [await hybrid_candidates(
            conn, question, vecs[0], vendor=[side[1]] if side and side[0] == "vendor" else None,
            product=[side[1]] if side and side[0] == "product" else None, limit=own * 3)]

    # (rerank query, candidates, aspect id, None for an own-passages list), in output order
    jobs: list[tuple[str, list[Hit], str | None]] = [(question, c, None) for c in own_lists]
    own_docs = sum(len(c) for _, c, _ in jobs)
    allowance = max(2, (budget - own_docs) // (len(sides) * len(topic.aspects)))
    nearest = await vector_candidates(conn, vecs[1:], sides, limit=allowance)
    for aspect, by_side in zip(topic.aspects, nearest):
        for side, near in zip(sides, by_side):
            query = aspect.query if side is None else f"{side[1]}: {aspect.query}"
            facts = ([] if side is None or version is None else
                     await fact_chunks(conn, aspect, side, version, allowance // 2))
            seen = {h.chunk_id for h in facts}
            merged = (facts + [h for h in near if h.chunk_id not in seen])[:allowance]
            jobs.append((query, [replace(h, rrf_rank=n) for n, h in enumerate(merged, start=1)],
                         aspect.id))
    gate = asyncio.Semaphore(TOPIC_RERANKS_IN_FLIGHT)

    async def rerank(query: str, candidates: list[Hit]) -> tuple[list[Hit], bool]:
        async with gate:
            return await _rerank(reranker, query, candidates)

    results = await asyncio.gather(*(rerank(q, c) for q, c, _ in jobs))

    hits: list[Hit] = []
    aspect_of: dict[str, str] = {}
    placed: dict[str, Hit] = {}  # text -> the hit kept for it
    taken: set[str] = set()  # texts kept for an aspect

    def place(hit: Hit, aspect_id: str | None) -> None:
        kept = placed.setdefault(_key(hit.text), hit)
        if kept is hit:
            hits.append(hit)
        if aspect_id is not None:
            aspect_of.setdefault(kept.chunk_id, aspect_id)

    for (_, _, aspect_id), (ranked, _) in zip(jobs, results):
        if aspect_id is None:
            for hit in _distinct(ranked, own):
                place(hit, None)
        else:
            for hit in _distinct([h for h in ranked if _key(h.text) not in taken], per_aspect):
                taken.add(_key(hit.text))
                place(hit, aspect_id)
    return TopicHits(
        hits=hits, aspect_of=aspect_of,
        rerank_ok=reranker is not None and all(ok for _, ok in results),
        reranked_docs=sum(len(c) for (_, c, _), (_, ok) in zip(jobs, results) if ok))
