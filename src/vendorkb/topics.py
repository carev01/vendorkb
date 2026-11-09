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

``topic_grid`` answers a "which vendors" question as a table, built in code: a row for each
vendor with a mark, a column for each aspect, a cell holding the best capability fact on the
aspect's dimensions (a ``Cell``, cited), or, for a cell no fact fills, a passage from a page no
extraction has reached yet, when the reranker finds it on point. ``render_grid`` writes it.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import lru_cache
from typing import Literal, cast

import httpx
import psycopg

from vendorkb.clients import Embedder, Reranker
from vendorkb.modes import ENUMERATION_MIN_SCORE, _distinct
from vendorkb.search import (
    _COLS, VEC_LIMIT, Hit, _hit, facet_candidates, hybrid_candidates, unextracted_candidates,
    vector_candidates)
from vendorkb.shape import Facet
from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import Aspect, Topic, current_version

log = logging.getLogger(__name__)

# Documents one topic question sends to the reranker in all, its own passages' reranks included:
# the hard cap on the main cost (rerank tokens).
TOPIC_RERANK_DOCS = 240
# Rerank calls of one topic question in flight at once (the rest wait their turn).
TOPIC_RERANKS_IN_FLIGHT = 4
# Candidates per aspect the grid's catch-up search sends to the reranker: at most 8 aspects x 20
# = 160 documents, inside TOPIC_RERANK_DOCS (``topic_grid`` shrinks it for a longer topic).
CATCH_UP_CANDIDATES = 20

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


def _label(text: str) -> str:
    """``_norm`` with underscores as spaces too: a topic id and a topic name compare equal."""
    return _norm(text.replace("_", " "))


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


def match_topics(term: str, topics: list[Topic]) -> list[Topic]:
    """The ``topics`` that ``term`` names, best match first, ties in ``topics`` order. Compared
    as ``_label`` text: a topic whose id, name or a trigger is the term, then one holding it
    as a whole word there. An aspect's name names no topic. MCP tools turn the wording of an
    agent into topics with it."""
    text = _label(term)
    if not text:
        return []
    word = re.compile(rf"(?<!\w){re.escape(text)}(?!\w)")
    ranked: list[tuple[int, int]] = []
    for i, topic in enumerate(topics):
        names = [_label(x) for x in (topic.id, topic.name, *topic.triggers)]
        if text in names:
            ranked.append((0, i))
        elif any(word.search(x) for x in names):
            ranked.append((1, i))
    return [topics[i] for _, i in sorted(ranked)]


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


async def _rerank_lists(reranker: Reranker | None, jobs: list[tuple[str, list[Hit]]]
                        ) -> list[tuple[list[Hit], bool]]:
    """``_rerank`` of each (query, candidates) job, in job order. The calls run concurrently,
    at most ``TOPIC_RERANKS_IN_FLIGHT`` at a time, none retried."""
    gate = asyncio.Semaphore(TOPIC_RERANKS_IN_FLIGHT)

    async def rerank(query: str, candidates: list[Hit]) -> tuple[list[Hit], bool]:
        async with gate:
            return await _rerank(reranker, query, candidates)

    return await asyncio.gather(*(rerank(q, c) for q, c in jobs))


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
    results = await _rerank_lists(reranker, [(q, c) for q, c, _ in jobs])

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


Mark = Literal["supported", "partial", "recent"]


@dataclass
class Cell:
    """One mark of the grid, cited by ``hit``. ``supported`` and ``partial`` are the support of
    the best capability fact; ``recent`` is a passage from a page no extraction has reached,
    judged on point by the reranker and verified by nothing else. ``superseded``: the fact's
    chunk is an earlier version of its page (read now, not as of the last capabilities build)."""
    mark: Mark
    hit: Hit
    superseded: bool


@dataclass
class Grid:
    vendors: list[str]  # the rows: the vendors with at least one cell, by name
    cells: dict[tuple[str, str], Cell]  # (vendor, aspect id) -> cell; a blank has no entry
    facts_as_of: datetime | None  # newest page among the fact cells (not the recent ones)
    no_fact: list[str]  # the vendors asked about with no cell, by name
    rerank_ok: bool = True  # the catch-up reranker judged every list it needed to
    reranked_docs: int = 0  # documents the catch-up's reranker scored (calls that succeeded)
    catch_up_ok: bool = True  # False: the embedder failed and the catch-up was skipped


async def _best_facts(conn: psycopg.AsyncConnection, topic: Topic, vendors: list[str],
                      version: str) -> list[tuple[str, str, str, bool, datetime, Hit]]:
    """(vendor, aspect id, support, superseded, doc_version, hit) for the best supported or
    partial fact of each vendor and aspect, in one statement over the ``capabilities`` rows of
    taxonomy ``version`` on the aspects' dimensions (a dimension two aspects list counts for
    both). Best is: supported over partial, then a fact on a current chunk over one on an
    earlier version of its page (``chunks.superseded_at``, read now), then ``is_current``, the
    newest page, the chunk. The vendor is the page's, as the citation shows it.

    The pick sorts narrow rows (ids, flags, dates: ``c.text`` in them made three 11 MB sorts
    for 239 cells); the chunk columns are read only for the rows picked."""
    pairs = [(a.id, dimension) for a in topic.aspects for dimension in a.dimensions]
    cur = await conn.execute(
        "SELECT p.vendor, p.aspect_id, p.support, p.superseded, p.doc_version, " + _COLS +
        " FROM (SELECT DISTINCT ON (d.vendor, a.aspect_id) d.vendor, a.aspect_id, cap.support,"
        " c.superseded_at IS NOT NULL AS superseded, cap.doc_version, c.chunk_id"
        " FROM capabilities cap JOIN raw_claims r ON r.claim_id = cap.claim_id"
        " JOIN chunks c ON c.chunk_id = r.chunk_id JOIN documents d ON d.doc_id = c.doc_id"
        " JOIN unnest(%s::text[], %s::text[]) AS a(aspect_id, dimension_id)"
        " ON a.dimension_id = cap.dimension_id"
        " WHERE d.vendor = ANY(%s) AND cap.taxonomy_version = %s"
        " AND cap.support IN ('supported', 'partial')"
        " ORDER BY d.vendor, a.aspect_id, (cap.support = 'supported') DESC,"
        " (c.superseded_at IS NULL) DESC, cap.is_current DESC, cap.doc_version DESC,"
        " c.chunk_id, cap.claim_id) p"
        " JOIN chunks c ON c.chunk_id = p.chunk_id JOIN documents d ON d.doc_id = c.doc_id"
        " ORDER BY p.vendor, p.aspect_id",
        [[aspect for aspect, _ in pairs], [dimension for _, dimension in pairs], vendors,
         version])
    return [(vendor, aspect, support, superseded, doc_version, _hit(tuple(row)))
            for vendor, aspect, support, superseded, doc_version, *row in await cur.fetchall()]


async def _catch_up(conn: psycopg.AsyncConnection, topic: Topic, vendors: list[str],
                    cells: dict[tuple[str, str], Cell], *, embedder: Embedder,
                    reranker: Reranker | None) -> tuple[bool, bool, int]:
    """Fill the empty cells of ``cells`` (in place) from the chunks no extraction has reached
    (``search.unextracted_candidates``): per aspect, the best candidates for the aspect's query
    (``VEC_LIMIT`` of them), minus those of vendors not asked about and of vendors whose cell is
    already filled (a vendor with facts must not crowd the others out of the cut), the first
    ``CATCH_UP_CANDIDATES`` of what is left reranked with that query. The best passage of a
    vendor that scores at least ``ENUMERATION_MIN_SCORE`` becomes its ``recent`` cell. A list
    whose rerank fails (or no reranker) marks nothing: a page nobody verified never fills a
    cell unscored.

    Returns (the embedder answered, every list judged, documents scored). An embedder that
    fails over HTTP (``httpx.HTTPError``) skips the catch-up with a warning: the grid is the
    facts alone. Any other embedder error, a wrong count or dimension, is a defect and
    propagates."""
    aspects = topic.aspects
    limit = max(1, min(CATCH_UP_CANDIDATES, TOPIC_RERANK_DOCS // len(aspects)))
    queries = [a.query for a in aspects]
    try:
        qvecs = await embedder.embed(queries, input_type="query")
    except httpx.HTTPError as exc:
        log.warning("vendorkb grid catch-up skipped, the embedder failed: %s: %s",
                    type(exc).__name__, exc)
        return False, True, 0
    lists = await unextracted_candidates(conn, queries, qvecs, limit=VEC_LIMIT)
    asked = set(vendors)
    jobs = [(a.query, [h for h in found
                       if h.vendor in asked and (h.vendor, a.id) not in cells][:limit])
            for a, found in zip(aspects, lists)]
    results = await _rerank_lists(reranker, jobs)
    docs = 0
    for aspect, (_, candidates), (ranked, judged) in zip(aspects, jobs, results):
        if not judged:
            continue
        docs += len(candidates)
        for hit in sorted(ranked, key=lambda h: -(h.rerank_score or 0.0)):
            if ((hit.rerank_score or 0.0) >= ENUMERATION_MIN_SCORE
                    and (hit.vendor, aspect.id) not in cells):
                cells[(hit.vendor, aspect.id)] = Cell("recent", hit, False)
    return True, reranker is not None and all(judged for _, judged in results), docs


async def topic_grid(conn: psycopg.AsyncConnection, topic: Topic, *, vendors: list[str],
                     embedder: Embedder, reranker: Reranker | None,
                     catch_up: bool = True) -> Grid:
    """The grid of ``vendors`` by the aspects of ``topic``: each cell the best capability fact
    (``_best_facts``), then, with ``catch_up``, the ``recent`` cells (``_catch_up``) for the
    cells no fact fills. A fact never gives way to a recent page. ``facts_as_of`` is the newest
    ``doc_version`` of the fact cells; no taxonomy loaded means no fact cells. An embedder that
    is down costs the catch-up only (``catch_up_ok`` False), never the facts."""
    vendors = list(dict.fromkeys(vendors))
    version = await current_version(conn) if vendors else None
    cells: dict[tuple[str, str], Cell] = {}
    as_of: datetime | None = None
    if version is not None:
        for vendor, aspect, support, superseded, doc_version, hit in await _best_facts(
                conn, topic, vendors, version):
            cells[(vendor, aspect)] = Cell(
                cast(Mark, support), replace(hit, superseded=True) if superseded else hit,
                superseded)
            as_of = doc_version if as_of is None else max(as_of, doc_version)
    catch_up_ok, rerank_ok, docs = True, True, 0
    if catch_up and vendors:
        catch_up_ok, rerank_ok, docs = await _catch_up(
            conn, topic, vendors, cells, embedder=embedder, reranker=reranker)
    rows = sorted({vendor for vendor, _ in cells})
    return Grid(vendors=rows, cells=cells, facts_as_of=as_of,
                no_fact=sorted(v for v in vendors if v not in rows), rerank_ok=rerank_ok,
                reranked_docs=docs, catch_up_ok=catch_up_ok)


_MARKS = {"supported": "✓", "partial": "◐", "recent": "○"}
LEGEND = ("✓ supported · ◐ partial · ○ found in a page not yet "
          "analysed (not verified) · † from an earlier version of the page.")
EARLIER = "†"  # after the marker of a fact from an earlier version of its page
BLANK_NOTE = "A blank cell means no fact was found, not that the product lacks the capability."


def _escaped(text: str) -> str:
    return text.replace("|", "\\|")


def render_grid(grid: Grid, topic: Topic) -> tuple[str, list[Hit]]:
    """The grid as a markdown table and the hits its [n] marks cite, ``hits[n - 1]`` for [n]:
    the rows by number of marks (most first), then name; each cell ``✓ [n]``, ``◐ [n]``
    or ``○ [n]``, with a ``†`` after the marker for a fact of an earlier version of its page,
    numbered row by row and aspect by aspect (one number for each cell); blank when no fact
    was found. Then the legend, the date the facts are current to and the vendors with no
    fact. A grid with no cell at all is a short answer saying so, not an empty table."""
    aspects = topic.aspects
    rows = sorted(grid.vendors, key=lambda v: (-sum((v, a.id) in grid.cells for a in aspects), v))
    hits: list[Hit] = []
    lines = ["| Vendor | " + " | ".join(_escaped(a.name) for a in aspects) + " |",
             "| --- |" + " --- |" * len(aspects)]
    for vendor in rows:
        texts = []
        for aspect in aspects:
            cell = grid.cells.get((vendor, aspect.id))
            if cell is None:
                texts.append("")
                continue
            hits.append(cell.hit)
            texts.append(f"{_MARKS[cell.mark]} [{len(hits)}]{EARLIER if cell.superseded else ''}")
        lines.append(f"| {_escaped(vendor)} | " + " | ".join(texts) + " |")
    as_of = grid.facts_as_of
    if as_of is None:
        current = "Facts current to: no facts found."
    else:
        day = as_of.astimezone(UTC) if as_of.tzinfo else as_of
        current = f"Facts current to {day:%Y-%m-%d}."
    footer = [LEGEND, f"{current} {BLANK_NOTE}"]
    if grid.no_fact:
        footer.append(f"No fact found for: {', '.join(sorted(grid.no_fact))}.")
    if not hits:
        lines = [f"No fact was found for {topic.name.lower()} in the documented vendors' pages."]
    return "\n".join([*lines, "", *footer]), hits
