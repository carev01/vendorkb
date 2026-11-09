import asyncio
import re
import uuid
from datetime import UTC, datetime, timedelta
from itertools import groupby
from pathlib import Path

import httpx
import pytest
import yaml
from pgvector import Vector

from tests.integration.test_capabilities_v2 import T0, _claim, _doc_of, _fresh, _tail_chunk
from vendorkb import search
from vendorkb import topics as topics_mod
from vendorkb.db import connect, init_schema
from vendorkb.search import Hit, hybrid_candidates
from vendorkb.taxonomy import capabilities, load
from vendorkb.taxonomy.load import Aspect, Topic
from vendorkb.topics import (
    TOPIC_RERANK_DOCS, Cell, Grid, detect_topic, fact_chunks, load_topics, render_grid,
    topic_grid, topic_hits)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
V2_TABLES = ("dimensions", "classes", "object_types", "targets", "variants", "offerings",
             "editions", "aliases", "topics", "topic_aspects")


def _topic(topic_id: str, triggers: list[str]) -> dict:
    """A topic over the fixture's dimensions: two aspects, the second without cues."""
    return {"id": topic_id, "name": f"Topic {topic_id}", "triggers": triggers,
            "aspects": [
                {"id": "keeping", "name": "Keeping copies", "query": "How long copies are kept",
                 "cues": ["Retention", "keep  copies"],
                 "dimensions": ["retention.maximum_period", "schedule.backup_frequency"]},
                {"id": "getting_back", "name": "Getting data back",
                 "query": "Restoring objects, and writing them out of a backup",
                 "dimensions": ["coverage.restore", "coverage.export"]}]}


@pytest.fixture
async def conn(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await conn.execute(f"DROP TABLE IF EXISTS capabilities, features, {', '.join(V2_TABLES)}"
                           " CASCADE")
        await init_schema(conn, embed_dim=768)
        await conn.execute("DELETE FROM taxonomy_meta")
        yield conn
    finally:
        await conn.close()


async def _load(conn, tmp_path: Path, topics: list[dict]) -> None:
    doc = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    if topics:
        doc["topics"] = topics
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    await load.load_taxonomy(conn, path)


async def test_load_topics_reads_the_loaded_taxonomy(conn, tmp_path):
    # YAML order is b before a: load_topics orders by topic id, so "a" wins a double match
    await _load(conn, tmp_path, [_topic("b_loss", ["Data Loss", "lost data"]),
                                 _topic("a_loss", ["Data-Loss"])])
    topics = await load_topics(conn)
    assert [t.id for t in topics] == ["a_loss", "b_loss"]
    b = topics[1]
    assert isinstance(b, load.Topic) and b.name == "Topic b_loss"
    assert b.triggers == ("data loss", "lost data")  # stored normalized
    assert [a.id for a in b.aspects] == ["keeping", "getting_back"]  # in ordinal order
    assert all(isinstance(a, load.Aspect) for a in b.aspects)
    keeping, getting_back = b.aspects
    assert keeping == load.Aspect(
        "keeping", "Keeping copies", "How long copies are kept", ("retention", "keep copies"),
        ("retention.maximum_period", "schedule.backup_frequency"))
    assert getting_back.cues == ()
    assert getting_back.dimensions == ("coverage.restore", "coverage.export")
    assert getting_back.query == "Restoring objects, and writing them out of a backup"
    # what the database holds detects the topic, and a cue of its aspects still blocks it
    assert detect_topic("Is there protection against data loss?", topics) is topics[0]
    assert detect_topic("Is there lost data in the retention period?", topics) is None


async def test_load_topics_is_empty_for_a_taxonomy_without_topics(conn, tmp_path):
    await _load(conn, tmp_path, [])
    assert await load_topics(conn) == []


# --- topic_hits: evidence per aspect within a rerank budget (dim-8 vectors, fakes only) ---

KEEPING = Aspect("keeping", "Keeping copies", "retention period for kept copies", (),
                 ("retention.maximum_period",))
GETTING_BACK = Aspect("getting_back", "Getting data back", "restoring objects from a backup",
                      (), ("coverage.restore", "coverage.export"))
SCHEDULING = Aspect("scheduling", "Scheduling", "backup frequency and schedule", (),
                    ("schedule.backup_frequency",))
TOPIC = Topic("data_loss", "Data loss", ("data loss",), (KEEPING, GETTING_BACK, SCHEDULING))
QUESTION = "How does Alpha Backup handle data loss?"
ALPHA = ("vendor", "Alpha")
BETA = ("vendor", "Beta")

# a theme is a vector axis and the words of its chunks; a text with a keyword embeds on its axis
THEMES = {"loss": (0, "data loss plan"), "retention": (1, "retention period"),
          "restoring": (2, "restoring objects"), "frequency": (3, "frequency schedule")}
OTHER_AXIS = 7
# opposite to the three aspect themes: behind every plain chunk for each aspect's query, so only
# a fact can bring its chunk in
FAR = [0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]


def _unit(axis: int) -> list[float]:
    return [1.0 if n == axis else 0.0 for n in range(8)]


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _score(query: str, doc: str) -> float:
    """Words the query and the passage share, plus one link no word makes (a stand-in for the
    judgement a real reranker has): "ninety days" answers a question about retention."""
    return len(_words(query) & _words(doc)) + (
        1.5 if "ninety" in doc and "retention" in query else 0)


class _Embedder:
    """A text embeds on the axis of the first theme keyword it holds (OTHER_AXIS for none)."""

    def __init__(self):
        self.calls = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append((list(texts), input_type))
        return [_unit(next((axis for key, (axis, _) in THEMES.items() if key in t.lower()),
                           OTHER_AXIS)) for t in texts]


class _Reranker:
    """Scores by ``_score``; ``fail(query)`` true makes that call return None (an outage)."""

    def __init__(self, fail=lambda query: False):
        self.fail = fail
        self.calls = []

    async def rerank(self, query, documents, top_k):
        self.calls.append((query, list(documents), top_k))
        if self.fail(query):
            return None
        scores = [_score(query, d) for d in documents]
        order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in order[:top_k]]


class _Gauge(_Reranker):
    """Remembers the most calls it had in flight at once."""

    def __init__(self):
        super().__init__()
        self.now = self.most = 0

    async def rerank(self, query, documents, top_k):
        self.now += 1
        self.most = max(self.most, self.now)
        try:
            await asyncio.sleep(0.01)  # long enough for every call that may start to start
            return await super().rerank(query, documents, top_k)
        finally:
            self.now -= 1


class _Recorder:
    """A connection that remembers the statements and transactions it was asked to run."""

    def __init__(self, conn):
        self._conn = conn
        self.statements = []
        self.transactions = 0

    async def execute(self, sql, params=None):
        self.statements.append(sql)
        return await self._conn.execute(sql, params)

    def transaction(self):
        self.transactions += 1
        return self._conn.transaction()

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.fixture
async def kb(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        yield conn, await _fresh(conn)
    finally:
        await conn.close()


async def _doc(conn, vendor: str, product: str) -> uuid.UUID:
    doc_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s,%s,%s,'S',%s,%s,%s,%s)",
        (doc_id, vendor, product, f"{product} guide", f"http://docs/{doc_id}", str(doc_id), T0))
    return doc_id


async def _chunk(conn, doc_id: uuid.UUID, text: str, vec: list[float]) -> str:
    chunk_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) SELECT %s, %s, %s, coalesce(max(ordinal) + 1, 0), %s, %s, %s FROM chunks"
        " WHERE doc_id = %s",
        (chunk_id, doc_id, ["A"], text, len(text) // 4, Vector(vec), doc_id))
    return str(chunk_id)


async def _themed(conn, doc_id: uuid.UUID, theme: str, count: int, tag: str) -> list[str]:
    axis, words = THEMES[theme]
    return [await _chunk(conn, doc_id, f"{words} {tag}{n}", _unit(axis)) for n in range(count)]


async def _vendor_corpus(conn, vendor: str, product: str, *, loss=4, per_theme=3) -> uuid.UUID:
    doc_id = await _doc(conn, vendor, product)
    for theme, count in (("loss", loss), ("retention", per_theme), ("restoring", per_theme),
                         ("frequency", per_theme)):
        await _themed(conn, doc_id, theme, count, vendor.lower())
    return doc_id


async def _retext(conn, claim_id: uuid.UUID, text: str, vec: list[float]) -> str:
    """Give the chunk behind a claim (``_claim`` makes it with a fixed text) its own text."""
    cur = await conn.execute(
        "UPDATE chunks SET text = %s, embedding = %s WHERE chunk_id ="
        " (SELECT chunk_id FROM raw_claims WHERE claim_id = %s) RETURNING chunk_id::text",
        (text, Vector(vec), claim_id))
    return (await cur.fetchone())[0]


def _aspects_of(res) -> list[str]:
    return [res.aspect_of[h.chunk_id] for h in res.hits if h.chunk_id in res.aspect_of]


async def test_every_aspect_gets_its_own_passages(kb):
    conn, _ = kb
    await _vendor_corpus(conn, "Alpha", "Alpha Backup")
    await _vendor_corpus(conn, "Beta", "Beta Vault")
    emb, rr = _Embedder(), _Reranker()
    res = await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA], embedder=emb, reranker=rr)

    # one embedder call for the question and every aspect query
    assert emb.calls == [([QUESTION, *(a.query for a in TOPIC.aspects)], "query")]
    # the question's own top passages are all one theme, and none is an aspect passage
    own, rest = res.hits[:3], res.hits[3:]
    assert all("data loss" in h.text for h in own)
    assert not any(h.chunk_id in res.aspect_of for h in own)
    # yet every aspect contributes its quota, in aspect order, from its own pages
    assert [res.aspect_of[h.chunk_id] for h in rest] == [
        "keeping", "keeping", "getting_back", "getting_back", "scheduling", "scheduling"]
    assert [h.text.split()[0] for h in rest] == ["retention"] * 2 + ["restoring"] * 2 + [
        "frequency"] * 2
    assert {h.vendor for h in res.hits} == {"Alpha"}
    assert len({h.chunk_id for h in res.hits}) == len(res.hits)
    assert set(res.aspect_of) <= {h.chunk_id for h in res.hits}
    assert all(h.rerank_score is not None for h in res.hits)
    # the side's name leads the aspect queries sent to the reranker; the question goes as is
    assert {q for q, _, _ in rr.calls} == {QUESTION, *(f"Alpha: {a.query}" for a in
                                                       TOPIC.aspects)}
    assert res.rerank_ok is True
    assert res.reranked_docs == sum(len(docs) for _, docs, _ in rr.calls) > 0


async def test_fact_chunks_join_the_candidates(kb):
    conn, v = kb
    await _vendor_corpus(conn, "Alpha", "Alpha Backup", per_theme=2)
    doc = await _doc(conn, "Alpha", "Alpha Backup")
    for n in range(8):
        await _chunk(conn, doc, f"plain filler words {n}", _unit(OTHER_AXIS))
    fact = await _claim(conn, "retention.maximum_period", vendor="Alpha", product="Alpha Backup",
                        mapped_version=v)
    fact_chunk = await _retext(conn, fact, "ninety days, then it expires", FAR)
    old = await _claim(conn, "retention.maximum_period", vendor="Alpha", product="Alpha Backup",
                       superseded=True, mapped_version=v)
    old_chunk = await _retext(conn, old, "ninety days in the guide's earlier version", FAR)
    await _tail_chunk(conn, await _doc_of(conn, old))  # its page is not retired
    await capabilities.build(conn)
    cur = await conn.execute("SELECT claim_id FROM capabilities WHERE claim_id = ANY(%s)",
                             ([fact, old],))
    assert {r[0] for r in await cur.fetchall()} == {fact, old}

    async def run(reranker):
        # a budget that leaves 7 candidates per aspect of 20 chunks: search alone cannot reach
        # the chunk, whose text shares no word with the query and whose vector is far off
        return await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA], embedder=_Embedder(),
                                reranker=reranker, budget=30, per_aspect=4)

    res = await run(_Reranker())
    ids = [h.chunk_id for h in res.hits]
    assert fact_chunk in ids and res.aspect_of[fact_chunk] == "keeping"
    assert old_chunk not in ids  # it scores as well as the fact chunk: only its page keeps it out
    assert res.rerank_ok and res.reranked_docs <= 30
    # unscored, the fact chunks lead their aspect's candidates, ranked as they stand
    down = await run(_Reranker(fail=lambda query: True))
    keeping = [h for h in down.hits if down.aspect_of.get(h.chunk_id) == "keeping"]
    assert (keeping[0].chunk_id, keeping[0].rrf_rank) == (fact_chunk, 1)
    assert [h.rrf_rank for h in keeping] == [1, 2, 3, 4]

    await conn.execute("DELETE FROM capabilities")  # without its fact, the chunk is not found
    assert fact_chunk not in [h.chunk_id for h in (await run(_Reranker())).hits]


async def test_fact_chunks_select_and_order(kb):
    conn, v = kb
    kw = {"vendor": "Alpha", "product": "P1", "mapped_version": v}
    s_new = await _claim(conn, "retention.maximum_period", capability="a", **kw,
                         fetched_at=T0 + timedelta(days=5))
    s_old = await _claim(conn, "retention.maximum_period", capability="b", **kw)
    # the same capability as s_new from an older page: not the current row of its key
    s_stale = await _claim(conn, "retention.maximum_period", capability="a", **kw,
                           fetched_at=T0 + timedelta(days=1))
    p_new = await _claim(conn, "retention.maximum_period", capability="c", support="partial",
                         **kw, fetched_at=T0 + timedelta(days=10))
    on_restore = await _claim(conn, "coverage.restore", capability="d", **kw)
    other_product = await _claim(conn, "retention.maximum_period", capability="g",
                                 vendor="Alpha", product="P2", mapped_version=v)
    # left out: not supported, another dimension, another vendor, and a chunk that is an
    # earlier version of its page
    await _claim(conn, "retention.maximum_period", capability="e", support="not_supported", **kw)
    await _claim(conn, "coverage.backup", capability="f", **kw)
    await _claim(conn, "retention.maximum_period", capability="h", vendor="Beta", product="Q1",
                 mapped_version=v)
    gone = await _claim(conn, "retention.maximum_period", capability="i", superseded=True, **kw)
    await _tail_chunk(conn, await _doc_of(conn, gone))
    # two facts on one chunk: the chunk comes back once
    await conn.execute("CREATE TEMP TABLE twin AS SELECT * FROM raw_claims WHERE claim_id = %s",
                       (s_old,))
    await conn.execute("UPDATE twin SET claim_id = gen_random_uuid(), capability_raw = 'b2'")
    await conn.execute("INSERT INTO raw_claims SELECT * FROM twin")
    await conn.execute("DROP TABLE twin")
    await capabilities.build(conn)
    cur = await conn.execute("SELECT count(*) FROM capabilities WHERE vendor = 'Alpha'"
                             " AND dimension_id = 'retention.maximum_period'")
    assert (await cur.fetchone())[0] == 8  # the four, the twin, and the other product's, not supported and superseded

    async def chunk_of(claim):
        cur = await conn.execute("SELECT chunk_id::text FROM raw_claims WHERE claim_id = %s",
                                 (claim,))
        return (await cur.fetchone())[0]

    async def got(aspect, facet, limit=10, version=v):
        return [h.chunk_id for h in await fact_chunks(conn, aspect, facet, version, limit)]

    new, old, stale, partial = [await chunk_of(c) for c in (s_new, s_old, s_stale, p_new)]
    # supported before partial, current before superseded, newest page first
    assert await got(KEEPING, ("product", "P1")) == [new, old, stale, partial]
    assert await got(KEEPING, ("product", "P1"), limit=2) == [new, old]
    # a vendor facet takes every product of the vendor
    vendor_wide = await got(KEEPING, ("vendor", "Alpha"))
    assert len(vendor_wide) == 5
    assert set(vendor_wide) == {new, old, stale, partial, await chunk_of(other_product)}
    # several dimensions of an aspect, and a taxonomy version other than the built one
    assert await got(GETTING_BACK, ("vendor", "Alpha")) == [await chunk_of(on_restore)]
    assert await got(KEEPING, ("product", "P1"), version="2-stale000") == []
    hit = (await fact_chunks(conn, KEEPING, ("product", "P1"), v, 1))[0]
    assert (hit.vendor, hit.product, hit.title, hit.url) == ("Alpha", "P1", "Doc", "http://docs/x")


async def test_the_rerank_budget_holds(kb):
    conn, _ = kb
    await _vendor_corpus(conn, "Alpha", "Alpha Backup", loss=10, per_theme=10)
    await _vendor_corpus(conn, "Beta", "Beta Vault", loss=10, per_theme=10)
    rr = _Reranker()
    res = await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA, ("product", "Beta Vault")],
                           embedder=_Embedder(), reranker=rr, budget=120)
    # the own passages' 9 candidates per side count first (18), the 6 (side, aspect) lists
    # share the rest: (120 - 18) // 6 = 17 each
    assert len(rr.calls) == 2 + 6
    assert res.reranked_docs == sum(len(docs) for _, docs, _ in rr.calls) == 18 + 6 * 17
    assert 0 < res.reranked_docs <= 120
    assert all(top_k == len(docs) for _, docs, top_k in rr.calls)
    # own passages in side order, then each aspect's passages side by side, in aspect order
    labelled = [(res.aspect_of[h.chunk_id], h.vendor) for h in res.hits
                if h.chunk_id in res.aspect_of]
    assert [k for k, _ in groupby(labelled)] == [
        (a.id, vendor) for a in TOPIC.aspects for vendor in ("Alpha", "Beta")]
    assert [h.vendor for h in res.hits[:6]] == ["Alpha"] * 3 + ["Beta"] * 3


@pytest.mark.parametrize("facets, spent", [
    ([], 9 + 8 * 28),  # one side: its 9 own candidates, (240 - 9) // 8 = 28 per aspect
    ([ALPHA], 9 + 8 * 28),
    ([ALPHA, BETA], 18 + 16 * 13),  # (240 - 18) // 16 = 13 per (side, aspect)
])
async def test_the_default_budget_holds(kb, facets, spent):
    conn, _ = kb
    eight = Topic("t", "T", ("data loss",), tuple(
        Aspect(f"a{n}", f"Aspect {n}", f"aspect {n} words", (), ("retention.maximum_period",))
        for n in range(8)))
    await _vendor_corpus(conn, "Alpha", "Alpha Backup", loss=10, per_theme=10)
    await _vendor_corpus(conn, "Beta", "Beta Vault", loss=10, per_theme=10)
    assert TOPIC_RERANK_DOCS == 240
    rr = _Reranker()
    res = await topic_hits(conn, QUESTION, eight, facets=facets, embedder=_Embedder(),
                           reranker=rr)
    assert res.reranked_docs == sum(len(docs) for _, docs, _ in rr.calls) == spent
    assert res.reranked_docs <= TOPIC_RERANK_DOCS


async def test_a_two_facet_question_scans_the_text_once(kb):
    conn, v = kb
    await _vendor_corpus(conn, "Alpha", "Alpha Backup")
    await _vendor_corpus(conn, "Beta", "Beta Vault")
    fact = await _claim(conn, "retention.maximum_period", vendor="Alpha", product="Alpha Backup",
                        mapped_version=v)
    await _retext(conn, fact, "ninety days, then it expires", FAR)
    await capabilities.build(conn)
    rec = _Recorder(conn)
    res = await topic_hits(rec, QUESTION, TOPIC, facets=[ALPHA, BETA], embedder=_Embedder(),
                           reranker=_Reranker())
    assert {h.vendor for h in res.hits} == {"Alpha", "Beta"}
    # one full-text statement, the question's own passages' (both sides at once); the aspects'
    # candidates are vector queries (one per aspect and side) in one transaction
    assert sum("ts_rank_cd" in sql for sql in rec.statements) == 1
    assert sum("<=>" in sql for sql in rec.statements) == 2 + 6  # own passages, then aspects
    assert rec.transactions == 2


async def test_aspect_candidates_come_from_vector_search_alone(kb, monkeypatch):
    conn, _ = kb
    monkeypatch.setattr(search, "FTS_WEIGHT", 1.0)  # a full-text match counts as a vector one
    await _vendor_corpus(conn, "Alpha", "Alpha Backup", per_theme=2)
    doc = await _doc(conn, "Alpha", "Alpha Backup")
    for n in range(8):
        await _chunk(conn, doc, f"plain filler words {n}", _unit(OTHER_AXIS))
    # every word of the keeping query, and an embedding behind every plain chunk
    words = "retention period for kept copies, in so many words"
    await _chunk(conn, doc, words, FAR)
    [avec] = (await _Embedder().embed([KEEPING.query]))
    found = await hybrid_candidates(conn, KEEPING.query, avec, vendor=["Alpha"], limit=7)
    assert words in [h.text for h in found]  # a hybrid search for the aspect would bring it in
    rr = _Reranker()
    res = await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA], embedder=_Embedder(),
                           reranker=rr, budget=30)  # 7 candidates per aspect
    aspect_calls = [docs for query, docs, _ in rr.calls if query.startswith("Alpha: ")]
    assert len(aspect_calls) == 3 and not any(words in docs for docs in aspect_calls)
    assert words not in [h.text for h in res.hits]


async def test_at_most_four_reranks_are_in_flight(kb):
    conn, _ = kb
    await _vendor_corpus(conn, "Alpha", "Alpha Backup")
    await _vendor_corpus(conn, "Beta", "Beta Vault")
    gauge = _Gauge()
    res = await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA, BETA], embedder=_Embedder(),
                           reranker=gauge)
    # 2 own lists and 6 (aspect, side) lists: all 8 are reranked, 4 at a time
    assert len(gauge.calls) == 8 and gauge.most == 4
    assert res.rerank_ok and res.reranked_docs == sum(len(d) for _, d, _ in gauge.calls)


async def test_a_budget_the_own_passages_use_up_still_gives_each_list_two(kb):
    conn, _ = kb
    await _vendor_corpus(conn, "Alpha", "Alpha Backup", loss=10, per_theme=10)
    rr = _Reranker()
    res = await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA], embedder=_Embedder(),
                           reranker=rr, budget=10)
    # 9 own candidates leave 1 for 3 aspects: the floor of 2 each applies, over the budget
    assert [len(docs) for _, docs, _ in rr.calls] == [9, 2, 2, 2]
    assert res.reranked_docs == 15


@pytest.mark.parametrize("reranker", [_Reranker(fail=lambda query: True), None])
async def test_topic_hits_without_a_reranker(kb, reranker):
    conn, _ = kb
    await _vendor_corpus(conn, "Alpha", "Alpha Backup")
    emb = _Embedder()
    res = await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA], embedder=emb,
                           reranker=reranker)
    assert res.rerank_ok is False and res.reranked_docs == 0
    assert all(h.rerank_score is None for h in res.hits)
    assert len(res.hits) == 3 + 6
    # every list in the order the search gave it: the question's own passages by hybrid search,
    # each aspect's by nearest chunk (candidates per aspect: (240 - 9) // 3 = 77)
    qvec, *avecs = (await emb.embed([QUESTION, *(a.query for a in TOPIC.aspects)]))
    expected = await hybrid_candidates(conn, QUESTION, qvec, vendor=["Alpha"], limit=9)
    assert [h.chunk_id for h in res.hits[:3]] == [h.chunk_id for h in expected[:3]]
    for aspect, avec in zip(TOPIC.aspects, avecs):
        cur = await conn.execute(
            "SELECT c.chunk_id::text FROM chunks c JOIN documents d USING (doc_id)"
            " WHERE d.vendor = 'Alpha' ORDER BY c.embedding <=> %s, c.chunk_id LIMIT 77",
            (Vector(avec),))
        assert [h.chunk_id for h in res.hits if res.aspect_of.get(h.chunk_id) == aspect.id] == [
            r[0] for r in (await cur.fetchall())[:2]]


async def test_a_failed_rerank_falls_back_for_that_list_only(kb):
    conn, _ = kb
    await _vendor_corpus(conn, "Alpha", "Alpha Backup")
    rr = _Reranker(fail=lambda query: "frequency" in query)
    res = await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA], embedder=_Embedder(),
                           reranker=rr)
    scheduling = [h for h in res.hits if res.aspect_of.get(h.chunk_id) == "scheduling"]
    assert len(scheduling) == 2
    assert all(h.rerank_score is None for h in scheduling)
    assert [h.rrf_rank for h in scheduling] == [1, 2]
    assert all(h.rerank_score is not None for h in res.hits if h not in scheduling)
    assert res.rerank_ok is False
    sent = [len(docs) for query, docs, _ in rr.calls if "frequency" not in query]
    assert res.reranked_docs == sum(sent) > 0  # the failed call's documents are not counted


async def test_no_facet_searches_the_whole_corpus(kb):
    conn, v = kb
    alpha = await _doc(conn, "Alpha", "Alpha Backup")
    await _themed(conn, alpha, "retention", 3, "alpha")
    await _themed(conn, alpha, "restoring", 1, "alpha")
    for n in range(8):
        await _chunk(conn, alpha, f"plain filler words {n}", _unit(OTHER_AXIS))
    beta = await _doc(conn, "Beta", "Beta Vault")
    best_retention = await _chunk(conn, beta, "retention period for kept copies",
                                  _unit(THEMES["retention"][0]))
    await _themed(conn, beta, "restoring", 1, "beta")
    await _themed(conn, beta, "frequency", 1, "beta")
    fact = await _claim(conn, "retention.maximum_period", vendor="Alpha", product="Alpha Backup",
                        mapped_version=v)
    fact_chunk = await _retext(conn, fact, "ninety days, then it expires", FAR)
    await capabilities.build(conn)
    question = "How is data kept safe?"

    async def run(facets, rr=None):
        # 7 candidates per aspect of 20 chunks: only a fact brings in the far-off chunk
        return await topic_hits(conn, question, TOPIC, facets=facets, embedder=_Embedder(),
                                reranker=rr or _Reranker(), budget=30, per_aspect=5)

    # with a side the fact chunk joins the keeping aspect (the control: it can be found) ...
    assert fact_chunk in [h.chunk_id for h in (await run([ALPHA])).hits]
    # ... with none, the whole corpus is searched and no fact chunk is added
    rr = _Reranker()
    res = await run([], rr)
    ids = [h.chunk_id for h in res.hits]
    assert fact_chunk not in ids
    assert {h.vendor for h in res.hits} == {"Alpha", "Beta"}
    assert best_retention in ids and res.aspect_of[best_retention] == "keeping"
    # the question once, and each aspect query as it is (no side name to put before it)
    assert {q for q, _, _ in rr.calls} == {question, *(a.query for a in TOPIC.aspects)}


async def test_a_page_covering_two_aspects_does_not_starve_the_second(kb):
    conn, _ = kb
    doc = await _doc(conn, "Alpha", "Alpha Backup")
    await _chunk(conn, doc, "retention period for kept copies restoring objects from",
                 [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    await _themed(conn, doc, "retention", 2, "alpha")
    await _themed(conn, doc, "restoring", 2, "alpha")
    await _themed(conn, doc, "loss", 3, "alpha")
    # a copy of the same page in another guide: a repeat of the text, kept once
    await _chunk(conn, await _doc(conn, "Alpha", "Alpha Backup"),
                 "retention  period for kept copies restoring objects from",
                 [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA], embedder=_Embedder(),
                           reranker=_Reranker())
    # the page leads both aspects; the first one to choose it keeps it, and the second still
    # gets its own two passages
    pages = [h for h in res.hits if "kept copies restoring objects" in " ".join(h.text.split())]
    assert len(pages) == 1 and res.aspect_of[pages[0].chunk_id] == "keeping"
    assert _aspects_of(res).count("keeping") == 2
    assert _aspects_of(res).count("getting_back") == 2
    assert len({" ".join(h.text.split()) for h in res.hits}) == len(res.hits)


async def test_an_own_passage_chosen_for_an_aspect_keeps_its_place(kb):
    conn, _ = kb
    doc = await _doc(conn, "Alpha", "Alpha Backup")
    axis = _unit(THEMES["retention"][0])
    texts = ["retention period for kept copies", "retention period for kept", "retention period"]
    for text in texts:
        await _chunk(conn, doc, text, axis)
    # the best page again, in another guide
    await _chunk(conn, await _doc(conn, "Alpha", "Alpha Backup"), texts[0], axis)
    await _themed(conn, doc, "restoring", 3, "alpha")
    await _themed(conn, doc, "frequency", 3, "alpha")
    # the question is about retention: its own passages are all the keeping aspect's pages
    res = await topic_hits(conn, "How long is retention for Alpha Backup?", TOPIC,
                           facets=[ALPHA], embedder=_Embedder(), reranker=_Reranker())
    assert {h.text for h in res.hits[:3]} == set(texts)
    # the aspect chose the two best of them: labelled where they stand, not repeated later
    assert {h.text for h in res.hits if res.aspect_of.get(h.chunk_id) == "keeping"} == set(
        texts[:2])
    assert set(res.aspect_of) <= {h.chunk_id for h in res.hits}
    assert len(res.hits) == 3 + 2 + 2  # keeping adds none; the other two aspects add two each
    assert _aspects_of(res) == ["keeping", "keeping", "getting_back", "getting_back",
                                "scheduling", "scheduling"]


async def test_fact_chunks_take_at_most_half_of_a_list(kb):
    conn, v = kb
    await _vendor_corpus(conn, "Alpha", "Alpha Backup", per_theme=2)
    for n in range(5):
        fact = await _claim(conn, "retention.maximum_period", capability=f"c{n}",
                            vendor="Alpha", product="Alpha Backup", mapped_version=v)
        await _retext(conn, fact, f"ninety days, case {n}", FAR)
    await capabilities.build(conn)
    rr = _Reranker()
    # 9 own candidates, then (27 - 9) // 3 = 6 candidates per aspect, 3 of them fact chunks
    res = await topic_hits(conn, QUESTION, TOPIC, facets=[ALPHA], embedder=_Embedder(),
                           reranker=rr, budget=27)
    (keeping,) = [docs for query, docs, _ in rr.calls if "retention" in query]
    assert len(keeping) == 6 and sum("ninety" in d for d in keeping) == 3
    # the other aspects have no facts on their dimensions
    assert all(not any("ninety" in d for d in docs) for query, docs, _ in rr.calls
               if query.startswith("Alpha: ") and "retention" not in query)
    assert res.reranked_docs == 27


# --- the vendor-by-aspect grid ------------------------------------------------------------------

def _day(n: int) -> datetime:
    return T0 + timedelta(days=n)


async def _chunk_of(conn, claim_id: uuid.UUID) -> str:
    cur = await conn.execute("SELECT chunk_id::text FROM raw_claims WHERE claim_id = %s",
                             (claim_id,))
    return (await cur.fetchone())[0]


async def test_grid_cells_take_the_best_fact(kb):
    conn, v = kb
    alpha = {"vendor": "Alpha", "product": "P1", "mapped_version": v}
    beta = {"vendor": "Beta", "product": "Q1", "mapped_version": v}
    # Alpha keeping: supported beats partial however new the partial's page; of the supported
    # facts (each its own key, so each current) the newest page wins
    await _claim(conn, "retention.maximum_period", capability="a1", support="partial",
                 fetched_at=_day(12), **alpha)
    days = {}
    for day in (7, 3, 10, 1, 5, 2, 8, 4, 6, 9):  # ten, so chunk order cannot pass for date
        days[day] = await _claim(conn, "retention.maximum_period", capability=f"a-{day}",
                                 fetched_at=_day(day), **alpha)
    keeping = days[10]
    # Alpha getting_back: b2 is supported on a newer page than b3, but a newer page of its own
    # key (b1, the same capability) makes it the key's old row: the current row of another
    # dimension (b3) wins
    await _claim(conn, "coverage.restore", capability="restoring", support="partial",
                 fetched_at=_day(9), **alpha)
    await _claim(conn, "coverage.restore", capability="restoring", fetched_at=_day(8), **alpha)
    getting_back = await _claim(conn, "coverage.export", capability="b3",
                                fetched_at=_day(1), **alpha)
    # Alpha scheduling: a partial fact is a partial mark
    scheduling = await _claim(conn, "schedule.backup_frequency", capability="a4",
                              support="partial", fetched_at=_day(4), **alpha)
    # Beta getting_back: a current chunk beats a superseded one (an earlier version of its
    # page) whatever the is_current flag and the page date say: c1 is supported on a current
    # chunk but is its key's old row (c3 is newer), c2 is on a superseded chunk of a newer page
    # and the only row of its key
    current_chunk = await _claim(conn, "coverage.restore", capability="restoring",
                                 fetched_at=_day(0), **beta)
    await _claim(conn, "coverage.restore", capability="restoring", support="partial",
                 fetched_at=_day(5), **beta)
    earlier = await _claim(conn, "coverage.export", capability="c2", fetched_at=_day(20),
                           superseded=True, **beta)
    await _tail_chunk(conn, await _doc_of(conn, earlier))  # its page is not retired
    # Beta scheduling: only a fact on a superseded chunk: shown, as from an earlier version
    only_earlier = await _claim(conn, "schedule.backup_frequency", capability="c4",
                                fetched_at=_day(2), superseded=True, **beta)
    await _tail_chunk(conn, await _doc_of(conn, only_earlier))
    # Beta keeping: a fact of another taxonomy version is no fact
    stale = await _claim(conn, "retention.maximum_period", capability="c5", **beta)
    # Gamma has nothing on the topic: a fact that says no, and a dimension the topic lacks
    await _claim(conn, "retention.maximum_period", capability="g1", support="not_supported",
                 vendor="Gamma", product="R1", mapped_version=v)
    await _claim(conn, "coverage.backup", capability="g2", vendor="Gamma", product="R1",
                 mapped_version=v)
    # Delta has facts but is not among the vendors asked for
    await _claim(conn, "retention.maximum_period", capability="d1", vendor="Delta",
                 product="S1", mapped_version=v)
    await capabilities.build(conn)
    await conn.execute("UPDATE capabilities SET taxonomy_version = '2-old' WHERE claim_id = %s",
                       (stale,))

    rec = _Recorder(conn)
    grid = await topic_grid(rec, TOPIC, vendors=["Gamma", "Beta", "Alpha"], embedder=None,
                            reranker=None, catch_up=False)
    assert len([s for s in rec.statements if "FROM capabilities" in s]) == 1  # the facts: one SQL
    assert {k: (c.mark, c.hit.chunk_id, c.superseded) for k, c in grid.cells.items()} == {
        ("Alpha", "keeping"): ("supported", await _chunk_of(conn, keeping), False),
        ("Alpha", "getting_back"): ("supported", await _chunk_of(conn, getting_back), False),
        ("Alpha", "scheduling"): ("partial", await _chunk_of(conn, scheduling), False),
        ("Beta", "getting_back"): ("supported", await _chunk_of(conn, current_chunk), False),
        ("Beta", "scheduling"): ("supported", await _chunk_of(conn, only_earlier), True)}
    # the hit is the chunk as a search returns it; one on a superseded chunk says so
    hit = grid.cells[("Beta", "scheduling")].hit
    assert hit.superseded and (hit.vendor, hit.product) == ("Beta", "Q1")
    assert (hit.title, hit.url, hit.text) == ("Doc", "http://docs/x", "synthetic text")
    assert not grid.cells[("Alpha", "keeping")].hit.superseded
    # a row for each vendor with a fact, the others listed; the date is the newest page of the
    # facts the grid shows (not Alpha's day-12 partial, nor Beta's day-20 earlier version)
    assert grid.vendors == ["Alpha", "Beta"] and grid.no_fact == ["Gamma"]
    assert grid.facts_as_of == _day(10)
    assert grid.reranked_docs == 0 and grid.rerank_ok is True and grid.catch_up_ok is True


async def test_a_grid_of_no_vendors_has_no_rows(kb):
    conn, v = kb
    await _claim(conn, "retention.maximum_period", vendor="Alpha", product="P1", mapped_version=v)
    await capabilities.build(conn)
    for vendors in ([], ["Nobody"]):
        grid = await topic_grid(conn, TOPIC, vendors=vendors, embedder=None, reranker=None,
                                catch_up=False)
        assert (grid.vendors, grid.cells, grid.facts_as_of) == ([], {}, None)
        assert grid.no_fact == vendors
    await conn.execute("DELETE FROM taxonomy_meta")  # no taxonomy loaded: no fact, no error
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha"], embedder=None, reranker=None,
                            catch_up=False)
    assert (grid.vendors, grid.no_fact) == ([], ["Alpha"])


async def _waiting(conn, doc_id: uuid.UUID, text: str, axis: int) -> str:
    """A chunk no extraction has reached."""
    return await _chunk(conn, doc_id, text, _unit(axis))


async def _catch_up_corpus(conn, v):
    """Alpha has a fact for keeping; four unanalysed chunks and one analysed chunk, each a
    theme of its own. Returns the doc ids and the waiting chunk ids by name."""
    fact = await _claim(conn, "retention.maximum_period", vendor="Alpha", product="Alpha Backup",
                        mapped_version=v)
    await capabilities.build(conn)
    alpha = await _doc(conn, "Alpha", "Alpha Product")
    beta = await _doc(conn, "Beta", "Beta Product")
    gamma = await _doc(conn, "Gamma", "Gamma Product")
    waiting = {
        # scores on keeping, but Alpha's keeping cell is a fact: never replaced
        "alpha_keeping": await _waiting(conn, alpha, "retention period of one week", 1),
        "alpha_restore": await _waiting(conn, alpha, "restoring objects from one week", 2),
        "beta_schedule": await _waiting(conn, beta, "frequency schedule hourly", 3),
        # shares no word with any aspect query
        "gamma_noise": await _waiting(conn, gamma, "zzz yyy", OTHER_AXIS)}
    # a perfect page for getting_back, but extracted (no claim): the catch-up does not look
    analysed = await _waiting(conn, beta, "restoring objects from the analysed page", 2)
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s,%s,'m','claims-v3.1','done')", (uuid.uuid4(), analysed))
    return fact, waiting


async def test_catch_up_fills_only_empty_cells(kb):
    conn, v = kb
    fact, waiting = await _catch_up_corpus(conn, v)
    emb, rr = _Embedder(), _Reranker()
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta", "Gamma"], embedder=emb,
                            reranker=rr)
    assert {k: (c.mark, c.hit.chunk_id, c.superseded) for k, c in grid.cells.items()} == {
        ("Alpha", "keeping"): ("supported", await _chunk_of(conn, fact), False),
        ("Alpha", "getting_back"): ("recent", waiting["alpha_restore"], False),
        ("Beta", "scheduling"): ("recent", waiting["beta_schedule"], False)}
    # a recent mark keeps the score that earned it; the date is of facts only
    recent = grid.cells[("Alpha", "getting_back")].hit
    assert recent.rerank_score == 3 and recent.vendor == "Alpha"
    assert grid.vendors == ["Alpha", "Beta"] and grid.no_fact == ["Gamma"]
    assert grid.facts_as_of == T0 and grid.rerank_ok is True
    # one embedding call for the aspect queries; reranked with the aspect query as it is, over
    # the empty cells' candidates only
    assert emb.calls == [([a.query for a in TOPIC.aspects], "query")]
    assert {q for q, _, _ in rr.calls} == {a.query for a in TOPIC.aspects}
    (keeping,) = [docs for q, docs, _ in rr.calls if q == KEEPING.query]
    assert "retention period of one week" not in keeping  # Alpha's cell is filled
    assert not any("analysed" in d for _, docs, _ in rr.calls for d in docs)
    assert grid.reranked_docs == sum(len(docs) for _, docs, _ in rr.calls) > 0


async def test_catch_up_marks_need_the_enumeration_floor(kb, monkeypatch):
    conn, v = kb
    _, waiting = await _catch_up_corpus(conn, v)
    # scores here are counts of shared words: the schedule page scores 2, the restore page 3
    monkeypatch.setattr(topics_mod, "ENUMERATION_MIN_SCORE", 2.5)
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta"], embedder=_Embedder(),
                            reranker=_Reranker())
    assert {k: c.hit.chunk_id for k, c in grid.cells.items() if c.mark == "recent"} == {
        ("Alpha", "getting_back"): waiting["alpha_restore"]}
    monkeypatch.setattr(topics_mod, "ENUMERATION_MIN_SCORE", 2.0)  # at the floor counts
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta"], embedder=_Embedder(),
                            reranker=_Reranker())
    assert ("Beta", "scheduling") in grid.cells


async def test_a_failed_catch_up_rerank_marks_nothing_for_that_aspect(kb):
    conn, v = kb
    _, waiting = await _catch_up_corpus(conn, v)
    rr = _Reranker(fail=lambda query: query == GETTING_BACK.query)
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta"], embedder=_Embedder(),
                            reranker=rr)
    # an unscored page never fills a cell; the other aspect is unharmed
    assert {k for k, c in grid.cells.items() if c.mark == "recent"} == {("Beta", "scheduling")}
    assert grid.rerank_ok is False
    # the failed call is not counted: nothing was scored
    assert grid.reranked_docs == sum(len(docs) for q, docs, _ in rr.calls
                                     if q != GETTING_BACK.query)
    # no reranker at all: no catch-up marks
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta"], embedder=_Embedder(),
                            reranker=None)
    assert [k for k, c in grid.cells.items() if c.mark == "recent"] == []
    assert grid.rerank_ok is False and grid.reranked_docs == 0


async def test_no_catch_up_means_no_search(kb):
    conn, v = kb
    await _catch_up_corpus(conn, v)
    emb, rr = _Embedder(), _Reranker()
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta"], embedder=emb, reranker=rr,
                            catch_up=False)
    assert emb.calls == [] and rr.calls == []
    assert list(grid.cells) == [("Alpha", "keeping")] and grid.rerank_ok is True
    assert grid.catch_up_ok is True  # not asked for is not failed


@pytest.mark.parametrize("aspects, per_list", [(8, 20), (13, 18)])
async def test_the_catch_up_rerank_spend_is_capped(kb, aspects, per_list):
    conn, _ = kb
    # two vendors with 15 waiting chunks each: every aspect has more candidates than it may send
    for vendor in ("Alpha", "Beta"):
        doc = await _doc(conn, vendor, f"{vendor} Product")
        for n in range(15):
            await _waiting(conn, doc, f"plain filler words {n}", OTHER_AXIS)
    topic = Topic("big", "Big", ("big",), tuple(
        Aspect(f"a{n}", f"Aspect {n}", KEEPING.query, (), ("retention.maximum_period",))
        for n in range(aspects)))
    gauge = _Gauge()
    grid = await topic_grid(conn, topic, vendors=["Alpha", "Beta"], embedder=_Embedder(),
                            reranker=gauge)
    assert [len(docs) for _, docs, _ in gauge.calls] == [per_list] * aspects
    assert grid.reranked_docs == aspects * per_list <= TOPIC_RERANK_DOCS
    assert gauge.most == 4  # the shared limit on rerank calls in flight
    assert grid.cells == {} and grid.no_fact == ["Alpha", "Beta"]  # the filler scores nothing


class _DownEmbedder:
    """An embedder that fails with ``error``."""

    def __init__(self, error):
        self.error = error
        self.calls = 0

    async def embed(self, texts, *, input_type="document"):
        self.calls += 1
        raise self.error


async def test_an_embedder_outage_leaves_the_grid_of_facts(kb, caplog):
    conn, v = kb
    fact, _ = await _catch_up_corpus(conn, v)
    rr = _Reranker()
    down = _DownEmbedder(httpx.ConnectError("embedder is down"))
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta", "Gamma"], embedder=down,
                            reranker=rr)
    # the facts alone, no search: no recent cell, no rerank, and the flag says why
    assert {k: (c.mark, c.hit.chunk_id) for k, c in grid.cells.items()} == {
        ("Alpha", "keeping"): ("supported", await _chunk_of(conn, fact))}
    assert (grid.vendors, grid.no_fact) == (["Alpha"], ["Beta", "Gamma"])
    assert grid.catch_up_ok is False and grid.facts_as_of == T0
    assert down.calls == 1 and rr.calls == [] and grid.reranked_docs == 0
    assert grid.rerank_ok is True  # the reranker was not at fault
    assert "catch-up skipped" in caplog.text and "embedder is down" in caplog.text
    # an answer that is no outage (a wrong count or dimension) is a defect: it still raises
    with pytest.raises(RuntimeError, match="dimension"):
        await topic_grid(conn, TOPIC, vendors=["Alpha"], reranker=rr,
                         embedder=_DownEmbedder(RuntimeError("embedding has dimension 3")))


async def test_catch_up_looks_past_the_pages_of_vendors_with_a_fact(kb):
    conn, v = kb
    await _claim(conn, "retention.maximum_period", vendor="Alpha", product="Alpha Backup",
                 mapped_version=v)
    await capabilities.build(conn)
    alpha = await _doc(conn, "Alpha", "Alpha Product")
    beta = await _doc(conn, "Beta", "Beta Product")
    # 25 waiting pages of Alpha (whose keeping cell a fact fills) lead both rankings for the
    # keeping query; the one page of Beta is farther in the vectors and ranks below them in text
    for n in range(25):
        await _waiting(conn, alpha, f"retention period for kept copies {n}", 1)
    wanted = await _chunk(conn, beta, "retention period of one week",
                          [0.0, 0.6, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0])
    rr = _Reranker()
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta"], embedder=_Embedder(),
                            reranker=rr)
    recent = {k: c.hit.chunk_id for k, c in grid.cells.items() if c.mark == "recent"}
    assert recent == {("Beta", "keeping"): wanted}
    assert grid.cells[("Alpha", "keeping")].mark == "supported"
    # Alpha's pages are dropped before the cut to 20, not after: only Beta's page is reranked
    # for keeping, and no call sends more than 20 documents
    (keeping,) = [docs for q, docs, _ in rr.calls if q == KEEPING.query]
    assert keeping == ["retention period of one week"]
    assert max(len(docs) for _, docs, _ in rr.calls) == 20
    assert grid.reranked_docs == sum(len(docs) for _, docs, _ in rr.calls)


async def test_catch_up_with_nothing_waiting_changes_nothing(kb):
    conn, v = kb
    await _claim(conn, "retention.maximum_period", vendor="Alpha", product="Alpha Backup",
                 mapped_version=v)
    await _claim(conn, "coverage.restore", vendor="Beta", product="Beta Vault", mapped_version=v,
                 support="partial")
    await capabilities.build(conn)  # every chunk is extracted: the pool is empty
    rr = _Reranker()
    plain = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta"], embedder=None,
                             reranker=None, catch_up=False)
    grid = await topic_grid(conn, TOPIC, vendors=["Alpha", "Beta"], embedder=_Embedder(),
                            reranker=rr)
    assert rr.calls == [] and grid.reranked_docs == 0
    assert grid.rerank_ok is True and grid.catch_up_ok is True
    assert grid.cells == plain.cells and grid.vendors == plain.vendors
    assert (grid.no_fact, grid.facts_as_of) == (plain.no_fact, plain.facts_as_of)


async def test_a_dimension_two_aspects_list_fills_both_cells(kb):
    conn, v = kb
    overlap = Topic("overlap", "Overlap", ("overlap",), (
        Aspect("first", "First", "q one", (), ("retention.maximum_period",)),
        Aspect("second", "Second", "q two", (), ("coverage.restore", "retention.maximum_period"))))
    both = await _claim(conn, "retention.maximum_period", vendor="Alpha", product="P1",
                        mapped_version=v)
    await _claim(conn, "coverage.restore", vendor="Alpha", product="P1", mapped_version=v,
                 support="partial")
    await _claim(conn, "coverage.restore", vendor="Beta", product="Q1", mapped_version=v,
                 support="partial")
    await capabilities.build(conn)
    grid = await topic_grid(conn, overlap, vendors=["Alpha", "Beta"], embedder=None,
                            reranker=None, catch_up=False)
    chunk = await _chunk_of(conn, both)
    # one fact fills the cell of each aspect that lists its dimension (and, supported, it beats
    # the second aspect's partial fact on its other dimension)
    assert {k: (c.mark, c.hit.chunk_id) for k, c in grid.cells.items()} == {
        ("Alpha", "first"): ("supported", chunk), ("Alpha", "second"): ("supported", chunk),
        ("Beta", "second"): ("partial", grid.cells[("Beta", "second")].hit.chunk_id)}
    text, hits = render_grid(grid, overlap)
    assert "| Alpha | ✓ [1] | ✓ [2] |" in text.splitlines()  # a number for each cell
    assert [h.chunk_id for h in hits[:2]] == [chunk, chunk]


def _hit(n: int, vendor: str = "Alpha") -> Hit:
    return Hit(chunk_id=f"c{n}", doc_id=f"d{n}", text=f"text {n}", section_path=["A"],
               vendor=vendor, product="P", title=f"Title {n}", url=f"http://docs/{n}",
               rrf_rank=0, rerank_score=None)


LEGEND = ("✓ supported · ◐ partial · ○ found in a page not yet "
          "analysed (not verified) · † from an earlier version of the page.")
BLANK_NOTE = "A blank cell means no fact was found, not that the product lacks the capability."


def test_render_grid():
    cells = {
        ("Alpha", "keeping"): Cell("supported", _hit(1), False),
        ("Alpha", "getting_back"): Cell("partial", _hit(2), False),
        ("Zed", "keeping"): Cell("supported", _hit(3, "Zed"), True),
        ("Zed", "scheduling"): Cell("recent", _hit(4, "Zed"), False),
        ("Beta", "getting_back"): Cell("recent", _hit(5, "Beta"), False)}
    # rows go by marks, then name; the vendors come in no order of their own
    grid = Grid(vendors=["Beta", "Zed", "Alpha"], cells=cells,
                facts_as_of=datetime(2026, 11, 5, 13, 30, tzinfo=UTC), no_fact=["Delta", "Gamma"])
    text, hits = render_grid(grid, TOPIC)
    assert text == "\n".join([
        "| Vendor | Keeping copies | Getting data back | Scheduling |",
        "| --- | --- | --- | --- |",
        "| Alpha | ✓ [1] | ◐ [2] |  |",
        "| Zed | ✓ [3]† |  | ○ [4] |",
        "| Beta |  | ○ [5] |  |",
        "",
        LEGEND,
        f"Facts current to 2026-11-05. {BLANK_NOTE}",
        "No fact found for: Delta, Gamma."])
    # [n] is hits[n - 1]: row by row, aspect by aspect
    assert [h.chunk_id for h in hits] == ["c1", "c2", "c3", "c4", "c5"]
    # no vendor left out: no such line; no fact cell: no date
    only_recent = Grid(vendors=["Beta"], cells={("Beta", "getting_back"): cells[
        ("Beta", "getting_back")]}, facts_as_of=None, no_fact=[])
    text, hits = render_grid(only_recent, TOPIC)
    assert text.splitlines()[-2:] == [LEGEND, f"Facts current to: no facts found. {BLANK_NOTE}"]
    assert "No fact found for" not in text and [h.chunk_id for h in hits] == ["c5"]


def test_render_grid_escapes_pipes_in_names():
    grid = Grid(vendors=["A|B"], cells={("A|B", "keeping"): Cell("partial", _hit(1), False)},
                facts_as_of=T0, no_fact=[])
    topic = Topic("t", "T", (), (Aspect("keeping", "Keep | copy", "q", (), ()),))
    text, _ = render_grid(grid, topic)
    assert text.splitlines()[:3] == ["| Vendor | Keep \\| copy |", "| --- | --- |",
                                     "| A\\|B | ◐ [1] |"]


def test_a_grid_with_no_marks_is_a_refusal_not_an_empty_table():
    grid = Grid(vendors=[], cells={}, facts_as_of=None, no_fact=["Alpha", "Beta"])
    text, hits = render_grid(grid, TOPIC)
    assert hits == [] and "| Vendor" not in text
    assert text.splitlines() == [
        "No fact was found for data loss in the documented vendors' pages.", "", LEGEND,
        f"Facts current to: no facts found. {BLANK_NOTE}", "No fact found for: Alpha, Beta."]
