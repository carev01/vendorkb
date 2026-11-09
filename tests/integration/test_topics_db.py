import asyncio
import re
import uuid
from datetime import timedelta
from itertools import groupby
from pathlib import Path

import pytest
import yaml
from pgvector import Vector

from tests.integration.test_capabilities_v2 import T0, _claim, _doc_of, _fresh, _tail_chunk
from vendorkb import search
from vendorkb.db import connect, init_schema
from vendorkb.search import hybrid_candidates
from vendorkb.taxonomy import capabilities, load
from vendorkb.taxonomy.load import Aspect, Topic
from vendorkb.topics import (
    TOPIC_RERANK_DOCS, detect_topic, fact_chunks, load_topics, topic_hits)

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
