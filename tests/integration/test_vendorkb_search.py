import uuid

import pytest

from pgvector import Vector

from vendorkb.db import connect, init_schema
from vendorkb import search
from vendorkb.modes import compare_facets, enumerate_vendors
from vendorkb.search import hybrid_candidates, read_section, search_docs, vendor_candidates


@pytest.fixture(autouse=True)
def _equal_fusion_weights(monkeypatch):
    """These tests pin the fusion mechanics, written for equal FTS/vector weights; the shipped
    FTS_WEIGHT is a tuning value (tests that vary it set it themselves)."""
    monkeypatch.setattr(search, "FTS_WEIGHT", 1.0)


def _vec(i):
    v = [0.0] * 768
    v[i] = 1.0
    return v


class FakeEmbedder:
    def __init__(self, vec):
        self.vec = vec
        self.calls = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append((texts, input_type))
        return [self.vec for _ in texts]


class FakeReranker:
    def __init__(self, scores=None, fail=False):
        self.scores = scores
        self.fail = fail
        self.calls = []

    async def rerank(self, query, documents, top_k):
        self.calls.append((query, list(documents), top_k))
        if self.fail:
            return None
        order = sorted(range(len(documents)), key=lambda i: -self.scores(documents[i]))
        return [(i, self.scores(documents[i])) for i in order[:top_k]]


async def _seed(conn):
    await conn.execute("DELETE FROM documents")
    ids = {}
    docs = [("V1", "P1", "Doc one"), ("V2", "P2", "Doc two")]
    for vendor, product, title in docs:
        did = str(uuid.uuid4())
        ids[vendor] = did
        await conn.execute(
            "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
            " fetched_at) VALUES (%s,%s,%s,'S',%s,%s,'h', now())",
            (did, vendor, product, title, f"http://u/{vendor}"))
    return ids


async def _chunk(conn, doc, ordinal, text, vec_i, path=("A",)):
    cid = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (cid, doc, list(path), ordinal, text, len(text) // 4, Vector(_vec(vec_i))))
    return cid


async def test_search_merges_fts_and_vector_and_reranks(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        fts_only = await _chunk(conn, ids["V1"], 0, "zebra stripes unique keyword", 5)
        vec_only = await _chunk(conn, ids["V1"], 1, "completely other words", 0)
        await _chunk(conn, ids["V1"], 2, "filler content", 7)
        for n in range(60):
            v = _vec(0)
            v[1] = 0.1  # nearer the query than the keyword chunk, farther than vec_only
            await conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
                " embedding) VALUES (%s,%s,%s,%s,%s,1,%s)",
                (str(uuid.uuid4()), ids["V2"], ["D"], n, f"distractor {n}", Vector(v)))
        rr = FakeReranker(lambda t: 0.9 if "other" in t else 0.5)
        res = await search_docs(conn, "zebra", embedder=FakeEmbedder(_vec(0)), reranker=rr, k=5)
        got = [h.chunk_id for h in res.hits]
        assert fts_only in got and vec_only in got
        assert got[0] == vec_only and got[1] == fts_only
        cur = await conn.execute(
            "SELECT chunk_id::text FROM chunks ORDER BY embedding <=> %s, chunk_id LIMIT 50",
            (Vector(_vec(0)),))
        assert fts_only not in [r[0] for r in await cur.fetchall()]
        assert res.rerank_ok is True
        assert res.hits[0].rerank_score == 0.9 and res.hits[0].rrf_rank >= 1
        assert res.hits[0].vendor == "V1" and res.hits[0].url == "http://u/V1"
    finally:
        await conn.close()


async def test_filtered_vector_search_is_not_starved_by_hnsw(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        for n in range(2000):
            v = _vec(0)
            v[10 + n % 700] = 0.1
            cid = str(uuid.uuid4())
            await conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
                " embedding) VALUES (%s,%s,%s,%s,%s,1,%s)",
                (cid, ids["V1"], ["A"], n, f"aaa {n}", Vector(v)))
        b = []
        for n in range(5):
            v = [0.0] * 768
            v[0] = -1.0
            v[3 + n] = 1.0
            cid = str(uuid.uuid4())
            await conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
                " embedding) VALUES (%s,%s,%s,%s,%s,1,%s)",
                (cid, ids["V2"], ["B"], n, f"bbb {n}", Vector(v)))
            b.append(cid)
        await conn.execute("ANALYZE")
        await conn.execute("SET enable_seqscan = off")
        await conn.execute("SET enable_sort = off")
        await conn.execute("SET enable_hashjoin = off")
        await conn.execute("SET enable_mergejoin = off")
        rr = FakeReranker(lambda t: 0.9)
        res = await search_docs(conn, "qqqabsent", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                vendor=["V2"], k=10)
        assert sorted(h.chunk_id for h in res.hits) == sorted(b)
    finally:
        await conn.close()


async def test_empty_rerank_reply_falls_back(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        await _chunk(conn, ids["V1"], 0, "alpha text", 0)

        class Empty(FakeReranker):
            async def rerank(self, query, documents, top_k):
                return []

        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                reranker=Empty(lambda t: 0), k=3)
        assert res.rerank_ok is False and len(res.hits) == 1
    finally:
        await conn.close()


async def test_vendor_filter_restricts_hits(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        await _chunk(conn, ids["V1"], 0, "alpha text", 0)
        await _chunk(conn, ids["V2"], 0, "alpha text", 0)
        rr = FakeReranker(lambda t: 0.9)
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                vendor=["V2"])
        assert [h.vendor for h in res.hits] == ["V2"]
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                product=["P2"])
        assert [h.vendor for h in res.hits] == ["V2"]
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                vendor=[])
        assert len(res.hits) == 2
    finally:
        await conn.close()


async def test_use_rerank_false_makes_no_reranker_call(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        await _chunk(conn, ids["V1"], 0, "alpha text", 0)
        rr = FakeReranker(lambda t: 0.9)
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                use_rerank=False)
        assert rr.calls == [] and res.rerank_ok is False and len(res.hits) == 1
        assert res.hits[0].rerank_score is None
    finally:
        await conn.close()


async def test_rerank_outage_falls_back_to_rrf_order(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        for i in range(5):
            await _chunk(conn, ids["V1"], i, f"alpha item {i}", i)
        rr = FakeReranker(fail=True)
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr, k=3)
        assert res.rerank_ok is False and len(res.hits) == 3
        assert [h.rrf_rank for h in res.hits] == [1, 2, 3]
        assert all(h.rerank_score is None for h in res.hits)
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr, k=50)
        assert len(res.hits) == 5
    finally:
        await conn.close()


async def test_read_section_expand_modes(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        c = [await _chunk(conn, ids["V1"], 0, "zero", 0, ("A",)),
             await _chunk(conn, ids["V1"], 1, "one", 1, ("B",)),
             await _chunk(conn, ids["V1"], 2, "two", 2, ("B",)),
             await _chunk(conn, ids["V1"], 3, "three", 3, ("C",)),
             await _chunk(conn, ids["V1"], 4, "four", 4, ("B",))]
        h0 = await read_section(conn, c[2])
        assert [h.chunk_id for h in h0] == [c[2]]
        assert h0[0].rrf_rank == 0 and h0[0].rerank_score is None
        h1 = await read_section(conn, c[2], "1")
        assert [h.chunk_id for h in h1] == [c[1], c[2], c[3]]
        hf = await read_section(conn, c[2], "full")
        assert [h.chunk_id for h in hf] == [c[1], c[2], c[4]]
        assert await read_section(conn, str(uuid.uuid4())) == []
    finally:
        await conn.close()


async def test_scope_index_from_db_fills_source_products(vkb_dsn):
    from vendorkb.scope import scope_index_from_db
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        await _seed(conn)
        idx = await scope_index_from_db(conn)
        assert idx.vendors == ["V1", "V2"]
        assert idx.products == ["P1", "P2"]
        assert idx.source_products == {"S": ["P1", "P2"]}
        assert idx.source_vendors == {}  # "S" is shared by V1 and V2, so not unique
        assert "VBR" in idx.aliases
    finally:
        await conn.close()


async def test_vendor_and_product_filters_are_ored(vkb_dsn):
    """answer-api scopes with OR: vendor named OR product named (I1)."""
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        await _chunk(conn, ids["V1"], 0, "alpha text", 0)
        await _chunk(conn, ids["V2"], 0, "alpha text", 0)
        rr = FakeReranker(lambda t: 0.9)
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                vendor=["V1"], product=["P2"])
        assert sorted(h.vendor for h in res.hits) == ["V1", "V2"]
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                vendor=["V1"], product=["P1"])
        assert [h.vendor for h in res.hits] == ["V1"]
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                vendor=["V9"], product=["P9"])
        assert res.hits == []
    finally:
        await conn.close()


async def test_fts_matches_a_question_sharing_only_some_terms(vkb_dsn):
    """A natural question ANDs to nothing; FTS must OR its terms (C1)."""
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        kw = await _chunk(conn, ids["V1"], 0,
                          "Exchange mailbox retention is configured per database.", 5)
        for n in range(60):
            v = _vec(0)
            v[1] = 0.1
            await conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
                " embedding) VALUES (%s,%s,%s,%s,%s,1,%s)",
                (str(uuid.uuid4()), ids["V2"], ["D"], n, f"distractor {n}", Vector(v)))
        q = "How long can my backup job keep Exchange mailboxes under retention?"
        res = await search_docs(conn, q, embedder=FakeEmbedder(_vec(0)), reranker=None,
                                use_rerank=False, k=40)
        assert kw in [h.chunk_id for h in res.hits]
        assert res.hits[0].chunk_id == kw  # FTS rank 1 ties vector rank 1, FTS listed first
    finally:
        await conn.close()


async def test_stopword_only_query_matches_nothing_by_fts(vkb_dsn):
    """An empty tsquery must not error and must match no chunk (C1 guard)."""
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        ids = await _seed(conn)
        await _chunk(conn, ids["V1"], 0, "the and of", 3)
        res = await search_docs(conn, "the of and", embedder=FakeEmbedder(_vec(0)),
                                reranker=None, use_rerank=False, k=5)
        # no error; at most the vector side's one hit (HNSW may skip it after the
        # previous test's deletes, so the count is not asserted exactly)
        assert len(res.hits) <= 1
    finally:
        await conn.close()


async def _doc(conn, vendor, product, title="T"):
    did = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s,%s,%s,'S',%s,%s,'h', now())",
        (did, vendor, product, title, f"http://u/{vendor}/{did}"))
    return did


async def _fresh(conn):
    await init_schema(conn, embed_dim=768)
    await conn.execute("DELETE FROM documents")


async def test_hybrid_candidates_reads_fts_weight_at_call_time(vkb_dsn, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        d1, d2 = await _doc(conn, "V1", "P1"), await _doc(conn, "V2", "P2")
        kw = await _chunk(conn, d1, 0, "Exchange mailbox retention keyword", 5)
        near = await _chunk(conn, d2, 0, "unrelated words", 0)
        got = await hybrid_candidates(conn, "retention", _vec(0))
        assert [h.chunk_id for h in got] == [kw, near]  # FTS rank 1 ties vector rank 1
        assert [h.rrf_rank for h in got] == [1, 2] and got[0].rerank_score is None
        monkeypatch.setattr(search, "FTS_WEIGHT", 0.0)
        got = await hybrid_candidates(conn, "retention", _vec(0))
        assert got[0].chunk_id == near
        got = await hybrid_candidates(conn, "retention", _vec(0), vendor=["V1"], limit=1)
        assert [h.chunk_id for h in got] == [kw]
    finally:
        await conn.close()


async def test_enumeration_covers_every_vendor(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        for v in ("VA", "VB", "VC"):
            d = await _doc(conn, v, f"{v}-P")
            await _chunk(conn, d, 0, f"alpha backup {v}", 0)
        await _doc(conn, "VD", "VD-P")  # a document with no chunks
        emb = FakeEmbedder(_vec(0))
        res = await enumerate_vendors(conn, "alpha", embedder=emb,
                                      reranker=FakeReranker(lambda t: 0.9),
                                      vendors=["VA", "VB", "VC", "VD", "VE"])
        assert sorted(res.kept + res.dropped + res.empty) == ["VA", "VB", "VC", "VD", "VE"]
        assert len(res.kept + res.dropped + res.empty) == 5
        assert res.empty == ["VD", "VE"] and sorted(res.kept) == ["VA", "VB", "VC"]
        assert len(emb.calls) == 1 and res.rerank_ok is True
        assert sorted(h.vendor for h in res.hits) == ["VA", "VB", "VC"]
    finally:
        await conn.close()


def _graded(rank):
    """A vector whose distance to _vec(0) grows with ``rank``: a deterministic vector order."""
    v = _vec(0)
    v[1] = 0.05 * rank
    return v


async def _ranked_chunk(conn, doc, rank, text):
    """A chunk at vector rank ``rank`` whose text matches no query word used with it."""
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s,%s,%s,%s,%s,1,%s)",
        (str(uuid.uuid4()), doc, ["A"], rank, text, Vector(_graded(rank))))


async def _ranked_vendor(conn, vendor, layout):
    """``layout`` lists (product, n_chunks) in rank order: the vendor's candidates come back
    in exactly this order (no FTS match, graded vectors)."""
    rank = 0
    for product, n in layout:
        d = await _doc(conn, vendor, product)
        for _ in range(n):
            await _ranked_chunk(conn, d, rank, f"{vendor} {product} r{rank}")
            rank += 1


async def test_enumeration_per_vendor_cap(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _ranked_vendor(conn, "VS", [("Only", 10)])
        # the top product holds the best ranks, so a plain top-5 would miss P2/P3
        await _ranked_vendor(conn, "VM3", [("P1", 4), ("P2", 1), ("P3", 1)])
        # more products than the cap: the best chunk of each of the 5 best products
        await _ranked_vendor(conn, "VM6", [("P1", 5), ("P2", 1), ("P3", 1), ("P4", 1),
                                          ("P5", 1), ("P6", 1)])
        rr = FakeReranker(lambda t: 0.9)
        res = await enumerate_vendors(conn, "qqqabsent", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=["VS", "VM3", "VM6"],
                                      per_vendor=3, per_vendor_multi=5)
        by = {v: [h for h in res.hits if h.vendor == v] for v in ("VS", "VM3", "VM6")}
        assert len(by["VS"]) == 3
        assert len(by["VM3"]) == 5 and {h.product for h in by["VM3"]} == {"P1", "P2", "P3"}
        assert sorted(h.text for h in by["VM6"]) == [
            "VM6 P1 r0", "VM6 P2 r5", "VM6 P3 r6", "VM6 P4 r7", "VM6 P5 r8"]
        assert len(rr.calls) == 1 and len(rr.calls[0][1]) == 13  # one rerank over all kept
    finally:
        await conn.close()


async def test_enumeration_fallback_truncation_keeps_product_coverage(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _ranked_vendor(conn, "VM", [("P1", 4), ("P2", 1), ("P3", 1)])
        await _ranked_vendor(conn, "VS", [("Only", 3)])
        res = await enumerate_vendors(conn, "qqqabsent", embedder=FakeEmbedder(_vec(0)),
                                      reranker=None, vendors=["VM", "VS"], max_hits=5)
        vm = [h for h in res.hits if h.vendor == "VM"]
        assert len(res.hits) == 5 and res.rerank_ok is False
        assert [h.product for h in vm] == ["P1", "P2", "P3"]
    finally:
        await conn.close()


async def test_enumeration_floor_drops_low_vendor(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        for v, score in (("VA", 0.9), ("VB", 0.5), ("VE", 0.3), ("VC", 0.1)):
            d = await _doc(conn, v, f"{v}-P")
            for i in range(2):
                await _chunk(conn, d, i, f"alpha {v} s{score}", 0)
        rr = FakeReranker(lambda t: float(t.split(" s")[-1]))
        # passed in reverse score order: kept must come back ordered by best score, and a
        # vendor scoring exactly the floor survives
        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=["VC", "VE", "VB", "VA"],
                                      min_score=0.3)
        assert res.kept == ["VA", "VB", "VE"] and res.dropped == ["VC"] and res.empty == []
        assert [h.vendor for h in res.hits] == ["VA", "VA", "VB", "VB", "VE", "VE"]
        assert res.hits[0].rerank_score == 0.9 and res.rerank_ok is True
    finally:
        await conn.close()


class CountingConn:
    """Counts statements sent through ``execute``; everything else is the real connection."""

    def __init__(self, conn):
        self._conn = conn
        self.executes = 0

    async def execute(self, *args, **kwargs):
        self.executes += 1
        return await self._conn.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


async def test_enumeration_statement_count_is_independent_of_vendor_count(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        vendors = [f"V{n}" for n in range(6)]
        for v in vendors:
            d = await _doc(conn, v, f"{v}-P")
            for i in range(12):  # more than per_vendor_multi * 2: no vector top-up needed
                await _chunk(conn, d, i, f"alpha {v} {i}", 0)
        rr = FakeReranker(lambda t: 0.9)
        counts = []
        for vs in (vendors[:1], vendors[:2], vendors):
            cc = CountingConn(conn)
            res = await enumerate_vendors(cc, "alpha", embedder=FakeEmbedder(_vec(0)),
                                          reranker=rr, vendors=vs)
            assert res.kept == vs and len(res.hits) == 3 * len(vs)
            counts.append(cc.executes)
        # FTS + (SET LOCAL + pool query) + text fetch, + extversion once per connection
        assert counts[0] == counts[1] == counts[2] <= 5
    finally:
        await conn.close()


async def test_enumeration_tops_up_vendors_short_of_vector_rows(vkb_dsn, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        big = await _doc(conn, "VB", "VB-P")
        for i in range(15):
            await _chunk(conn, big, i, f"big {i}", 0)
        small = await _doc(conn, "VS", "VS-P")
        small_ids = []
        for i in range(2):  # far from the query and no keyword match: outside the pool
            v = [0.0] * 768
            v[0] = -1.0
            v[5 + i] = 1.0
            cid = str(uuid.uuid4())
            await conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
                " embedding) VALUES (%s,%s,%s,%s,%s,1,%s)",
                (cid, small, ["A"], i, f"small {i}", Vector(v)))
            small_ids.append(cid)
        monkeypatch.setattr(search, "ENUM_VEC_POOL", 10)
        rr = FakeReranker(lambda t: 0.9)
        base = CountingConn(conn)
        await enumerate_vendors(base, "qqqabsent", embedder=FakeEmbedder(_vec(0)),
                                reranker=rr, vendors=["VB"])
        cc = CountingConn(conn)
        res = await enumerate_vendors(cc, "qqqabsent", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=["VB", "VS"])
        assert sorted(res.kept) == ["VB", "VS"] and res.empty == []
        assert sorted(h.chunk_id for h in res.hits if h.vendor == "VS") == sorted(small_ids)
        assert cc.executes - base.executes <= 2  # one scoped vector query (+ SET LOCAL)
    finally:
        await conn.close()


async def test_enumeration_edge_inputs(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        d = await _doc(conn, "VA", "VA-P")
        await _chunk(conn, d, 0, "alpha VA", 0)
        rr = FakeReranker(lambda t: 0.9)
        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=["VA", "VA", "VX", "VX"])
        assert res.kept == ["VA"] and res.empty == ["VX"] and len(res.hits) == 1
        # nothing to rerank is not a reranker failure
        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=["VX", "VY"])
        assert res.empty == ["VX", "VY"] and res.hits == [] and res.rerank_ok is True
        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=["VX"], use_rerank=False)
        assert res.rerank_ok is False
    finally:
        await conn.close()


async def test_enumeration_partial_rerank_reply_keeps_unscored_vendor(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        for v in ("VA", "VB"):
            d = await _doc(conn, v, f"{v}-P")
            for i in range(2):
                await _chunk(conn, d, i, f"alpha {v} {i}", 0)

        class Partial(FakeReranker):
            async def rerank(self, query, documents, top_k):
                out = [(i, 0.8) for i, t in enumerate(documents) if "VA" in t]
                return out + out[:1]  # VB unscored, and one index repeated

        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=Partial(), vendors=["VB", "VA"],
                                      min_score=0.5)
        assert res.kept == ["VA", "VB"] and res.dropped == [] and res.rerank_ok is False
        assert [h.vendor for h in res.hits] == ["VA", "VA", "VB", "VB"]
        assert len({h.chunk_id for h in res.hits}) == 4
        assert [h.rerank_score for h in res.hits if h.vendor == "VB"] == [None, None]
    finally:
        await conn.close()


async def test_enumeration_rerank_failure_keeps_all(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        for v in ("VA", "VB"):
            d = await _doc(conn, v, f"{v}-P")
            for i in range(4):
                await _chunk(conn, d, i, f"alpha {v} {i}", 0)
        for reranker, use_rerank in ((FakeReranker(fail=True), True),
                                     (None, True), (FakeReranker(lambda t: 0.0), False)):
            res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                          reranker=reranker, vendors=["VA", "VB", "VX"],
                                          min_score=0.3, use_rerank=use_rerank)
            assert res.kept == ["VA", "VB"] and res.dropped == [] and res.empty == ["VX"]
            assert res.rerank_ok is False
            assert all(h.rerank_score is None for h in res.hits)
            for v in ("VA", "VB"):
                assert [h.rrf_rank for h in res.hits if h.vendor == v] == [1, 2, 3]
        assert reranker.calls == []  # use_rerank=False makes no reranker call
    finally:
        await conn.close()


async def test_enumeration_max_hits(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        vendors = [f"V{n:02d}" for n in range(30)]
        for n, v in enumerate(vendors):
            d = await _doc(conn, v, f"{v}-P")
            for i in range(3):
                await _chunk(conn, d, i, f"alpha {v} c{i} s{0.9 - n / 100 - i / 1000}", 0)
        rr = FakeReranker(lambda t: float(t.split(" s")[-1]))
        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=vendors, max_hits=60)
        assert len(res.hits) == 60 and sorted(res.kept) == vendors
        counts = {v: sum(h.vendor == v for h in res.hits) for v in vendors}
        assert all(c >= 1 for c in counts.values())
        assert all(" c2 " not in h.text for h in res.hits)  # each vendor's worst chunk went
        assert res.kept == vendors  # ordered by best score
        assert [h.vendor for h in res.hits[:2]] == ["V00", "V00"]
    finally:
        await conn.close()


async def test_comparison_quota_survives_rerank(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        da, db = await _doc(conn, "A", "PA"), await _doc(conn, "B", "PB")
        for i in range(30):
            await _chunk(conn, da, i, f"alpha high {i}", 0)
        for i in range(5):
            await _chunk(conn, db, i, f"alpha low {i}", 0)
        rr = FakeReranker(lambda t: 0.9 if "high" in t else 0.5)
        emb = FakeEmbedder(_vec(0))
        res = await compare_facets(conn, "alpha", embedder=emb, reranker=rr,
                                   facets=[("vendor", "A"), ("product", "PB")])
        assert [h.vendor for h in res.hits] == ["A", "B"] * 5
        assert res.facets_empty == [] and res.rerank_ok is True
        assert len(emb.calls) == 1 and len(rr.calls) == 2
        assert len(rr.calls[0][1]) == 20  # RERANK_CANDIDATES // 2
    finally:
        await conn.close()


async def test_comparison_drops_repeated_passages_within_a_facet(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        a1, a2, b = await _doc(conn, "A", "PA"), await _doc(conn, "A", "PA2"), await _doc(conn, "B", "PB")
        # the same page published twice (two guides): identical text, identical score
        for d in (a1, a2):
            for i in range(3):
                await _chunk(conn, d, i, f"alpha page{i} s0.9{i}", 0)
        await _chunk(conn, a1, 3, "alpha other s0.10", 0)
        for i in range(5):
            await _chunk(conn, b, i, f"alpha bee{i} s0.5{i}", 0)
        rr = FakeReranker(lambda t: float(t.split(" s")[-1]))
        res = await compare_facets(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                   facets=[("vendor", "A"), ("vendor", "B")])
        a_texts = [h.text for h in res.hits if h.vendor == "A"]
        assert len(a_texts) == len(set(a_texts)) == 4  # three pages once each, then the other
        assert sum("alpha other" in t for t in a_texts) == 1
    finally:
        await conn.close()


async def test_comparison_empty_facet(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        da = await _doc(conn, "A", "PA")
        for i in range(8):
            await _chunk(conn, da, i, f"alpha {i}", 0)
        rr = FakeReranker(lambda t: 0.9)
        res = await compare_facets(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                   facets=[("vendor", "A"), ("vendor", "B")])
        assert res.facets_empty == ["B"]
        assert [h.vendor for h in res.hits] == ["A"] * 5 and res.rerank_ok is True
        res = await compare_facets(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                   reranker=FakeReranker(fail=True),
                                   facets=[("vendor", "A"), ("vendor", "B")])
        assert res.rerank_ok is False and [h.rrf_rank for h in res.hits] == [1, 2, 3, 4, 5]
    finally:
        await conn.close()


async def _raw_chunk(conn, doc, ordinal, text, vec):
    cid = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s,%s,%s,%s,%s,1,%s)",
        (cid, doc, ["A"], ordinal, text, Vector(vec)))
    return cid


def _fine(rank):
    """Like _graded, with steps small enough for 60 distinct ranks."""
    v = _vec(0)
    v[1] = 0.02 * rank
    return v


async def _assert_matches_scoped(conn, query, vendors, limit=10):
    batched = await vendor_candidates(conn, query, _vec(0), vendors, limit=limit)
    for v in vendors:
        scoped = await hybrid_candidates(conn, query, _vec(0), vendor=[v], limit=limit)
        got = [(h.chunk_id, h.rrf_rank) for h in batched.get(v, [])]
        assert got == [(h.chunk_id, h.rrf_rank) for h in scoped], v
        assert all(h.vendor == v for h in batched.get(v, []))
    return batched


async def test_vendor_candidates_equal_scoped_hybrid_search(vkb_dsn, monkeypatch):
    """Every vendor fits in the pool, so each vendor's batched list is exactly the scoped
    hybrid search's: same 50 + 50 fusion, same ties, same cut."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        # full-text and vector orders disagree; BIG and MID exceed the per-list cut of 50
        for vendor, n, mult in (("BIG", 60, 7), ("MID", 30, 11), ("SMALL", 6, 1)):
            d = await _doc(conn, vendor, f"{vendor}-P")
            for j in range(n):
                text = " ".join(["alpha"] * max(1, (n - j) // 3) + ["filler"] * j)
                await _raw_chunk(conn, d, j, f"{vendor} {j} {text}", _fine((j * mult) % n))
        d = await _doc(conn, "NOKW", "NOKW-P")  # vector rows only
        for j in range(12):
            await _raw_chunk(conn, d, j, f"nokw {j}", _fine(j))
        vendors = ["BIG", "MID", "SMALL", "NOKW", "ABSENT"]
        for weight in (1.0, 0.5):
            monkeypatch.setattr(search, "FTS_WEIGHT", weight)
            batched = await _assert_matches_scoped(conn, "alpha", vendors)
            assert "ABSENT" not in batched and all(len(batched[v]) == 10 for v in vendors[:4]
                                                   if v != "SMALL")
    finally:
        await conn.close()


async def test_vendor_candidates_partition_full_text_per_vendor(vkb_dsn):
    """A vendor whose keyword matches all rank below another vendor's still gets its own
    full-text rows: the cut is per vendor, not global."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        d = await _doc(conn, "KW", "KW-P")
        for j in range(80):
            await _raw_chunk(conn, d, j, "alpha alpha alpha alpha " * 3, _fine(j % 40))
        d = await _doc(conn, "LO", "LO-P")
        weak = [await _raw_chunk(conn, d, j, "alpha " + "filler " * 40, _fine(45 + j))
                for j in range(3)]  # far vectors: only full text can lift them
        for j in range(12):
            await _raw_chunk(conn, d, 3 + j, f"near {j}", _fine(j))
        batched = await _assert_matches_scoped(conn, "alpha", ["KW", "LO"])
        assert set(weak) <= {h.chunk_id for h in batched["LO"][:6]}
    finally:
        await conn.close()


async def test_vendor_candidates_top_up_replaces_short_pool_rows(vkb_dsn, monkeypatch):
    """A vendor with 1-9 rows in the pool is topped up by its own scoped query, which
    replaces (never adds to) its pooled rows."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        d = await _doc(conn, "BIG", "BIG-P")
        for j in range(20):
            await _raw_chunk(conn, d, j, f"big {j}", _fine(5 + j))
        d = await _doc(conn, "SMALL", "SMALL-P")
        near = [await _raw_chunk(conn, d, j, f"small near {j}", _fine(j)) for j in range(2)]
        for j in range(12):
            await _raw_chunk(conn, d, 2 + j, f"small mid {j}", _fine(20 + j))
        for j in range(3):  # keyword matches at vector ranks 15-17: a top-up must reach them
            await _raw_chunk(conn, d, 14 + j, f"small alpha {j}", _fine(40 + j))
        monkeypatch.setattr(search, "ENUM_VEC_POOL", 10)
        batched = await vendor_candidates(conn, "alpha", _vec(0), ["BIG", "SMALL"], limit=10)
        assert set(near) <= {h.chunk_id for h in batched["SMALL"]}
        await _assert_matches_scoped(conn, "alpha", ["BIG", "SMALL"])
        assert len(batched["SMALL"]) == 10
        assert [h.text for h in batched["SMALL"][:3]] == [
            h.text for h in batched["SMALL"] if "alpha" in h.text]
    finally:
        await conn.close()


async def test_vendor_candidates_fuse_fifty_rows_per_list(vkb_dsn):
    """Each list is cut at 50 before fusion, as in the scoped search: a chunk ranked past 50
    in one list adds nothing there (opposed full-text and vector orders over 60 chunks)."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        d = await _doc(conn, "LONG", "LONG-P")
        for j in range(60):  # full-text rank j + 1, vector rank 60 - j
            await _raw_chunk(conn, d, j, "alpha " * (60 - j) + "filler " * j, _fine(59 - j))
        batched = await _assert_matches_scoped(conn, "alpha", ["LONG"])
        texts = [h.text for h in batched["LONG"]]
        # the head of each list scores 1/61 alone, below chunks both lists rank 11-50
        assert texts[0] != "alpha " * 60 and len(texts) == 10
    finally:
        await conn.close()


# --- final fix wave: batched comparison candidates, best scores --------------------------

async def test_enumeration_best_scores_cover_dropped_vendors(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        for v, score in (("VA", 0.9), ("VC", 0.1)):
            d = await _doc(conn, v, f"{v}-P")
            await _chunk(conn, d, 0, f"alpha {v} s{score}", 0)
        rr = FakeReranker(lambda t: float(t.split(" s")[-1]))
        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=["VA", "VC"], min_score=0.5)
        assert res.dropped == ["VC"] and res.best_scores == {"VA": 0.9, "VC": 0.1}
        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=FakeReranker(fail=True), vendors=["VA", "VC"])
        assert res.best_scores == {}
    finally:
        await conn.close()


async def _facet_corpus(conn):
    for vendor, product, n, mult in (("BIG", "BIG-P", 60, 7), ("MID", "MID-P", 30, 11),
                                     ("MID", "MID-Q", 8, 3), ("SMALL", "SMALL-P", 6, 1)):
        d = await _doc(conn, vendor, product)
        for j in range(n):
            text = " ".join(["alpha"] * max(1, (n - j) // 3) + ["filler"] * j)
            await _raw_chunk(conn, d, j, f"{vendor} {product} {j} {text}", _fine((j * mult) % n))


async def test_facet_candidates_equal_scoped_hybrid_search(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _facet_corpus(conn)
        facets = [("vendor", "BIG"), ("product", "MID-Q"), ("vendor", "MID"),
                  ("vendor", "ABSENT")]
        for limit in (20, 10):
            got = await search.facet_candidates(conn, "alpha", _vec(0), facets, limit=limit)
            assert len(got) == len(facets)
            for (kind, name), hits in zip(facets, got):
                scoped = await hybrid_candidates(
                    conn, "alpha", _vec(0), limit=limit,
                    vendor=[name] if kind == "vendor" else None,
                    product=[name] if kind == "product" else None)
                assert [(h.chunk_id, h.rrf_rank) for h in hits] == [
                    (h.chunk_id, h.rrf_rank) for h in scoped], name
            assert got[3] == []
    finally:
        await conn.close()


async def test_vector_candidates_are_the_nearest_chunks_within_each_side(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        rank = 0  # every chunk its own distance from the query: no tie for the index to break
        for vendor, product, n in (("BIG", "BIG-P", 40), ("MID", "MID-P", 20),
                                   ("MID", "MID-Q", 8), ("SMALL", "SMALL-P", 4)):
            d = await _doc(conn, vendor, product)
            for j in range(n):
                rank += 1
                await _raw_chunk(conn, d, j, f"{product} {j}", _fine(rank))
        # a page's earlier version, the nearest chunk of all: never a candidate
        old = await _raw_chunk(conn, await _doc(conn, "BIG", "BIG-P"), 0, "old", _fine(0))
        await conn.execute("UPDATE chunks SET superseded_at = now() WHERE chunk_id = %s", (old,))
        # the approximate index, rebuilt: the rows the earlier tests deleted would still be in it
        await conn.execute("REINDEX INDEX chunks_embedding_idx")
        qvecs = [_fine(0), _fine(45)]
        sides = [("vendor", "BIG"), ("product", "MID-Q"), ("vendor", "MID"), None,
                 ("vendor", "ABSENT")]
        for limit in (20, 5):
            cc = CountingConn(conn)
            await search._pgvector_version(cc)  # read once per connection
            cc.executes = 0
            got = await search.vector_candidates(cc, qvecs, sides, limit=limit)
            # one SET LOCAL and one vector query per pair: no full-text statement
            assert cc.executes == 1 + len(qvecs) * len(sides)
            assert len(got) == len(qvecs) and all(len(g) == len(sides) for g in got)
            for qvec, per_side in zip(qvecs, got):
                for side, hits in zip(sides, per_side):
                    kind, name = side or (None, None)
                    col = {"vendor": "d.vendor", "product": "d.product"}.get(kind, "true")
                    cur = await conn.execute(
                        "SELECT c.chunk_id::text FROM chunks c JOIN documents d USING (doc_id)"
                        f" WHERE c.superseded_at IS NULL AND ({col} = %s OR %s::text IS NULL)"
                        " ORDER BY c.embedding <=> %s, c.chunk_id LIMIT %s",
                        (name, name, Vector(qvec), limit))
                    assert [(h.chunk_id, h.rrf_rank) for h in hits] == [
                        (r[0], n) for n, r in enumerate(await cur.fetchall(), start=1)], side
                    assert all(h.rerank_score is None and h.chunk_id != old for h in hits)
            assert got[0][4] == [] and len(got[0][3]) == limit
            assert {h.vendor for h in got[1][0]} == {"BIG"}
            assert {h.product for h in got[0][1]} == {"MID-Q"}
    finally:
        await conn.close()


async def test_vector_candidates_edge_inputs(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _facet_corpus(conn)
        side = [("vendor", "BIG")]
        assert await search.vector_candidates(conn, [], side, limit=5) == []
        assert await search.vector_candidates(conn, [_vec(0)], [], limit=5) == [[]]
        with pytest.raises(ValueError, match="unknown vector column"):
            await search.vector_candidates(conn, [_vec(0)], side, limit=5, vector_column="text")
    finally:
        await conn.close()


async def test_comparison_statement_count_has_one_full_text_query(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _facet_corpus(conn)
        rr = FakeReranker(lambda t: 0.9)
        for facets in ([("vendor", "BIG"), ("vendor", "MID")],
                       [("vendor", "BIG"), ("vendor", "MID"), ("vendor", "SMALL")]):
            cc = CountingConn(conn)
            await search._pgvector_version(cc)  # read once per connection
            cc.executes = 0
            res = await compare_facets(cc, "alpha", embedder=FakeEmbedder(_vec(0)),
                                       reranker=rr, facets=facets)
            assert len(res.hits) == 5 * len(facets)
            # ONE full-text query, one SET LOCAL, one vector query per facet
            assert cc.executes == 2 + len(facets)
    finally:
        await conn.close()


async def test_comparison_reranks_run_concurrently(vkb_dsn):
    import asyncio

    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _facet_corpus(conn)

        class Barrier(FakeReranker):
            """Each call waits until both facets' calls have started: a serial loop hangs."""
            def __init__(self):
                super().__init__(lambda t: 0.5)
                self.started = 0
                self.both = asyncio.Event()

            async def rerank(self, query, documents, top_k):
                self.started += 1
                if self.started == 2:
                    self.both.set()
                await self.both.wait()
                return await super().rerank(query, documents, top_k)

        rr = Barrier()
        res = await asyncio.wait_for(
            compare_facets(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                           facets=[("vendor", "BIG"), ("vendor", "MID")]), timeout=10)
        assert [h.vendor for h in res.hits] == ["BIG", "MID"] * 5 and res.rerank_ok is True
    finally:
        await conn.close()


async def test_superseded_chunks_are_never_retrieved(vkb_dsn):
    """An earlier version of a page, kept only for its claims, is the best full-text and
    vector match here and still comes back from no retrieval."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc = await _doc(conn, "V1", "P1")
        old = [await _chunk(conn, doc, 0, "retention keyword old", 0),
               await _chunk(conn, doc, 1, "retention keyword old tail", 1)]
        await conn.execute("UPDATE chunks SET superseded_at = now() WHERE doc_id = %s", (doc,))
        new = [await _chunk(conn, doc, 0, "retention new", 2),
               await _chunk(conn, doc, 1, "retention new tail", 3)]
        other = await _doc(conn, "V2", "P2")
        await _chunk(conn, other, 0, "retention elsewhere", 4)
        query = "retention keyword"

        def ids(hits):
            return {h.chunk_id for h in hits}

        assert ids(await hybrid_candidates(conn, query, _vec(0))) >= set(new)
        for kwargs in ({}, {"vendor": ["V1"]}, {"product": ["P1"]}):
            got = await hybrid_candidates(conn, query, _vec(0), **kwargs)
            assert ids(got) and not ids(got) & set(old), kwargs
        for kind, name in (("vendor", "V1"), ("product", "P1")):
            [got] = await search.facet_candidates(conn, query, _vec(0), [(kind, name)], limit=10)
            assert ids(got) == set(new), name
        by_vendor = await vendor_candidates(conn, query, _vec(0), ["V1"], limit=10)
        assert ids(by_vendor["V1"]) == set(new)
        res = await search_docs(conn, query, embedder=FakeEmbedder(_vec(0)), reranker=None,
                                use_rerank=False, vendor=["V1"])
        assert ids(res.hits) == set(new)

        # a current anchor's neighbours and section are all current; a superseded anchor's
        # are all superseded
        assert [h.chunk_id for h in await read_section(conn, new[0], "1")] == new
        assert [h.chunk_id for h in await read_section(conn, new[1], "full")] == new
        assert [h.chunk_id for h in await read_section(conn, old[0], "1")] == old
        assert [h.chunk_id for h in await read_section(conn, old[1], "full")] == old
        assert [h.chunk_id for h in await read_section(conn, old[1])] == [old[1]]
    finally:
        await conn.close()


async def _extraction(conn, chunk, status, prompt_version="claims-v3.1", model="m"):
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s,%s,%s,%s,%s)", (uuid.uuid4(), chunk, model, prompt_version, status))


async def test_unextracted_candidates(vkb_dsn):
    """Only current chunks with no final extraction of the v3 prompt family are candidates:
    ranked per query by vector distance and full text (RRF), in one statement."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        d1, d2 = await _doc(conn, "V1", "P1"), await _doc(conn, "V2", "P2")
        # the nearest chunks to the first query, every one of them out of the running
        done = await _raw_chunk(conn, d1, 0, "stripes extracted", _fine(0))
        await _extraction(conn, done, "done")
        failed = await _raw_chunk(conn, d1, 1, "stripes given up", _fine(0))
        await _extraction(conn, failed, "failed", "claims-v3")
        by_another_model = await _raw_chunk(conn, d2, 0, "stripes by the teacher", _fine(0))
        await _extraction(conn, by_another_model, "done", "claims-v3.1", "teacher")
        old = await _raw_chunk(conn, d2, 4, "stripes of an earlier version", _fine(0))
        await conn.execute("UPDATE chunks SET superseded_at = now() WHERE chunk_id = %s", (old,))
        # still waiting to be extracted: never analysed, not final, or not of the v3 family
        fresh = await _raw_chunk(conn, d2, 1, "zebra stripes never analysed", _fine(1))
        errored = await _raw_chunk(conn, d1, 2, "to retry", _fine(2))
        await _extraction(conn, errored, "errored")
        older_prompt = await _raw_chunk(conn, d1, 3, "once read by v2", _fine(3))
        await _extraction(conn, older_prompt, "done", "claims-v2")
        vec_only = await _raw_chunk(conn, d2, 3, "other words entirely", _fine(4))
        fts_only = await _raw_chunk(conn, d2, 2, "zebra elsewhere", _fine(20))

        queries, qvecs = ["zebra", "quokka"], [_fine(0), _fine(20)]
        cc = CountingConn(conn)
        got = await search.unextracted_candidates(cc, queries, qvecs, limit=20)
        assert cc.executes == 1  # one statement for every query
        # the first query fuses a full-text match (fresh, fts_only) with the vector order; the
        # second matches no text, so it is the vector order alone
        by_query = [[fresh, fts_only, errored, older_prompt, vec_only],
                    [fts_only, vec_only, older_prompt, errored, fresh]]
        assert [[h.chunk_id for h in hits] for hits in got] == by_query
        for hits in got:
            assert [h.rrf_rank for h in hits] == [1, 2, 3, 4, 5]
            assert all(h.rerank_score is None and not h.superseded for h in hits)
        top = got[0][0]
        assert (top.vendor, top.product, top.title) == ("V2", "P2", "T")
        assert top.url.startswith("http://u/V2/") and top.text == "zebra stripes never analysed"
        short = await search.unextracted_candidates(conn, queries, qvecs, limit=2)
        assert [[h.chunk_id for h in hits] for hits in short] == [q[:2] for q in by_query]
        # a query of stopwords has no full-text side, and an empty batch no statement
        [stop] = await search.unextracted_candidates(conn, ["the of and"], [_fine(0)], limit=20)
        assert [h.chunk_id for h in stop] == [fresh, errored, older_prompt, vec_only, fts_only]
        nothing = CountingConn(conn)
        assert await search.unextracted_candidates(nothing, [], [], limit=20) == []
        assert nothing.executes == 0
    finally:
        await conn.close()
