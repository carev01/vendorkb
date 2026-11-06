import uuid

from pgvector import Vector

from vendorkb.db import connect, init_schema
from vendorkb import search
from vendorkb.modes import compare_facets, enumerate_vendors
from vendorkb.search import hybrid_candidates, read_section, search_docs


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
        rr = FakeReranker(lambda t: 0.5)
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
        rr = FakeReranker(lambda t: 0.5)
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
        rr = FakeReranker(lambda t: 0.5)
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
        rr = FakeReranker(lambda t: 0.5)
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
                                      reranker=FakeReranker(lambda t: 0.5),
                                      vendors=["VA", "VB", "VC", "VD", "VE"])
        assert sorted(res.kept + res.dropped + res.empty) == ["VA", "VB", "VC", "VD", "VE"]
        assert len(res.kept + res.dropped + res.empty) == 5
        assert res.empty == ["VD", "VE"] and sorted(res.kept) == ["VA", "VB", "VC"]
        assert len(emb.calls) == 1 and res.rerank_ok is True
        assert sorted(h.vendor for h in res.hits) == ["VA", "VB", "VC"]
    finally:
        await conn.close()


async def test_enumeration_per_vendor_cap(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        single = await _doc(conn, "VS", "Only")
        for i in range(10):
            await _chunk(conn, single, i, f"alpha single {i}", 0)
        for p in ("P1", "P2", "P3"):
            d = await _doc(conn, "VM", p)
            for i in range(3):
                await _chunk(conn, d, i, f"alpha multi {p} {i}", 0)
        rr = FakeReranker(lambda t: 0.5)
        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=["VS", "VM"])
        vs = [h for h in res.hits if h.vendor == "VS"]
        vm = [h for h in res.hits if h.vendor == "VM"]
        assert len(vs) == 3
        assert len(vm) == 5 and {h.product for h in vm} == {"P1", "P2", "P3"}
        assert len(rr.calls) == 1 and len(rr.calls[0][1]) == 8  # one rerank over all kept
    finally:
        await conn.close()


async def test_enumeration_floor_drops_low_vendor(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        for v, score in (("VA", 0.9), ("VB", 0.5), ("VC", 0.1)):
            d = await _doc(conn, v, f"{v}-P")
            for i in range(2):
                await _chunk(conn, d, i, f"alpha {v} s{score}", 0)
        rr = FakeReranker(lambda t: float(t.split(" s")[-1]))
        res = await enumerate_vendors(conn, "alpha", embedder=FakeEmbedder(_vec(0)),
                                      reranker=rr, vendors=["VA", "VB", "VC"], min_score=0.3)
        assert res.kept == ["VA", "VB"] and res.dropped == ["VC"] and res.empty == []
        assert [h.vendor for h in res.hits] == ["VA", "VA", "VB", "VB"]
        assert res.hits[0].rerank_score == 0.9 and res.rerank_ok is True
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


async def test_comparison_empty_facet(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        da = await _doc(conn, "A", "PA")
        for i in range(8):
            await _chunk(conn, da, i, f"alpha {i}", 0)
        rr = FakeReranker(lambda t: 0.5)
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
