import uuid

from pgvector import Vector

from vendorkb.db import connect, init_schema
from vendorkb.search import read_section, search_docs


def _vec(i):
    v = [0.0] * 768
    v[i] = 1.0
    return v


class FakeEmbedder:
    def __init__(self, vec):
        self.vec = vec
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)
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
        await init_schema(conn)
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
        await init_schema(conn)
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
        await init_schema(conn)
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
        await init_schema(conn)
        ids = await _seed(conn)
        await _chunk(conn, ids["V1"], 0, "alpha text", 0)
        await _chunk(conn, ids["V2"], 0, "alpha text", 0)
        rr = FakeReranker(lambda t: 0.5)
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                vendor=["V2"])
        assert [h.vendor for h in res.hits] == ["V2"]
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                vendor=["V1"], product=["P2"])
        assert res.hits == []
        res = await search_docs(conn, "alpha", embedder=FakeEmbedder(_vec(0)), reranker=rr,
                                vendor=[])
        assert len(res.hits) == 2
    finally:
        await conn.close()


async def test_use_rerank_false_makes_no_reranker_call(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn)
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
        await init_schema(conn)
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
        await init_schema(conn)
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
        await init_schema(conn)
        await _seed(conn)
        idx = await scope_index_from_db(conn)
        assert idx.vendors == ["V1", "V2"]
        assert idx.products == ["P1", "P2"]
        assert idx.source_products == {"S": ["P1", "P2"]}
        assert idx.source_vendors == {}  # "S" is shared by V1 and V2, so not unique
        assert "VBR" in idx.aliases
    finally:
        await conn.close()
