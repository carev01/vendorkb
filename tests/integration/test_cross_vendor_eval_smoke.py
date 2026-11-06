"""Smoke of eval/cross_vendor.py against a seeded DB with fake clients (no network)."""
import argparse
import json
import uuid

from pgvector import Vector

from eval import cross_vendor
from vendorkb.db import connect, init_schema
from vendorkb.scope import scope_index_from_db


class FakeEmbedder:
    tokens_used = 0

    async def embed(self, texts, *, input_type="document"):
        v = [0.0] * 768
        v[0] = 1.0
        return [v for _ in texts]


async def _doc(conn, vendor, product, text):
    did = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s,%s,%s,'S',%s,%s,'h', now())",
        (did, vendor, product, f"{vendor} doc", f"http://u/{vendor}"))
    v = [0.0] * 768
    v[0] = 1.0
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s,%s,%s,0,%s,5,%s)",
        (str(uuid.uuid4()), did, ["A"], text, Vector(v)))
    return did


async def test_score_paths_and_candidates(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        await conn.execute("DELETE FROM documents")
        d1 = await _doc(conn, "Alpha", "AlphaBackup", "immutability object lock supported")
        d2 = await _doc(conn, "Beta", "BetaVault", "immutability retention lock")
        await _doc(conn, "Gamma", "GammaSafe", "unrelated tape handling")
        index = await scope_index_from_db(conn)
        args = argparse.Namespace(rerank=False, legacy=False, min_score=0.0)
        kw = {"embedder": FakeEmbedder(), "reranker": None, "scope_index": index, "args": args}

        enum_q = {"id": "e", "type": "cross_vendor", "shape": "enumeration",
                  "question": "immutability", "holdout": True, "gold_sources": [],
                  "expected_vendors": [{"vendor": "Alpha"}, {"vendor": "Beta"}],
                  "unclear_vendors": ["Gamma"]}
        r = await cross_vendor._run_question(conn, enum_q, **kw)
        assert r["vendor_recall"] == 1.0 and r["vendor_noise"] == [] and "error" not in r
        assert r["kept"] == 3 and r["empty"] == 0

        comp_q = {"id": "c", "type": "cross_vendor", "shape": "comparison",
                  "question": "Compare Alpha and Beta immutability", "gold_sources": [
                      {"article_id": d1}, {"article_id": d2}]}
        r = await cross_vendor._run_question(conn, comp_q, **kw)
        assert r["facet_recall"] == 1.0, r

        single_q = {"id": "s", "type": "single_vendor", "question": "Alpha immutability",
                    "gold_sources": [{"article_id": d1}]}
        r = await cross_vendor._run_question(conn, single_q, **kw)
        assert r["pool_rank"] == 1, r

        args.legacy = True
        r = await cross_vendor._run_question(conn, single_q, **kw)
        assert r["pool_rank"] == 1, r

        records = await cross_vendor._fts_hits(conn, "immutability", "Alpha", 5)
        assert [h.vendor for h in records] == ["Alpha"]
    finally:
        await conn.close()
    json.dumps(r)  # results are JSON-serialisable
