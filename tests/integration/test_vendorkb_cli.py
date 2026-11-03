import asyncio
import json
import uuid

from pgvector import Vector

from vendorkb.cli import main
from vendorkb.db import connect, init_schema


async def test_stats_reports_counts_and_percentiles(vkb_dsn, monkeypatch, capsys):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn)
        await conn.execute("DELETE FROM documents")
        did = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
            " fetched_at) VALUES (%s,'V1','P1','S','Doc','http://u','h', now())", (did,))
        vec = Vector([0.1] * 768)
        for n, tokens in enumerate((10, 20, 1500)):
            await conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
                " embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (str(uuid.uuid4()), did, ["A"], n, f"text {n}", tokens, vec))
    finally:
        await conn.close()
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)
    code = await asyncio.to_thread(main, ["stats"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["documents"] == 1
    assert out["chunks"] == 3
    assert out["chunks_per_vendor"] == {"V1": 3}
    assert set(out["token_percentiles"]) == {"p50", "p90", "p99", "max"}
    assert out["token_percentiles"]["max"] == 1500
    assert out["over_max_tokens"] == 1
    assert out["token_counting"] == "approximate tokens (chars/4)"
