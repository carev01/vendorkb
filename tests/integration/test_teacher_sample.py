"""eval/teacher_sample.py against a synthetic DB: three vendors with uneven documents."""
import json
import uuid

import pytest
import yaml
from pgvector import Vector

from eval import teacher_sample
from vendorkb.db import connect, init_schema

# vendor -> list of (product, title, chunks in that document)
LAYOUT = {
    "big": [("Big Core", "Big intro", 120), ("Big Core", "Big tuning", 40),
            ("Big Edge", "Big edge", 10)],
    "mid": [("Mid One", "Mid a", 30), ("Mid One", "Mid b", 6)],
    "tiny": [("Tiny", "Tiny doc", 3)],
}


async def _seed(conn, layout=LAYOUT) -> dict[str, list[str]]:
    await conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")
    await init_schema(conn, embed_dim=768)
    by_vendor: dict[str, list[str]] = {}
    for vendor, docs in layout.items():
        by_vendor[vendor] = []
        for product, title, n in docs:
            doc_id = str(uuid.uuid4())
            await conn.execute(
                "INSERT INTO documents (doc_id, vendor, product, source, title, url,"
                " content_hash, fetched_at) VALUES (%s,%s,%s,'S',%s,'http://u',%s, now())",
                (doc_id, vendor, product, title, doc_id))
            for ordinal in range(n):
                cid = str(uuid.uuid4())
                await conn.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text,"
                    " token_count, embedding) VALUES (%s,%s,%s,%s,%s,5,%s)",
                    (cid, doc_id, ["A"], ordinal, f"{title} {ordinal}", Vector([0.1] * 768)))
                by_vendor[vendor].append(cid)
    return by_vendor


async def _select(conn, **kw):
    args = {"must": [], "total": 100, "heldout_frac": 0.2, "seed": 7, "exclude": set()}
    args.update(kw)
    return await teacher_sample.select_chunks(conn, **args)


def _vendor_of(by_vendor, cid):
    return next(v for v, ids in by_vendor.items() if cid in ids)


async def test_totals_and_disjoint(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed(conn)
        train, heldout, stats = await _select(conn)
        assert len(train) + len(heldout) == 100 == stats["total"]
        assert len(heldout) == 20 == stats["heldout"]
        assert not set(train) & set(heldout)
        assert len(set(train)) == len(train)
    finally:
        await conn.close()


async def test_every_vendor_gets_its_minimum(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        by_vendor = await _seed(conn)
        train, heldout, stats = await _select(conn, total=120)
        assert stats["per_vendor"]["tiny"] == 3  # all it has: min(50, 3)
        assert stats["per_vendor"]["mid"] <= 36
        counts = {v: sum(1 for c in train + heldout if c in ids)
                  for v, ids in by_vendor.items()}
        assert counts == stats["per_vendor"]
        assert counts["big"] >= 50
        # sqrt weighting: the big vendor gets more than mid, but not in proportion to count
        assert counts["big"] > counts["mid"]
    finally:
        await conn.close()


async def test_sqrt_allocation_and_round_robin_over_documents(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        by_vendor = await _seed(conn)
        train, heldout, stats = await _select(conn, total=60, heldout_frac=0.0)
        chosen = set(train)
        # big has 3 documents; a 50-chunk floor taken round-robin touches all of them
        docs = {}
        cur = await conn.execute("SELECT chunk_id::text, title FROM chunks JOIN documents"
                                 " USING (doc_id) WHERE vendor = 'big'")
        for cid, title in await cur.fetchall():
            docs[cid] = title
        assert {docs[c] for c in chosen if c in docs} == {"Big intro", "Big tuning", "Big edge"}
        assert stats["per_vendor"]["big"] >= 50
        assert chosen <= set().union(*map(set, by_vendor.values()))
    finally:
        await conn.close()


async def test_sqrt_allocation_with_the_floor_in_play(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        layout = {"a": [("P", "a1", 250), ("P", "a2", 150)], "b": [("P", "b1", 100)],
                  "c": [("P", "c1", 30)], "d": [("P", "d1", 10)]}
        await _seed(conn, layout)
        _, _, stats = await _select(conn, total=200, heldout_frac=0.0)
        got = stats["per_vendor"]
        # c and d sit at their floors (all they have); a and b split the rest by sqrt: 2 : 1
        assert (got["c"], got["d"]) == (30, 10)
        assert got["a"] + got["b"] == 160
        assert abs(got["a"] - 2 * got["b"]) <= 1
    finally:
        await conn.close()


async def test_must_include_present_and_capped(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        by_vendor = await _seed(conn)
        must = [{"vendor": "MID", "product_like": "%one", "title_like": "Mid a"},
                {"vendor": "big", "product_like": "big edge", "title_like": None}]
        train, heldout, stats = await _select(conn, must=must, total=100)
        chosen = set(train) | set(heldout)
        cur = await conn.execute(
            "SELECT chunk_id::text FROM chunks JOIN documents USING (doc_id)"
            " WHERE title IN ('Mid a', 'Big edge')")
        wanted = {r[0] for r in await cur.fetchall()}
        assert len(wanted) == 40 and wanted <= chosen
        assert stats["must_include"] == 40
        # overflow: cap is total // 2, sampled across documents
        everything_big = [{"vendor": "big", "product_like": None, "title_like": None}]
        train, heldout, stats = await _select(conn, must=everything_big, total=40,
                                              heldout_frac=0.0)
        assert stats["must_include"] == 20
        assert stats["must_include_matched"] == 170
        cur = await conn.execute("SELECT doc_id::text, chunk_id::text FROM chunks")
        doc_of = {c: d for d, c in await cur.fetchall()}
        # with no held-out split, train lists the must-include chunks first
        per_doc: dict[str, int] = {}
        for c in train[:stats["must_include"]]:
            per_doc[doc_of[c]] = per_doc.get(doc_of[c], 0) + 1
        assert sorted(per_doc.values()) == [6, 7, 7]  # 20 chunks over big's 3 documents
        assert all(_vendor_of(by_vendor, c) == "big" for c in train[:20])
    finally:
        await conn.close()


async def test_must_include_vendor_null_means_any_vendor(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed(conn)
        must = [{"vendor": None, "product_like": None, "title_like": "%tuning%"}]
        train, heldout, stats = await _select(conn, must=must, total=100)
        cur = await conn.execute("SELECT chunk_id::text FROM chunks JOIN documents"
                                 " USING (doc_id) WHERE title ILIKE '%tuning%'")
        wanted = {r[0] for r in await cur.fetchall()}
        assert len(wanted) == 40 and wanted <= set(train) | set(heldout)
        assert stats["must_include"] == 40
        # "any vendor" crosses vendors
        both = [{"vendor": None, "product_like": "%one", "title_like": None}]
        _, _, stats = await _select(conn, must=both, total=100)
        assert stats["must_include"] == 36
    finally:
        await conn.close()


async def test_all_null_must_entry_is_invalid(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed(conn)
        bad = [{"vendor": None, "product_like": None, "title_like": None}]
        with pytest.raises(ValueError):
            await _select(conn, must=bad)
    finally:
        await conn.close()


async def test_excluded_ids_never_appear(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        by_vendor = await _seed(conn)
        banned = set(by_vendor["tiny"]) | set(by_vendor["big"][:100])
        must = [{"vendor": "tiny", "product_like": None, "title_like": None}]
        train, heldout, stats = await _select(conn, must=must, exclude=banned, total=150)
        assert not (set(train) | set(heldout)) & banned
        assert "tiny" not in stats["per_vendor"]
    finally:
        await conn.close()


async def test_same_seed_same_output_other_seed_differs(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed(conn)
        a = await _select(conn, seed=3)
        b = await _select(conn, seed=3)
        c = await _select(conn, seed=4)
        assert a == b
        assert a[0] != c[0]
    finally:
        await conn.close()


async def test_total_beyond_the_corpus_returns_everything(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        by_vendor = await _seed(conn)
        train, heldout, stats = await _select(conn, total=10_000, heldout_frac=0.1)
        assert set(train) | set(heldout) == set().union(*map(set, by_vendor.values()))
    finally:
        await conn.close()


async def test_cli_writes_files_and_prints_stats(vkb_dsn, tmp_path, monkeypatch, capsys):
    conn = await connect(vkb_dsn)
    try:
        by_vendor = await _seed(conn)
    finally:
        await conn.close()
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)
    must = tmp_path / "must.yaml"
    must.write_text(yaml.safe_dump([{"vendor": "mid", "product_like": None,
                                     "title_like": "Mid b"}]))
    excl = tmp_path / "excl.txt"
    excl.write_text("\n".join(by_vendor["big"][:5]) + "\n\n")
    out_t, out_h = tmp_path / "t.txt", tmp_path / "h.txt"
    rc = await teacher_sample.main([
        "--must-include", str(must), "--total", "120", "--heldout-frac", "0.25", "--seed", "1",
        "--exclude", str(excl), "--out-train", str(out_t), "--out-heldout", str(out_h)])
    assert rc == 0
    train = out_t.read_text().split()
    held = out_h.read_text().split()
    stats = json.loads(capsys.readouterr().out)
    assert (len(train), len(held)) == (stats["train"], stats["heldout"]) == (90, 30)
    assert not set(train) & set(held)
    assert not (set(train) | set(held)) & set(by_vendor["big"][:5])
