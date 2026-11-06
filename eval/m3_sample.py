"""Build the M3 extraction pilot: a stratified chunk sample (spec 2026-11-06 §4).

(a) PER_VENDOR random chunks for each pilot vendor, spread across its products in proportion to
    their chunk counts (seeded); (b) every chunk of every document cited in `expected_vendors`
    of the enumeration questions; (c) for each enumeration question and each vendor, the top
    CAND_PER_VENDOR hybrid candidates (`search.vendor_candidates`; one query embedding per
    question). Writes one chunk id per line, plus a JSON sidecar saying which part(s) each id
    came from.

    uv run python eval/m3_sample.py --out eval/m3_pilot_ids.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from vendorkb.clients import Embedder
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect
from vendorkb.scope import scope_index_from_db
from vendorkb.search import vendor_candidates

PILOT_VENDORS = ["Commvault", "Cohesity", "Rubrik", "Druva", "AvePoint"]
PER_VENDOR = 3000
CAND_PER_VENDOR = 5
SEED = "m3-pilot-1"  # md5(chunk_id || SEED) orders the random draw deterministically


async def _part_a(conn) -> set[str]:
    out: set[str] = set()
    for vendor in PILOT_VENDORS:
        cur = await conn.execute(
            "SELECT coalesce(d.product, ''), count(*) FROM chunks c JOIN documents d USING (doc_id)"
            " WHERE d.vendor = %s GROUP BY 1", (vendor,))
        counts = dict(await cur.fetchall())
        total = sum(counts.values())
        for product, n in sorted(counts.items()):
            take = max(1, round(PER_VENDOR * n / total))
            cur = await conn.execute(
                "SELECT c.chunk_id::text FROM chunks c JOIN documents d USING (doc_id)"
                " WHERE d.vendor = %s AND coalesce(d.product, '') = %s"
                " ORDER BY md5(c.chunk_id::text || %s) LIMIT %s",
                (vendor, product, SEED, take))
            out.update(r[0] for r in await cur.fetchall())
    return out


async def _part_b(conn, questions: list[dict]) -> set[str]:
    docs = sorted({a for q in questions for e in q.get("expected_vendors", [])
                   for a in e.get("article_ids", [])})
    cur = await conn.execute("SELECT chunk_id::text FROM chunks WHERE doc_id::text = ANY(%s)",
                             (docs,))
    return {r[0] for r in await cur.fetchall()}


async def _part_c(conn, questions: list[dict], settings: VendorkbSettings) -> set[str]:
    vendors = (await scope_index_from_db(conn)).vendors
    out: set[str] = set()
    async with Embedder(settings) as embedder:
        for q in questions:
            qvec = (await embedder.embed([q["question"]], input_type="query"))[0]
            tops = await vendor_candidates(conn, q["question"], qvec, vendors,
                                           limit=CAND_PER_VENDOR)
            out.update(h.chunk_id for hits in tops.values() for h in hits)
    return out


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    settings = VendorkbSettings()
    questions = [q for q in yaml.safe_load(Path(args.questions).read_text())
                 if q.get("shape") == "enumeration"]
    conn = await connect(settings.vendorkb_dsn)
    try:
        parts = {"a": await _part_a(conn), "b": await _part_b(conn, questions),
                 "c": await _part_c(conn, questions, settings)}
    finally:
        await conn.close()
    ids = sorted(set().union(*parts.values()))
    Path(args.out).write_text("\n".join(ids) + "\n", encoding="utf-8")
    sidecar = {i: [p for p, s in parts.items() if i in s] for i in ids}
    Path(args.out).with_suffix(".parts.json").write_text(json.dumps(sidecar), encoding="utf-8")
    print(json.dumps({"total": len(ids), **{k: len(v) for k, v in parts.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
