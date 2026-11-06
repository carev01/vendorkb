"""Stratified chunk sample for the Opus teacher extraction run, plus a held-out set.

The selection, in order: (1) every chunk matched by a ``--must-include`` entry, capped at
``total // 2`` (an overflow is taken round-robin across the matched documents); (2) the rest of
``--total`` spread over all vendors in proportion to ``sqrt(vendor chunk count)``, at least
``min(50, available)`` per vendor, taken round-robin across each vendor's documents. A seeded
random ``--heldout-frac`` of the selection becomes the held-out set, disjoint from train.

``--must-include`` is a YAML list of ``{vendor: str | null, product_like: str | null,
title_like: str | null}`` (the patterns are SQL ILIKE; the vendor matches case-insensitively and
null means any vendor; an entry with all three null is an error). It lives outside the
repo when it is derived from internal tables. Each ``--exclude`` file holds one chunk id per line
(chunks that must never be sampled, e.g. an earlier pilot). Writes one chunk id per line to
``--out-train`` and ``--out-heldout`` and prints the stats as JSON.

    uv run python eval/teacher_sample.py --must-include PATH --total 10000 --heldout-frac 0.1 \
        --seed 1 --exclude pilot_ids.txt --out-train train.txt --out-heldout heldout.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from vendorkb.config import VendorkbSettings
from vendorkb.db import connect

MIN_PER_VENDOR = 50


def _round_robin(rng: random.Random, rows: list[tuple[str, str]], n: int) -> list[str]:
    """Take ``n`` chunk ids one per document in turn; ``rows`` is ``(doc_id, chunk_id)``."""
    by_doc: dict[str, list[str]] = defaultdict(list)
    for doc_id, chunk_id in sorted(rows):
        by_doc[doc_id].append(chunk_id)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for chunks in by_doc.values():
        rng.shuffle(chunks)
    out: list[str] = []
    depth = 0
    while len(out) < n:
        progressed = False
        for doc_id in docs:
            if depth < len(by_doc[doc_id]):
                out.append(by_doc[doc_id][depth])
                progressed = True
                if len(out) == n:
                    break
        if not progressed:
            break
        depth += 1
    return out


def _allocate(available: dict[str, int], budget: int) -> dict[str, int]:
    """Split ``budget`` over vendors by ``sqrt(available)``, floor ``min(50, available)``, cap
    ``available``. The floors win over the budget when they cannot both be met."""
    floor = {v: min(MIN_PER_VENDOR, n) for v, n in available.items()}
    weight = {v: math.sqrt(n) for v, n in available.items()}
    fixed: dict[str, int] = {}
    active = sorted(available)
    while active:
        spare = budget - sum(fixed.values())
        total_w = sum(weight[v] for v in active)
        share = {v: spare * weight[v] / total_w for v in active}
        capped = {v: available[v] for v in active if share[v] >= available[v]}
        floored = {v: floor[v] for v in active if share[v] < floor[v] and v not in capped}
        if not capped and not floored:
            break
        fixed.update(capped)
        fixed.update(floored)
        active = [v for v in active if v not in fixed]
    if active:
        spare = budget - sum(fixed.values())
        total_w = sum(weight[v] for v in active)
        raw = {v: spare * weight[v] / total_w for v in active}
        alloc = {v: min(int(raw[v]), available[v]) for v in active}
        left = spare - sum(alloc.values())
        for v in sorted(active, key=lambda v: (-(raw[v] - int(raw[v])), v)):
            if left <= 0:
                break
            if alloc[v] < available[v]:
                alloc[v] += 1
                left -= 1
        fixed.update(alloc)
    return fixed


async def _must_rows(conn, must: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(doc_id, chunk_id)`` of every chunk matched by any must-include entry."""
    found: dict[str, str] = {}
    for entry in must:
        if not any(entry.get(k) for k in ("vendor", "product_like", "title_like")):
            raise ValueError(f"must-include entry {entry!r} has no vendor, product_like or"
                             " title_like")
        cur = await conn.execute(
            "SELECT c.doc_id::text, c.chunk_id::text FROM chunks c JOIN documents d USING (doc_id)"
            " WHERE (%s::text IS NULL OR lower(d.vendor) = lower(%s))"
            " AND (%s::text IS NULL OR d.product ILIKE %s)"
            " AND (%s::text IS NULL OR d.title ILIKE %s)",
            (entry.get("vendor"), entry.get("vendor"), entry.get("product_like"), entry.get("product_like"),
             entry.get("title_like"), entry.get("title_like")))
        for doc_id, chunk_id in await cur.fetchall():
            found[chunk_id] = doc_id
    return sorted((doc_id, chunk_id) for chunk_id, doc_id in found.items())


async def select_chunks(conn, *, must: list[dict[str, Any]], total: int, heldout_frac: float,
                        seed: int, exclude: set[str]
                        ) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return ``(train_ids, heldout_ids, stats)``; see the module docstring."""
    rng = random.Random(seed)
    must_rows = [r for r in await _must_rows(conn, must) if r[1] not in exclude]
    cap = total // 2
    must_ids = (_round_robin(rng, must_rows, cap) if len(must_rows) > cap
                else [chunk_id for _, chunk_id in must_rows])
    taken = set(must_ids)

    cur = await conn.execute(
        "SELECT d.vendor, c.doc_id::text, c.chunk_id::text FROM chunks c"
        " JOIN documents d USING (doc_id)")
    pool: dict[str, list[tuple[str, str]]] = defaultdict(list)
    vendor_of: dict[str, str] = {}
    for vendor, doc_id, chunk_id in await cur.fetchall():
        vendor_of[chunk_id] = vendor
        if chunk_id not in exclude and chunk_id not in taken:
            pool[vendor].append((doc_id, chunk_id))
    alloc = _allocate({v: len(rows) for v, rows in pool.items()},
                      max(total - len(must_ids), 0)) if pool else {}
    rest: list[str] = []
    for vendor in sorted(pool):
        rest.extend(_round_robin(rng, pool[vendor], alloc[vendor]))

    selection = must_ids + rest
    n_held = min(round(heldout_frac * len(selection)), len(selection))
    held = set(rng.sample(sorted(selection), n_held))
    train = [i for i in selection if i not in held]
    heldout = sorted(held)

    per_vendor: dict[str, int] = defaultdict(int)
    for chunk_id in selection:
        per_vendor[vendor_of[chunk_id]] += 1
    stats = {"must_include": len(must_ids), "must_include_matched": len(must_rows),
             "per_vendor": dict(sorted(per_vendor.items())), "total": len(selection),
             "train": len(train), "heldout": len(heldout)}
    return train, heldout, stats


def _read_ids(paths: list[str]) -> set[str]:
    return {line.strip() for p in paths for line in Path(p).read_text().splitlines()
            if line.strip()}


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--must-include", required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--heldout-frac", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-heldout", required=True)
    args = ap.parse_args(argv)
    must = yaml.safe_load(Path(args.must_include).read_text()) or []
    conn = await connect(VendorkbSettings().vendorkb_dsn)
    try:
        train, heldout, stats = await select_chunks(
            conn, must=must, total=args.total, heldout_frac=args.heldout_frac, seed=args.seed,
            exclude=_read_ids(args.exclude))
    finally:
        await conn.close()
    Path(args.out_train).write_text("".join(f"{i}\n" for i in train), encoding="utf-8")
    Path(args.out_heldout).write_text("".join(f"{i}\n" for i in heldout), encoding="utf-8")
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
