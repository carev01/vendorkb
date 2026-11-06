"""Findability checks for a comparison-table coverage map (the map stays outside the repo).

The map is a YAML list of ``{table, row, vendor, ids, out_of_scope}``: ``vendor`` is the
table's vendor (or null), ``ids`` the taxonomy v2 ids the row maps to, ``out_of_scope`` a
reason text (or null). An id is a dimension id, ``target:<target_id>`` or
``object:<class_id>:<object_type_id>`` (the object type must be in that class's lineage).

    uv run python eval/findability.py coverage --map PATH --taxonomy taxonomy/taxonomy.yaml \
        [--json OUT]

``coverage`` checks that every in-scope row has at least one id and that every id exists in the
taxonomy; it exits 1 if a row is unmapped or an id is unknown. It prints only table names, row
labels and ids, never anything else from the map.

    uv run python eval/findability.py claims --map PATH [--dsn CAPABILITIES_DSN] [--json OUT]

``claims`` asks ``capabilities.query`` (in the database ``--dsn`` names, the one holding the
capabilities table) whether each in-scope row with a vendor and ids has at least one fact for
that vendor. A row's queries are its targets (``target:`` ids, or none) crossed with its
aspects (each ``object:`` id gives one query per required coverage dimension with the object
type filtered, each dimension id one query; none means the target alone). The row has a fact
if any query returns a row in ``facts``, ``product_level`` or ``product_wide``; the report
counts each row once, in the most specific group that returned something.

    uv run python eval/findability.py chunks-candidates --map PATH --claims-report JSON \\
        --taxonomy taxonomy/taxonomy.yaml --out JSONL [--dsn SEARCH_DSN] [--limit N]

``chunks-candidates`` takes the rows ``claims --json`` found no fact for, builds the question
"<vendor> <target name> <dimension or object name>" for each, runs hybrid search scoped to the
row's vendor with Voyage rerank (paid: ``--limit`` caps the rows, the call counts are printed)
and writes the top 10 chunks as JSONL ``{row_key, chunk_id, text}``. It does not touch the
capabilities table, so ``--dsn`` is the search database.

    uv run python eval/findability.py chunks-score --candidates JSONL --labels JSONL [--json OUT]

``chunks-score`` reads the relevance labels ``{row_key, chunk_id, relevant}`` and reports the
rows with at least one relevant chunk among their candidates.

Every report prints only table names, row labels (as ``table|row`` keys) and ids.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import psycopg
import yaml

from vendorkb.clients import Embedder, Reranker
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect
from vendorkb.search import search_docs
from vendorkb.taxonomy.capabilities import CapFilter, query
from vendorkb.taxonomy.load import Taxonomy, parse_doc, validate

COVERAGE_DIMENSIONS = ("coverage.backup", "coverage.restore", "coverage.export",
                       "coverage.compare")
GROUPS = ("facts", "product_level", "product_wide")  # most specific first
TOP_K = 10


def _known_ids(taxonomy: Taxonomy) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    dimensions = {d.id for d in taxonomy.dimensions}
    targets = {t.id for t in taxonomy.targets}
    objects = {(c.id, o.id) for c in taxonomy.classes for o in taxonomy.object_types(c.id)}
    return dimensions, targets, objects


def _id_is_known(id_: str, dimensions: set[str], targets: set[str],
                 objects: set[tuple[str, str]]) -> bool:
    if id_.startswith("target:"):
        return id_.removeprefix("target:") in targets
    if id_.startswith("object:"):
        class_id, sep, object_type = id_.removeprefix("object:").rpartition(":")
        return bool(sep) and (class_id, object_type) in objects
    return id_ in dimensions


def coverage_report(map_rows: list[dict[str, Any]], taxonomy: Taxonomy) -> dict[str, Any]:
    """Count the map's rows and list the unmapped rows and unknown ids.

    A row with a non-empty ``out_of_scope`` is out of scope (its ids are not checked). An
    in-scope row is ``mapped`` if it has any id and ``unmapped`` if it has none; an id the
    taxonomy lacks is listed in ``unknown_ids`` whether or not the row has other ids.
    """
    dimensions, targets, objects = _known_ids(taxonomy)
    out_of_scope = 0
    mapped = 0
    unmapped: list[dict[str, str]] = []
    unknown: list[dict[str, str]] = []
    for row in map_rows:
        if row.get("out_of_scope"):
            out_of_scope += 1
            continue
        ids = row.get("ids") or []
        if not ids:
            unmapped.append({"table": row["table"], "row": row["row"]})
            continue
        mapped += 1
        for id_ in ids:
            if not isinstance(id_, str) or not _id_is_known(id_, dimensions, targets, objects):
                unknown.append({"table": row["table"], "row": row["row"], "id": str(id_)})
    return {"rows": len(map_rows), "out_of_scope": out_of_scope,
            "in_scope": len(map_rows) - out_of_scope, "mapped": mapped,
            "unmapped": unmapped, "unknown_ids": unknown}


def load_map(path: str | Path) -> list[dict[str, Any]]:
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: the map must be a YAML list of rows")
    for n, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("table") or not row.get("row"):
            raise ValueError(f"{path}: row {n} needs a table and a row label")
        if not isinstance(row.get("ids") or [], list):
            raise ValueError(f"{path}: row {n} ids must be a list")
    return rows


def cmd_coverage(args: argparse.Namespace) -> int:
    taxonomy = validate(parse_doc(Path(args.taxonomy).read_bytes(), args.taxonomy))
    report = coverage_report(load_map(args.map), taxonomy)
    for key in ("rows", "out_of_scope", "in_scope", "mapped"):
        print(f"{key}: {report[key]}")
    print(f"unmapped: {len(report['unmapped'])}")
    for u in report["unmapped"]:
        print(f"  {u['table']} | {u['row']}")
    print(f"unknown_ids: {len(report['unknown_ids'])}")
    for u in report["unknown_ids"]:
        print(f"  {u['table']} | {u['row']} | {u['id']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if report["unmapped"] or report["unknown_ids"] else 0


def row_key(row: dict[str, Any]) -> str:
    return f"{row['table']}|{row['row']}"


def _split_object_id(id_: str) -> tuple[str, str]:
    class_id, _, object_type = id_.removeprefix("object:").rpartition(":")
    return class_id, object_type


def row_filters(vendor: str, ids: Iterable[str]) -> list[CapFilter]:
    """The capability queries for one row: its targets crossed with its aspects."""
    targets: list[str | None] = []
    aspects: list[tuple[str | None, str | None]] = []  # (dimension, object_type)
    for id_ in ids:
        if id_.startswith("target:"):
            targets.append(id_.removeprefix("target:"))
        elif id_.startswith("object:"):
            object_type = _split_object_id(id_)[1]
            aspects.extend((d, object_type) for d in COVERAGE_DIMENSIONS)
        else:
            aspects.append((id_, None))
    return [CapFilter(vendor=vendor, target=t, dimension=d, object_type=o)
            for t in targets or [None] for d, o in aspects or [(None, None)]]


async def claims_report(conn: psycopg.AsyncConnection,
                        map_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which in-scope rows have at least one capability fact for their vendor.

    Rows out of scope, without a vendor or without ids are skipped (counted, not queried).
    ``with_fact`` is split by the most specific group that answered; ``without_fact`` lists
    the rows no query answered (what ``chunks-candidates`` takes)."""
    out_of_scope = no_vendor = no_ids = 0
    by_group = dict.fromkeys(GROUPS, 0)
    without: list[dict[str, str]] = []
    checked = 0
    for row in map_rows:
        if row.get("out_of_scope"):
            out_of_scope += 1
        elif not row.get("vendor"):
            no_vendor += 1
        elif not row.get("ids"):
            no_ids += 1
        else:
            checked += 1
            found: set[str] = set()
            for f in row_filters(row["vendor"], row["ids"]):
                result = await query(conn, f)
                found.update(g for g in GROUPS if getattr(result, g))
            best = next((g for g in GROUPS if g in found), None)
            if best is None:
                without.append({"table": row["table"], "row": row["row"],
                                "row_key": row_key(row)})
            else:
                by_group[best] += 1
    return {"rows": len(map_rows), "out_of_scope": out_of_scope, "no_vendor": no_vendor,
            "no_ids": no_ids, "checked": checked, "with_fact": checked - len(without),
            "with_fact_by_group": by_group, "without_fact": without}


def _names(taxonomy: Taxonomy, ids: Iterable[str]) -> tuple[str | None, str | None]:
    """The first target's name and the first aspect's name (an object type before a
    dimension), None where the row has none."""
    dimensions = {d.id: d.name for d in taxonomy.dimensions}
    targets = {t.id: t.name for t in taxonomy.targets}
    objects = {(c.id, o.id): o.name for c in taxonomy.classes
               for o in taxonomy.object_types(c.id)}
    target = obj = dim = None
    for id_ in ids:
        if id_.startswith("target:"):
            target = target or targets.get(id_.removeprefix("target:"))
        elif id_.startswith("object:"):
            obj = obj or objects.get(_split_object_id(id_))
        else:
            dim = dim or dimensions.get(id_)
    return target, obj or dim


def build_question(vendor: str, taxonomy: Taxonomy, ids: Iterable[str]) -> str:
    target, aspect = _names(taxonomy, ids)
    return " ".join(p for p in (vendor, target, aspect) if p)


class _Counting:
    """Counts the paid calls of an embedder or reranker."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0


class CountingEmbedder(_Counting):
    async def embed(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return await self._inner.embed(*args, **kwargs)


class CountingReranker(_Counting):
    async def rerank(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return await self._inner.rerank(*args, **kwargs)


async def chunk_candidates(conn: psycopg.AsyncConnection, map_rows: list[dict[str, Any]],
                           claims: dict[str, Any], taxonomy: Taxonomy, *,
                           embedder: Any, reranker: Any,
                           limit: int | None = None) -> dict[str, Any]:
    """The top 10 reranked chunks for each row the claims report found no fact for.

    Returns ``{"candidates": [{row_key, chunk_id, text}], "rows": n searched, "no_hits":
    [row_key], "rerank_failed": [row_key]}``; a row whose rerank failed gets no candidates
    (an unreranked pool would not be the pool being measured). ``limit`` caps the rows."""
    wanted = [w["row_key"] for w in claims["without_fact"]]
    by_key = {row_key(r): r for r in map_rows}
    unknown = [k for k in wanted if k not in by_key]
    if unknown:
        raise ValueError(f"claims report rows not in the map: {unknown}")
    candidates: list[dict[str, str]] = []
    no_hits: list[str] = []
    rerank_failed: list[str] = []
    searched = 0
    for key in wanted[:limit]:
        row = by_key[key]
        question = build_question(row["vendor"], taxonomy, row.get("ids") or [])
        res = await search_docs(conn, question, embedder=embedder, reranker=reranker,
                                vendor=[row["vendor"]], k=TOP_K)
        searched += 1
        if not res.hits:
            no_hits.append(key)
        elif not res.rerank_ok:
            rerank_failed.append(key)
        else:
            candidates.extend({"row_key": key, "chunk_id": h.chunk_id, "text": h.text}
                              for h in res.hits)
    return {"candidates": candidates, "rows": searched, "no_hits": no_hits,
            "rerank_failed": rerank_failed}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def chunks_score(candidates: list[dict[str, Any]],
                 labels: list[dict[str, Any]]) -> dict[str, Any]:
    """The rows with at least one chunk labelled relevant, among the rows with candidates.

    A candidate without a label counts in ``unlabelled`` (and is not relevant); a label for a
    chunk that is not a candidate of its row counts in ``stray_labels`` and is ignored. A chunk
    labelled twice takes the last label."""
    label_of = {(lab["row_key"], lab["chunk_id"]): bool(lab["relevant"]) for lab in labels}
    pairs = {(c["row_key"], c["chunk_id"]) for c in candidates}
    rows: dict[str, bool] = {}
    for key, chunk in sorted(pairs):
        rows[key] = rows.get(key, False) or label_of.get((key, chunk), False)
    return {"rows": len(rows), "with_relevant": sum(rows.values()),
            "without_relevant": sorted(k for k, ok in rows.items() if not ok),
            "unlabelled": len(pairs - label_of.keys()),
            "stray_labels": len(label_of.keys() - pairs)}


def _dsn(args: argparse.Namespace) -> str:
    return str(args.dsn or VendorkbSettings().vendorkb_dsn)


async def _cmd_claims(args: argparse.Namespace) -> int:
    conn = await connect(_dsn(args))
    try:
        report = await claims_report(conn, load_map(args.map))
    finally:
        await conn.close()
    for key in ("rows", "out_of_scope", "no_vendor", "no_ids", "checked", "with_fact"):
        print(f"{key}: {report[key]}")
    for group, n in report["with_fact_by_group"].items():
        print(f"  {group}: {n}")
    print(f"without_fact: {len(report['without_fact'])}")
    for w in report["without_fact"]:
        print(f"  {w['row_key']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


async def _cmd_chunks_candidates(args: argparse.Namespace) -> int:
    taxonomy = validate(parse_doc(Path(args.taxonomy).read_bytes(), args.taxonomy))
    claims = json.loads(Path(args.claims_report).read_text(encoding="utf-8"))
    settings = VendorkbSettings()
    conn = await connect(_dsn(args))
    try:
        async with Embedder(settings) as embedder:
            counted_embedder = CountingEmbedder(embedder)
            counted_reranker = CountingReranker(Reranker(settings))
            result = await chunk_candidates(
                conn, load_map(args.map), claims, taxonomy, embedder=counted_embedder,
                reranker=counted_reranker, limit=args.limit)
    finally:
        await conn.close()
    Path(args.out).write_text(
        "".join(json.dumps(c) + "\n" for c in result["candidates"]), encoding="utf-8")
    print(f"rows searched: {result['rows']}")
    print(f"candidates written: {len(result['candidates'])}")
    print(f"embed calls: {counted_embedder.calls}")
    print(f"rerank calls: {counted_reranker.calls}")
    for name in ("no_hits", "rerank_failed"):
        print(f"{name}: {len(result[name])}")
        for key in result[name]:
            print(f"  {key}")
    return 1 if result["rerank_failed"] else 0


def _cmd_chunks_score(args: argparse.Namespace) -> int:
    report = chunks_score(_read_jsonl(args.candidates), _read_jsonl(args.labels))
    for key in ("rows", "with_relevant", "unlabelled", "stray_labels"):
        print(f"{key}: {report[key]}")
    print(f"without_relevant: {len(report['without_relevant'])}")
    for key in report["without_relevant"]:
        print(f"  {key}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage", help="check every map row points at taxonomy v2 ids")
    c.add_argument("--map", required=True, help="the coverage map YAML (kept outside the repo)")
    c.add_argument("--taxonomy", required=True)
    c.add_argument("--json", help="also write the report here")
    cl = sub.add_parser("claims", help="do the capabilities facts answer each row's vendor")
    cl.add_argument("--map", required=True)
    cl.add_argument("--dsn", help="the database holding the capabilities table "
                                  "(default: the settings' vendorkb_dsn)")
    cl.add_argument("--json", help="also write the report here (chunks-candidates reads it)")
    cc = sub.add_parser("chunks-candidates",
                        help="top-10 reranked chunks for the rows with no fact (paid: Voyage)")
    cc.add_argument("--map", required=True)
    cc.add_argument("--claims-report", required=True, help="the JSON written by claims --json")
    cc.add_argument("--taxonomy", required=True)
    cc.add_argument("--out", required=True, help="candidates JSONL")
    cc.add_argument("--dsn", help="the search database (default: the settings' vendorkb_dsn)")
    cc.add_argument("--limit", type=int, help="search only the first N rows (smoke runs)")
    cs = sub.add_parser("chunks-score", help="rows with at least one relevant labelled chunk")
    cs.add_argument("--candidates", required=True)
    cs.add_argument("--labels", required=True, help="JSONL {row_key, chunk_id, relevant}")
    cs.add_argument("--json", help="also write the report here")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "coverage":
        return cmd_coverage(args)
    if args.cmd == "chunks-score":
        return _cmd_chunks_score(args)
    cmd = _cmd_claims if args.cmd == "claims" else _cmd_chunks_candidates
    return asyncio.run(cmd(args))


if __name__ == "__main__":
    sys.exit(main())
