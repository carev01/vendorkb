"""M4 measurement: does the capability table find the vendors that document a capability?

For each enumeration question in eval/questions.yaml, resolve its `key_terms` (and the question
itself) to taxonomy features and targets: exact alias or name match first, else the nearest
alias/name by voyage-4 embedding at cosine >= --nn-threshold. The table's answer is every vendor
with a capabilities row whose feature is one of those features, or whose target is one of those
targets or a descendant, with support supported/partial. Scored against `expected_vendors`
(unclear vendors excluded), next to retrieval enumeration (a scored rerank run's
vendor_best_scores at --floor) and the union of the two.

    uv run python eval/m4_measure.py --rerank-run eval/runs/cross-vendor-enum-fts025-2026-11-06.json \
        --floor 0.54 --out eval/runs/m4-measure-2026-11-06.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

import yaml

from vendorkb.clients import Embedder
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect
from vendorkb.taxonomy.mapper import _exact_index, nearest
from vendorkb.taxonomy.cluster import normalize

SUPPORTED = ("supported", "partial")


class _Index:
    """Alias/name keys of one kind, embedded ONCE and reused for every question (re-embedding
    the ~20k keys per question cost millions of tokens)."""

    def __init__(self, exact: dict[str, str], keys: list[str], vectors: list[list[float]]):
        self.exact, self.keys, self.vectors = exact, keys, vectors

    @classmethod
    async def build(cls, conn, embedder: Embedder, kind: str) -> _Index:
        exact = await _exact_index(conn, kind)  # type: ignore[arg-type]
        keys = sorted(exact)
        return cls(exact, keys, await embedder.embed(keys, input_type="document") if keys else [])

    def resolve(self, phrases: list[str], vectors: list[list[float]],
                threshold: float) -> dict[str, tuple[str, str, float]]:
        out: dict[str, tuple[str, str, float]] = {}
        for p, v in zip(phrases, vectors):
            if normalize(p) in self.exact:
                out[p] = (self.exact[normalize(p)], "alias", 1.0)
            elif self.keys:
                j, cos = nearest([v], self.vectors)[0]
                if cos >= threshold:
                    out[p] = (self.exact[self.keys[j]], "nn", round(cos, 3))
        return out


async def _table_vendors(conn, features: set[str], targets: set[str]) -> dict[str, list[str]]:
    """vendor -> chunk ids of its supporting capability rows."""
    if not features and not targets:
        return {}
    cur = await conn.execute(
        "SELECT c.vendor, r.chunk_id::text FROM capabilities c JOIN raw_claims r USING (claim_id)"
        " WHERE c.support = ANY(%s) AND (c.feature_id = ANY(%s) OR c.target_id = ANY(%s)"
        " OR EXISTS (SELECT 1 FROM unnest(%s::text[]) t WHERE starts_with(c.target_id, t || '.')))",
        (list(SUPPORTED), sorted(features), sorted(targets), sorted(targets)))
    out: dict[str, list[str]] = {}
    for vendor, chunk_id in await cur.fetchall():
        out.setdefault(vendor, []).append(chunk_id)
    return out


def _score(found: set[str], expected: set[str], unclear: set[str]) -> dict:
    exp = expected - unclear
    listed = found - unclear
    tp = listed & exp
    return {"recall": round(len(tp) / len(exp), 3) if exp else None,
            "precision": round(len(tp) / len(listed), 3) if listed else None,
            "found": len(listed), "tp": len(tp), "missed": sorted(exp - listed),
            "extra": sorted(listed - exp)}


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    return round(statistics.fmean(vals), 3) if vals else None


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.yaml")
    ap.add_argument("--rerank-run", required=True)
    ap.add_argument("--floor", type=float, default=0.54)
    ap.add_argument("--nn-threshold", type=float, default=0.75)
    ap.add_argument("--parts", default="eval/m3_pilot_ids.parts.json")
    ap.add_argument("--no-question", action="store_true",
                    help="resolve from key_terms only, not the question text")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    settings = VendorkbSettings()
    questions = [q for q in yaml.safe_load(Path(args.questions).read_text())
                 if q.get("shape") == "enumeration"]
    rerank = {r["id"]: r for r in json.load(open(args.rerank_run))["per_question"]}
    parts = json.load(open(args.parts)) if Path(args.parts).exists() else {}
    conn = await connect(settings.vendorkb_dsn)
    rows = []
    try:
        async with Embedder(settings) as embedder:
            findex = await _Index.build(conn, embedder, "feature")
            tindex = await _Index.build(conn, embedder, "target")
            all_phrases = [[*(q.get("key_terms") or [])] + ([] if args.no_question else [q["question"]])
                           for q in questions]
            flat = [p for ps in all_phrases for p in ps]
            flat_vecs = await embedder.embed(flat, input_type="document")
            vec_of = dict(zip(flat, flat_vecs))
            for q, phrases in zip(questions, all_phrases):
                pv = [vec_of[p] for p in phrases]
                feats = findex.resolve(phrases, pv, args.nn_threshold)
                tgts = tindex.resolve(phrases, pv, args.nn_threshold)
                table = await _table_vendors(conn, {v[0] for v in feats.values()},
                                             {v[0] for v in tgts.values()})
                expected = {e["vendor"] for e in q["expected_vendors"]}
                unclear = set(q.get("unclear_vendors") or [])
                best = rerank.get(q["id"], {}).get("vendor_best_scores") or {}
                retrieval = {v for v, s in best.items() if s >= args.floor}
                # a found vendor whose supporting chunks all came only from retrieval candidates
                only_c = {v for v, cids in table.items()
                          if cids and all(parts.get(c) == ["c"] for c in cids)}
                rows.append({
                    "id": q["id"], "holdout": bool(q.get("holdout")),
                    "features": feats, "targets": tgts,
                    "table": _score(set(table), expected, unclear),
                    "table_without_c_only": _score(set(table) - only_c, expected, unclear),
                    "retrieval": _score(retrieval, expected, unclear),
                    "union": _score(set(table) | retrieval, expected, unclear)})
    finally:
        await conn.close()
    summary: dict = {}
    for name in ("table", "table_without_c_only", "retrieval", "union"):
        for split, rs in (("all", rows), ("calibration", [r for r in rows if not r["holdout"]]),
                          ("holdout", [r for r in rows if r["holdout"]])):
            sub = [r[name] for r in rs]
            summary.setdefault(name, {})[split] = {
                "recall": _mean(sub, "recall"), "precision": _mean(sub, "precision"),
                "found_per_q": round(statistics.fmean(s["found"] for s in sub), 1)}
    summary["unresolved_questions"] = [r["id"] for r in rows
                                       if not r["features"] and not r["targets"]]
    Path(args.out).write_text(json.dumps({"summary": summary, "per_question": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
