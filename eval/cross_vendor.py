"""Cross-vendor eval: labelling candidates and vendor/facet recall scoring.

New question fields (cross-vendor questions in ``eval/questions.yaml``):
``shape`` (enumeration | comparison; absent -> ``detect_shape``), ``expected_vendors``
([{vendor, product, article_ids}]), ``unclear_vendors``, ``holdout`` (true for new questions)
and ``key_terms`` (enumeration; the labelling keyword search).

    uv run python eval/cross_vendor.py candidates --question-ids xv-01 xv-02 --out eval/runs/cand
    uv run python eval/cross_vendor.py score --out eval/runs/xv-free.json
    uv run python eval/cross_vendor.py score --legacy --out eval/runs/xv-legacy.json
    uv run python eval/cross_vendor.py score --rerank --out eval/runs/xv-rerank.json

Settings come from .env. ``candidates`` and a plain ``score`` spend only one query embedding
per question; only ``--rerank`` calls the reranker.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

import yaml

from vendorkb import search
from vendorkb.clients import Embedder, Reranker
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect
from vendorkb.modes import ENUMERATION_MIN_SCORE, compare_facets, enumerate_vendors
from vendorkb.scope import ScopeIndex, detect_scope, scope_index_from_db
from vendorkb.search import (
    _COLS,
    RERANK_CANDIDATES,
    _hit,
    hybrid_candidates,
    search_docs,
    vendor_candidates,
)
from vendorkb.shape import Facet, detect_shape

# ---------------------------------------------------------------- pure scoring

def vendor_recall(expected: set[str], unclear: set[str], hit_vendors: set[str]) -> float | None:
    """Share of expected vendors found among the hits; unclear vendors count on neither
    side. None (not 0) when nothing is left to find."""
    exp = expected - unclear
    if not exp:
        return None
    return len(exp & (hit_vendors - unclear)) / len(exp)


def vendor_noise(expected: set[str], unclear: set[str], hit_vendors: set[str]) -> list[str]:
    """Hit vendors that are neither expected nor unclear, sorted."""
    return sorted(hit_vendors - expected - unclear)


def facet_recall(gold_doc_ids_by_facet: dict[str, set[str]], hits) -> float | None:
    """Mean over facets of 1 if any hit is one of the facet's gold documents, else 0.
    Facets without gold documents are not scored; None when none is."""
    gold = {f: ids for f, ids in gold_doc_ids_by_facet.items() if ids}
    if not gold:
        return None
    got = {h.doc_id for h in hits}
    return sum(1 for ids in gold.values() if ids & got) / len(gold)


def gold_by_facet(facets: list[Facet], gold_doc_ids: set[str],
                  docs: dict[str, tuple[str | None, str | None]]) -> dict[str, set[str]]:
    """Gold doc ids per facet; ``docs`` maps doc_id -> (vendor, product)."""
    out: dict[str, set[str]] = {name: set() for _, name in facets}
    for d in gold_doc_ids:
        vendor, product = docs.get(d, (None, None))
        for kind, name in facets:
            if (vendor if kind == "vendor" else product) == name:
                out[name].add(d)
    return out


def first_gold(doc_ids: list[str], gold: set[str]) -> int | None:
    """Pool recall rank (same rule as eval/retrieval_compare._first_gold)."""
    return next((i for i, d in enumerate(doc_ids, start=1) if d in gold), None)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[max(1, math.ceil(p / 100 * len(s))) - 1]


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 3) if xs else None


def summarize(results: list[dict], *, rerank: bool) -> dict:
    """Metrics per split (calibration = holdout false, holdout = holdout true)."""
    out: dict = {}
    for split, flag in (("calibration", False), ("holdout", True)):
        all_rs = [r for r in results if bool(r.get("holdout")) == flag]
        if not all_rs:
            continue
        rs = [r for r in all_rs if "error" not in r]
        s: dict = {"n_errors": len(all_rs) - len(rs)}
        enum = [r for r in rs if r.get("shape") == "enumeration" and "vendor_recall" in r]
        if enum:
            # recall and noise cover labelled questions only (expected vendors beyond the
            # unclear ones); an unlabelled question's every hit vendor would count as noise
            labelled = [r for r in enum if r["vendor_recall"] is not None]
            s["enumeration"] = {
                "n": len(enum), "n_scored": len(labelled),
                "n_unlabelled": len(enum) - len(labelled),
                "vendor_recall": _mean([r["vendor_recall"] for r in labelled]),
                "noise_per_question": _mean([len(r["vendor_noise"]) for r in labelled])}
        comp = [r for r in rs if r.get("shape") == "comparison" and "facet_recall" in r]
        if comp:
            scored = [r["facet_recall"] for r in comp if r["facet_recall"] is not None]
            s["comparison"] = {"n": len(comp), "n_scored": len(scored),
                               "facet_recall": _mean(scored)}
        for typ in ("single_vendor", "multi_article"):
            pool = [r for r in rs if r["type"] == typ and "pool_rank" in r]
            if pool:
                ranks = [r["pool_rank"] for r in pool]
                s[typ] = {"n": len(pool),
                          "pool_recall": _mean([1.0 if v else 0.0 for v in ranks]),
                          "recall@10": _mean([1.0 if v and v <= 10 else 0.0 for v in ranks])}
        if rerank:
            s["rerank_tokens_per_question"] = _mean([r.get("rerank_tokens", 0) for r in rs])
            s["rerank_failures"] = sum(1 for r in rs if r.get("rerank_ok") is False)
        lat: dict = {}
        for shape in sorted({r["shape"] for r in rs if "latency_s" in r}):
            vals = [r["latency_s"] for r in rs if r["shape"] == shape and "latency_s" in r]
            lat[shape] = {"p50": percentile(vals, 50), "p90": percentile(vals, 90)}
        s["latency_s"] = lat
        out[split] = s
    return out


# ---------------------------------------------------------------- DB-backed pieces

def _record(h) -> dict:
    return {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "vendor": h.vendor,
            "product": h.product, "title": h.title,
            "section_path": h.section_path, "text": h.text, "url": h.url}


async def _fts_hits_by_vendor(conn, term: str, vendors: list[str], limit: int) -> dict[str, list]:
    """Top ``limit`` full-text hits per vendor for one key term, in one query. The term is a
    phrase (phraseto_tsquery), so "pre-job script" matches the phrase, not any of its words."""
    sql = (f"SELECT * FROM (SELECT {_COLS}, row_number() OVER (PARTITION BY d.vendor "
           "ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.chunk_id) AS rn "
           "FROM chunks c JOIN documents d USING (doc_id), "
           "(SELECT phraseto_tsquery('english', %s) AS tsq) q "
           "WHERE c.tsv @@ q.tsq AND d.vendor = ANY(%s)) t WHERE rn <= %s")
    cur = await conn.execute(sql, [term, list(vendors), limit])
    out: dict[str, list] = {}
    for r in await cur.fetchall():
        out.setdefault(r[4], []).append(_hit(r[:-1], r[-1]))
    return out


async def _doc_rows(conn, doc_ids: set[str]) -> dict[str, tuple[str | None, str | None]]:
    if not doc_ids:
        return {}
    cur = await conn.execute("SELECT doc_id::text, vendor, product FROM documents"
                             " WHERE doc_id::text = ANY(%s)", [sorted(doc_ids)])
    return {r[0]: (r[1], r[2]) for r in await cur.fetchall()}


def _gold_ids(q: dict) -> set[str]:
    return {str(g["article_id"]).lower() for g in q.get("gold_sources") or []}


async def _single(conn, embedder, question, scope_index, *, legacy: bool,
                  reranker, rerank: bool):
    """The single-facet path: detected scope, relaxed when it grounds nothing."""
    vendors, products = detect_scope(question, scope_index)
    if legacy or rerank:
        k = 10 if rerank else RERANK_CANDIDATES

        async def run(v, p):
            return await search_docs(conn, question, embedder=embedder, reranker=reranker,
                                     vendor=v or None, product=p or None, k=k,
                                     use_rerank=rerank)
        res = await run(vendors, products)
        if (vendors or products) and not res.hits:
            res = await run([], [])
        return res.hits, res.rerank_ok or not res.hits
    qvec = (await embedder.embed([question], input_type="query"))[0]
    hits = await hybrid_candidates(conn, question, qvec, vendor=vendors or None,
                                   product=products or None)
    if (vendors or products) and not hits:
        hits = await hybrid_candidates(conn, question, qvec)
    return hits, True


async def _run_question(conn, q: dict, *, embedder, reranker, scope_index: ScopeIndex,
                        args) -> dict:
    question = q["question"]
    rerank = bool(args.rerank)
    res: dict = {"id": q["id"], "type": q["type"], "holdout": bool(q.get("holdout")),
                 "shape": q.get("shape")}
    e0 = embedder.tokens_used
    r0 = reranker.tokens_used if reranker else 0
    t0 = time.perf_counter()
    try:
        detected = detect_shape(question, scope_index)
        shape = q.get("shape") or detected.kind
        res.update(shape=shape, detected_shape=detected.kind)
        rerank_ok: bool | None = None
        hits: list = []
        if args.legacy:
            hits, rerank_ok = await _single(conn, embedder, question, scope_index,
                                            legacy=True, reranker=reranker, rerank=rerank)
        elif shape == "enumeration":
            er = await enumerate_vendors(conn, question, embedder=embedder, reranker=reranker,
                                         vendors=scope_index.vendors, min_score=args.min_score,
                                         use_rerank=rerank)
            hits, rerank_ok = er.hits, er.rerank_ok
            res.update(kept=len(er.kept), dropped=len(er.dropped), empty=len(er.empty),
                       dropped_vendors=er.dropped)
        elif shape == "comparison":
            cr = await compare_facets(conn, question, embedder=embedder, reranker=reranker,
                                      facets=detected.facets, use_rerank=rerank)
            hits, rerank_ok = cr.hits, cr.rerank_ok or not cr.hits
            res.update(facets=[n for _, n in detected.facets], facets_empty=cr.facets_empty)
            if len(detected.facets) < 2:
                res["shape_mismatch"] = "comparison with fewer than 2 detected facets"
        else:
            hits, rerank_ok = await _single(conn, embedder, question, scope_index,
                                            legacy=False, reranker=reranker, rerank=rerank)
        if rerank:  # only meaningful when a rerank was asked for
            res["rerank_ok"] = bool(rerank_ok)
        res["pool_depth"] = "top10-reranked" if rerank else f"top{RERANK_CANDIDATES}-rrf"
        res["hit_doc_ids"] = [h.doc_id for h in hits]
        res["hit_vendors"] = sorted({h.vendor for h in hits})

        gold = _gold_ids(q)
        if q["type"] == "cross_vendor" and shape == "enumeration":
            expected = {e["vendor"] for e in q.get("expected_vendors") or []}
            unclear = set(q.get("unclear_vendors") or [])
            got = {h.vendor for h in hits}
            res["vendor_recall"] = vendor_recall(expected, unclear, got)
            res["vendor_noise"] = vendor_noise(expected, unclear, got)
            res["missing_vendors"] = sorted(expected - unclear - got)
        elif q["type"] == "cross_vendor" and shape == "comparison":
            gbf = gold_by_facet(detected.facets, gold, await _doc_rows(conn, gold))
            res["facet_recall"] = facet_recall(gbf, hits)
            res["facets_without_gold"] = sorted(f for f, ids in gbf.items() if not ids)
        elif q["type"] == "cross_vendor":
            res["shape_mismatch"] = f"cross_vendor question resolved to shape {shape}"
        if q["type"] in ("single_vendor", "multi_article") and gold:
            res["pool_rank"] = first_gold([h.doc_id for h in hits], gold)
    except Exception as exc:  # recorded in the output, never swallowed
        res["error"] = f"{type(exc).__name__}: {exc}"
    res["latency_s"] = round(time.perf_counter() - t0, 3)
    res["rerank_tokens"] = (reranker.tokens_used - r0) if reranker else 0
    res["embed_tokens"] = embedder.tokens_used - e0
    return res


def load_done(path: Path) -> list[dict]:
    """Results already written to the JSONL (errored and rerank-failed ones are not done: they run again).
    A torn last line (killed mid-write) is ignored."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" not in row and row.get("rerank_ok") is not False:
            rows.append(row)
    return rows


# ---------------------------------------------------------------- commands

def _load(path: str) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)


async def cmd_candidates(args, base: VendorkbSettings) -> int:
    wanted = set(args.question_ids)
    questions = [q for q in _load(args.questions) if q["id"] in wanted]
    missing = wanted - {q["id"] for q in questions}
    if missing:
        print(f"unknown question ids: {sorted(missing)}", file=sys.stderr)
        return 2
    conn = await connect(base.vendorkb_dsn)
    try:
        vendors = (await scope_index_from_db(conn)).vendors
        async with Embedder(base) as embedder:
            for q in questions:
                qvec = (await embedder.embed([q["question"]], input_type="query"))[0]
                # Batched: per-vendor scoped searches cost ~4 s each on the full corpus.
                tops = await vendor_candidates(conn, q["question"], qvec, vendors, limit=10)
                by_term = {t: await _fts_hits_by_vendor(conn, t, vendors, 5)
                           for t in q.get("key_terms") or []}
                for v in vendors:
                    out = Path(args.out) / q["id"] / f"{_safe(v)}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(
                        {"question_id": q["id"], "vendor": v,
                         "candidates": [_record(h) for h in tops.get(v, [])],
                         "key_term_hits": {t: [_record(h) for h in hits.get(v, [])]
                                           for t, hits in by_term.items()}},
                        indent=2), encoding="utf-8")
                print(f"{q['id']}: {len(vendors)} vendors written")
    finally:
        await conn.close()
    return 0


async def cmd_score(args, base: VendorkbSettings) -> int:
    search.FTS_WEIGHT = args.fts_weight
    if args.min_score != ENUMERATION_MIN_SCORE and not args.rerank:
        print("warning: --min-score has no effect without --rerank", file=sys.stderr)
    questions = [q for q in _load(args.questions) if q["type"] != "unanswerable"]
    if args.question_ids:
        questions = [q for q in questions if q["id"] in set(args.question_ids)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.with_suffix(out.suffix + "l")  # FILE.json -> FILE.jsonl
    results: list[dict] = load_done(jsonl) if args.resume else []
    # rewrite the JSONL so it holds exactly the kept rows, then append as questions finish
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    done = {r["id"] for r in results}
    conn = await connect(base.vendorkb_dsn)
    try:
        scope_index = await scope_index_from_db(conn)
        async with Embedder(base) as embedder:
            reranker = Reranker(base) if args.rerank else None
            try:
                for q in questions:
                    if q["id"] in done:
                        continue
                    r = await _run_question(conn, q, embedder=embedder, reranker=reranker,
                                            scope_index=scope_index, args=args)
                    results.append(r)
                    with jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(r) + "\n")
            finally:
                if reranker:
                    await reranker.aclose()
    finally:
        await conn.close()
    order = {q["id"]: i for i, q in enumerate(questions)}
    results.sort(key=lambda r: order.get(r["id"], len(order)))
    report = {"mode": "legacy" if args.legacy else "new", "rerank": bool(args.rerank),
              "min_score": args.min_score, "fts_weight": args.fts_weight,
              "summary": summarize(results, rerank=bool(args.rerank)),
              "per_question": results}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    errors = [r["id"] for r in results if "error" in r]
    if errors:
        print(f"errors on: {errors}", file=sys.stderr)
    failed = [r["id"] for r in results if r.get("rerank_ok") is False]
    if failed:
        print(f"RERANK FAILED (results not reranked) on: {failed}", file=sys.stderr)
    return 1 if errors or failed else 0


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--question-ids", nargs="+", required=True)
    c.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--rerank", action="store_true", help="call the reranker (paid)")
    s.add_argument("--legacy", action="store_true",
                   help="every question through search_docs with scope detection")
    s.add_argument("--min-score", type=float, default=ENUMERATION_MIN_SCORE)
    s.add_argument("--fts-weight", type=float, default=search.FTS_WEIGHT)
    s.add_argument("--question-ids", nargs="+")
    s.add_argument("--resume", action="store_true",
                   help="skip questions already (successfully) in OUT's .jsonl")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    base = VendorkbSettings()
    return await (cmd_candidates if args.cmd == "candidates" else cmd_score)(args, base)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
