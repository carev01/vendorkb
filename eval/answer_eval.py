"""Answer-quality runner: every eval question through ``answer()``, outputs to JSONL.

    uv run python eval/answer_eval.py run --out eval/runs/answers-2026-11-06.jsonl \
        [--ids sv-a-01 xv-01] [--limit N] [--max-usd 10] [--max-rerank-tokens 3000000]
    uv run python eval/answer_eval.py summarize --answers eval/runs/answers-2026-11-06.jsonl \
        --grades eval/runs/answer-grades-2026-11-06.jsonl

``run`` is resumable (ids already in --out are skipped, a torn last line is ignored) and
stops before the next question once summed ``usage.cost`` reaches --max-usd or the rerank
tokens used this process reach --max-rerank-tokens. A missing cost counts as 0 and is
warned about. Each citation line carries the cited chunk text, url and vendor (looked up by
doc_id, one query per question) so graders need no DB access. Settings come from .env.

Grades JSONL (written by the grading step), one object per question::

    {"id": str, "shape": "single"|"comparison"|"enumeration",
     "facts_total": int, "facts_covered": int, "unsupported_claims": int,
     "refusal_correct": bool|null,      # unanswerable questions (and wrongly refused ones)
     "vendor_precision": float|null, "vendor_recall": float|null,   # enumeration
     "closing_line_wrong": int|null,    # enumeration: expected vendors the closing line lists
     "facets_answered": bool|null}      # comparison

``summarize`` prints per-shape, per-holdout (holdout flag read from the answers file) and
overall means; fact_coverage is sum(covered)/sum(total).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import yaml

from vendorkb.answer import answer
from vendorkb.clients import Embedder, Reranker
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect
from vendorkb.scope import scope_index_from_db

_ENUM_KEYS = ("vendors_listed", "vendors_checked", "facets_empty", "lines_dropped",
              "lines_naming_other_vendors")


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


async def run_questions(questions: list[dict], answer_fn: Callable[[str], Awaitable[dict]],
                        out: Path, *, embedder, reranker,
                        vendor_lookup: Callable[[list[str]], Awaitable[dict[str, str]]],
                        ids: set[str] | None = None, limit: int | None = None,
                        max_usd: float = 10.0, max_rerank_tokens: int = 3_000_000,
                        log: Callable[[str], None] = print) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out)
    todo = [q for q in questions if (ids is None or q["id"] in ids) and q["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    totals: dict = {"cost": 0.0, "rerank_tokens": 0, "embed_tokens": 0, "questions": 0,
                    "missing_cost": 0, "stopped": None}
    for q in todo:
        if totals["cost"] >= max_usd:
            totals["stopped"] = "max_usd"
            break
        if totals["rerank_tokens"] >= max_rerank_tokens:
            totals["stopped"] = "max_rerank_tokens"
            break
        e0, r0, t0 = embedder.tokens_used, reranker.tokens_used, time.perf_counter()
        try:
            res = await answer_fn(q["question"])
        except Exception as exc:  # recorded in the output, never swallowed
            res = {"answer": None, "citations": [], "usage": None, "shape": None,
                   "error": f"{type(exc).__name__}: {exc}"}
        seconds = round(time.perf_counter() - t0, 3)
        cites = res.get("citations") or []
        vendors = await vendor_lookup(sorted({c["doc_id"] for c in cites})) if cites else {}
        row = {"id": q["id"], "type": q["type"], "holdout": bool(q.get("holdout")),
               "shape": res.get("shape"), "question": q["question"],
               "answer": res.get("answer"),
               "citations": [{"marker": c.get("marker"), "chunk_id": c.get("chunk_id"),
                              "doc_id": c.get("doc_id"), "vendor": vendors.get(c["doc_id"]),
                              "url": c.get("url"), "title": c.get("title"),
                              "text": c.get("text")} for c in cites]}
        for k in _ENUM_KEYS:
            row[k] = res.get(k)
        row.update(rerank_ok=res.get("rerank_ok"), scope=res.get("scope"),
                   usage=res.get("usage"), error=res.get("error"), seconds=seconds,
                   rerank_tokens=reranker.tokens_used - r0,
                   embed_tokens=embedder.tokens_used - e0)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        cost = (row["usage"] or {}).get("cost")
        if cost is None:
            totals["missing_cost"] += 1
            if row["usage"] is not None:
                log(f"warning: {q['id']}: usage carries no cost (counted as 0)")
        totals["cost"] += cost or 0.0
        totals["rerank_tokens"] += row["rerank_tokens"]
        totals["embed_tokens"] += row["embed_tokens"]
        totals["questions"] += 1
        log(f"{q['id']}: {seconds}s error={row['error']} | totals cost=${totals['cost']:.4f} "
            f"rerank_tokens={totals['rerank_tokens']}")
    if totals["missing_cost"]:
        log(f"warning: {totals['missing_cost']} question(s) had no cost; the cap undercounts")
    if totals["stopped"]:
        log(f"STOPPED: {totals['stopped']} cap reached after {totals['questions']} questions")
    return totals


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 3) if xs else None


def _agg(rows: list[dict]) -> dict:
    def vals(k: str) -> list[float]:
        return [float(r[k]) for r in rows if r.get(k) is not None]
    total = sum(r.get("facts_total") or 0 for r in rows)
    covered = sum(r.get("facts_covered") or 0 for r in rows)
    wrong = vals("closing_line_wrong")
    return {"n": len(rows),
            "fact_coverage": round(covered / total, 3) if total else None,
            "unsupported_claims": sum(r.get("unsupported_claims") or 0 for r in rows),
            "refusal_correct": _mean(vals("refusal_correct")),
            "vendor_precision": _mean(vals("vendor_precision")),
            "vendor_recall": _mean(vals("vendor_recall")),
            "closing_line_wrong": int(sum(wrong)) if wrong else None,
            "facets_answered": _mean(vals("facets_answered"))}


def summarize_grades(answers: Path, grades: Path) -> dict:
    holdout = {}
    for line in answers.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            holdout[r["id"]] = bool(r.get("holdout"))
        except (json.JSONDecodeError, KeyError):
            continue
    rows = [json.loads(x) for x in grades.read_text(encoding="utf-8").splitlines() if x.strip()]
    out: dict = {"overall": _agg(rows), "by_shape": {}, "by_holdout": {}}
    for shape in sorted({r["shape"] for r in rows}):
        out["by_shape"][shape] = _agg([r for r in rows if r["shape"] == shape])
    for name, flag in (("calibration", False), ("holdout", True)):
        sub = [r for r in rows if holdout.get(r["id"], False) == flag]
        if sub:
            out["by_holdout"][name] = _agg(sub)
    return out


async def cmd_run(args, base: VendorkbSettings) -> int:
    from openai import AsyncOpenAI
    questions = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))
    conn = await connect(base.vendorkb_dsn)
    client = AsyncOpenAI(base_url=base.judge_base_url, api_key=base.judge_api_key)

    async def vendor_lookup(doc_ids: list[str]) -> dict[str, str]:
        cur = await conn.execute("SELECT doc_id::text, vendor FROM documents"
                                 " WHERE doc_id::text = ANY(%s)", [doc_ids])
        return {r[0]: r[1] for r in await cur.fetchall()}

    try:
        scope_index = await scope_index_from_db(conn)
        async with Embedder(base) as embedder, Reranker(base) as reranker:
            async def answer_fn(question: str) -> dict:
                return await answer(conn, question, settings=base, embedder=embedder,
                                    reranker=reranker, scope_index=scope_index, client=client)
            totals = await run_questions(
                questions, answer_fn, Path(args.out), embedder=embedder, reranker=reranker,
                vendor_lookup=vendor_lookup, ids=set(args.ids) if args.ids else None,
                limit=args.limit, max_usd=args.max_usd,
                max_rerank_tokens=args.max_rerank_tokens)
    finally:
        await client.close()
        await conn.close()
    print(json.dumps(totals, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--ids", nargs="+")
    r.add_argument("--limit", type=int)
    r.add_argument("--max-usd", type=float, default=10.0)
    r.add_argument("--max-rerank-tokens", type=int, default=3_000_000)
    s = sub.add_parser("summarize")
    s.add_argument("--answers", required=True)
    s.add_argument("--grades", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "summarize":
        print(json.dumps(summarize_grades(Path(args.answers), Path(args.grades)), indent=2))
        return 0
    return asyncio.run(cmd_run(args, VendorkbSettings()))


if __name__ == "__main__":
    sys.exit(main())
