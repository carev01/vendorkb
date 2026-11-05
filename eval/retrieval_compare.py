"""Free retrieval A/B between the live embedding column and the reembed candidate.

For every answerable question in ``eval/questions.yaml`` it runs retrieval the way
``vendorkb.answer.answer`` does: detect the scope, run hybrid search (relaxing an
empty scoped result), and stop before the paid rerank. It records the rank of the
first gold article in the 40-chunk rerank pool, and in the vector-only top 50.
Rerank and answer calls are never made; the only cost is one query embedding per
question per side.

Run (settings from .env; ``--candidate-*`` describe the embedding_next model):

    uv run python eval/retrieval_compare.py --candidate-model voyage-4 \
        --candidate-dim 1024 --candidate-base-url https://api.voyageai.com/v1 \
        --candidate-key-from rerank_api_key --out eval/runs/retrieval-ab.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

import yaml
from pgvector import Vector

from vendorkb.clients import Embedder
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect
from vendorkb.scope import detect_scope, scope_index_from_db
from vendorkb.search import RERANK_CANDIDATES, VEC_LIMIT, _filters, _vector_rows, search_docs


async def _vector_only(conn, embedder: Embedder, question: str, column: str,
                       vendors: list[str], products: list[str]) -> list[str]:
    where, fparams = _filters(vendors or None, products or None)
    qvec = Vector((await embedder.embed([question], input_type="query"))[0])
    sql = (f"SELECT doc_id::text FROM (SELECT c.doc_id, c.chunk_id, c.{column} <=> %s AS dist "
           f"FROM chunks c JOIN documents d USING (doc_id) WHERE true{where} "
           f"ORDER BY c.{column} <=> %s LIMIT %s) t ORDER BY dist, chunk_id")
    return [r[0] for r in await _vector_rows(conn, sql, [qvec, *fparams, qvec, VEC_LIMIT])]


def _first_gold(doc_ids: list[str], gold: set[str]) -> int | None:
    return next((i for i, d in enumerate(doc_ids, start=1) if d in gold), None)


async def _side(conn, embedder: Embedder, column: str, questions: list[dict],
                scope_index) -> dict[str, dict]:
    out = {}
    for q in questions:
        gold = {str(g["article_id"]).lower() for g in q["gold_sources"]}
        vendors, products = detect_scope(q["question"], scope_index)
        res = await search_docs(conn, q["question"], embedder=embedder, reranker=None,
                                vendor=vendors or None, product=products or None,
                                k=RERANK_CANDIDATES, use_rerank=False, vector_column=column)
        if (vendors or products) and not res.hits:
            vendors, products = [], []
            res = await search_docs(conn, q["question"], embedder=embedder, reranker=None,
                                    k=RERANK_CANDIDATES, use_rerank=False,
                                    vector_column=column)
        vec = await _vector_only(conn, embedder, q["question"], column, vendors, products)
        out[q["id"]] = {"pool_rank": _first_gold([h.doc_id for h in res.hits], gold),
                        "vector_rank": _first_gold(vec, gold)}
    return out


def _summary(ranks: dict[str, dict], key: str) -> dict:
    vals = [r[key] for r in ranks.values()]
    n = len(vals)
    return {"n": n,
            "recall@5": round(sum(1 for v in vals if v and v <= 5) / n, 3),
            "recall@10": round(sum(1 for v in vals if v and v <= 10) / n, 3),
            "recall@all": round(sum(1 for v in vals if v) / n, 3),
            "mrr": round(statistics.fmean(1 / v if v else 0 for v in vals), 3)}


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.yaml")
    ap.add_argument("--candidate-model", required=True)
    ap.add_argument("--candidate-dim", type=int, required=True)
    ap.add_argument("--candidate-base-url", required=True)
    ap.add_argument("--candidate-key-from", default="embed_api_key",
                    help="settings field holding the candidate provider's key")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    base = VendorkbSettings()
    cand = base.model_copy(update={
        "embed_model": args.candidate_model, "embed_dim": args.candidate_dim,
        "embed_base_url": args.candidate_base_url,
        "embed_api_key": getattr(base, args.candidate_key_from),
        "embed_input_type": True, "embed_output_dimension": args.candidate_dim})
    questions = [q for q in yaml.safe_load(Path(args.questions).read_text())
                 if q["type"] != "unanswerable" and q.get("gold_sources")]
    conn = await connect(base.vendorkb_dsn)
    try:
        scope_index = await scope_index_from_db(conn)
        async with Embedder(base) as e_live, Embedder(cand) as e_cand:
            live = await _side(conn, e_live, "embedding", questions, scope_index)
            new = await _side(conn, e_cand, "embedding_next", questions, scope_index)
            cand_tokens = e_cand.tokens_used
    finally:
        await conn.close()

    report: dict = {"live_model": base.embed_model, "candidate_model": args.candidate_model,
                    "candidate_query_tokens": cand_tokens, "groups": {}}
    groups = {"all": [q["id"] for q in questions],
              "deciding": [q["id"] for q in questions if q.get("decides")]}
    for t in sorted({q["type"] for q in questions}):
        groups[t] = [q["id"] for q in questions if q["type"] == t and q.get("decides")]
    for g, ids in groups.items():
        report["groups"][g] = {
            side: {key: _summary({i: ranks[i] for i in ids}, key)
                   for key in ("pool_rank", "vector_rank")}
            for side, ranks in (("live", live), ("candidate", new))}
    report["per_question"] = {i: {"live": live[i], "candidate": new[i]} for i in live}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for g, sides in report["groups"].items():
        for side, m in sides.items():
            print(f"{g:15} {side:9} pool {m['pool_rank']}  vector {m['vector_rank']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
