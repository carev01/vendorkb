"""``python -m vendorkb.cli`` -- search / answer / stats / sync / reembed / promote-embedding."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys

import psycopg
from openai import AsyncOpenAI

from vendorkb.answer import answer
from vendorkb.clients import DocExtractor, Embedder, Reranker
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema
from vendorkb.reembed import promote, reembed
from vendorkb.scope import scope_index_from_db
from vendorkb.search import search_docs
from vendorkb.sync import sync

MAX_TOKENS = 1200


def _snippet(text: str) -> str:
    """First 200 chars of the chunk body, after the 'title / trail' prefix."""
    head, sep, body = text.partition("\n\n")
    return (body if sep else head)[:200]


async def _search(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    conn = await connect(settings.vendorkb_dsn)
    try:
        async with Embedder(settings) as embedder:
            if args.rerank:
                async with Reranker(settings) as reranker:
                    result = await search_docs(
                        conn, args.query, embedder=embedder, reranker=reranker,
                        vendor=args.vendor or None, product=args.product or None,
                        k=args.k, use_rerank=True)
            else:
                result = await search_docs(
                    conn, args.query, embedder=embedder, reranker=None,
                    vendor=args.vendor or None, product=args.product or None,
                    k=args.k, use_rerank=False)
    finally:
        await conn.close()
    for rank, h in enumerate(result.hits, 1):
        print(json.dumps({
            "rank": rank, "chunk_id": h.chunk_id, "vendor": h.vendor, "product": h.product,
            "title": h.title, "section_path": h.section_path, "rrf_rank": h.rrf_rank,
            "rerank_score": h.rerank_score, "snippet": _snippet(h.text)}))
    return 0


async def _answer(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    conn = await connect(settings.vendorkb_dsn)
    client = AsyncOpenAI(base_url=settings.judge_base_url, api_key=settings.judge_api_key)
    try:
        scope_index = await scope_index_from_db(conn)
        async with Embedder(settings) as embedder, Reranker(settings) as reranker:
            out = await answer(conn, args.question, settings=settings, embedder=embedder,
                               reranker=reranker, scope_index=scope_index, client=client)
    finally:
        await client.close()
        await conn.close()
    print(json.dumps(out, indent=2, default=str))
    return 0


async def _stats(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    conn = await connect(settings.vendorkb_dsn)
    try:
        cur = await conn.execute("SELECT count(*) FROM documents")
        documents = (await cur.fetchone())[0]  # type: ignore[index]
        cur = await conn.execute(
            "SELECT count(*), percentile_disc(ARRAY[0.5,0.9,0.99]) "
            "WITHIN GROUP (ORDER BY token_count), max(token_count), "
            "count(*) FILTER (WHERE token_count > %s) FROM chunks", (MAX_TOKENS,))
        chunks, pcts, tmax, over = await cur.fetchone()  # type: ignore[misc]
        cur = await conn.execute(
            "SELECT d.vendor, count(*) FROM chunks c JOIN documents d USING (doc_id) "
            "GROUP BY d.vendor ORDER BY d.vendor")
        per_vendor = {v: n for v, n in await cur.fetchall()}
        sync_info = await _sync_stats(conn)
    finally:
        await conn.close()
    p50, p90, p99 = pcts if pcts else (None, None, None)
    print(json.dumps({
        "documents": documents, "chunks": chunks, "chunks_per_vendor": per_vendor,
        "token_percentiles": {"p50": p50, "p90": p90, "p99": p99, "max": tmax},
        "token_counting": "approximate tokens (chars/4)", "over_max_tokens": over,
        "sync": sync_info}, indent=2))
    return 0


async def _sync_stats(conn: psycopg.AsyncConnection) -> dict | None:
    """Sync progress, or None on a database that predates the sync tables."""
    cur = await conn.execute("SELECT to_regclass('sync_state'), to_regclass('sync_failures')")
    if None in (await cur.fetchone() or (None,)):
        return None
    cur = await conn.execute("SELECT key, value FROM sync_state")
    state = {k: v for k, v in await cur.fetchall()}
    cur = await conn.execute("SELECT count(*) FROM sync_failures")
    failures = (await cur.fetchone())[0]  # type: ignore[index]
    return {"mode": state.get("mode"), "cursor_set": "cursor" in state,
            "bootstrap_after": state.get("bootstrap_after"), "failures": failures,
            "abort_count": int(state.get("abort_count", 0))}


async def _sync(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    conn = await connect(settings.vendorkb_dsn)
    try:
        await init_schema(conn, embed_dim=settings.embed_dim)
        async with DocExtractor(settings) as docext, Embedder(settings) as embedder:
            report = await sync(conn, docext=docext, embedder=embedder,
                                page_size=settings.sync_page_size, max_tokens=args.max_tokens)
    finally:
        await conn.close()
    print(json.dumps(dataclasses.asdict(report)))
    # A held lock or a budget stop is a clean exit; a dropped stream or error resumes next run.
    return 0 if report.complete or report.stopped_by_budget or report.locked else 1


async def _reembed(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    conn = await connect(settings.vendorkb_dsn)
    try:
        async with Embedder(settings) as embedder:
            report = await reembed(conn, embedder, dim=settings.embed_dim,
                                   batch=settings.embed_max_batch,
                                   concurrency=settings.embed_concurrency, limit=args.limit)
    finally:
        await conn.close()
    print(json.dumps({"model": settings.embed_model, "dim": settings.embed_dim,
                      **dataclasses.asdict(report)}))
    return 0


async def _promote(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    if not args.yes:
        print("promote-embedding replaces chunks.embedding with chunks.embedding_next; "
              "pass --yes", file=sys.stderr)
        return 2
    conn = await connect(settings.vendorkb_dsn)
    try:
        column_type = await promote(conn)
    finally:
        await conn.close()
    print(f"chunks.embedding is now {column_type}; set EMBED_* in .env to the new model")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vendorkb")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--vendor", action="append", default=[])
    s.add_argument("--product", action="append", default=[])
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--rerank", action="store_true")
    s = sub.add_parser("answer")
    s.add_argument("question")
    sub.add_parser("stats")
    s = sub.add_parser("sync", help="pull the DocExtractor delta feed (bootstrap, then "
                                    "incremental; resumable)")
    s.add_argument("--max-tokens", type=int, default=None,
                   help="stop after the page on which embedding tokens reach N (bootstrap: "
                        "the total across attempts; incremental: this run)")
    s = sub.add_parser("reembed", help="embed every chunk into chunks.embedding_next "
                                       "with the configured EMBED_* model (resumable)")
    s.add_argument("--limit", type=int, default=None, help="stop after N chunks (smoke)")
    s = sub.add_parser("promote-embedding",
                       help="replace chunks.embedding with chunks.embedding_next")
    s.add_argument("--yes", action="store_true")
    return p


_COMMANDS = {"search": _search, "answer": _answer, "stats": _stats, "sync": _sync,
             "reembed": _reembed, "promote-embedding": _promote}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = VendorkbSettings()  # fresh each call: never a cached copy of another env
    return asyncio.run(_COMMANDS[args.cmd](settings, args))


if __name__ == "__main__":
    sys.exit(main())
