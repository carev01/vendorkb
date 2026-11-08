"""``python -m vendorkb.cli`` -- search / answer / stats / sync / reembed / promote-embedding /
extract / claims (regate) / taxonomy (propose, load, map) / capabilities (build, query)."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    import anthropic

# The claims and taxonomy stacks (anthropic, numpy, scipy, sklearn) are imported inside their
# command handlers: `vendorkb sync` runs in a 1 Gi pod and never needs them.

MAX_TOKENS = 1200
# `capabilities query` choices, copied so the parser needs no claims/taxonomy import (a unit
# test keeps them equal to extract.SUPPORT_VALUES, AVAILABILITY_VALUES and load.FAMILIES).
SUPPORT_VALUES = ("supported", "partial", "not_supported", "unclear")
AVAILABILITY_VALUES = ("generally_available", "preview", "early_access", "limited_availability",
                       "planned", "deprecated", "unstated")
FAMILIES = ("coverage", "protection", "schedule", "retention", "storage", "recovery",
            "security", "cyber", "operations", "governance", "data_use", "vendor")


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
                                page_size=settings.sync_page_size, max_tokens=args.max_tokens,
                                allow_prune=args.allow_prune)
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


EXTRACT_MAX_RETRIES = 5  # the SDKs back off on 429, 5xx and connection errors


def _extract_client(settings: VendorkbSettings) -> anthropic.AsyncAnthropic | AsyncOpenAI:
    """The extraction client for ``extract_backend``. anthropic: with no ANTHROPIC_API_KEY
    setting the SDK reads the env var. openai: EXTRACT_BASE_URL + EXTRACT_API_KEY."""
    if settings.extract_backend == "openai":
        return AsyncOpenAI(base_url=settings.extract_base_url or None,
                           api_key=settings.extract_api_key or None,
                           max_retries=EXTRACT_MAX_RETRIES)
    import anthropic

    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None,
                                    max_retries=EXTRACT_MAX_RETRIES)


def _map_client(settings: VendorkbSettings) -> anthropic.AsyncAnthropic | AsyncOpenAI:
    """The adjudication client for MAP_BACKEND (openai: MAP_BASE_URL / MAP_API_KEY, else the
    EXTRACT_ ones)."""
    return _extract_client(settings.model_copy(update={
        "extract_backend": settings.map_backend,
        "extract_base_url": settings.map_base_url or settings.extract_base_url,
        "extract_api_key": settings.map_api_key or settings.extract_api_key}))


def _read_ids(path: str, limit: int | None) -> list[str]:
    with open(path, encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return ids[:limit] if limit is not None else ids


async def _extract(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    from vendorkb.claims import extract

    if args.backend:
        settings = settings.model_copy(update={"extract_backend": args.backend})
    mode: Literal["sync", "batch"] = args.mode or (
        "batch" if settings.extract_backend == "anthropic" else "sync")
    conn = await connect(settings.vendorkb_dsn)
    client = _extract_client(settings)
    try:
        await init_schema(conn, embed_dim=settings.embed_dim)
        ids = (_read_ids(args.ids_file, args.limit) if args.ids_file else
               await extract.unextracted_ids(conn, vendor=args.vendor, limit=args.limit))
        report = await extract.run(conn, ids, client=client, settings=settings, mode=mode,
                                   retry_truncated=args.retry_errored)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()
        await conn.close()
    print(json.dumps(dataclasses.asdict(report)))
    if report.locked:
        print("another `vendorkb extract` run holds the extract lock; nothing done",
              file=sys.stderr)
        return 0
    # Errored chunks are retried by the next run; a budget stop is a clean exit.
    return 0 if report.errored == 0 else 1


async def _refresh(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    from vendorkb.claims import refresh

    async def ready() -> None:
        if settings.extract_backend == "openai" and settings.extract_base_url:
            await refresh.wait_until_ready(settings.extract_base_url, settings.extract_api_key,
                                           timeout_s=args.warmup_seconds)

    conn = await connect(settings.vendorkb_dsn)
    extract_client, map_client = _extract_client(settings), _map_client(settings)
    try:
        await init_schema(conn, embed_dim=settings.embed_dim)
        async with Embedder(settings) as embedder:
            report = await refresh.refresh(
                conn, settings=settings, extract_client=extract_client, map_client=map_client,
                embedder=embedder, ready=ready, settle_minutes=args.settle_minutes,
                extract_limit=args.extract_limit, map_max_usd=args.map_max_usd)
    finally:
        for client in (extract_client, map_client):
            close = getattr(client, "close", None)
            if close is not None:
                await close()
        await conn.close()
    print(json.dumps(dataclasses.asdict(report)))
    # Errored chunks and failed adjudications are retried by the next due run, but they are
    # worth a failed Job; nothing to do is a clean exit.
    failed = (report.extract or {}).get("errored", 0) or (report.map or {}).get("llm_invalid", 0)
    return 1 if failed else 0


async def _claims(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    from vendorkb.claims import regate

    conn = await connect(settings.vendorkb_dsn)
    try:
        await init_schema(conn, embed_dim=settings.embed_dim)
        report = await regate.regate(conn, batch_size=args.batch_size, dry_run=args.dry_run)
    finally:
        await conn.close()
    print(json.dumps(dataclasses.asdict(report)))
    return 0


async def _taxonomy(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    from vendorkb.taxonomy import aspects as aspects_mod
    from vendorkb.taxonomy import load, mapper, propose

    if args.backend:
        key = "map_backend" if args.taxonomy_cmd == "map" else "extract_backend"
        settings = settings.model_copy(update={key: args.backend})
    conn = await connect(settings.vendorkb_dsn)
    try:
        if args.taxonomy_cmd == "load":
            await init_schema(conn, embed_dim=settings.embed_dim)
            try:
                loaded = await load.load_taxonomy(conn, args.path)
            except load.TaxonomyError as e:  # nothing was written
                print(e, file=sys.stderr)
                return 1
            print(json.dumps(loaded))  # the version and each table's row count
            return 0
        if args.taxonomy_cmd == "aspects":
            try:
                found = await aspects_mod.aspects(conn, target=args.target, cls=args.cls)
            except LookupError as e:
                print(e, file=sys.stderr)
                return 2
            for aspect in found:
                print(json.dumps(dataclasses.asdict(aspect)))
            return 0
        if args.taxonomy_cmd == "map":
            await init_schema(conn, embed_dim=settings.embed_dim)
            client = _map_client(settings)
            try:
                async with Embedder(settings) as embedder:
                    report = await mapper.map_claims(conn, embedder=embedder, client=client,
                                                     settings=settings, max_usd=args.max_usd,
                                                     remap=args.remap)
            finally:
                close = getattr(client, "close", None)
                if close is not None:
                    await close()
            print(json.dumps(dataclasses.asdict(report)))
            # A budget stop is a clean exit; failed calls or unusable replies are not.
            return 0 if report.llm_invalid == 0 else 1
        client = _extract_client(settings)
        try:
            async with Embedder(settings) as embedder:
                prep = await propose.propose(
                    conn, client=client, embedder=embedder, settings=settings,
                    kind=args.kind, distance_threshold=args.threshold, out_path=args.out,
                    max_usd=args.max_usd, limit=args.limit)
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()
    finally:
        await conn.close()
    print(json.dumps(dataclasses.asdict(prep)))
    # A budget stop is a clean exit (re-run with a higher --max-usd to continue).
    return 0 if prep.invalid == 0 else 1


async def _capabilities(settings: VendorkbSettings, args: argparse.Namespace) -> int:
    from vendorkb.taxonomy import capabilities, load

    conn = await connect(settings.vendorkb_dsn)
    try:
        if args.capabilities_cmd == "build":
            await init_schema(conn, embed_dim=settings.embed_dim)
            rows = await capabilities.build(conn)
            print(json.dumps({"rows": rows,
                              "taxonomy_version": await load.current_version(conn)}))
            return 0
        found = await capabilities.query(conn, capabilities.CapFilter(
            vendor=args.vendor, offering=args.offering, edition=args.edition,
            dimension=args.dimension, family=args.family, target=args.target,
            object_type=args.object_type, support=args.support,
            availability=args.availability, value_min=args.value_min,
            value_max=args.value_max))
    finally:
        await conn.close()
    print(json.dumps(dataclasses.asdict(found)))
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
    s.add_argument("--allow-prune", type=int, default=None, metavar="N",
                   help="let the bootstrap's end prune up to N documents the stream never "
                        "served, above the safety ceiling (use the count a refusal logged)")
    s = sub.add_parser("reembed", help="embed every chunk into chunks.embedding_next "
                                       "with the configured EMBED_* model (resumable)")
    s.add_argument("--limit", type=int, default=None, help="stop after N chunks (smoke)")
    s = sub.add_parser("promote-embedding",
                       help="replace chunks.embedding with chunks.embedding_next")
    s.add_argument("--yes", action="store_true")
    s = sub.add_parser("extract", help="extract capability claims from chunks with "
                                       "EXTRACT_MODEL (resumable; budget-capped)")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids-file", help="chunk ids, one per line")
    src.add_argument("--unextracted", action="store_true",
                     help="every chunk with no done or failed extraction of the claims-v3 family"
                          " by any model: new and re-synced chunks, and errored ones")
    s.add_argument("--vendor", default=None, help="with --unextracted: only this vendor")
    s.add_argument("--backend", choices=("anthropic", "openai"), default=None,
                   help="override EXTRACT_BACKEND (openai: EXTRACT_BASE_URL/EXTRACT_API_KEY)")
    s.add_argument("--mode", choices=("sync", "batch"), default=None,
                   help="sync: one call per chunk, EXTRACT_CONCURRENCY at once; batch: the "
                        "Message Batches API (anthropic only). Default: batch for anthropic, "
                        "sync for openai")
    s.add_argument("--limit", type=int, default=None, help="only the first N ids (smoke)")
    s.add_argument("--retry-errored", action="store_true",
                   help="first reset the chunks (of these ids) whose last error was a "
                        "max_tokens truncation, so they run again (raise EXTRACT_MAX_TOKENS)")
    s = sub.add_parser("refresh", help="when the sync brought new chunks: wake the extraction"
                                       " endpoint, extract them, map their claims and rebuild"
                                       " capabilities (otherwise contact nothing)")
    s.add_argument("--settle-minutes", type=float, default=60.0,
                   help="wait until no new document arrived for this long (default 60)")
    s.add_argument("--extract-limit", type=int, default=20000,
                   help="chunks per run; a backlog this large runs without settling")
    s.add_argument("--map-max-usd", type=float, default=1.0,
                   help="adjudication budget per run (default 1.0)")
    s.add_argument("--warmup-seconds", type=float, default=1500.0,
                   help="how long to wait for a scaled-to-zero endpoint (default 1500)")
    s = sub.add_parser("claims", help="maintain stored claims")
    cl = s.add_subparsers(dest="claims_cmd", required=True)
    c = cl.add_parser("regate", help="re-run the current evidence gate over every raw claim; "
                                     "claims that now fail move to rejected_claims")
    c.add_argument("--batch-size", type=int, default=1000)
    c.add_argument("--dry-run", action="store_true", help="count only; change nothing")
    s = sub.add_parser("taxonomy", help="grow (propose), load and apply (map) the taxonomy")
    tax = s.add_subparsers(dest="taxonomy_cmd", required=True)
    t = tax.add_parser("propose", help="cluster the v3 claim strings of one section and merge "
                                       "EXTRACT_MODEL's alias / new-entry answers into the "
                                       "v2 YAML for review (budget-capped)")
    t.add_argument("--kind", choices=("dimension", "target", "object_type", "offering"),
                   required=True, help="dimension: capability; target: target; object_type:"
                   " target scope per target class; offering: offering and edition per vendor")
    t.add_argument("--threshold", type=float, default=0.25,
                   help="cosine distance threshold for clustering (default 0.25)")
    t.add_argument("--out", default="taxonomy/taxonomy.yaml",
                   help="the taxonomy YAML; the long tail goes next to it (long_tail.yaml)")
    t.add_argument("--max-usd", type=float, default=2.0,
                   help="stop before the estimated spend exceeds this (default 2.0)")
    t.add_argument("--backend", choices=("anthropic", "openai"), default=None,
                   help="override EXTRACT_BACKEND")
    t.add_argument("--limit", type=int, default=None,
                   help="send at most N clusters, biggest first (smoke run)")
    t = tax.add_parser("load", help="validate the reviewed v2 YAML and replace the taxonomy"
                                    " tables with it (every problem listed, nothing written)")
    t.add_argument("path")
    t.set_defaults(backend=None)
    t = tax.add_parser("aspects", help="the aspects that apply to a workload (JSON lines)")
    group = t.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", default=None, help="target id, alias or name")
    group.add_argument("--class", dest="cls", default=None, help="workload class id")
    t.set_defaults(backend=None)
    t = tax.add_parser("map", help="map the v3 claims not yet mapped under the current taxonomy"
                                   " version (exact aliases, then MAP_MODEL adjudication of the"
                                   " nearest candidates; budget-capped)")
    t.add_argument("--max-usd", type=float, default=1.0,
                   help="stop adjudicating before the estimated spend exceeds this; the rest"
                        " stay unmapped (default 1.0)")
    t.add_argument("--backend", choices=("anthropic", "openai"), default=None,
                   help="override MAP_BACKEND")
    t.add_argument("--all", dest="remap", action="store_true",
                   help="map every v3 claim again, including those already mapped under the"
                        " current taxonomy version")
    s = sub.add_parser("capabilities", help="build or query the capabilities table")
    cap = s.add_subparsers(dest="capabilities_cmd", required=True)
    cap.add_parser("build", help="rebuild from mapped raw claims (current taxonomy version)")
    c = cap.add_parser("query", help="facts matching the filters, grouped (one JSON object:"
                                     " facts, product_level, product_wide, conflicts, hint)")
    c.add_argument("--vendor", default=None, help="vendor (case-insensitive)")
    c.add_argument("--offering", default=None, help="offering id or alias")
    c.add_argument("--edition", default=None, help="edition id or alias")
    c.add_argument("--dimension", default=None, help="dimension id or alias")
    c.add_argument("--family", default=None, choices=FAMILIES, help="dimension family")
    c.add_argument("--target", default=None, help="target id or alias (with descendants)")
    c.add_argument("--object-type", default=None, help="object type id")
    c.add_argument("--support", default=None, choices=SUPPORT_VALUES)
    c.add_argument("--availability", default=None, choices=AVAILABILITY_VALUES)
    c.add_argument("--value-min", type=float, default=None)
    c.add_argument("--value-max", type=float, default=None)
    return p


_COMMANDS = {"search": _search, "answer": _answer, "stats": _stats, "sync": _sync,
             "reembed": _reembed, "promote-embedding": _promote, "extract": _extract,
             "refresh": _refresh, "claims": _claims, "taxonomy": _taxonomy,
             "capabilities": _capabilities}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.cmd == "extract" and args.vendor and not args.unextracted:
        parser.error("extract: --vendor narrows --unextracted only")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpcore"):  # they log every request at INFO
        logging.getLogger(name).setLevel(logging.WARNING)
    settings = VendorkbSettings()  # fresh each call: never a cached copy of another env
    return asyncio.run(_COMMANDS[args.cmd](settings, args))


if __name__ == "__main__":
    sys.exit(main())
