"""The reactive claims refresh (``vendorkb refresh``): run it often, it acts only on new data.

The sync writes the chunks DocExtractor added or changed. A refresh counts the chunks that no
extraction of the v3 prompt family has ever touched. With none it returns at once and contacts
nothing, so the extraction endpoint (scaled to zero between runs) stays down; a chunk that is
only 'errored' never triggers a run. With some, it waits until the feed has been quiet for
``settle_minutes`` (a DocExtractor refresh lands over hours: one run per burst, not one cold
start per sync), unless the backlog already fills a run (``extract_limit``). A due run then:

1. waits until the endpoint answers (``ready``; a cold start takes minutes);
2. extracts every chunk with no finished extraction (``extract.unextracted_ids``: errored ones
   ride along), at most ``extract_limit``, then lets the endpoint go (``release``: on Vast,
   destroy its workers, whether the extraction succeeded or not; a failure to release is
   logged, never fatal);
3. maps the claims not yet mapped under the current taxonomy version (``map_max_usd``; claims
   left over wait for the next due run, so an idle refresh never spends);
4. support-checks the claims of topic dimensions (``support.run`` with ``support_client``: the
   checker, then the reviewer for every flagged claim; ``support_max_usd`` and
   ``support_review_max_usd``; claims left over wait for the next due run; a failure is
   reported in ``support.error`` and does not stop the rebuild);
5. rebuilds the capabilities table when anything was extracted, mapped or support-checked.

Batch mode (``batch``; BACKLOG 37): map and support run on the Message Batches API at half
price, one round per run (``map_claims`` and ``support.run`` with ``wait=False``). A run is then
also due while a map or support batch is open, and it extracts only when new chunks are due
(never with ``extract_new`` false: the frequent collect run, which never wakes the GPU). The
stages land over several runs: map targets, map dimensions, check, review. Capabilities are
rebuilt only once support work is quiescent (no batch open, no check or review waiting, items
at their attempt cap counting as done) and something was written, so an unchecked fact is
never published (``capabilities_deferred`` says when a rebuild waits).
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import psycopg

from vendorkb import llm_batch
from vendorkb.claims import extract, support
from vendorkb.config import VendorkbSettings
from vendorkb.task_runs import track
from vendorkb.taxonomy import capabilities, mapper
from vendorkb.taxonomy.load import current_version

log = logging.getLogger(__name__)
FALLBACK_POLL_S = 30.0  # how often the fallback polls its batch


@dataclass
class RefreshReport:
    new_chunks: int = 0  # chunks no v3 extraction has touched (the trigger)
    newest_fetch: str | None = None  # the latest fetched_at among their documents
    due: bool = False
    reason: str = ""
    extract: dict[str, Any] | None = None  # extract.ExtractReport of a due run
    # extract.ExtractReport of the fallback model's batch over the chunks the extractor
    # looped on (extract.FALLBACK_ERRORS); None when there were none or it is off
    fallback: dict[str, Any] | None = None
    fallback_due: int = 0  # looping chunks waiting for the fallback when the run started
    # chunks this run tried (extractor and fallback) still without a done extraction
    unresolved: int | None = None
    map: dict[str, Any] | None = None  # mapper.MapReport of a due run
    support: dict[str, Any] | None = None  # support.SupportReport of a due run
    capabilities_rows: int | None = None  # set when the table was rebuilt
    mode: str = "sync"
    open_batches: int = 0  # batch mode: map and support batches open when the run started
    capabilities_deferred: bool = False  # batch mode: changed, but support work still waits
    # what ``release`` returned after the extraction (on Vast: the destroyed worker ids);
    # None when nothing was released or the release failed
    released: Any = None


async def new_chunks(conn: psycopg.AsyncConnection) -> tuple[int, datetime | None]:
    """(count, newest documents.fetched_at) of the current chunks with no v3 extraction row at
    all (a superseded chunk, kept for its claims, is never new)."""
    cur = await conn.execute(
        "SELECT count(*), max(d.fetched_at) FROM chunks c JOIN documents d USING (doc_id)"
        " WHERE c.superseded_at IS NULL AND NOT EXISTS (SELECT 1 FROM extractions e"
        " WHERE e.chunk_id = c.chunk_id AND e.prompt_version LIKE %s)",
        (extract.PROMPT_FAMILY,))
    count, newest = await cur.fetchone()  # type: ignore[misc]
    return int(count), newest


async def wait_until_ready(base_url: str, api_key: str, *, timeout_s: float,
                           poll_s: float = 15.0,
                           transport: httpx.AsyncBaseTransport | None = None) -> None:
    """Poll ``{base_url}/models`` until it answers 200; TimeoutError after ``timeout_s``."""
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=30.0, transport=transport) as http:
        while True:
            try:
                if (await http.get(url, headers=headers)).status_code == 200:
                    return
            except httpx.HTTPError as e:
                log.info("refresh: endpoint not up yet: %s", e)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"the extraction endpoint did not answer in {timeout_s:.0f} s")
            await asyncio.sleep(poll_s)


def _details(report: Any) -> dict[str, Any]:
    """A stage report as its task run's details: list fields cut to 20 items."""
    data = dataclasses.asdict(report)
    return {k: v[:20] if isinstance(v, list) else v for k, v in data.items()}


def failed(report: RefreshReport) -> bool:
    """Errored chunks and failed adjudications are retried by the next due run, but they are
    worth a failed Job; nothing to do is a clean exit."""
    support = report.support or {}
    # a chunk the extractor looped on and the fallback extracted is not a failure
    extract_failed = (report.unresolved > 0 if report.unresolved is not None
                      else (report.extract or {}).get("errored", 0))
    return bool(extract_failed or (report.map or {}).get("llm_invalid", 0)
                or support.get("check", {}).get("invalid", 0)
                or support.get("review", {}).get("invalid", 0) or "error" in support)


async def _release(release: Callable[[], Awaitable[Any]]) -> Any:
    """Let the extraction endpoint go; best effort (a worker left behind stops on its own
    after the endpoint's inactivity timeout, as before)."""
    try:
        return await release()
    except Exception as e:  # never fail a refresh over its cleanup
        log.warning("refresh: could not release the extraction endpoint: %s", e)
        return None


async def _fallback(conn: psycopg.AsyncConnection, ids: list[str], *,
                    settings: VendorkbSettings, client: Any) -> extract.ExtractReport:
    """Extract ``ids`` (chunks the extractor looped on) with the fallback model in one
    Anthropic Message Batch, waiting for it; at most ``extract_fallback_max_usd`` of new spend
    (on top of the ledger, under the overall ``extract_budget_usd``)."""
    spent = await extract.spent_usd(conn)
    fb_settings = settings.model_copy(update={
        "extract_backend": "anthropic", "extract_model": settings.extract_fallback_model,
        "extract_max_tokens": settings.extract_fallback_max_tokens,
        "extract_effort": settings.extract_fallback_effort,
        # the extractor's nominal GPU-host prices must never price an API model
        "extract_price_in_per_mtok": 0.0, "extract_price_out_per_mtok": 0.0,
        "extract_budget_usd": min(settings.extract_budget_usd,
                                  spent + settings.extract_fallback_max_usd)})
    log.info("refresh: %d looping chunks go to %s", len(ids), settings.extract_fallback_model)
    async with track(settings.vendorkb_dsn, "extract", stage="fallback") as run:
        fb = await extract.run(conn, ids, client=client, settings=fb_settings, mode="batch",
                               poll_seconds=FALLBACK_POLL_S)
        await run.finish("failed" if fb.errored else "succeeded", _details(fb))
    return fb


async def refresh(conn: psycopg.AsyncConnection, *, settings: VendorkbSettings,
                  extract_client: Any, map_client: Any, support_client: Any, embedder: Any,
                  ready: Callable[[], Awaitable[None]], settle_minutes: float,
                  extract_limit: int, map_max_usd: float, support_max_usd: float = 0.0,
                  support_review_max_usd: float = 0.0, batch: bool = False,
                  extract_new: bool = True, now: datetime | None = None,
                  fallback_client: Any = None,
                  release: Callable[[], Awaitable[Any]] | None = None) -> RefreshReport:
    """One refresh run, recorded as one ``refresh`` task run whose stages are its children.
    The collect run (``extract_new`` false) is recorded with stage ``collect``, so its hourly
    successes never stand for the extracting run's."""
    stage = None if extract_new else "collect"
    async with track(settings.vendorkb_dsn, "refresh", stage=stage) as run:
        report = await _refresh(
            conn, settings=settings, extract_client=extract_client, map_client=map_client,
            support_client=support_client, embedder=embedder, ready=ready,
            settle_minutes=settle_minutes, extract_limit=extract_limit,
            map_max_usd=map_max_usd, support_max_usd=support_max_usd,
            support_review_max_usd=support_review_max_usd, batch=batch,
            extract_new=extract_new, now=now, fallback_client=fallback_client,
            release=release)
        await run.finish("failed" if failed(report) else "succeeded", dataclasses.asdict(report))
        return report


async def _refresh(conn: psycopg.AsyncConnection, *, settings: VendorkbSettings,
                   extract_client: Any, map_client: Any, support_client: Any, embedder: Any,
                   ready: Callable[[], Awaitable[None]], settle_minutes: float,
                   extract_limit: int, map_max_usd: float, support_max_usd: float = 0.0,
                   support_review_max_usd: float = 0.0, batch: bool = False,
                   extract_new: bool = True, now: datetime | None = None,
                   fallback_client: Any = None,
                   release: Callable[[], Awaitable[Any]] | None = None) -> RefreshReport:
    dsn = settings.vendorkb_dsn
    report = RefreshReport(mode="batch" if batch else "sync")
    count, newest = await new_chunks(conn)
    report.new_chunks = count
    report.newest_fetch = newest.isoformat() if newest is not None else None
    now = now or datetime.now(UTC)
    extract_due = False
    if count == 0:
        report.reason = "no new chunks"
    elif not extract_new:
        report.reason = "extraction is left to the extracting run"
    elif count >= extract_limit:
        extract_due, report.reason = True, "the backlog fills a run"
    elif newest is None or now - newest >= timedelta(minutes=settle_minutes):
        extract_due, report.reason = True, "the feed is quiet"
    else:
        minutes = (now - newest).total_seconds() / 60
        report.reason = f"settling: the newest document arrived {minutes:.0f} min ago"
    kinds = (*mapper.BATCH_KINDS, *support.BATCH_KINDS)
    if batch:
        report.open_batches = await llm_batch.in_flight(conn, kinds)
    use_fallback = bool(extract_new and settings.extract_fallback_model
                        and fallback_client is not None)
    if use_fallback:
        report.fallback_due = len(await extract.fallback_ids(
            conn, settings.extract_model, settings.extract_fallback_model, limit=extract_limit))
    report.due = extract_due or report.open_batches > 0 or report.fallback_due > 0
    if report.due and not extract_due and report.open_batches:
        report.reason += f"; {report.open_batches} batches open"
    if report.due and not extract_due and report.fallback_due:
        report.reason += f"; {report.fallback_due} chunks for the fallback"
    if not report.due:
        log.info("refresh: nothing to do (%d new chunks; %s)", count, report.reason)
        return report

    changed = False
    if extract_due:
        try:
            await ready()
            ids = await extract.unextracted_ids(conn, limit=extract_limit)
            async with track(dsn, "extract") as run:
                ext = await extract.run(conn, ids, client=extract_client, settings=settings,
                                        mode="sync")
                # as `vendorkb extract` ends it: errored chunks make a failed run
                await run.finish("failed" if ext.errored else "succeeded", _details(ext))
        finally:
            if release is not None:
                report.released = await _release(release)
        report.extract = dataclasses.asdict(ext)
        changed = ext.done > 0
    if use_fallback:
        fb_ids = await extract.fallback_ids(conn, settings.extract_model,
                                            settings.extract_fallback_model, limit=extract_limit)
        if fb_ids:
            fb = await _fallback(conn, fb_ids, settings=settings, client=fallback_client)
            report.fallback = dataclasses.asdict(fb)
            changed = changed or fb.done > 0
        tried = list(dict.fromkeys([*(ids if extract_due else []), *fb_ids]))
        report.unresolved = await extract.unresolved(conn, tried)
    mode: Any = "batch" if batch else "sync"
    version = await current_version(conn)
    if version is not None:
        async with track(dsn, "map") as run:
            mapped = await mapper.map_claims(conn, embedder=embedder, client=map_client,
                                             settings=settings, max_usd=map_max_usd, mode=mode,
                                             wait=False)
            await run.finish("failed" if mapped.llm_invalid else "succeeded", _details(mapped))
        report.map = dataclasses.asdict(mapped)
        report.map["errors"] = report.map["errors"][:20]
        # batch mode: claims read but waiting on a batch are not written yet
        changed = changed or mapped.claims - (mapped.claims_pending if batch else 0) > 0 \
            or mapped.batch.collected > 0
        rejected = await support.rejected_count(conn)
        try:
            async with track(dsn, "support") as run:
                checked = await support.run(conn, client=support_client, settings=settings,
                                            check_max_usd=support_max_usd,
                                            review_max_usd=support_review_max_usd, mode=mode,
                                            wait=False)
                bad = checked.check.invalid or checked.review.invalid
                await run.finish("failed" if bad else "succeeded", _details(checked))
        except Exception as e:  # what was extracted, mapped or rejected is still built; the
            log.exception("refresh: the support check failed")  # Job fails
            report.support = {"error": f"{type(e).__name__}: {e}"}
            changed = changed or await support.rejected_count(conn) != rejected
        else:
            report.support = dataclasses.asdict(checked)
            changed = changed or checked.check.calls > 0 or checked.review.calls > 0
        if batch and changed and (await llm_batch.in_flight(conn, kinds) > 0
                                  or await support.waiting(conn, version) > 0):
            report.capabilities_deferred, changed = True, False
            log.info("refresh: capabilities wait for the open batches and the support check")
    if changed:
        async with track(dsn, "capabilities"):
            report.capabilities_rows = (await capabilities.build(conn)).rows
    log.info("refresh: %s", report)
    return report
