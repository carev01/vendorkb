"""The reactive claims refresh (``vendorkb refresh``): run it often, it acts only on new data.

The sync writes the chunks DocExtractor added or changed. A refresh counts the chunks that no
extraction of the v3 prompt family has ever touched. With none it returns at once and contacts
nothing, so the extraction endpoint (scaled to zero between runs) stays down; a chunk that is
only 'errored' never triggers a run. With some, it waits until the feed has been quiet for
``settle_minutes`` (a DocExtractor refresh lands over hours: one run per burst, not one cold
start per sync), unless the backlog already fills a run (``extract_limit``). A due run then:

1. waits until the endpoint answers (``ready``; a cold start takes minutes);
2. extracts every chunk with no finished extraction (``extract.unextracted_ids``: errored ones
   ride along), at most ``extract_limit``;
3. maps the claims not yet mapped under the current taxonomy version (``map_max_usd``; claims
   left over wait for the next due run, so an idle refresh never spends);
4. rebuilds the capabilities table when anything was extracted or mapped.
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

from vendorkb.claims import extract
from vendorkb.config import VendorkbSettings
from vendorkb.taxonomy import capabilities, mapper
from vendorkb.taxonomy.load import current_version

log = logging.getLogger(__name__)


@dataclass
class RefreshReport:
    new_chunks: int = 0  # chunks no v3 extraction has touched (the trigger)
    newest_fetch: str | None = None  # the latest fetched_at among their documents
    due: bool = False
    reason: str = ""
    extract: dict[str, Any] | None = None  # extract.ExtractReport of a due run
    map: dict[str, Any] | None = None  # mapper.MapReport of a due run
    capabilities_rows: int | None = None  # set when the table was rebuilt


async def new_chunks(conn: psycopg.AsyncConnection) -> tuple[int, datetime | None]:
    """(count, newest documents.fetched_at) of the chunks with no v3 extraction row at all."""
    cur = await conn.execute(
        "SELECT count(*), max(d.fetched_at) FROM chunks c JOIN documents d USING (doc_id)"
        " WHERE NOT EXISTS (SELECT 1 FROM extractions e WHERE e.chunk_id = c.chunk_id"
        " AND e.prompt_version LIKE %s)", (extract.PROMPT_FAMILY,))
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


async def refresh(conn: psycopg.AsyncConnection, *, settings: VendorkbSettings,
                  extract_client: Any, map_client: Any, embedder: Any,
                  ready: Callable[[], Awaitable[None]], settle_minutes: float,
                  extract_limit: int, map_max_usd: float,
                  now: datetime | None = None) -> RefreshReport:
    report = RefreshReport()
    count, newest = await new_chunks(conn)
    report.new_chunks = count
    report.newest_fetch = newest.isoformat() if newest is not None else None
    now = now or datetime.now(UTC)
    if count == 0:
        report.reason = "no new chunks"
    elif count >= extract_limit:
        report.due, report.reason = True, "the backlog fills a run"
    elif newest is None or now - newest >= timedelta(minutes=settle_minutes):
        report.due, report.reason = True, "the feed is quiet"
    else:
        minutes = (now - newest).total_seconds() / 60
        report.reason = f"settling: the newest document arrived {minutes:.0f} min ago"
    if not report.due:
        log.info("refresh: nothing to do (%d new chunks; %s)", count, report.reason)
        return report

    await ready()
    ids = await extract.unextracted_ids(conn, limit=extract_limit)
    ext = await extract.run(conn, ids, client=extract_client, settings=settings, mode="sync")
    report.extract = dataclasses.asdict(ext)
    changed = ext.done > 0
    if await current_version(conn) is not None:
        mapped = await mapper.map_claims(conn, embedder=embedder, client=map_client,
                                         settings=settings, max_usd=map_max_usd)
        report.map = dataclasses.asdict(mapped)
        report.map["errors"] = report.map["errors"][:20]
        changed = changed or mapped.claims > 0
    if changed:
        report.capabilities_rows = await capabilities.build(conn)
    log.info("refresh: %s", report)
    return report
