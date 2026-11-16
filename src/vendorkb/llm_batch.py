"""Message Batches for the map and support calls (BACKLOG 37; spec
docs/superpowers/specs/2026-11-09-message-batches-design.md). Half price, results within 24 h.

The same rules as ``claims/extract.py``'s batch mode, for any caller that has a request builder,
a reply parser and a writer:

- **Ledger first.** ``submit`` writes an ``llm_batches`` row under a ``pending-`` id before
  ``messages.batches.create``, then renames it to the provider's id. A crash between the two
  keeps the spend counted, and the row's custom ids stay held (``held_ids``) until a person
  checks the provider's batch list and fixes or deletes the row.
- **No double asks.** An item is held while a submitted batch lists it, and once ``failed``.
- **Size.** Requests are split under the API's count and size caps.
- **Budget.** Each request is reserved at its full ``max_tokens`` of output, at the batch price;
  ``submit`` stops before the reservations would pass ``max_usd``.
- **Collecting.** ``collect`` processes every *ended* batch of a kind: each succeeded result is
  parsed (``llm.parse_message``, priced at the batch price) and handed to the caller's
  ``handle``, which stores it and returns None, or returns why it could not be used. Then the
  batch row is marked processed with its actual cost. Unended batches wait for a later call.
- **Failures.** An unusable, errored, expired or missing result writes nothing: the item is due
  again, until ``MAX_ATTEMPTS`` asks have failed; then it is ``failed`` and never asked again
  until ``reset_failed``.

``handle`` must be idempotent: a crash after it stored a result but before the batch row was
marked processed hands the same result to it again on the next ``collect``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
import psycopg
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from vendorkb.claims.extract import (
    CHARS_PER_TOKEN, MAX_BATCH_BYTES, MAX_BATCH_REQUESTS, cost_usd)
from vendorkb.taxonomy.llm import parse_message

log = logging.getLogger(__name__)

PENDING = "pending-"
MAX_ATTEMPTS = 3  # as extract: an item is failed after this many unusable asks


@dataclass
class Item:
    custom_id: str  # 1-64 characters of [A-Za-z0-9_-]
    params: dict[str, Any]  # Messages API params (llm.build_request, anthropic backend)
    payload: dict[str, Any] = field(default_factory=dict)  # what ``handle`` needs back


@dataclass
class Result:
    custom_id: str
    payload: dict[str, Any]
    text: str | None
    cost: float


Handler = Callable[[Result], Awaitable[str | None]]


@dataclass
class BatchReport:
    submitted: int = 0  # requests submitted by this call
    batches: int = 0  # batches submitted by this call
    reserved_usd: float = 0.0  # their budget reservation (worst case)
    skipped_by_budget: int = 0  # due items left for a later run by max_usd
    collected: int = 0  # batches processed
    used: int = 0  # results stored
    errored: int = 0  # results unusable or missing: due again
    failed: int = 0  # of those, now at the attempt cap
    spent_usd: float = 0.0  # actual cost of the collected batches
    unconfirmed: int = 0  # pending- rows: a crash between the ledger row and the create

    def add(self, other: BatchReport) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def reserve_usd(params: dict[str, Any], price: tuple[float, float]) -> float:
    """A request's worst case at the batch price: its input estimated from its length, its
    output at ``max_tokens``."""
    chars = len(json.dumps(params.get("system", ""))) + len(json.dumps(params["messages"]))
    return cost_usd(chars // CHARS_PER_TOKEN + 1, int(params["max_tokens"]), batch=True,
                    price=price)


def _groups(items: list[Item]) -> list[list[Item]]:
    groups: list[list[Item]] = [[]]
    size = 0
    for item in items:
        n = len(json.dumps(item.params)) + 100
        if groups[-1] and (len(groups[-1]) >= MAX_BATCH_REQUESTS or size + n > MAX_BATCH_BYTES):
            groups.append([])
            size = 0
        groups[-1].append(item)
        size += n
    return [g for g in groups if g]


async def held_ids(conn: psycopg.AsyncConnection, kind: str) -> set[str]:
    """The custom ids not to submit: listed by a submitted batch (pending- rows included), or
    failed."""
    cur = await conn.execute(
        "SELECT unnest(custom_ids) FROM llm_batches WHERE kind = %s AND status = 'submitted'"
        " UNION SELECT custom_id FROM llm_batch_items WHERE kind = %s AND status = 'failed'",
        (kind, kind))
    return {row[0] for row in await cur.fetchall()}


async def in_flight(conn: psycopg.AsyncConnection, kinds: Iterable[str]) -> int:
    """Submitted batches of ``kinds`` that will end: pending- rows are left out (they wait for
    a person, see ``unconfirmed``), so a run waiting on this count always finishes."""
    cur = await conn.execute("SELECT count(*) FROM llm_batches WHERE status = 'submitted'"
                             " AND kind = ANY(%s) AND batch_id NOT LIKE %s",
                             (list(kinds), PENDING + "%"))
    return int((await cur.fetchone())[0])  # type: ignore[index]


async def unconfirmed(conn: psycopg.AsyncConnection, kind: str) -> int:
    cur = await conn.execute("SELECT batch_id, cardinality(custom_ids) FROM llm_batches"
                             " WHERE kind = %s AND status = 'submitted' AND batch_id LIKE %s",
                             (kind, PENDING + "%"))
    rows = await cur.fetchall()
    for batch_id, n in rows:
        log.warning("batch: %s row %s (%d items) never learned its batch id: check the"
                    " provider's batch list, then set its batch_id or delete it; its items"
                    " are held back", kind, batch_id, n)
    return len(rows)


async def reset_failed(conn: psycopg.AsyncConnection, kind: str) -> int:
    """Make the failed items of ``kind`` due again (``--retry-failed``)."""
    cur = await conn.execute("UPDATE llm_batch_items SET status = 'errored', attempts = 0,"
                             " updated_at = now() WHERE kind = %s AND status = 'failed'",
                             (kind,))
    return cur.rowcount


async def submit(conn: psycopg.AsyncConnection, client: Any, *, kind: str, model: str,
                 items: list[Item], price: tuple[float, float], max_usd: float,
                 report: BatchReport) -> None:
    """Submit ``items`` (the caller has already left out ``held_ids``) up to ``max_usd`` of
    reservations, in as many batches as the caps need."""
    admitted: list[tuple[Item, float]] = []
    total = 0.0
    for n, item in enumerate(items):
        cost = reserve_usd(item.params, price)
        if total + cost > max_usd:
            report.skipped_by_budget += len(items) - n
            log.warning("batch: %s: $%.4f admits %d of %d items; the rest wait for the next"
                        " run", kind, max_usd, len(admitted), len(items))
            break
        total += cost
        admitted.append((item, cost))
    reserved = {item.custom_id: cost for item, cost in admitted}
    for group in _groups([item for item, _ in admitted]):
        ids = [item.custom_id for item in group]
        est = sum(reserved[i] for i in ids)
        pending = f"{PENDING}{uuid.uuid4()}"
        await conn.execute(  # ledger first: a crash after the create must not lose the spend
            "INSERT INTO llm_batches (batch_id, kind, model, custom_ids, est_cost_usd, status)"
            " VALUES (%s, %s, %s, %s, %s, 'submitted')", (pending, kind, model, ids, est))
        try:
            batch = await client.messages.batches.create(requests=[
                Request(custom_id=item.custom_id,
                        params=cast(MessageCreateParamsNonStreaming, item.params))
                for item in group])
        except anthropic.APIStatusError as e:
            if e.status_code < 500:  # refused outright: no batch exists, nothing is owed
                await conn.execute("DELETE FROM llm_batches WHERE batch_id = %s", (pending,))
            raise  # a 5xx or a dropped connection may still have created it: the row stays
        await conn.execute("UPDATE llm_batches SET batch_id = %s WHERE batch_id = %s",
                           (batch.id, pending))
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO llm_batch_items (kind, custom_id, batch_id, payload, attempts,"
                " status) VALUES (%s, %s, %s, %s, 1, 'in_flight') ON CONFLICT (kind, custom_id)"
                " DO UPDATE SET batch_id = EXCLUDED.batch_id, payload = EXCLUDED.payload,"
                " attempts = llm_batch_items.attempts + 1, status = 'in_flight',"
                " updated_at = now()",
                [(kind, item.custom_id, batch.id, json.dumps(item.payload)) for item in group])
        report.submitted += len(group)
        report.batches += 1
        report.reserved_usd += est
        log.info("batch: submitted %s batch %s (%d requests, reserved $%.4f)", kind, batch.id,
                 len(group), est)


def _batch_error(result: Any) -> str:
    kind = getattr(result, "type", "unknown")
    if kind != "errored":
        return f"batch:{kind}"
    inner = getattr(getattr(result, "error", None), "error", None)
    etype = inner.get("type") if isinstance(inner, dict) else getattr(inner, "type", None)
    return f"batch:errored:{etype}" if etype else "batch:errored"


async def _settle(conn: psycopg.AsyncConnection, kind: str, custom_id: str,
                  error: str | None, report: BatchReport) -> None:
    if error is None:
        await conn.execute("UPDATE llm_batch_items SET status = 'done', last_error = NULL,"
                           " updated_at = now() WHERE kind = %s AND custom_id = %s",
                           (kind, custom_id))
        report.used += 1
        return
    cur = await conn.execute(
        "UPDATE llm_batch_items SET status = CASE WHEN attempts >= %s THEN 'failed'"
        " ELSE 'errored' END, last_error = %s, updated_at = now()"
        " WHERE kind = %s AND custom_id = %s RETURNING status",
        (MAX_ATTEMPTS, error[:500], kind, custom_id))
    row = await cur.fetchone()
    report.errored += 1
    if row is not None and row[0] == "failed":
        report.failed += 1
        log.warning("batch: %s item %s failed %d times (%s): not asked again until reset",
                    kind, custom_id, MAX_ATTEMPTS, error)


async def _ended(client: Any, batch_id: str) -> bool:
    return (await client.messages.batches.retrieve(batch_id)).processing_status == "ended"


async def collect(conn: psycopg.AsyncConnection, client: Any, *, kind: str,
                  price: tuple[float, float], handle: Handler, report: BatchReport) -> None:
    """Process every ended batch of ``kind`` (see the module docstring)."""
    report.unconfirmed += await unconfirmed(conn, kind)
    cur = await conn.execute(
        "SELECT batch_id, custom_ids FROM llm_batches WHERE kind = %s AND status = 'submitted'"
        " AND batch_id NOT LIKE %s ORDER BY submitted_at", (kind, PENDING + "%"))
    for batch_id, custom_ids in await cur.fetchall():
        try:
            if not await _ended(client, batch_id):
                continue
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            log.warning("batch: %s batch %s: status unknown (%s); next run", kind, batch_id, e)
            continue
        expected = set(custom_ids)
        payloads = {row[0]: row[1] for row in await (await conn.execute(
            "SELECT custom_id, payload FROM llm_batch_items WHERE kind = %s AND batch_id = %s",
            (kind, batch_id))).fetchall()}
        seen: set[str] = set()
        actual = 0.0
        async for result in await client.messages.batches.results(batch_id):
            cid = result.custom_id
            if cid not in expected or cid in seen:
                log.warning("batch: %s batch %s: unexpected custom_id %s", kind, batch_id, cid)
                continue
            seen.add(cid)
            if result.result.type != "succeeded":
                await _settle(conn, kind, cid, _batch_error(result.result), report)
                continue
            text, error, cost = parse_message(result.result.message, price=price, batch=True)
            actual += cost
            if error is None:
                error = await handle(Result(cid, payloads.get(cid) or {}, text, cost))
            await _settle(conn, kind, cid, error, report)
        for cid in sorted(expected - seen):
            await _settle(conn, kind, cid, "batch:no-result", report)
        await conn.execute("UPDATE llm_batches SET status = 'processed', actual_cost_usd = %s,"
                           " ended_at = now() WHERE batch_id = %s", (actual, batch_id))
        report.collected += 1
        report.spent_usd += actual
        log.info("batch: %s batch %s processed (%d results, $%.4f)", kind, batch_id, len(seen),
                 actual)


async def wait_ended(conn: psycopg.AsyncConnection, client: Any, kinds: Iterable[str], *,
                     poll_seconds: float) -> None:
    """Poll until every submitted batch of ``kinds`` has ended (no lock is held: the ledger
    already keeps a waiting run from double asks). pending- rows are not waited for."""
    kinds = list(kinds)
    while True:
        cur = await conn.execute("SELECT batch_id FROM llm_batches WHERE status = 'submitted'"
                                 " AND kind = ANY(%s) AND batch_id NOT LIKE %s",
                                 (kinds, PENDING + "%"))
        open_ids = [row[0] for row in await cur.fetchall()]
        ended = []
        for batch_id in open_ids:
            try:
                ended.append(await _ended(client, batch_id))
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                log.warning("batch: polling %s failed (%s)", batch_id, type(e).__name__)
                ended.append(False)
        if all(ended):
            return
        await asyncio.sleep(poll_seconds)
