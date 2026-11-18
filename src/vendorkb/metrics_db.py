"""The pipeline's state as Prometheus gauges, read from the database once a minute.

``snapshot(conn)`` runs a handful of aggregate queries; ``PipelineCollector`` serves the last
good snapshot (a failed refresh keeps it and flips ``metrics_refresh_ok`` to 0);
``refresh_loop`` refreshes it on an interval, one short-lived connection per refresh. No label
carries question text: kinds, stages, sources and statuses are bounded sets, and the only
unbounded ones are run ids, table names and the taxonomy version.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field

import psycopg
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from vendorkb.claims import support
from vendorkb.claims.extract import PROMPT_FAMILY
from vendorkb.task_runs import CONNECT_TIMEOUT_S, STALLED_AFTER_S
from vendorkb.taxonomy import load

log = logging.getLogger(__name__)

P = "vendorkb_"
SPEND_WINDOWS = {"24h": "24 hours", "7d": "7 days"}
REFRESH_TIMEOUT_S = 20.0  # statement_timeout of a refresh; wait_for allows a little more


@dataclass
class Snapshot:
    # (kind, stage, run, origin, stalled, done, total, spent_usd, started_ts)
    runs: list[tuple] = field(default_factory=list)
    last_success: dict[str, float] = field(default_factory=dict)
    last_failure: dict[str, float] = field(default_factory=dict)
    backlog: dict[str, int] = field(default_factory=dict)
    batches_open: dict[str, int] = field(default_factory=dict)
    batch_oldest_age: dict[str, float] = field(default_factory=dict)
    batch_items: dict[tuple[str, str], int] = field(default_factory=dict)
    corpus: dict[str, int] = field(default_factory=dict)
    extraction_errors: int = 0
    taxonomy_version: str | None = None
    spend: dict[tuple[str, str], float] = field(default_factory=dict)  # (kind, window)
    table_sizes: dict[str, int] = field(default_factory=dict)
    sync_cursor_ts: float | None = None


async def _rows(conn: psycopg.AsyncConnection, query: str, params=None) -> list[tuple]:
    cur = await conn.execute(query, params)
    return await cur.fetchall()


async def _support_waiting(conn: psycopg.AsyncConnection, version: str) -> int:
    """``support.waiting`` as one COUNT: the in-scope claims with no verdict that no batch
    holds, plus the flagged, unreviewed verdicts no batch holds."""
    dims = await support.topic_dimensions(conn)
    held = ("(SELECT unnest(custom_ids) FROM llm_batches WHERE kind = %(kind)s"
            " AND status = 'submitted' UNION SELECT custom_id FROM llm_batch_items"
            " WHERE kind = %(kind)s AND status = 'failed')")
    (n,), = await _rows(
        conn,
        "SELECT (SELECT count(*)" + support._SOURCE + " AND NOT EXISTS (SELECT 1 FROM"
        " claim_support s WHERE s.claim_id = r.claim_id)"
        " AND r.claim_id::text NOT IN " + held.replace("%(kind)s", "%(check)s") + ")"
        " + (SELECT count(*) FROM claim_support WHERE NOT supported AND review_supported IS NULL"
        " AND claim_id::text NOT IN " + held.replace("%(kind)s", "%(review)s") + ")",
        {"family": PROMPT_FAMILY, "ver": version, "dims": dims,
         "check": support.CHECK_KIND, "review": support.REVIEW_KIND})
    return int(n)


async def snapshot(conn: psycopg.AsyncConnection) -> Snapshot:
    s = Snapshot()
    s.runs = [
        (kind, stage or "", str(run_id), origin, stalled, done, total, spent, started)
        for kind, stage, run_id, origin, stalled, done, total, spent, started in await _rows(
            conn,
            "SELECT kind, stage, run_id, origin, heartbeat_at < now() - make_interval(secs"
            " => %s), done, total, spent_usd, extract(epoch FROM started_at)::float8"
            " FROM task_runs WHERE state = 'running' ORDER BY started_at", (STALLED_AFTER_S,))]
    for kind, state, ts in await _rows(
            conn, "SELECT kind, state, extract(epoch FROM max(ended_at))::float8 FROM task_runs"
            " WHERE state IN ('succeeded', 'failed') AND ended_at IS NOT NULL"
            " GROUP BY kind, state"):
        (s.last_success if state == "succeeded" else s.last_failure)[kind] = ts

    version = await load.current_version(conn)
    s.taxonomy_version = version
    (unextracted,), = await _rows(
        conn,
        "SELECT count(*) FROM chunks c WHERE c.superseded_at IS NULL AND NOT EXISTS"
        " (SELECT 1 FROM extractions e WHERE e.chunk_id = c.chunk_id"
        " AND e.prompt_version LIKE %s AND e.status IN ('done', 'failed'))", (PROMPT_FAMILY,))
    (unmapped,), = await _rows(
        conn, "SELECT count(*) FROM raw_claims"
        " WHERE %s::text IS NULL OR mapped_version IS DISTINCT FROM %s::text",
        (version, version))
    s.backlog = {"extract": unextracted, "map": unmapped,
                 "support": await _support_waiting(conn, version) if version else 0}

    for kind, n, oldest in await _rows(
            conn,
            "SELECT kind, count(*), extract(epoch FROM now() - min(submitted_at))::float8"
            " FROM (SELECT kind, submitted_at FROM llm_batches WHERE status = 'submitted'"
            " UNION ALL SELECT 'extract', submitted_at FROM extract_batches"
            " WHERE status = 'submitted') b GROUP BY kind"):
        s.batches_open[kind] = n
        s.batch_oldest_age[kind] = oldest
    s.batch_items = {(k, st): n for k, st, n in await _rows(
        conn, "SELECT kind, status, count(*) FROM llm_batch_items GROUP BY kind, status")}

    (docs, chunks, claims, caps, errors), = await _rows(
        conn,
        "SELECT (SELECT count(*) FROM documents),"
        " (SELECT count(*) FROM chunks WHERE superseded_at IS NULL),"
        " (SELECT count(*) FROM raw_claims), (SELECT count(*) FROM capabilities),"
        " (SELECT count(*) FROM extractions WHERE status IN ('errored', 'failed'))")
    s.corpus = {"documents": docs, "chunks": chunks, "claims": claims, "capabilities": caps}
    s.extraction_errors = errors

    for window, interval in SPEND_WINDOWS.items():
        for kind, usd in await _rows(
                conn,
                "SELECT kind, coalesce(sum(spent_usd), 0) FROM task_runs t WHERE NOT EXISTS"
                " (SELECT 1 FROM task_runs c WHERE c.parent_id = t.run_id)"
                " AND started_at > now() - %s::interval GROUP BY kind", (interval,)):
            s.spend[(kind, window)] = float(usd)
    s.table_sizes = {name: size for name, size in await _rows(
        conn,
        "SELECT c.relname, pg_total_relation_size(c.oid)::bigint FROM pg_class c"
        " JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public'"
        " AND c.relkind = 'r' ORDER BY 2 DESC LIMIT 15")}
    rows = await _rows(
        conn, "SELECT extract(epoch FROM updated_at)::float8 FROM sync_state WHERE key = 'cursor'")
    s.sync_cursor_ts = rows[0][0] if rows else None
    return s


def _gauge(name: str, doc: str, labels: list[str], rows) -> GaugeMetricFamily:
    g = GaugeMetricFamily(P + name, doc, labels=labels)
    for label_values, value in rows:
        g.add_metric(label_values, value)
    return g


class PipelineCollector(Collector):
    """Serves the last good snapshot; ``update(None, False)`` keeps it and marks the refresh bad."""

    def __init__(self) -> None:
        self._snap: Snapshot | None = None
        self._ok = False
        self._ok_ts = 0.0

    def update(self, s: Snapshot | None, ok: bool) -> None:
        if s is not None:
            self._snap = s
        if ok:
            self._ok_ts = time.time()
        self._ok = ok

    def collect(self):
        s, ok, ok_ts = self._snap, self._ok, self._ok_ts
        if s is not None:
            yield from self._families(s)
        yield _gauge("metrics_refresh_ok", "1 after a good refresh, 0 after a failed one",
                     [], [([], 1 if ok else 0)])
        yield _gauge("metrics_refresh_timestamp_seconds", "Time of the last good refresh",
                     [], [([], ok_ts)])

    @staticmethod
    def _families(s: Snapshot):
        lab = ["kind", "stage", "run", "origin"]
        for name, stalled, doc in (("task_running", False, "Runs with a live heartbeat"),
                                   ("task_stalled", True, "Runs whose heartbeat is stale")):
            yield _gauge(name, doc, lab, [([r[0], r[1], r[2], r[3]], 1)
                                          for r in s.runs if r[4] == stalled])
        rl = ["kind", "stage", "run"]
        yield _gauge("task_done", "Items done by a running run", rl,
                     [([r[0], r[1], r[2]], r[5]) for r in s.runs])
        yield _gauge("task_total", "Items a running run expects", rl,
                     [([r[0], r[1], r[2]], r[6]) for r in s.runs if r[6] is not None])
        yield _gauge("task_spent_usd", "Spend of a running run", rl,
                     [([r[0], r[1], r[2]], r[7]) for r in s.runs])
        yield _gauge("task_started_timestamp_seconds", "Start of a running run", rl,
                     [([r[0], r[1], r[2]], r[8]) for r in s.runs])
        yield _gauge("task_last_success_timestamp_seconds", "Last successful end per kind",
                     ["kind"], [([k], v) for k, v in s.last_success.items()])
        yield _gauge("task_last_failure_timestamp_seconds", "Last failed end per kind",
                     ["kind"], [([k], v) for k, v in s.last_failure.items()])
        yield _gauge("backlog", "Work waiting per stage", ["kind"],
                     [([k], v) for k, v in s.backlog.items()])
        yield _gauge("batches_open", "Submitted batches not yet processed", ["kind"],
                     [([k], v) for k, v in s.batches_open.items()])
        yield _gauge("batch_oldest_age_seconds", "Age of the oldest open batch", ["kind"],
                     [([k], v) for k, v in s.batch_oldest_age.items()])
        yield _gauge("batch_items", "Batch items by status", ["kind", "status"],
                     [([k, st], n) for (k, st), n in s.batch_items.items()])
        yield _gauge("corpus", "Corpus size", ["what"],
                     [([k], v) for k, v in s.corpus.items()])
        yield _gauge("extraction_errors", "Extractions errored or failed", [],
                     [([], s.extraction_errors)])
        yield _gauge("taxonomy_info", "Current taxonomy version", ["version"],
                     [([s.taxonomy_version], 1)] if s.taxonomy_version else [])
        yield _gauge("pipeline_spend_usd", "Spend of leaf runs started in the window",
                     ["kind", "window"], [([k, w], v) for (k, w), v in s.spend.items()])
        yield _gauge("table_size_bytes", "Total relation size of the largest tables", ["table"],
                     [([k], v) for k, v in s.table_sizes.items()])
        yield _gauge("sync_cursor_timestamp_seconds", "Last update of the sync cursor", [],
                     [([], s.sync_cursor_ts)] if s.sync_cursor_ts is not None else [])


async def refresh_loop(dsn: str, collector: PipelineCollector, interval_s: float,
                       timeout_s: float = REFRESH_TIMEOUT_S) -> None:
    """Refresh the collector every ``interval_s`` until cancelled. A failure keeps the last
    snapshot, marks the refresh bad and is logged once per streak."""
    failing = False
    while True:
        try:
            conn = await psycopg.AsyncConnection.connect(
                dsn, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S)
            try:
                await conn.execute(f"SET statement_timeout = {int(timeout_s * 1000)}")
                collector.update(
                    await asyncio.wait_for(snapshot(conn), timeout_s + 5), True)
            finally:
                await conn.close()
            failing = False
        except Exception as exc:
            collector.update(None, False)
            if not failing:
                log.warning("metrics refresh failed: %s", exc)
            failing = True
        await asyncio.sleep(interval_s)
