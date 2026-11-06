"""``vendorkb claims regate``: re-run the current evidence gate over every stored claim.

Each raw claim is gated again with its ``model_quote`` (or, for claims stored before that
column, its ``evidence_quote``) against its chunk's current text. A claim that now fails moves
to ``rejected_claims`` with the reason prefixed ``regate:`` (its ``capabilities`` rows
cascade away); a claim that passes gets the span the current gate derives as its
``evidence_quote`` and ``gate``. Work goes in pages of ``batch_size`` claims (keyset on
claim_id), one transaction per page, and each touched extraction's ``n_claims`` is recounted.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import psycopg

from vendorkb.claims import gate


@dataclass
class RegateReport:
    checked: int = 0
    kept: int = 0
    updated: int = 0  # kept, with a new evidence span or gate reason
    moved: int = 0
    dry_run: bool = False


async def regate(conn: psycopg.AsyncConnection, *, batch_size: int = 1000,
                 dry_run: bool = False) -> RegateReport:
    report = RegateReport(dry_run=dry_run)
    last = uuid.UUID(int=0)
    while True:
        cur = await conn.execute(
            "SELECT r.claim_id, r.extraction_id, coalesce(r.model_quote, r.evidence_quote),"
            " r.evidence_quote, r.gate, c.text FROM raw_claims r"
            " JOIN chunks c ON c.chunk_id = r.chunk_id"
            " WHERE r.claim_id > %s ORDER BY r.claim_id LIMIT %s", (last, batch_size))
        rows = await cur.fetchall()
        if not rows:
            return report
        last = rows[-1][0]
        moved: list[tuple[str, uuid.UUID]] = []
        updated: list[tuple[str | None, str, uuid.UUID]] = []
        touched: set[uuid.UUID] = set()
        for claim_id, extraction_id, quote, evidence, reason, text in rows:
            report.checked += 1
            r = gate.match(quote, text)
            if not r.ok:
                moved.append((f"regate:{r.reason}", claim_id))
                touched.add(extraction_id)
                continue
            report.kept += 1
            if (r.span, r.reason) != (evidence, reason):
                updated.append((r.span, r.reason, claim_id))
        report.moved += len(moved)
        report.updated += len(updated)
        if dry_run or not (moved or updated):
            continue
        async with conn.transaction(), conn.cursor() as c:
            if moved:
                await c.executemany(
                    "INSERT INTO rejected_claims (claim_id, extraction_id, chunk_id, vendor,"
                    " product, version, capability_raw, target_raw, support, conditions,"
                    " evidence_quote, model_quote, reason, extractor_model, prompt_version)"
                    " SELECT claim_id, extraction_id, chunk_id, vendor, product, version,"
                    " capability_raw, target_raw, support, conditions, NULL,"
                    " coalesce(model_quote, evidence_quote), %s, extractor_model,"
                    " prompt_version FROM raw_claims WHERE claim_id = %s"
                    " ON CONFLICT (claim_id) DO NOTHING", moved)
                await c.execute("DELETE FROM raw_claims WHERE claim_id = ANY(%s)",
                                ([cid for _, cid in moved],))
                await c.execute(
                    "UPDATE extractions e SET n_claims = (SELECT count(*) FROM raw_claims r"
                    " WHERE r.extraction_id = e.extraction_id)"
                    " WHERE e.extraction_id = ANY(%s)", (list(touched),))
            if updated:
                await c.executemany(
                    "UPDATE raw_claims SET evidence_quote = %s, gate = %s WHERE claim_id = %s",
                    updated)
