"""The derived ``capabilities`` table: rebuilt from mapped ``raw_claims``, and queried.

``build`` replaces every row (delete, then insert, in one transaction) with one row per raw
claim mapped to a feature under the current taxonomy version. Conflicting claims for the
same (vendor, product, feature, target) stay separate rows (spec §5). A claim whose target
phrase did not map keeps a NULL ``target_id``; ``CapRow.target_raw`` shows the phrase.
``mapping_conf`` is the lower of the feature's and the target's confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import psycopg

from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import current_version


@dataclass(frozen=True)
class CapRow:
    vendor: str
    product: str | None
    version: str | None
    feature_id: str
    target_id: str | None
    target_raw: str | None
    support: str
    conditions: str | None
    evidence_quote: str
    chunk_id: str
    url: str | None
    title: str | None
    mapping_conf: float | None
    mapping_method: str | None


async def build(conn: psycopg.AsyncConnection) -> int:
    """Rebuild ``capabilities`` for the current taxonomy version; returns the row count."""
    version = await current_version(conn)
    async with conn.transaction():
        await conn.execute("DELETE FROM capabilities")
        if version is None:
            return 0
        cur = await conn.execute(
            "INSERT INTO capabilities (cap_id, vendor, product, version, feature_id, target_id,"
            " support, conditions, claim_id, mapping_conf, mapping_method, taxonomy_version)"
            " SELECT gen_random_uuid(), r.vendor, r.product, r.version, f.feature_id,"
            " t.target_id, r.support, r.conditions, r.claim_id,"
            " CASE WHEN t.target_id IS NULL THEN r.feature_conf"
            "  ELSE least(r.feature_conf, r.target_conf) END,"
            " CASE WHEN r.feature_method = 'alias'"
            "  AND (t.target_id IS NULL OR r.target_method = 'alias') THEN 'alias' ELSE 'nn' END,"
            " %s FROM raw_claims r JOIN features f ON f.feature_id = r.feature_id"
            " LEFT JOIN targets t ON t.target_id = r.target_id"
            " WHERE r.mapped_version = %s", (version, version))
        return cur.rowcount


async def _resolve(conn: psycopg.AsyncConnection, kind: str, term: str) -> str | None:
    """An id as given, else the entity whose alias or name matches ``term`` (normalized)."""
    table, col = ("features", "feature_id") if kind == "feature" else ("targets", "target_id")
    cur = await conn.execute(f"SELECT {col} FROM {table} WHERE {col}=%s", (term.strip(),))
    if (row := await cur.fetchone()) is not None:
        return row[0]
    key = normalize(term)
    cur = await conn.execute(
        "SELECT entity_id FROM aliases WHERE entity_type=%s AND alias=%s", (kind, key))
    if (row := await cur.fetchone()) is not None:
        return row[0]
    cur = await conn.execute(
        f"SELECT {col} FROM {table} WHERE lower(name)=%s ORDER BY {col} LIMIT 1", (key,))
    row = await cur.fetchone()
    return None if row is None else row[0]


async def query(conn: psycopg.AsyncConnection, feature: str, target: str | None = None,
                vendor: str | None = None, support: str | None = None) -> list[CapRow]:
    """Rows for ``feature`` (an id, alias or name); ``target`` matches itself and its
    descendants (``aws`` matches ``aws.rds``); ``vendor`` is case-insensitive."""
    fid = await _resolve(conn, "feature", feature)
    tid = None if target is None else await _resolve(conn, "target", target)
    if fid is None or (target is not None and tid is None):
        return []
    cur = await conn.execute(
        "SELECT c.vendor, c.product, c.version, c.feature_id, c.target_id, r.target_raw,"
        " c.support, c.conditions, r.evidence_quote, r.chunk_id::text, d.url, d.title,"
        " c.mapping_conf, c.mapping_method"
        " FROM capabilities c JOIN raw_claims r USING (claim_id)"
        " JOIN chunks ch ON ch.chunk_id = r.chunk_id JOIN documents d ON d.doc_id = ch.doc_id"
        " WHERE c.feature_id = %(f)s"
        " AND (%(t)s::text IS NULL OR c.target_id = %(t)s"
        "      OR starts_with(c.target_id, %(t)s || '.'))"
        " AND (%(v)s::text IS NULL OR lower(c.vendor) = lower(%(v)s))"
        " AND (%(s)s::text IS NULL OR c.support = %(s)s)"
        " ORDER BY c.vendor, c.product NULLS LAST, c.target_id NULLS FIRST, c.support,"
        " r.chunk_id",
        {"f": fid, "t": tid, "v": vendor, "s": support})
    return [CapRow(*row) for row in await cur.fetchall()]
