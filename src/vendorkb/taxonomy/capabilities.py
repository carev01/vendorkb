"""``capabilities`` v2 (spec §6.2, §6.3): the fact table, rebuilt from mapped claims, and queried.

``build`` replaces every row (delete, then insert, in one transaction) with one row per claim
of the claims-v3 family (``propose.PROMPT_FAMILY``) mapped under the current taxonomy version
with a dimension. Each row is keyed on its claim and gets:

- ``scope``: ``offering`` when an offering is named; else ``product_wide`` when there is no
  target and the dimension is a workload one (not ``platform``); else ``product``.
- ``conflict_group``: rows sharing the conflict key (vendor, product, offering, edition,
  dimension, target, variant, object type; NULLs equal) that differ in support, or in their
  stated values (a row with no value contradicts none), share a group id derived from the key,
  so a rebuild reproduces it.
- ``is_current``: the key's rows from its most recently fetched document
  (``documents.fetched_at``, also kept as ``doc_version``); the latest document wins and the
  older rows stay, so the conflict stays visible.

``query`` serves only rows built under the current taxonomy version and sorts every matching
row into one of three groups, so nothing is presented as more specific than its evidence:

- ``product_wide``: rows with no target whose dimension is a workload one. With a ``target``
  filter only the dimensions that apply to the target's class are kept (and the target filter
  does not apply to them); with an ``object_type`` filter rows with no object type are kept.
  They are never in ``facts``.
- ``product_level``: with an ``offering`` or ``edition`` filter, the rows less specific than it:
  those with no offering, of the filtered offerings' vendors, and (with an edition filter) the
  filtered offerings' rows with no edition. Without either filter it is empty.
- ``facts``: the rest.

``conflicts`` lists each conflict group met in the result with all its members, the ones the
filters left out included. When all three groups are empty, ``hint`` says what to search the
chunks for instead. Filters naming taxonomy entities (target, dimension, offering, edition)
take an id or an alias; one that resolves to nothing matches nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import psycopg

from vendorkb.taxonomy.aspects import _lineage, dimension_applies
from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import current_version
from vendorkb.taxonomy.propose import PROMPT_FAMILY

__all__ = ["CapFilter", "CapResult", "Conflict", "Fact", "build", "query"]

_KEY = ("vendor", "product", "offering_id", "edition_id", "dimension_id", "target_id",
        "variant_id", "object_type_id")


@dataclass(frozen=True)
class CapFilter:
    vendor: str | None = None
    offering: str | None = None
    edition: str | None = None
    dimension: str | None = None
    family: str | None = None
    target: str | None = None  # descendants included
    object_type: str | None = None
    support: str | None = None
    availability: str | None = None
    value_min: float | None = None
    value_max: float | None = None


@dataclass(frozen=True)
class Fact:
    claim_id: str
    vendor: str
    product: str | None
    offering_id: str | None
    edition_id: str | None
    dimension_id: str
    dimension_family: str | None
    class_id: str | None
    target_id: str | None
    variant_id: str | None
    object_type_id: str | None
    kind: str
    support: str
    availability: str | None
    default_state: str | None
    value: float | None
    unit: str | None
    bound: str | None
    conditions: str | None
    scope: str
    conflict_group: str | None
    is_current: bool
    capability_raw: str
    target_raw: str | None
    evidence_quote: str
    chunk_id: str
    url: str | None
    title: str | None
    doc_version: str | None  # documents.fetched_at, ISO 8601
    taxonomy_version: str


@dataclass(frozen=True)
class Conflict:
    conflict_group: str
    facts: list[Fact]


@dataclass
class CapResult:
    facts: list[Fact] = field(default_factory=list)
    product_level: list[Fact] = field(default_factory=list)
    product_wide: list[Fact] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    hint: str | None = None


async def build(conn: psycopg.AsyncConnection) -> int:
    """Rebuild ``capabilities`` for the current taxonomy version; returns the row count."""
    version = await current_version(conn)
    key = ", ".join(_KEY)
    async with conn.transaction():
        await conn.execute("DELETE FROM capabilities")
        if version is None:
            return 0
        cur = await conn.execute(
            "INSERT INTO capabilities (claim_id, vendor, product, offering_id, edition_id,"
            " dimension_id, class_id, target_id, variant_id, object_type_id, kind, support,"
            " availability, default_state, value, unit, bound, conditions, doc_version,"
            " taxonomy_version, scope, conflict_group, is_current)"
            " SELECT claim_id, vendor, product, offering_id, edition_id, dimension_id,"
            " class_id, target_id, variant_id, object_type_id, kind, support, availability,"
            " default_state, value, unit, bound, conditions, fetched_at, %(ver)s,"
            " CASE WHEN offering_id IS NOT NULL THEN 'offering'"
            "  WHEN target_id IS NULL AND NOT platform THEN 'product_wide' ELSE 'product' END,"
            " CASE WHEN min(support) OVER k <> max(support) OVER k"
            "  OR min(value) OVER k <> max(value) OVER k"
            f" THEN md5(jsonb_build_array({key})::text)::uuid END,"
            " fetched_at = max(fetched_at) OVER k"
            " FROM (SELECT r.*, d.fetched_at, dim.platform FROM raw_claims r"
            "  JOIN chunks ch ON ch.chunk_id = r.chunk_id"
            "  JOIN documents d ON d.doc_id = ch.doc_id"
            "  JOIN dimensions dim ON dim.dimension_id = r.dimension_id"
            "  WHERE r.prompt_version LIKE %(family)s AND r.mapped_version = %(ver)s) s"
            f" WINDOW k AS (PARTITION BY {key})",
            {"ver": version, "family": PROMPT_FAMILY})
        return cur.rowcount


async def _resolve(conn: psycopg.AsyncConnection, table: str, col: str, entity_type: str,
                   term: str) -> list[str]:
    """The ids whose id is ``term``, plus those with ``term`` as a normalized alias (any
    scope)."""
    cur = await conn.execute(
        f"SELECT {col} FROM {table} WHERE {col} = %(t)s UNION"
        " SELECT entity_id FROM aliases WHERE entity_type = %(e)s AND alias = %(a)s",
        {"t": term.strip(), "e": entity_type, "a": normalize(term)})
    return sorted({str(r[0]) for r in await cur.fetchall()})


async def _editions(conn: psycopg.AsyncConnection, term: str,
                    offerings: list[str] | None) -> list[tuple[str, str]]:
    """The (offering, edition) pairs whose edition id or alias is ``term``, within
    ``offerings`` when given."""
    cur = await conn.execute(
        "SELECT offering_id, edition_id FROM editions WHERE edition_id = %(t)s UNION"
        " SELECT scope, entity_id FROM aliases WHERE entity_type = 'edition' AND alias = %(a)s",
        {"t": term.strip(), "a": normalize(term)})
    pairs = {(str(o), str(e)) for o, e in await cur.fetchall()}
    return sorted(p for p in pairs if offerings is None or p[0] in offerings)


async def _names(conn: psycopg.AsyncConnection, table: str, col: str,
                 ids: list[str]) -> list[str]:
    cur = await conn.execute(f"SELECT name FROM {table} WHERE {col} = ANY(%s) ORDER BY {col}",
                             (ids,))
    return [str(r[0]) for r in await cur.fetchall()]


async def _hint(conn: psycopg.AsyncConnection, f: CapFilter, targets: list[str] | None,
                dimensions: list[str] | None) -> str:
    """Where to look instead: the vendor, target and dimension, by name where known."""
    parts = []
    if f.vendor:
        parts.append(f"vendor {f.vendor!r}")
    if f.offering:
        parts.append(f"offering {f.offering!r}")
    if f.target:
        names = await _names(conn, "targets", "target_id", targets or [])
        parts.append("target " + (" / ".join(names) if names else repr(f.target)))
    if f.dimension:
        names = await _names(conn, "dimensions", "dimension_id", dimensions or [])
        parts.append("dimension " + (" / ".join(names) if names else repr(f.dimension)))
    elif f.family:
        parts.append(f"dimension family {f.family!r}")
    if f.object_type:
        parts.append(f"object type {f.object_type!r}")
    where = ", ".join(parts) if parts else "the question's vendor and aspect"
    return (f"No capability facts match. Search the document chunks (hybrid retrieval) for "
            f"{where}; the citation-checked answer path is the fallback.")


_FACT_COLUMNS = (
    "c.claim_id::text, c.vendor, c.product, c.offering_id, c.edition_id, c.dimension_id,"
    " dim.family, c.class_id, c.target_id, c.variant_id, c.object_type_id, c.kind, c.support,"
    " c.availability, c.default_state, c.value, c.unit, c.bound, c.conditions, c.scope,"
    " c.conflict_group::text, c.is_current, r.capability_raw, r.target_raw, r.evidence_quote,"
    " r.chunk_id::text, d.url, d.title, c.doc_version, c.taxonomy_version")
_FROM = (" FROM capabilities c JOIN raw_claims r ON r.claim_id = c.claim_id"
         " JOIN chunks ch ON ch.chunk_id = r.chunk_id JOIN documents d ON d.doc_id = ch.doc_id"
         " LEFT JOIN dimensions dim ON dim.dimension_id = c.dimension_id")
_ORDER = (" ORDER BY c.vendor, c.product NULLS FIRST, c.offering_id NULLS FIRST,"
          " c.edition_id NULLS FIRST, c.dimension_id, c.target_id NULLS FIRST,"
          " c.variant_id NULLS FIRST, c.object_type_id NULLS FIRST, c.is_current DESC,"
          " c.doc_version DESC, c.claim_id")


def _fact(row: tuple) -> Fact:
    values = list(row)
    doc_version = values[28]
    values[28] = None if doc_version is None else doc_version.isoformat()
    return Fact(*values)


async def query(conn: psycopg.AsyncConnection, f: CapFilter) -> CapResult:
    """The rows matching ``f``, grouped as the module docstring describes."""
    version = await current_version(conn)
    targets = dimensions = offerings = None
    if f.target is not None:
        targets = await _resolve(conn, "targets", "target_id", "target", f.target)
    if f.dimension is not None:
        dimensions = await _resolve(conn, "dimensions", "dimension_id", "dimension",
                                    f.dimension)
    if f.offering is not None:
        offerings = await _resolve(conn, "offerings", "offering_id", "offering", f.offering)
    editions = None if f.edition is None else await _editions(conn, f.edition, offerings)
    if version is None:
        return CapResult(hint=await _hint(conn, f, targets, dimensions))

    params: dict[str, Any] = {
        "ver": version, "vendor": f.vendor, "family": f.family, "support": f.support,
        "availability": f.availability, "vmin": f.value_min, "vmax": f.value_max,
        "ot": f.object_type, "dims": dimensions, "has_target": targets is not None,
        "targets": targets or [], "target_dims": [], "offerings": offerings,
        "editions": None if editions is None else [f"{o}/{e}" for o, e in editions],
        "general": offerings is not None or editions is not None,
        "edition_offerings": sorted({o for o, _ in editions or []}),
        "general_vendors": []}
    if targets:
        # the target and its descendants; the dimensions that apply to any of their classes
        cur = await conn.execute(
            "WITH RECURSIVE t AS (SELECT target_id, class_id FROM targets"
            " WHERE target_id = ANY(%s) UNION SELECT c.target_id, c.class_id FROM targets c"
            " JOIN t ON c.parent_id = t.target_id) SELECT target_id, class_id FROM t",
            (targets,))
        found = await cur.fetchall()
        params["targets"] = sorted({str(r[0]) for r in found})
        lineages = [await _lineage(conn, cls) for cls in sorted({str(r[1]) for r in found})]
        cur = await conn.execute("SELECT dimension_id, applies_to FROM dimensions")
        params["target_dims"] = sorted(
            str(d) for d, applies_to in await cur.fetchall()
            if any(dimension_applies(applies_to, lineage) for lineage in lineages))
    if params["general"]:
        related = sorted(set(offerings or []) | set(params["edition_offerings"]))
        cur = await conn.execute("SELECT DISTINCT lower(vendor) FROM offerings"
                                 " WHERE offering_id = ANY(%s)", (related,))
        params["general_vendors"] = [str(r[0]) for r in await cur.fetchall()]

    wide = "(c.target_id IS NULL AND NOT coalesce(dim.platform, false))"
    exact = ("((%(offerings)s::text[] IS NULL OR c.offering_id = ANY(%(offerings)s::text[]))"
             " AND (%(editions)s::text[] IS NULL"
             "  OR c.offering_id || '/' || c.edition_id = ANY(%(editions)s::text[])))")
    general = ("(%(general)s AND ((c.offering_id IS NULL"
               "  AND lower(c.vendor) = ANY(%(general_vendors)s::text[]))"
               " OR (c.edition_id IS NULL"
               "  AND c.offering_id = ANY(%(edition_offerings)s::text[]))))")
    sql = (
        f"SELECT CASE WHEN {wide} THEN 'product_wide' WHEN {exact} THEN 'facts'"
        f" ELSE 'product_level' END, {_FACT_COLUMNS}{_FROM}"
        " WHERE c.taxonomy_version = %(ver)s"
        " AND (%(vendor)s::text IS NULL OR lower(c.vendor) = lower(%(vendor)s::text))"
        " AND (%(dims)s::text[] IS NULL OR c.dimension_id = ANY(%(dims)s::text[]))"
        " AND (%(family)s::text IS NULL OR dim.family = %(family)s::text)"
        " AND (%(support)s::text IS NULL OR c.support = %(support)s::text)"
        " AND (%(availability)s::text IS NULL OR c.availability = %(availability)s::text)"
        " AND (%(vmin)s::float8 IS NULL OR c.value >= %(vmin)s::float8)"
        " AND (%(vmax)s::float8 IS NULL OR c.value <= %(vmax)s::float8)"
        " AND (%(ot)s::text IS NULL OR c.object_type_id = %(ot)s::text"
        f"  OR ({wide} AND c.object_type_id IS NULL))"
        " AND (NOT %(has_target)s OR c.target_id = ANY(%(targets)s::text[])"
        f"  OR ({wide} AND c.dimension_id = ANY(%(target_dims)s::text[])))"
        f" AND ({exact} OR {general})" + _ORDER)
    cur = await conn.execute(sql, params)
    result = CapResult()
    groups: dict[str, list[Fact]] = {"facts": result.facts,
                                     "product_level": result.product_level,
                                     "product_wide": result.product_wide}
    for row in await cur.fetchall():
        groups[row[0]].append(_fact(row[1:]))

    met = sorted({x.conflict_group for g in groups.values() for x in g
                  if x.conflict_group is not None})
    if met:
        cur = await conn.execute(
            f"SELECT {_FACT_COLUMNS}{_FROM} WHERE c.taxonomy_version = %s"
            " AND c.conflict_group = ANY(%s::uuid[])" + _ORDER, (version, met))
        members: dict[str, list[Fact]] = {g: [] for g in met}
        for row in await cur.fetchall():
            fact = _fact(row)
            members[str(fact.conflict_group)].append(fact)
        result.conflicts = [Conflict(g, members[g]) for g in met]
    if not (result.facts or result.product_level or result.product_wide):
        result.hint = await _hint(conn, f, targets, dimensions)
    return result
