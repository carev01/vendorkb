"""``capabilities`` v2 (spec §6.2, §6.3): the fact table, rebuilt from mapped claims, and queried.

``build`` replaces every row (delete, then insert, in one transaction) with one row per claim
of the claims-v3 family (``propose.PROMPT_FAMILY``) mapped under the current taxonomy version
with a dimension. Each row is keyed on its claim and gets:

- ``scope`` (workload dimensions, i.e. not ``platform``):
  - ``target_unmapped``: the claim names a target that matched no taxonomy target. It is not a
    product-wide statement: "backup of Lotus Notes is not supported" says nothing about backup
    in general.
  - ``product_wide``: no target text at all.
  - Otherwise ``offering`` when an offering is named, else ``product``.
  Each agrees with the ``query`` group of the same name.
- ``conflict_group``: rows that make the same statement (the conflict key: vendor, product,
  offering, edition, dimension, target, variant and object type, plus the normalized capability
  phrase, target scope text when no object type resolved, and conditions; NULLs equal) share a
  group id derived from the key when one says supported or partial and another not_supported,
  or when their stated values differ (a row with no value contradicts none). Supported against
  partial, and statements about different things under one dimension, are not conflicts. A
  rebuild reproduces the id.
- ``is_current``: the key's rows from its most recently fetched document
  (``documents.fetched_at``, also kept as ``doc_version``); the latest document wins and the
  older rows stay, so the conflict stays visible.

``query`` serves only rows built under the current taxonomy version and sorts every matching
row into one of four groups, so nothing is presented as more specific than its evidence:

- ``product_wide``: rows with no target whose dimension is a workload one. With a ``target``
  filter only the dimensions that apply to the target's class are kept (and the target filter
  does not apply to them); with an ``object_type`` filter rows with no object type are kept.
  They are never in ``facts``.
- ``product_level``: with an ``offering`` or ``edition`` filter, the rows less specific than it:
  those with no offering, of the filtered offerings' vendors, and (with an edition filter) the
  filtered offerings' rows with no edition. Without either filter it is empty.
- ``unresolved_target``: ``target_unmapped`` rows (``Fact.target_raw`` names the workload).
  Never returned with a ``target`` filter, which they cannot be shown to match.
- ``facts``: the rest.

``conflicts`` lists each conflict group met in the result with all its members, the ones the
filters left out included. When all three groups are empty, ``hint`` says what to search the
chunks for instead. Filters naming taxonomy entities (target, dimension, offering, edition)
take an id or an alias; when one resolves to nothing (an edition: nothing within the filtered
offerings), the result is empty in every group, with a ``hint`` naming what did not resolve,
rather than the less specific rows of the vendor.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import psycopg

from vendorkb.taxonomy.aspects import _lineage, dimension_applies
from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import current_version
from vendorkb.taxonomy.propose import PROMPT_FAMILY

__all__ = ["CapFilter", "CapResult", "Conflict", "Fact", "build", "query"]

_KEY = ("vendor", "product", "offering_id", "edition_id", "dimension_id", "target_id",
        "variant_id", "object_type_id", "cap_norm", "scope_norm", "cond_norm")


def _norm(expr: str) -> str:
    """SQL: ``expr`` lowercased with every run of non-word characters as one space."""
    return f"trim(lower(regexp_replace(coalesce({expr}, ''), '\\W+', ' ', 'g')))"


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
    facts: list[Fact]  # at most CONFLICT_MEMBERS of them, current first
    size: int = 0  # all members of the group


@dataclass
class CapResult:
    facts: list[Fact] = field(default_factory=list)
    product_level: list[Fact] = field(default_factory=list)
    product_wide: list[Fact] = field(default_factory=list)
    unresolved_target: list[Fact] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    hint: str | None = None
    totals: dict[str, int] = field(default_factory=dict)  # rows per group before ``limit``


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
            " CASE WHEN target_id IS NULL AND NOT platform AND target_raw IS NOT NULL"
            "  THEN 'target_unmapped' WHEN target_id IS NULL AND NOT platform THEN 'product_wide'"
            "  WHEN offering_id IS NOT NULL THEN 'offering' ELSE 'product' END,"
            " CASE WHEN (bool_or(support = 'not_supported') OVER k"
            "   AND bool_or(support IN ('supported', 'partial')) OVER k)"
            "  OR min(value) OVER k <> max(value) OVER k"
            f" THEN md5(jsonb_build_array({key})::text)::uuid END,"
            " fetched_at = max(fetched_at) OVER k"
            # vendor and product come from the document, not the claim: the sync re-tags a
            # document (store.update_metadata) without touching its chunks or their claims
            " FROM (SELECT r.claim_id, d.vendor, d.product, r.offering_id, r.edition_id,"
            "  r.dimension_id, r.class_id, r.target_id, r.variant_id, r.object_type_id, r.kind,"
            "  r.support, r.availability, r.default_state, r.value, r.unit, r.bound,"
            "  r.conditions, d.fetched_at, dim.platform, r.target_raw,"
            f"  {_norm('r.capability_raw')} AS cap_norm,"
            f"  CASE WHEN r.object_type_id IS NULL THEN {_norm('r.target_scope')} END AS scope_norm,"
            f"  {_norm('r.conditions')} AS cond_norm FROM raw_claims r"
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


def _unresolved(f: CapFilter, targets: list[str] | None, dimensions: list[str] | None,
                offerings: list[str] | None, editions: list[tuple[str, str]] | None) -> list[str]:
    """The given taxonomy filters that resolved to no entity, as the user wrote them."""
    missing = []
    if targets == []:
        missing.append(f"target {f.target!r}")
    if dimensions == []:
        missing.append(f"dimension {f.dimension!r}")
    if offerings == []:
        missing.append(f"offering {f.offering!r}")
    if editions == []:
        missing.append(f"edition {f.edition!r}"
                       + (f" of offering {f.offering!r}" if offerings else ""))
    return missing


async def _hint(conn: psycopg.AsyncConnection, f: CapFilter, targets: list[str] | None,
                dimensions: list[str] | None, unresolved: list[str] | None = None) -> str:
    """Where to look instead: the vendor, target and dimension, by name where known; first,
    the filters that named no taxonomy entity."""
    if unresolved:
        return (f"No taxonomy entity matches {', '.join(unresolved)} (by id or alias). "
                + await _hint(conn, f, targets, dimensions))
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


_N_FACT = len(fields(Fact))


def _fact(row: tuple) -> Fact:
    values = list(row)
    doc_version = values[28]
    values[28] = None if doc_version is None else doc_version.isoformat()
    return Fact(*values)


CONFLICT_MEMBERS = 10  # members listed per conflict group (``Conflict.size`` counts them all)


async def query(conn: psycopg.AsyncConnection, f: CapFilter,
                limit: int | None = None) -> CapResult:
    """The rows matching ``f``, grouped as the module docstring describes; with ``limit``, at
    most that many rows per group, ``totals`` keeping the full counts. A limited group is
    filled round-robin across vendors (every vendor's first row in the usual order, then every
    vendor's second, ...) and returned in that order, so a broad filter shows many vendors
    rather than the first few alphabetically. Each conflict group lists at most
    ``CONFLICT_MEMBERS`` members."""
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
    unresolved = _unresolved(f, targets, dimensions, offerings, editions)
    if unresolved:
        # never fall back to the less specific rows of a filter that named nothing
        return CapResult(hint=await _hint(conn, f, targets, dimensions, unresolved))
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

    wide = "(c.scope = 'product_wide')"
    off_taxonomy = "(c.scope = 'target_unmapped')"
    exact = ("((%(offerings)s::text[] IS NULL OR c.offering_id = ANY(%(offerings)s::text[]))"
             " AND (%(editions)s::text[] IS NULL"
             "  OR c.offering_id || '/' || c.edition_id = ANY(%(editions)s::text[])))")
    general = ("(%(general)s AND ((c.offering_id IS NULL"
               "  AND lower(c.vendor) = ANY(%(general_vendors)s::text[]))"
               " OR (c.edition_id IS NULL"
               "  AND c.offering_id = ANY(%(edition_offerings)s::text[]))))")
    grp = (f"CASE WHEN {off_taxonomy} THEN 'unresolved_target' WHEN {wide} THEN 'product_wide'"
           f" WHEN {exact} THEN 'facts' ELSE 'product_level' END")
    params["limit"] = limit
    sql = (
        f"SELECT {grp} AS grp, count(*) OVER (PARTITION BY {grp}) AS total,"
        f" row_number() OVER (PARTITION BY {grp}{_ORDER}) AS pos,"
        f" row_number() OVER (PARTITION BY {grp}, c.vendor{_ORDER}) AS vendor_pos,"
        f" {_FACT_COLUMNS}{_FROM}"
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
    # the window ranks rows inside each group; the outer filter keeps the first ``limit``
    if limit is None:
        sql = f"SELECT * FROM ({sql}) s ORDER BY s.pos"
    else:  # round-robin across vendors: rank by the row's position within its vendor
        sql = (f"SELECT * FROM (SELECT s.*, row_number() OVER (PARTITION BY s.grp ORDER BY"
               f" s.vendor_pos, s.pos) AS rr FROM ({sql}) s) t WHERE t.rr <= %(limit)s"
               " ORDER BY t.rr")
    cur = await conn.execute(sql, params)
    result = CapResult(totals={"facts": 0, "product_level": 0, "product_wide": 0,
                               "unresolved_target": 0})
    groups: dict[str, list[Fact]] = {"facts": result.facts,
                                     "product_level": result.product_level,
                                     "product_wide": result.product_wide,
                                     "unresolved_target": result.unresolved_target}
    for row in await cur.fetchall():
        groups[row[0]].append(_fact(row[4:4 + _N_FACT]))  # after grp, total, pos, vendor_pos
        result.totals[row[0]] = int(row[1])

    met = sorted({x.conflict_group for g in groups.values() for x in g
                  if x.conflict_group is not None})
    if met:
        # a conflict group can hold thousands of rows: list the first CONFLICT_MEMBERS
        cur = await conn.execute(
            f"SELECT * FROM (SELECT count(*) OVER (PARTITION BY c.conflict_group),"
            f" row_number() OVER (PARTITION BY c.conflict_group{_ORDER}), {_FACT_COLUMNS}{_FROM}"
            " WHERE c.taxonomy_version = %s AND c.conflict_group = ANY(%s::uuid[])) s"
            " WHERE s.row_number <= %s ORDER BY s.row_number", (version, met, CONFLICT_MEMBERS))
        members: dict[str, list[Fact]] = {g: [] for g in met}
        sizes: dict[str, int] = {}
        for row in await cur.fetchall():
            fact = _fact(row[2:])
            members[str(fact.conflict_group)].append(fact)
            sizes[str(fact.conflict_group)] = int(row[0])
        result.conflicts = [Conflict(g, members[g], sizes.get(g, 0)) for g in met]
    if not (result.facts or result.product_level or result.product_wide
            or result.unresolved_target):
        result.hint = await _hint(conn, f, targets, dimensions)
    return result
