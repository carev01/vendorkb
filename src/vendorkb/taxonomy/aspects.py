"""``aspects()``: which aspects apply to a workload, read from the v2 taxonomy tables.

A workload is a target (resolved by id or alias) or a class. Its aspects are, in order:

1. coverage aspects: each object type of the class, inherited ones included, root ancestor
   first and each class's in YAML order (the order of ``Taxonomy.object_types``), times
   ``REQUIRED_DIMENSIONS``;
2. the class's other, non-platform dimensions (any coverage-family dimension that is not one of
   the required four included), in ``FAMILIES`` order, then by id;
3. platform dimensions (about the backup product itself), same order.

A dimension applies to class C when its ``applies_to`` is NULL (all classes) or any entry is C
or an ancestor of C. Two targets of one class therefore get equal lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import psycopg

from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import FAMILIES, REQUIRED_DIMENSIONS


@dataclass(frozen=True)
class Aspect:
    kind: Literal["coverage", "dimension", "platform"]
    dimension_id: str
    family: str
    name: str
    definition: str
    object_type_id: str | None
    value_unit: str | None


def dimension_applies(applies_to: list[str] | None, lineage: list[str]) -> bool:
    """Whether a dimension with ``applies_to`` (NULL: every class) applies to the class whose
    lineage (the class and its ancestors) is ``lineage``."""
    return applies_to is None or any(entry in lineage for entry in applies_to)


async def _target_class(conn: psycopg.AsyncConnection, target: str) -> str:
    cur = await conn.execute("SELECT class_id FROM targets WHERE target_id = %s", (target,))
    row = await cur.fetchone()
    if row is None:
        cur = await conn.execute(
            "SELECT DISTINCT t.class_id FROM aliases a JOIN targets t ON t.target_id = a.entity_id"
            " WHERE a.entity_type = 'target' AND a.scope = '' AND a.alias = %s",
            (normalize(target),))
        found = await cur.fetchall()
        if not found:
            raise LookupError(f"unknown target {target!r}")
        if len(found) > 1:
            raise LookupError(f"target {target!r} is ambiguous: classes "
                              f"{', '.join(sorted(r[0] for r in found))}")
        row = found[0]
    return str(row[0])


async def _lineage(conn: psycopg.AsyncConnection, class_id: str) -> list[str]:
    """``class_id`` and its ancestors, most specific first (``classes.parent_id``)."""
    lineage: list[str] = []
    current: str | None = class_id
    while current is not None:
        cur = await conn.execute("SELECT parent_id FROM classes WHERE class_id = %s", (current,))
        row = await cur.fetchone()
        if row is None:
            if current == class_id:
                raise LookupError(f"unknown class {class_id!r}")
            break
        lineage.append(current)
        current = row[0]
    return lineage


async def aspects(conn: psycopg.AsyncConnection, *, target: str | None = None,
                  cls: str | None = None) -> list[Aspect]:
    """The aspects of a target (id or alias) or a class; exactly one must be given.
    Unknown input raises ``LookupError``."""
    if (target is None) == (cls is None):
        raise ValueError("give exactly one of target and cls")
    class_id = await _target_class(conn, target) if target is not None else str(cls)
    lineage = await _lineage(conn, class_id)

    cur = await conn.execute(
        "SELECT dimension_id, family, name, definition, applies_to, value_unit, platform"
        " FROM dimensions")
    dims = {r[0]: r for r in await cur.fetchall()}

    out: list[Aspect] = []
    for cid in reversed(lineage):  # validation refuses a redefined object type: ids are unique
        cur = await conn.execute(
            "SELECT object_type_id FROM object_types WHERE class_id = %s ORDER BY ordinal",
            (cid,))
        for (object_type,) in await cur.fetchall():
            for dimension_id in REQUIRED_DIMENSIONS:
                d = dims[dimension_id]
                out.append(Aspect("coverage", dimension_id, d[1], d[2], d[3], object_type, d[5]))

    def order(d: tuple) -> tuple[int, str]:
        return (FAMILIES.index(d[1]), d[0])

    rest = [d for d in dims.values()
            if d[0] not in REQUIRED_DIMENSIONS and dimension_applies(d[4], lineage)]
    for platform in (False, True):
        for d in sorted((d for d in rest if d[6] is platform), key=order):
            out.append(Aspect("platform" if platform else "dimension", d[0], d[1], d[2], d[3],
                              None, d[5]))
    return out
