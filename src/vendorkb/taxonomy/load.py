"""Load ``taxonomy.yaml`` into ``features`` / ``targets`` / ``aliases``.

The YAML (the source of truth, reviewed in git) has the top-level keys ``version`` (int),
``features`` and ``targets`` (lists of entries) and ``long_tail`` (ignored here). An entry is
``{id, name, category, definition, aliases, [parent], [support_count], [vendor_count]}``.
Feature ids are snake_case; target ids are dotted lowercase (``aws.rds``), and a target's
``parent`` must be another target in the file. ``category`` is one of ``CATEGORIES``; a
target's ``kind`` derives from it.

Loading validates the whole file first (a ValueError names the problem and nothing is
written), then replaces the tables' contents in one transaction: entries missing from the
file are deleted (with any ``capabilities`` rows that point at them), the rest upserted,
the aliases rewritten (normalized), and ``version`` recorded as the taxonomy version.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import yaml

from vendorkb.taxonomy.cluster import normalize

CATEGORIES = ("Workloads", "Platforms", "Storage targets", "Data protection", "Security",
              "Recovery", "Management")
FEATURE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TARGET_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
TARGET_KIND = {"Workloads": "workload", "Platforms": "platform", "Storage targets": "storage"}
VERSION_KEY = "taxonomy_version"


@dataclass(frozen=True)
class Entry:
    id: str
    name: str
    category: str
    definition: str | None
    parent: str | None
    aliases: tuple[str, ...]


def read_doc(path: str | Path) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a YAML mapping")
    return doc


def _entries(doc: dict[str, Any], kind: str) -> list[Entry]:
    key = f"{kind}s"
    raw = doc.get(key) or []
    if not isinstance(raw, list):
        raise ValueError(f"{key}: must be a list")
    id_re = FEATURE_ID_RE if kind == "feature" else TARGET_ID_RE
    out: list[Entry] = []
    seen: set[str] = set()
    for n, item in enumerate(raw):
        where = f"{key}[{n}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where}: must be a mapping")
        eid = item.get("id")
        if not isinstance(eid, str) or not id_re.match(eid):
            raise ValueError(f"{where}: invalid {kind} id {eid!r}")
        if eid in seen:
            raise ValueError(f"{where}: duplicate id {eid!r}")
        seen.add(eid)
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where} ({eid}): name is required")
        if item.get("category") not in CATEGORIES:
            raise ValueError(f"{where} ({eid}): category {item.get('category')!r} is not one"
                             f" of {', '.join(CATEGORIES)}")
        definition = item.get("definition")
        if definition is not None and not isinstance(definition, str):
            raise ValueError(f"{where} ({eid}): definition must be a string")
        parent = item.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f"{where} ({eid}): parent must be an id or null")
        aliases = item.get("aliases") or []
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ValueError(f"{where} ({eid}): aliases must be a list of strings")
        normalized = tuple(dict.fromkeys(a for a in map(normalize, aliases) if a))
        out.append(Entry(eid, name.strip(), item["category"], definition, parent, normalized))
    for e in out:
        if e.parent is not None and e.parent not in seen:
            raise ValueError(f"{key}: {e.id} has unknown parent {e.parent!r}")
    parent_of = {e.id: e.parent for e in out}
    for e in out:
        chain, node = {e.id}, e.parent
        while node is not None:
            if node in chain:
                raise ValueError(f"{key}: parent cycle through {e.id}")
            chain.add(node)
            node = parent_of[node]
    owner: dict[str, str] = {}
    for e in out:
        for a in e.aliases:
            if owner.setdefault(a, e.id) != e.id:
                raise ValueError(f"{key}: alias {a!r} is on both {owner[a]} and {e.id}")
    return out


def validate(doc: dict[str, Any]) -> tuple[int, list[Entry], list[Entry]]:
    version = doc.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"version must be an integer, got {version!r}")
    return version, _entries(doc, "feature"), _entries(doc, "target")


async def current_version(conn: psycopg.AsyncConnection) -> str | None:
    cur = await conn.execute("SELECT value FROM taxonomy_meta WHERE key=%s", (VERSION_KEY,))
    row = await cur.fetchone()
    return None if row is None else str(row[0])


async def _replace(conn: psycopg.AsyncConnection, table: str, id_col: str,
                   entries: list[Entry], cols: str, values: list[tuple]) -> None:
    ids = [e.id for e in entries]
    await conn.execute(f"UPDATE {table} SET parent_id = NULL")
    await conn.execute(f"DELETE FROM {table} WHERE {id_col} <> ALL(%s)", (ids,))
    placeholders = ",".join(["%s"] * (len(cols.split(",")) + 1))
    updates = ",".join(f"{c.strip()}=EXCLUDED.{c.strip()}" for c in cols.split(","))
    async with conn.cursor() as cur:
        if values:
            await cur.executemany(
                f"INSERT INTO {table} ({id_col}, {cols}) VALUES ({placeholders})"
                f" ON CONFLICT ({id_col}) DO UPDATE SET {updates}", values)
        parents = [(e.parent, e.id) for e in entries if e.parent is not None]
        if parents:
            await cur.executemany(
                f"UPDATE {table} SET parent_id=%s WHERE {id_col}=%s", parents)


async def load_taxonomy(conn: psycopg.AsyncConnection, path: str | Path) -> dict:
    """Validate and load the YAML at ``path``; returns the version and the row counts."""
    version, features, targets = validate(read_doc(path))
    feature_ids = [e.id for e in features]
    target_ids = [e.id for e in targets]
    aliases = ([(a, "feature", e.id) for e in features for a in e.aliases]
               + [(a, "target", e.id) for e in targets for a in e.aliases])
    async with conn.transaction():
        await conn.execute("DELETE FROM aliases")
        await conn.execute(
            "DELETE FROM capabilities WHERE feature_id <> ALL(%s) OR target_id <> ALL(%s)",
            (feature_ids, target_ids))
        await _replace(conn, "features", "feature_id", features, "name, category, definition",
                       [(e.id, e.name, e.category, e.definition) for e in features])
        await _replace(conn, "targets", "target_id", targets, "name, kind",
                       [(e.id, e.name, TARGET_KIND.get(e.category)) for e in targets])
        if aliases:
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO aliases (alias, entity_type, entity_id) VALUES (%s,%s,%s)",
                    aliases)
        await conn.execute(
            "INSERT INTO taxonomy_meta (key, value) VALUES (%s,%s) ON CONFLICT (key)"
            " DO UPDATE SET value=EXCLUDED.value, updated_at=now()",
            (VERSION_KEY, str(version)))
    return {"version": str(version), "features": len(features), "targets": len(targets),
            "aliases": len(aliases)}
