"""The v1 taxonomy YAML format (``features`` / ``targets``), kept only for ``propose``.

Taxonomy v2 (``load``) replaced it and the v1 tables are dropped; ``propose`` still writes and
re-validates v1 files until it is rewritten for v2, and then this module goes. The YAML has
the top-level keys ``version`` (int), ``features`` and ``targets`` (lists of entries); an entry
is ``{id, name, category, definition, aliases, [parent], ...}``. A mapping key given twice is
refused; a bare YAML number used as an alias is taken as text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import parse_doc

CATEGORIES = ("Workloads", "Platforms", "Storage targets", "Data protection", "Security",
              "Recovery", "Management")
FEATURE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TARGET_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


@dataclass(frozen=True)
class Entry:
    id: str
    name: str
    category: str
    definition: str | None
    parent: str | None
    aliases: tuple[str, ...]


def read_doc(path: str | Path) -> dict[str, Any]:
    return parse_doc(Path(path).read_bytes(), path)


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
        if not isinstance(aliases, list):
            raise ValueError(f"{where} ({eid}): aliases must be a list of strings")
        texts: list[str] = []
        for a in aliases:
            if isinstance(a, bool) or not isinstance(a, str | int | float):
                raise ValueError(f"{where} ({eid}): alias {a!r} is not text (quote it)")
            texts.append(str(a))  # a bare YAML number such as 365 is text here
        normalized = tuple(dict.fromkeys(a for a in map(normalize, texts) if a))
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
