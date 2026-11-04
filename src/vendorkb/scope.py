"""Question scope detection: which vendors/products does a question name.

Reads ``scope_aliases.json`` (bundled with the package; seeded from graph-rag's answer_api) as plain data.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ALIASES_PATH = Path(__file__).resolve().parent / "scope_aliases.json"

# Copied from graph-rag's answer_api.scope.
_CROSS_VENDOR = re.compile(
    r"\bvendors\b"
    r"|\b(which|what|any|other|different|competing) vendor\b(?!-)"
    r"|\bacross (\w+ )?(vendors|providers|clouds|platforms|products|solutions)\b"
    r"|\bthird[- ]party (vendors?|tools|products|solutions|backup)\b"
    r"|\b(all|other|alternative|competing) (backup )?(products|solutions|vendors)\b",
    re.IGNORECASE)


@dataclass
class ScopeIndex:
    vendors: list[str]
    products: list[str]
    aliases: dict[str, list[str]]
    source_products: dict[str, list[str]] = field(default_factory=dict)
    source_vendors: dict[str, str] = field(default_factory=dict)


def _read_aliases(path: Path | None) -> dict[str, list[str]]:
    raw = json.loads((path or DEFAULT_ALIASES_PATH).read_text(encoding="utf-8"))
    return {k: ([v] if isinstance(v, str) else list(v)) for k, v in raw.items()}


def load_scope_index(vendors: list[str], products: list[str], aliases_path: Path | None = None,
                     source_products: dict[str, list[str]] | None = None,
                     source_vendors: dict[str, str] | None = None) -> ScopeIndex:
    return ScopeIndex(list(vendors), list(products), _read_aliases(aliases_path),
                      dict(source_products or {}), dict(source_vendors or {}))


async def scope_index_from_db(conn, aliases_path: Path | None = None) -> ScopeIndex:
    cur = await conn.execute("SELECT DISTINCT vendor FROM documents WHERE vendor IS NOT NULL")
    vendors = sorted(r[0] for r in await cur.fetchall())
    cur = await conn.execute("SELECT DISTINCT product FROM documents WHERE product IS NOT NULL")
    products = sorted(r[0] for r in await cur.fetchall())
    cur = await conn.execute(
        "SELECT DISTINCT source, product FROM documents"
        " WHERE source IS NOT NULL AND product IS NOT NULL")
    source_products: dict[str, list[str]] = {}
    for source, product in sorted(await cur.fetchall()):
        source_products.setdefault(source, []).append(product)
    cur = await conn.execute(
        "SELECT DISTINCT source, vendor FROM documents"
        " WHERE source IS NOT NULL AND vendor IS NOT NULL")
    by_source: dict[str, set[str]] = {}
    for source, vendor in await cur.fetchall():
        by_source.setdefault(source, set()).add(vendor)
    # a source shared by several vendors is not unique, hence never a direct surface
    source_vendors = {s: next(iter(v)) for s, v in by_source.items() if len(v) == 1}
    return load_scope_index(vendors, products, aliases_path, source_products, source_vendors)


def _resolve(name: str, index: ScopeIndex) -> tuple[list[str], list[str]]:
    """Map one target name to (vendors, products); unknown names resolve to nothing."""
    if name in index.vendors:
        return [name], []
    if name in index.products:
        return [], [name]
    if name in index.source_products:
        return [], list(index.source_products[name])
    return [], []


def _pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)


def detect_scope(question: str, index: ScopeIndex) -> tuple[list[str], list[str]]:
    if _CROSS_VENDOR.search(question):
        return [], []
    # surface text -> list of target names (direct names target themselves)
    surfaces: dict[str, list[str]] = {}
    for name in [*index.vendors, *index.products]:
        surfaces.setdefault(name, [name])
    for source, vendor in index.source_vendors.items():
        # distinctive sources only: unique to one vendor and naming it ("Veeam Agent for Linux")
        if source in index.source_products and vendor.lower() in source.lower():
            surfaces.setdefault(source, [source])
    for alias, targets in index.aliases.items():
        surfaces[alias] = targets
    taken: list[tuple[int, int]] = []
    hits: list[tuple[int, str]] = []
    for surface in sorted(surfaces, key=len, reverse=True):
        for m in _pattern(surface).finditer(question):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            hits.append((m.start(), surface))
    vendors: list[str] = []
    products: list[str] = []
    for _, surface in sorted(hits):
        for target in surfaces[surface]:
            v, p = _resolve(target, index)
            vendors.extend(x for x in v if x not in vendors)
            products.extend(x for x in p if x not in products)
    return vendors, products
