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
    product_vendors: dict[str, str] = field(default_factory=dict)


def _read_aliases(path: Path | None) -> dict[str, list[str]]:
    raw = json.loads((path or DEFAULT_ALIASES_PATH).read_text(encoding="utf-8"))
    return {k: ([v] if isinstance(v, str) else list(v)) for k, v in raw.items()}


def load_scope_index(vendors: list[str], products: list[str], aliases_path: Path | None = None,
                     source_products: dict[str, list[str]] | None = None,
                     source_vendors: dict[str, str] | None = None,
                     product_vendors: dict[str, str] | None = None) -> ScopeIndex:
    return ScopeIndex(list(vendors), list(products), _read_aliases(aliases_path),
                      dict(source_products or {}), dict(source_vendors or {}),
                      dict(product_vendors or {}))


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
    cur = await conn.execute(
        "SELECT DISTINCT product, vendor FROM documents"
        " WHERE product IS NOT NULL AND vendor IS NOT NULL")
    by_product: dict[str, set[str]] = {}
    for product, vendor in await cur.fetchall():
        by_product.setdefault(product, set()).add(vendor)
    # a product name under several vendors maps to none
    product_vendors = {p: next(iter(v)) for p, v in by_product.items() if len(v) == 1}
    return load_scope_index(vendors, products, aliases_path, source_products, source_vendors,
                            product_vendors)


def _resolve(name: str, index: ScopeIndex) -> tuple[list[str], list[str]]:
    """Map one target name to (vendors, products); unknown names resolve to nothing."""
    if name in index.vendors:
        return [name], []
    if name in index.products:
        return [], [name]
    if name in index.source_products:
        return [], list(index.source_products[name])
    return [], []


# single-word names that are also ordinary words: matched with their exact capitalization
_WORD_NAMES = frozenset({
    "own", "fly", "slide", "spanning", "security", "platform", "insights", "support", "snapshot",
    "monday", "application", "infrastructure", "documentation", "docs", "notices", "sentinel"})


# A product name made only of these words ("Backup and Recovery", "Security Center") names
# nobody's product on its own: it is a surface only after its vendor's name. Alone it matched
# phrases such as "Microsoft's Entra Backup and Recovery" to Storware's product (cv-12).
_GENERIC_WORDS = frozenset({
    "backup", "backups", "recovery", "restore", "and", "&", "cloud", "data", "protection",
    "manager", "management", "console", "server", "agent", "suite", "platform", "service",
    "services", "for", "the", "online", "storage", "archive", "archiving", "replication",
    "security", "enterprise", "center", "edition"})


def _is_generic(product: str) -> bool:
    words = re.findall(r"[a-z0-9&]+", product.lower())
    return bool(words) and all(w in _GENERIC_WORDS for w in words)


def _vendor_names(vendor: str) -> list[str]:
    """A vendor's name and the parts around a parenthesis: "Hornetsecurity (Altaro)" ->
    that, "Hornetsecurity" and "Altaro"."""
    names = [vendor, re.sub(r"\s*\([^)]*\)", "", vendor).strip(),
             *re.findall(r"\(([^)]*)\)", vendor)]
    return list(dict.fromkeys(n for n in names if n))


def _pattern(name: str) -> re.Pattern[str]:
    flags = 0 if name.lower() in _WORD_NAMES else re.IGNORECASE
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", flags)


def is_cross_vendor(question: str) -> bool:
    return bool(_CROSS_VENDOR.search(question))


def detect_scope(question: str, index: ScopeIndex) -> tuple[list[str], list[str]]:
    if is_cross_vendor(question):
        return [], []
    return find_facets(question, index)


def find_facets(question: str, index: ScopeIndex) -> tuple[list[str], list[str]]:
    """(vendors, products) the question names, in question order; ignores cross-vendor wording."""
    # surface text -> list of target names (direct names target themselves)
    surfaces: dict[str, list[str]] = {}
    for name in index.vendors:
        surfaces.setdefault(name, [name])
    for name in index.products:
        if not _is_generic(name):
            surfaces.setdefault(name, [name])
        elif name in index.product_vendors:
            for vendor in _vendor_names(index.product_vendors[name]):
                surfaces.setdefault(f"{vendor} {name}", [name])
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
