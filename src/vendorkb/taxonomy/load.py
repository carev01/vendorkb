"""Load the taxonomy v2 YAML (``taxonomy/taxonomy.yaml``, reviewed in git) into its tables.

The YAML has the top-level keys ``version: 2``, ``dimensions``, ``classes``, ``targets`` and
``offerings`` (which may be omitted), and no others:

- **dimension** ``{id, name, definition, aliases, [lookup_aliases], applies_to, [value_unit],
  [platform]}``: one vendor-neutral comparable aspect. ``id`` is ``<family>.<snake>`` with the family one of
  ``FAMILIES``; ``applies_to`` is ``all`` or a list of class ids (an entry matches its class and
  every class beneath it); ``platform: true`` marks an aspect of the backup product itself.
  The ``REQUIRED_DIMENSIONS`` must be present.
- **class** ``{id, name, definition, object_types}``: a kind of workload. ``id`` is dotted and
  every dotted prefix must be a class (its ancestors); ``object_types`` is a list of
  ``{id, name, aliases}``, inherited by every class beneath, and a class may not redefine an
  object type of an ancestor.
- **target** ``{id, name, class, aliases, [lookup_aliases], [object_aliases], [variants]}``: a
  concrete workload, an instance of ``class``. ``id`` is dotted and every dotted prefix must be a target (its
  parent chain); ``object_aliases`` maps object types of the class's lineage to the target's
  own words for them; ``variants`` is a list of ``{id, name, aliases}``.
- **offering** ``{vendor, id, name, aliases, editions}``: ``id`` is ``<snake>.<snake>`` and
  ``editions`` a list of ``{id, name, aliases}``.

``aliases`` may be omitted (none). ``lookup_aliases`` (dimensions and targets) are matched like
aliases but are never candidate-search keys of the mapper (``aliases.search`` false): adding one
leaves every other phrase's candidates, and so its stored decision, as it was. Dimensions, targets, offerings, object types, variants and
editions may carry ``proposed: true``, the review marker ``taxonomy propose`` puts on the
entries it adds; it changes nothing else. Unknown keys are refused, so a misspelt key is not silently
ignored. ``validate`` collects every problem before raising ``TaxonomyError``, and nothing is
written unless the file is valid. Aliases are stored normalized (``cluster.normalize``) with
each entity's own name among them, keyed ``(alias, entity_type, scope)``: an alias may name
only one entity of its type within its scope (and an object-type alias only one object type in
a class's whole lineage). A load replaces every taxonomy row in one transaction and records the
version ``"2-<first 8 hex of the file's sha256>"``, so a hand edit changes it. A mapping key
given twice is refused; a bare YAML number used as an alias is taken as text.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
import yaml

from vendorkb.taxonomy.cluster import normalize

FAMILIES = ("coverage", "protection", "schedule", "retention", "storage", "recovery",
            "security", "cyber", "operations", "governance", "data_use", "vendor")
REQUIRED_DIMENSIONS = ("coverage.backup", "coverage.restore", "coverage.export",
                       "coverage.compare")
FORMAT_VERSION = 2
VERSION_KEY = "taxonomy_version"

_SNAKE = r"[a-z][a-z0-9_]*"
SNAKE_RE = re.compile(rf"^{_SNAKE}$")
DOTTED_RE = re.compile(rf"^{_SNAKE}(\.{_SNAKE})*$")
PAIR_RE = re.compile(rf"^{_SNAKE}\.{_SNAKE}$")  # dimension and offering ids

# (alias, entity_type, scope, entity_id): see the aliases table in schema.sql.
AliasRow = tuple[str, str, str, str]


class TaxonomyError(ValueError):
    """The taxonomy is invalid; ``problems`` lists every problem found."""

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"taxonomy has {len(self.problems)} problem(s):\n"
                         + "\n".join(f"- {p}" for p in self.problems))


@dataclass(frozen=True)
class Named:
    """An object type, variant or edition."""

    id: str
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Dimension:
    id: str
    family: str
    name: str
    definition: str
    aliases: tuple[str, ...]
    applies_to: tuple[str, ...] | None  # None: all classes
    value_unit: str | None
    platform: bool
    lookup_aliases: tuple[str, ...] = ()  # exact lookup only: never a candidate-search key


@dataclass(frozen=True)
class WorkloadClass:
    id: str
    parent: str | None
    name: str
    definition: str
    object_types: tuple[Named, ...]


@dataclass(frozen=True)
class Target:
    id: str
    parent: str | None
    name: str
    class_id: str
    aliases: tuple[str, ...]
    object_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    variants: tuple[Named, ...] = ()
    lookup_aliases: tuple[str, ...] = ()  # exact lookup only: never a candidate-search key


@dataclass(frozen=True)
class Offering:
    id: str
    vendor: str
    name: str
    aliases: tuple[str, ...]
    editions: tuple[Named, ...]


@dataclass(frozen=True)
class Taxonomy:
    version: int
    dimensions: list[Dimension]
    classes: list[WorkloadClass]
    targets: list[Target]
    offerings: list[Offering]

    def lineage(self, class_id: str) -> list[str]:
        """``class_id`` first, then its ancestors, nearest first. KeyError if unknown."""
        return _lineage({c.id for c in self.classes}, class_id)

    def object_types(self, class_id: str) -> list[Named]:
        """The object types of ``class_id``, inherited ones included: the root ancestor's
        first, each class's in YAML order."""
        by_id = {c.id: c for c in self.classes}
        return [o for cid in reversed(self.lineage(class_id)) for o in by_id[cid].object_types]

    def alias_rows(self) -> list[AliasRow]:
        """Every ``aliases`` row, deduplicated, in a stable order."""
        rows: list[AliasRow] = []

        def add(entity_type: str, scope: str, entity_id: str, name: str | None,
                aliases: Iterable[str]) -> None:
            texts = ([normalize(name)] if name is not None else []) + list(aliases)
            rows.extend((a, entity_type, scope, entity_id) for a in texts if a)

        for d in self.dimensions:
            add("dimension", "", d.id, d.name, (*d.aliases, *d.lookup_aliases))
        for c in self.classes:
            for o in c.object_types:
                add("object_type", c.id, o.id, o.name, o.aliases)
        for t in self.targets:
            add("target", "", t.id, t.name, (*t.aliases, *t.lookup_aliases))
            for ot, aliases in t.object_aliases.items():
                add("target_object_type", t.id, ot, None, aliases)
            for v in t.variants:
                add("variant", t.id, v.id, v.name, v.aliases)
        for f in self.offerings:
            add("offering", f.vendor, f.id, f.name, f.aliases)
            for e in f.editions:
                add("edition", f.id, e.id, e.name, e.aliases)
        return list(dict.fromkeys(rows))

    def lookup_only(self) -> set[tuple[str, str, str]]:
        """The (alias, entity_type, scope) of every ``lookup_aliases`` entry: matched exactly
        like any alias, but never a candidate-search key, so adding one does not change the
        candidates (and the stored decisions) of any other phrase."""
        return ({(a, "dimension", "") for d in self.dimensions for a in d.lookup_aliases}
                | {(a, "target", "") for t in self.targets for a in t.lookup_aliases})


def _parent(dotted_id: str) -> str | None:
    return dotted_id.rpartition(".")[0] or None


def _lineage(known: set[str], class_id: str) -> list[str]:
    if class_id not in known:
        raise KeyError(class_id)
    chain = [class_id]
    node = _parent(class_id)
    while node is not None:
        if node in known:  # a missing ancestor is a validation problem, skipped here
            chain.append(node)
        node = _parent(node)
    return chain


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses a mapping with the same key twice (a careless edit)."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            raise TaxonomyError(
                [f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"])
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                                 _unique_mapping)


def parse_doc(data: bytes, where: str | Path = "taxonomy") -> dict[str, Any]:
    try:
        doc = yaml.load(data.decode("utf-8"), Loader=_UniqueKeyLoader)  # a SafeLoader subclass
    except yaml.YAMLError:
        # the path only, and no chained cause: a YAML error quotes the offending text
        raise TaxonomyError([f"{where}: not valid YAML"]) from None
    if not isinstance(doc, dict):
        raise TaxonomyError([f"{where}: not a YAML mapping"])
    return doc


def read_doc(path: str | Path) -> dict[str, Any]:
    return parse_doc(Path(path).read_bytes(), path)


def version_string(version: int, data: bytes) -> str:
    """The taxonomy version: the YAML ``version`` plus the file's hash, so a hand edit
    without a version bump still invalidates mappings made under the old file."""
    return f"{version}-{hashlib.sha256(data).hexdigest()[:8]}"


class _Checker:
    """Collects problems while reading the document; each reader returns its best value."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def add(self, problem: str) -> None:
        if problem not in self.problems:
            self.problems.append(problem)

    def keys(self, where: str, item: dict, required: tuple[str, ...],
             optional: tuple[str, ...]) -> None:
        for key in item:
            if key not in required and key not in optional:
                self.add(f"{where}: unknown key {key!r}")
        for key in required:
            if key not in item:
                self.add(f"{where}: {key} is required")

    def text(self, where: str, item: dict, key: str, *, required: bool = True) -> str | None:
        value = item.get(key)
        if value is None and not required:
            return None
        if not isinstance(value, str) or not value.strip():
            if key in item or not required:
                self.add(f"{where}: {key} must be non-empty text, got {value!r}")
            return None
        return value.strip()

    def flag(self, where: str, item: dict, key: str) -> bool:
        """An optional boolean key (default false)."""
        value = item.get(key, False)
        if not isinstance(value, bool):
            self.add(f"{where}: {key} must be true or false, got {value!r}")
            return False
        return value

    def lookup_aliases(self, where: str, item: dict, name: str | None,
                       aliases: tuple[str, ...]) -> tuple[str, ...]:
        """An entry's ``lookup_aliases``; one also among its name and aliases is a problem."""
        lookup = self.aliases(where, item.get("lookup_aliases"))
        searched = {*aliases, *([normalize(name)] if name else [])}
        for a in lookup:
            if a in searched:
                self.add(f"{where}: {a!r} is in both aliases and lookup_aliases")
        return tuple(a for a in lookup if a not in searched)

    def aliases(self, where: str, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            self.add(f"{where}: aliases must be a list of text")
            return ()
        texts: list[str] = []
        for a in value:
            if isinstance(a, bool) or not isinstance(a, str | int | float):
                self.add(f"{where}: alias {a!r} is not text (quote it)")
            else:
                texts.append(str(a))  # a bare YAML number such as 365 is text here
        return tuple(dict.fromkeys(a for a in map(normalize, texts) if a))

    def entries(self, where: str, value: Any) -> list[tuple[str, dict]]:
        """The mappings of a list, each with its location; problems for the rest."""
        if value is None:
            value = []
        if not isinstance(value, list):
            self.add(f"{where}: must be a list")
            return []
        out = []
        for n, item in enumerate(value):
            if isinstance(item, dict):
                out.append((f"{where}[{n}]", item))
            else:
                self.add(f"{where}[{n}]: must be a mapping")
        return out

    def ident(self, where: str, item: dict, pattern: re.Pattern[str], what: str,
              seen: set[str]) -> str | None:
        eid = item.get("id")
        if not isinstance(eid, str) or not pattern.match(eid):
            self.add(f"{where}: invalid {what} id {eid!r}")
            return None
        if eid in seen:
            self.add(f"{where}: duplicate {what} id {eid!r}")
            return None
        seen.add(eid)
        return eid

    def named_list(self, where: str, value: Any, what: str) -> tuple[Named, ...]:
        out: list[Named] = []
        seen: set[str] = set()
        for loc, item in self.entries(where, value):
            self.keys(loc, item, ("id", "name"), ("aliases", "proposed"))
            eid = self.ident(loc, item, SNAKE_RE, what, seen)
            label = f"{loc} ({eid})" if eid else loc
            self.flag(label, item, "proposed")
            name = self.text(label, item, "name")
            aliases = self.aliases(label, item.get("aliases"))
            if eid is not None and name is not None:
                out.append(Named(eid, name, aliases))
        return tuple(out)


def _dimensions(ck: _Checker, raw: Any) -> list[tuple[Dimension, Any]]:
    out: list[tuple[Dimension, Any]] = []
    seen: set[str] = set()
    for where, item in ck.entries("dimensions", raw):
        ck.keys(where, item, ("id", "name", "definition", "applies_to"),
                ("aliases", "lookup_aliases", "value_unit", "platform", "proposed"))
        did = ck.ident(where, item, PAIR_RE, "dimension", seen)
        label = f"{where} ({did})" if did else where
        family = did.partition(".")[0] if did else None
        if family is not None and family not in FAMILIES:
            ck.add(f"{label}: unknown family {family!r} (one of {', '.join(FAMILIES)})")
        name = ck.text(label, item, "name")
        definition = ck.text(label, item, "definition")
        aliases = ck.aliases(label, item.get("aliases"))
        lookup = ck.lookup_aliases(label, item, name, aliases)
        applies = item.get("applies_to")
        if not (applies == "all" or (isinstance(applies, list) and applies
                                     and all(isinstance(a, str) for a in applies))):
            if "applies_to" in item:
                ck.add(f"{label}: applies_to must be 'all' or a non-empty list of class ids,"
                       f" got {applies!r}")
            applies = None
        value_unit = ck.text(label, item, "value_unit", required=False)
        platform = ck.flag(label, item, "platform")
        ck.flag(label, item, "proposed")
        if did is not None and family in FAMILIES and name and definition:
            applies_to = None if applies in (None, "all") else tuple(dict.fromkeys(applies))
            out.append((Dimension(did, family, name, definition, aliases, applies_to,
                                  value_unit, platform, lookup), label))
    return out


def _classes(ck: _Checker, raw: Any) -> list[WorkloadClass]:
    out: list[WorkloadClass] = []
    seen: set[str] = set()
    for where, item in ck.entries("classes", raw):
        ck.keys(where, item, ("id", "name", "definition"), ("object_types",))
        cid = ck.ident(where, item, DOTTED_RE, "class", seen)
        label = f"{where} ({cid})" if cid else where
        name = ck.text(label, item, "name")
        definition = ck.text(label, item, "definition")
        object_types = ck.named_list(f"{label} object_types", item.get("object_types"),
                                     "object type")
        if cid is not None and name and definition:
            out.append(WorkloadClass(cid, _parent(cid), name, definition, object_types))
    return out


def _targets(ck: _Checker, raw: Any) -> list[tuple[Target, str]]:
    out: list[tuple[Target, str]] = []
    seen: set[str] = set()
    for where, item in ck.entries("targets", raw):
        ck.keys(where, item, ("id", "name", "class"),
                ("aliases", "lookup_aliases", "object_aliases", "variants", "proposed"))
        tid = ck.ident(where, item, DOTTED_RE, "target", seen)
        label = f"{where} ({tid})" if tid else where
        ck.flag(label, item, "proposed")
        name = ck.text(label, item, "name")
        class_id = ck.text(label, item, "class")
        aliases = ck.aliases(label, item.get("aliases"))
        lookup = ck.lookup_aliases(label, item, name, aliases)
        raw_oa = item.get("object_aliases") or {}
        object_aliases: dict[str, tuple[str, ...]] = {}
        if not isinstance(raw_oa, dict):
            ck.add(f"{label}: object_aliases must map object type ids to alias lists")
        else:
            for ot, texts in raw_oa.items():
                object_aliases[str(ot)] = ck.aliases(f"{label} object_aliases.{ot}", texts)
        variants = ck.named_list(f"{label} variants", item.get("variants"), "variant")
        if tid is not None and name and class_id:
            out.append((Target(tid, _parent(tid), name, class_id, aliases, object_aliases,
                               variants, lookup), label))
    return out


def _offerings(ck: _Checker, raw: Any) -> list[Offering]:
    out: list[Offering] = []
    seen: set[str] = set()
    for where, item in ck.entries("offerings", raw):
        ck.keys(where, item, ("vendor", "id", "name"), ("aliases", "editions", "proposed"))
        oid = ck.ident(where, item, PAIR_RE, "offering", seen)
        label = f"{where} ({oid})" if oid else where
        ck.flag(label, item, "proposed")
        vendor = ck.text(label, item, "vendor")
        name = ck.text(label, item, "name")
        aliases = ck.aliases(label, item.get("aliases"))
        editions = ck.named_list(f"{label} editions", item.get("editions"), "edition")
        if oid is not None and vendor and name:
            out.append(Offering(oid, vendor, name, aliases, editions))
    return out


def _check_alias_owners(ck: _Checker, rows: list[AliasRow]) -> None:
    owners: dict[tuple[str, str, str], list[str]] = {}
    for alias, etype, scope, eid in rows:
        ids = owners.setdefault((alias, etype, scope), [])
        if eid not in ids:
            ids.append(eid)
    for (alias, etype, scope), ids in owners.items():
        if len(ids) > 1:
            where = f" in {scope}" if scope else ""
            ck.add(f"{etype} alias {alias!r}{where} is on {' and '.join(ids)}")


def validate(doc: dict[str, Any]) -> Taxonomy:
    """The taxonomy in ``doc``, or a ``TaxonomyError`` listing every problem."""
    ck = _Checker()
    for key in doc:
        if key not in ("version", "dimensions", "classes", "targets", "offerings"):
            ck.add(f"unknown top-level key {key!r}")
    version = doc.get("version")
    if version != FORMAT_VERSION or isinstance(version, bool):
        ck.add(f"version must be {FORMAT_VERSION}, got {version!r}")
    for key in ("dimensions", "classes", "targets"):
        if key not in doc:
            ck.add(f"{key}: missing")
    dimensions = _dimensions(ck, doc.get("dimensions"))
    classes = _classes(ck, doc.get("classes"))
    targets = _targets(ck, doc.get("targets"))
    offerings = _offerings(ck, doc.get("offerings"))

    class_by_id = {c.id: c for c in classes}
    known = set(class_by_id)
    dimension_ids = {d.id for d, _ in dimensions}
    for required in REQUIRED_DIMENSIONS:
        if required not in dimension_ids:
            ck.add(f"dimensions: missing required dimension {required}")
    for d, label in dimensions:
        for cid in d.applies_to or ():
            if cid not in known:
                ck.add(f"{label}: applies_to names unknown class {cid!r}")

    # Classes: every dotted prefix exists; no object type is redefined; one owner per alias
    # across the lineage (a lookup walks the lineage, so a clash there is ambiguous).
    lineage_aliases: dict[str, dict[str, str]] = {}
    for c in classes:
        node = c.parent
        while node is not None:
            if node not in known:
                ck.add(f"class {c.id}: ancestor class {node!r} is missing")
            node = _parent(node)
        chain = _lineage(known, c.id)
        for o in c.object_types:
            for ancestor in chain[1:]:
                if any(p.id == o.id for p in class_by_id[ancestor].object_types):
                    ck.add(f"class {c.id} redefines object type {o.id!r} of {ancestor}")
        owner: dict[str, str] = {}
        for cid in reversed(chain):
            for o in class_by_id[cid].object_types:
                for a in (normalize(o.name), *o.aliases):
                    if owner.setdefault(a, o.id) != o.id:
                        ck.add(f"class {c.id}: object type alias {a!r} is on both"
                               f" {owner[a]} and {o.id} in its lineage")
        lineage_aliases[c.id] = owner

    # Targets: every dotted prefix exists, the class exists, object_aliases name lineage
    # object types and agree with the class's own aliases.
    target_ids = {t.id for t, _ in targets}
    for t, label in targets:
        node = t.parent
        while node is not None:
            if node not in target_ids:
                ck.add(f"{label}: parent target {node!r} is missing")
            node = _parent(node)
        if t.class_id not in known:
            ck.add(f"{label}: unknown class {t.class_id!r}")
            continue
        lineage_types = {o.id for cid in _lineage(known, t.class_id)
                         for o in class_by_id[cid].object_types}
        class_owner = lineage_aliases[t.class_id]
        for ot, aliases in t.object_aliases.items():
            if ot not in lineage_types:
                ck.add(f"{label}: object_aliases key {ot!r} is not an object type of"
                       f" {t.class_id} or its ancestors")
                continue
            for a in aliases:
                if class_owner.get(a, ot) != ot:
                    ck.add(f"{label}: object alias {a!r} for {ot} is already the class's alias"
                           f" for {class_owner[a]}")

    taxonomy = Taxonomy(FORMAT_VERSION, [d for d, _ in dimensions], classes,
                        [t for t, _ in targets], offerings)
    _check_alias_owners(ck, taxonomy.alias_rows())
    if ck.problems:
        raise TaxonomyError(ck.problems)
    return taxonomy


async def current_version(conn: psycopg.AsyncConnection) -> str | None:
    cur = await conn.execute("SELECT value FROM taxonomy_meta WHERE key=%s", (VERSION_KEY,))
    row = await cur.fetchone()
    return None if row is None else str(row[0])


_TABLES = ("aliases", "editions", "offerings", "variants", "targets", "object_types",
           "classes", "dimensions")  # children first


async def load_taxonomy(conn: psycopg.AsyncConnection, path: str | Path) -> dict:
    """Validate the YAML at ``path`` and replace every taxonomy row with it, in one
    transaction; returns the version and each table's row count."""
    data = Path(path).read_bytes()
    tax = validate(parse_doc(data, path))
    version = version_string(tax.version, data)
    classes = sorted(tax.classes, key=lambda c: c.id.count("."))  # parents first
    targets = sorted(tax.targets, key=lambda t: t.id.count("."))
    lookup_only = tax.lookup_only()
    rows: dict[str, list[tuple]] = {
        "dimensions": [(d.id, d.family, d.name, d.definition,
                        None if d.applies_to is None else list(d.applies_to), d.value_unit,
                        d.platform) for d in tax.dimensions],
        "classes": [(c.id, c.parent, c.name, c.definition) for c in classes],
        "object_types": [(c.id, o.id, o.name, n) for c in tax.classes
                         for n, o in enumerate(c.object_types)],
        "targets": [(t.id, t.parent, t.class_id, t.name) for t in targets],
        "variants": [(t.id, v.id, v.name) for t in tax.targets for v in t.variants],
        "offerings": [(f.id, f.vendor, f.name) for f in tax.offerings],
        "editions": [(f.id, e.id, e.name) for f in tax.offerings for e in f.editions],
        "aliases": [(*r, r[:3] not in lookup_only) for r in tax.alias_rows()],
    }
    columns = {
        "dimensions": "dimension_id, family, name, definition, applies_to, value_unit, platform",
        "classes": "class_id, parent_id, name, definition",
        "object_types": "class_id, object_type_id, name, ordinal",
        "targets": "target_id, parent_id, class_id, name",
        "variants": "target_id, variant_id, name",
        "offerings": "offering_id, vendor, name",
        "editions": "offering_id, edition_id, name",
        "aliases": "alias, entity_type, scope, entity_id, search",
    }
    async with conn.transaction():
        for table in _TABLES:
            await conn.execute(f"DELETE FROM {table}")
        async with conn.cursor() as cur:
            for table in reversed(_TABLES):  # parents first
                if rows[table]:
                    marks = ",".join(["%s"] * len(rows[table][0]))
                    await cur.executemany(
                        f"INSERT INTO {table} ({columns[table]}) VALUES ({marks})", rows[table])
        await conn.execute(
            "INSERT INTO taxonomy_meta (key, value) VALUES (%s,%s) ON CONFLICT (key)"
            " DO UPDATE SET value=EXCLUDED.value, updated_at=now()",
            (VERSION_KEY, version))
    return {"version": version, **{t: len(rows[t]) for t in reversed(_TABLES)}}
