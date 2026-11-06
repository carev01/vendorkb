"""Grow the taxonomy v2 aliases bottom-up from the v3 claims (spec §5 step 3).

``propose`` reads the claims of the claims-v3 prompt family (``prompt_version LIKE
'claims-v3%'``) and clusters the distinct normalized strings of one section:

- ``dimension``: ``capability_raw``;
- ``target``: ``target_raw``;
- ``object_type``: ``target_scope``, per workload class. The class is the class of the target
  the claim's ``target_raw`` names (a target name or alias); a claim whose target names none is
  skipped and counted in ``unresolved``;
- ``offering``: ``offering`` and ``edition``, per vendor (matched to the YAML's ``vendor``
  case-insensitively). Editions are listed as ``<offering id>.<edition id>``.

Only strings with >= 2 claims or >= 2 vendors are embedded (input_type document) and
clustered (agglomerative, cosine, average linkage), at most ``MAX_CLUSTER_STRINGS`` of them
over all groups (the top by claim count, then vendor count): clustering memory is
quadratic. The rest go to the long tail file next to ``out_path`` (``long_tail_path``),
under the kind, replaced on every run of that kind. A cluster whose every string is already
a name or alias in its group is not sent; ``limit`` caps the clusters sent (biggest first).

For each cluster the extraction model (``extract_backend`` / ``extract_model``, strict JSON
schema) gets the strings and the ``TOP_CANDIDATES`` existing entries of the group nearest the
cluster's centroid by embedding (id, name, definition), and answers
``{"action": "alias" | "new" | "none", "id": ..., "new": ...}``. The model decides; the
nearest entry alone never maps a string. ``new`` carries its section's fields (``family``
for a dimension, ``class`` for a target, ``offering`` for an edition, ...).

The answers are merged into the YAML at ``out_path`` in place, keeping its comments, layout
and order for a PR review (a ruamel round trip):

- ``alias`` appends the cluster's strings to that entry's aliases; a string another entry of
  the same scope already owns stays with it.
- ``new`` appends an entry at the end of its section (a class's object types, an offering's
  editions) with ``proposed: true`` and the strings as its aliases. A new id that already
  exists merges into that entry instead (``merged_into_existing``).
- After each change the whole document is validated (``load.validate``); a change that would
  break it is undone and counted in ``invalid``. The file must exist and be valid before any
  paid work, and the file written is validated again as the loader reads it.

The budget guard stops dispatching before a call whose estimate (prompt chars / 3 in, the
full ``MAX_TOKENS`` out, at ``extract.price_per_mtok`` for the model), added to the spend so
far and the estimates in flight, would exceed ``max_usd``. Spend is the provider's
``usage.cost`` when reported, else the token price. Clusters not sent are counted in
``unproposed``; the file still gets what was proposed. A 400 stops dispatching and is raised
after the file is written; any other failure costs only its cluster (counted in ``invalid``).
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import anthropic
import numpy as np
import openai
import psycopg
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import SingleQuotedScalarString
from ruamel.yaml.util import load_yaml_guess_indent

from vendorkb.claims.extract import _reported_cost, cost_usd, price_per_mtok
from vendorkb.config import VendorkbSettings
from vendorkb.taxonomy.cluster import cluster_strings, normalize
from vendorkb.taxonomy.load import FAMILIES, Taxonomy, TaxonomyError, parse_doc, validate

log = logging.getLogger(__name__)

__all__ = ["KINDS", "ProposeReport", "long_tail_path", "propose"]

Kind = Literal["dimension", "target", "object_type", "offering"]
KINDS: tuple[Kind, ...] = ("dimension", "target", "object_type", "offering")
Section = Literal["dimension", "target", "object_type", "offering", "edition"]
PROMPT_FAMILY = "claims-v3%"  # claims-v3, claims-v3.1, ...
# OpenRouter turns reasoning effort into a thinking budget (>= 1024 tokens on Anthropic
# models), so the reply needs headroom above it.
MAX_TOKENS = 2000
MAX_CLUSTER_STRINGS = 15_000  # ~1.8 GB of distance matrices
CHARS_PER_TOKEN = 3
MAX_MEMBERS_IN_PROMPT = 40
MAX_VENDORS_IN_PROMPT = 10
TOP_CANDIDATES = 5

_TEXT: dict[str, Any] = {"type": "string"}
_NULLABLE_TEXT: dict[str, Any] = {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


# The fields of a "new" entry, per section.
NEW_FIELDS: dict[Section, dict[str, Any]] = {
    "dimension": {"family": {"type": "string", "enum": list(FAMILIES)}, "id": _TEXT,
                  "name": _TEXT, "definition": _TEXT,
                  "applies_to": {"type": "array", "items": _TEXT},
                  "value_unit": _NULLABLE_TEXT},
    "target": {"id": _TEXT, "name": _TEXT, "class": _TEXT},
    "object_type": {"id": _TEXT, "name": _TEXT},
    "offering": {"id": _TEXT, "name": _TEXT},
    "edition": {"offering": _TEXT, "id": _TEXT, "name": _TEXT},
}


def reply_schema(section: Section) -> dict[str, Any]:
    return _object({
        "action": {"type": "string", "enum": ["alias", "new", "none"]},
        "id": _NULLABLE_TEXT,
        "new": {"anyOf": [_object(NEW_FIELDS[section]), {"type": "null"}]},
    })


_RULES = """\
You get a cluster of phrasings extracted from several vendors' backup product documentation, \
and the existing taxonomy entries closest to it. Decide what the cluster names:
- "alias": the phrasings name one listed entry: the same thing, not merely a related one. \
Set "id" to that entry's id and "new" to null.
- "new": the phrasings name one thing that belongs in this section and that no listed entry \
covers. Fill "new" and set "id" to null.
- "none": the phrasings mix different things, are too vague to name one thing, are a \
vendor's brand name for its own feature, or do not belong in this section. Set "id" and \
"new" to null.
Choose an entry because it means the same thing, never only because it is listed first. \
Prefer "alias" when an entry fits; propose "new" only when none does. Names and definitions \
are vendor-neutral and in your own words."""


def _system(section: Section, tax: Taxonomy) -> str:
    classes = ", ".join(c.id for c in tax.classes)
    intro = {
        "dimension": f"""\
You maintain the dimensions of a vendor-neutral taxonomy of backup products. A dimension is \
one comparable aspect of a product, such as its maximum retention period or whether it can \
restore to another tenant.
For "new":
- family: one of {", ".join(FAMILIES)}.
- id: snake_case, the part after the family ("maximum_period" for \
"retention.maximum_period").
- name: short. definition: one sentence.
- applies_to: the workload classes it is limited to, from: {classes}; [] for all classes.
- value_unit: the unit when the dimension is a quantity, else null.""",
        "target": f"""\
You maintain the targets of a vendor-neutral taxonomy of backup products. A target is a \
concrete workload a backup product protects or stores to: an application, a cloud service, \
a hypervisor, a database or a storage system.
For "new":
- id: dotted, lowercase snake_case segments, general to specific; every shorter prefix must \
be an existing target id (a service of "aws" is "aws.<service>").
- name: the workload's name.
- class: one of {classes}.""",
        "object_type": """\
You maintain the object types of the workload classes of a vendor-neutral taxonomy of backup \
products. An object type is a kind of item inside a workload that a backup product backs up \
or restores, such as a mailbox, a user or a database. The message names the class.
For "new":
- id: snake_case, singular.
- name: short, singular.""",
        "offering": """\
You maintain the offerings of each vendor in a taxonomy of backup products. An offering is \
a product or service a vendor sells. The message names the vendor.
For "new":
- id: snake_case, the offering only (the vendor prefix is added).
- name: the offering's name.""",
        "edition": """\
You maintain the editions of each vendor's offerings in a taxonomy of backup products. An \
edition is a tier or license level of one offering. The message names the vendor and lists \
its offerings; listed editions have ids "<offering id>.<edition id>".
For "new":
- offering: the id of the listed offering it belongs to.
- id: snake_case, the edition only.
- name: the edition's name.""",
    }[section]
    return intro + "\n\n" + _RULES


class _Embedder(Protocol):
    async def embed(self, texts: list[str], *, input_type: Literal["document", "query"] = ...
                    ) -> list[list[float]]: ...


@dataclass
class _Stat:
    text: str
    count: int = 0
    vendors: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Entry:
    id: str  # the id the model answers with (editions: "<offering id>.<edition id>")
    name: str
    definition: str
    aliases: tuple[str, ...]

    @property
    def embed_text(self) -> str:
        return f"{self.name}: {self.definition}" if self.definition else self.name


@dataclass
class _Group:
    """The strings of one section within one scope, and the entries they may alias."""

    section: Section
    scope: str  # "" | class id | normalized vendor
    label: str
    entries: list[_Entry]
    covered: set[str]
    vendor: str = ""  # offering / edition: the vendor as the YAML (or else a claim) writes it
    offerings: list[tuple[str, str]] = field(default_factory=list)  # edition: (id, name)
    stats: dict[str, _Stat] = field(default_factory=dict)


@dataclass
class _Cluster:
    group: _Group
    members: list[_Stat]
    candidates: list[_Entry] = field(default_factory=list)


@dataclass
class ProposeReport:
    kind: str = ""
    out_path: str = ""
    long_tail_path: str = ""
    strings: int = 0
    unresolved: int = 0             # object_type: claims whose target names no target
    clustered: int = 0              # strings embedded and clustered
    long_tail_prefiltered: int = 0  # 1 claim, 1 vendor: never embedded
    long_tail_capped: int = 0       # clusterable but past MAX_CLUSTER_STRINGS
    clusters: int = 0
    eligible: int = 0
    already_covered: int = 0
    skipped_by_limit: int = 0
    proposed: int = 0              # aliased + added
    aliased: int = 0
    added: int = 0                 # new entries written (proposed: true)
    merged_into_existing: int = 0  # "new" whose id already existed: merged as an alias
    none: int = 0
    unproposed: int = 0  # eligible but not sent (budget stop or a 400)
    invalid: int = 0     # replies that failed validation or the call
    long_tail: int = 0   # strings in the long tail file under this kind
    spent_usd: float = 0.0
    stopped_by_budget: bool = False
    errors: list[str] = field(default_factory=list)


def long_tail_path(out_path: str | Path) -> Path:
    """``taxonomy/taxonomy.yaml`` -> ``taxonomy/long_tail.yaml``; ``x/t.yaml`` ->
    ``x/t.long_tail.yaml``."""
    out = Path(out_path)
    return out.with_name("long_tail.yaml" if out.stem == "taxonomy"
                         else f"{out.stem}.long_tail.yaml")


def _snake(raw: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if not out:
        raise ValueError(f"cannot make an id from {raw!r}")
    return out


def _dotted(raw: str) -> str:
    segments = [re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") for s in raw.split(".")]
    return ".".join(_snake(s) for s in segments if s)


# ---- groups ---------------------------------------------------------------------------------

class _Groups:
    """Builds each group on first use from the validated taxonomy."""

    def __init__(self, tax: Taxonomy) -> None:
        self.tax = tax
        self.groups: dict[tuple[Section, str], _Group] = {}
        self.target_class: dict[str, str] = {}
        for t in tax.targets:
            for a in (normalize(t.name), *t.aliases):
                self.target_class.setdefault(a, t.class_id)

    def get(self, section: Section, scope: str, raw_vendor: str = "") -> _Group:
        key = (section, scope)
        group = self.groups.get(key)
        if group is None:
            group = self.groups[key] = self._build(section, scope, raw_vendor)
        return group

    def _build(self, section: Section, scope: str, raw_vendor: str) -> _Group:
        tax = self.tax
        extra: set[str] = set()
        vendor, offerings_list = "", []
        if section == "dimension":
            entries = [_Entry(d.id, d.name, d.definition, d.aliases) for d in tax.dimensions]
            label = "dimensions"
        elif section == "target":
            entries = [_Entry(t.id, t.name, f"a workload of class {t.class_id}", t.aliases)
                       for t in tax.targets]
            label = "targets"
        elif section == "object_type":
            entries = [_Entry(o.id, o.name, "", o.aliases) for o in tax.object_types(scope)]
            cls = next(c for c in tax.classes if c.id == scope)
            label = f"object types of class {scope} ({cls.name})"
            for t in tax.targets:  # a target's own words for an object type are covered too
                if t.class_id == scope:
                    extra.update(a for texts in t.object_aliases.values() for a in texts)
        else:
            offerings = [f for f in tax.offerings if normalize(f.vendor) == scope]
            vendor = offerings[0].vendor if offerings else raw_vendor
            offerings_list = [(f.id, f.name) for f in offerings]
            if section == "offering":
                entries = [_Entry(f.id, f.name, f"an offering of {f.vendor}", f.aliases)
                           for f in offerings]
                label = f"offerings of vendor {vendor}"
            else:
                entries = [_Entry(f"{f.id}.{e.id}", e.name, f"an edition of {f.name}",
                                  e.aliases) for f in offerings for e in f.editions]
                label = f"editions of the offerings of vendor {vendor}"
        covered = {a for e in entries for a in (normalize(e.name), *e.aliases)} | extra
        return _Group(section, scope, label, entries, covered, vendor, offerings_list)


_COLUMNS: dict[Kind, str] = {"dimension": "capability_raw", "target": "target_raw",
                             "object_type": "target_raw, target_scope",
                             "offering": "offering, edition"}


async def _collect(conn: psycopg.AsyncConnection, kind: Kind, groups: _Groups,
                   report: ProposeReport) -> None:
    cur = await conn.execute(
        f"SELECT {_COLUMNS[kind]}, vendor FROM raw_claims WHERE prompt_version LIKE %s",
        (PROMPT_FAMILY,))

    def add(group: _Group, text: str | None, vendor: str) -> None:
        key = normalize(text or "")
        if key:
            st = group.stats.setdefault(key, _Stat(key))
            st.count += 1
            st.vendors.add(vendor)

    for row in await cur.fetchall():
        vendor = row[-1]
        if kind in ("dimension", "target"):
            add(groups.get(kind, ""), row[0], vendor)
        elif kind == "object_type":
            if not normalize(row[1] or ""):
                continue
            cls = groups.target_class.get(normalize(row[0] or ""))
            if cls is None:
                report.unresolved += 1
                continue
            add(groups.get("object_type", cls), row[1], vendor)
        else:
            scope = normalize(vendor)
            if normalize(row[0] or ""):
                add(groups.get("offering", scope, vendor), row[0], vendor)
            if normalize(row[1] or ""):
                add(groups.get("edition", scope, vendor), row[1], vendor)


# ---- the model call -------------------------------------------------------------------------

def _user_message(c: _Cluster) -> str:
    members = c.members
    vendors = sorted({v for m in members for v in m.vendors})
    shown = sorted(members, key=lambda m: (-m.count, m.text))[:MAX_MEMBERS_IN_PROMPT]
    lines = [f"- {m.text} ({m.count})" for m in shown]
    if len(members) > len(shown):
        lines.append(f"- ... and {len(members) - len(shown)} more")
    text = (f"Section: {c.group.label}\nVendors ({len(vendors)}): "
            f"{', '.join(vendors[:MAX_VENDORS_IN_PROMPT])}"
            f"{', ...' if len(vendors) > MAX_VENDORS_IN_PROMPT else ''}\n\n"
            "Phrasings (claim count):\n" + "\n".join(lines) + "\n\n"
            "Closest existing entries (id | name | definition):\n"
            + ("\n".join(f"- {e.id} | {e.name} | {e.definition}" for e in c.candidates)
               or "(none)"))
    if c.group.section == "edition":
        text += (f"\n\nOfferings of vendor {c.group.vendor} (id | name):\n"
                 + ("\n".join(f"- {i} | {n}" for i, n in c.group.offerings) or "(none)"))
    return text


def _request(system: str, schema: dict[str, Any], user: str, settings: VendorkbSettings
             ) -> dict[str, Any]:
    if settings.extract_backend == "openai":
        return {
            "model": settings.extract_model, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "taxonomy_proposal", "strict": True, "schema": schema}},
            "extra_body": {"usage": {"include": True}, "reasoning": {"effort": "low"}},
        }
    return {
        "model": settings.extract_model, "max_tokens": MAX_TOKENS, "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": schema}},
    }


async def _call(client: Any, request: dict[str, Any], settings: VendorkbSettings,
                price: tuple[float, float]) -> tuple[str | None, str | None, float]:
    """(reply text, error, cost) for one proposal call."""
    if settings.extract_backend == "openai":
        resp = await client.chat.completions.create(**request)
        usage = getattr(resp, "usage", None)
        cost = _reported_cost(usage)
        if cost is None:
            cost = cost_usd(int(getattr(usage, "prompt_tokens", 0) or 0),
                            int(getattr(usage, "completion_tokens", 0) or 0), batch=False,
                            price=price)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return None, "no-choices", cost
        if choices[0].finish_reason == "length":
            return None, "length", cost
        if getattr(choices[0].message, "refusal", None):
            return None, "refusal", cost
        return choices[0].message.content, None, cost
    msg = await client.messages.create(**request)
    usage = getattr(msg, "usage", None)
    cost = cost_usd(int(getattr(usage, "input_tokens", 0) or 0),
                    int(getattr(usage, "output_tokens", 0) or 0), batch=False, price=price)
    if getattr(msg, "stop_reason", None) in ("refusal", "max_tokens"):
        return None, str(msg.stop_reason), cost
    text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
    return text, None if text is not None else "no-text", cost


def _parse(text: str | None, group: _Group) -> dict[str, Any]:
    """The reply ``{action, id, new}`` with its fields checked, or a ValueError."""
    if text is None:
        raise ValueError("no reply text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid-json: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("invalid-shape")
    action = data.get("action")
    if action == "none":
        return {"action": "none"}
    if action == "alias":
        eid = data.get("id")
        if not isinstance(eid, str) or eid not in {e.id for e in group.entries}:
            raise ValueError(f"alias of unknown {group.section} id {eid!r}")
        return {"action": "alias", "id": eid}
    if action != "new":
        raise ValueError(f"action {action!r}")
    new = data.get("new")
    if not isinstance(new, dict):
        raise ValueError("new: not an object")
    out: dict[str, Any] = {}
    for key, spec in NEW_FIELDS[group.section].items():
        value: Any = new.get(key)
        if spec is _NULLABLE_TEXT:
            ok = value is None or isinstance(value, str)
        elif spec.get("type") == "array":
            ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
        else:
            ok = isinstance(value, str) and bool(value.strip())
        if not ok:
            raise ValueError(f"new.{key}: {value!r}")
        out[key] = (value.strip() or None) if isinstance(value, str) else value
    return {"action": "new", "new": out}


# ---- the YAML, edited in place --------------------------------------------------------------

def _scalar(text: str) -> str:
    """``text`` quoted when a YAML 1.1 reader (PyYAML, the loader) would not read it back as
    this same text (``yes``, ``1:20``, ``null``); ruamel itself writes YAML 1.2."""
    try:
        plain = yaml.safe_load(text)
    except yaml.YAMLError:
        plain = None
    return text if plain == text else SingleQuotedScalarString(text)


def _flow_seq(items: list[Any]) -> CommentedSeq:
    seq = CommentedSeq(items)
    seq.fa.set_flow_style()
    return seq


def _set(m: CommentedMap, *pairs: tuple[str, Any]) -> None:
    """Set keys one by one: ``CommentedMap.update`` with keywords corrupts its key
    bookkeeping (ruamel 0.19), which breaks a later ``insert``."""
    for key, value in pairs:
        m[key] = value


def _find(seq: Any, eid: str) -> Any:
    """The mapping with ``id == eid`` in ``seq``, or None."""
    return next((m for m in seq or [] if isinstance(m, dict) and m.get("id") == eid), None)


class _Doc:
    def __init__(self, text: str) -> None:
        _, indent, offset = load_yaml_guess_indent(text)
        self.yaml = YAML()
        self.yaml.width = 4096  # never re-wrap a long flow line
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=indent - offset if offset else indent, sequence=indent,
                         offset=offset)
        self.data: CommentedMap = self.yaml.load(text)

    def dump(self) -> str:
        buf = io.StringIO()
        self.yaml.dump(self.data, buf)
        return buf.getvalue()

    def entry(self, group: _Group, eid: str, tax: Taxonomy) -> CommentedMap:
        """The YAML mapping of the group's entry ``eid``."""
        if group.section == "object_type":
            owner = next(cid for cid in tax.lineage(group.scope)
                         if any(o.id == eid for o in _class(tax, cid).object_types))
            found = _find(_find(self.data["classes"], owner)["object_types"], eid)
        elif group.section == "edition":
            offering, _, edition = eid.rpartition(".")
            found = _find(_find(self.data["offerings"], offering)["editions"], edition)
        else:
            found = _find(self.data[f"{group.section}s"], eid)
        assert found is not None, eid
        return found


def _class(tax: Taxonomy, cid: str) -> Any:
    return next(c for c in tax.classes if c.id == cid)


def _take_trailing(node: Any) -> tuple[Callable[[], None], Any] | None:
    """Detach the comment (a blank line, say) that ends ``node``, looking into its last
    value: (put it back, the comment token), or None."""
    if isinstance(node, CommentedMap) and node:
        key = list(node)[-1]
        slot = node.ca.items.get(key)
        if slot and len(slot) > 2 and slot[2] is not None:
            token, slot[2] = slot[2], None
            return (lambda: slot.__setitem__(2, token)), token
        return _take_trailing(node[key])
    if isinstance(node, CommentedSeq) and node:
        idx = len(node) - 1
        slot = node.ca.items.get(idx)
        if slot and slot[0] is not None:
            token, slot[0] = slot[0], None
            return (lambda: slot.__setitem__(0, token)), token
        return _take_trailing(node[idx])
    return None


def _put_after(seq: CommentedSeq, entry: CommentedMap, token: Any) -> Callable[[], None]:
    """Attach ``token`` after ``entry``, the last item of ``seq``; returns the undo."""
    if entry.fa.flow_style():
        idx = len(seq) - 1
        seq.ca.items[idx] = [token, None]
        return lambda: seq.ca.items.pop(idx, None)
    key = list(entry)[-1]
    entry.ca.items[key] = [None, None, token, None]
    return lambda: entry.ca.items.pop(key, None)


def _add_aliases(entry: CommentedMap, texts: list[str]) -> Callable[[], None]:
    """Append ``texts`` not already there to the entry's aliases (the key added after the
    name / class / definition when missing); returns the undo."""
    seq = entry.get("aliases")
    created = not isinstance(seq, list)
    taken = None
    if created:
        keys = list(entry)
        after = next((k for k in ("definition", "class", "name") if k in keys), keys[-1])
        if after == keys[-1]:  # the new key ends the entry: it takes the entry's end comment
            taken = _take_trailing(entry)
        seq = _flow_seq([])
        entry.insert(keys.index(after) + 1, "aliases", seq)
        if taken is not None:
            entry.ca.items["aliases"] = [None, None, taken[1], None]
    start = len(seq)
    have = {normalize(str(a)) for a in seq}
    seq.extend(_scalar(t) for t in dict.fromkeys(texts) if t not in have)

    def undo() -> None:
        if created:
            entry.ca.items.pop("aliases", None)
            del entry["aliases"]
            if taken is not None:
                taken[0]()
        else:
            del seq[start:]
    return undo


def _append(owner: CommentedMap, key: str, entry: CommentedMap, *, named: bool
            ) -> Callable[[], None]:
    """Append ``entry`` to the list ``owner[key]`` (created if missing) in the style of its
    siblings, moving the comment (a blank line, say) that ended the list to after it;
    returns the undo."""
    seq = owner.get(key)
    created = not isinstance(seq, list)
    if created or not seq:  # the list's end is the owner's end only when key is its last
        taken = _take_trailing(owner) if created or list(owner)[-1] == key else None
    else:
        taken = _take_trailing(seq)
    if created:
        seq = owner[key] = CommentedSeq()
    if seq:
        last = seq[-1]
        flow = isinstance(last, CommentedMap) and bool(last.fa.flow_style())
    else:
        flow = named  # object types and editions: one line each, as in the seed
        if isinstance(seq, CommentedSeq):
            seq.fa.set_block_style()
    if flow:
        entry.fa.set_flow_style()
    seq.append(entry)
    unput = _put_after(seq, entry, taken[1]) if taken is not None else None

    def undo() -> None:
        if unput is not None:
            unput()
        seq.remove(entry)
        if created:
            del owner[key]
        if taken is not None:
            taken[0]()
    return undo


def _new_entry(doc: _Doc, group: _Group, new: dict[str, Any]
               ) -> tuple[str, CommentedMap, str, CommentedMap, bool]:
    """(entry id, the mapping and key of the list it goes in, the entry without aliases,
    named) for a "new" reply."""
    s = group.section
    m = CommentedMap()
    if s == "dimension":
        family = new["family"]
        raw = new["id"]
        raw = raw[len(family) + 1:] if raw.lower().startswith(f"{family}.") else raw
        eid = f"{family}.{_snake(raw)}"
        _set(m, ("id", eid), ("name", _scalar(new["name"])),
             ("definition", _scalar(new["definition"])))
        applies = list(dict.fromkeys(new["applies_to"]))
        m["applies_to"] = _flow_seq(applies) if applies else "all"
        if new["value_unit"]:
            m["value_unit"] = _scalar(new["value_unit"])
        return eid, doc.data, "dimensions", m, False
    if s == "target":
        eid = _dotted(new["id"])
        _set(m, ("id", eid), ("name", _scalar(new["name"])), ("class", _scalar(new["class"])))
        return eid, doc.data, "targets", m, False
    if s == "object_type":
        eid = _snake(new["id"])
        _set(m, ("id", eid), ("name", _scalar(new["name"])))
        owner = _find(doc.data["classes"], group.scope)
        assert owner is not None
        return eid, owner, "object_types", m, True
    if s == "offering":
        prefixes = {i.partition(".")[0] for i, _ in group.offerings}
        prefix = prefixes.pop() if len(prefixes) == 1 else _snake(group.vendor)
        eid = f"{prefix}.{_snake(new['id'].rpartition('.')[2])}"
        _set(m, ("vendor", _scalar(group.vendor)), ("id", eid), ("name", _scalar(new["name"])))
        return eid, doc.data, "offerings", m, False
    offering_id = new["offering"]
    if offering_id not in {i for i, _ in group.offerings}:
        raise ValueError(f"new edition of unknown offering {offering_id!r}")
    offering = _find(doc.data["offerings"], offering_id)
    assert offering is not None
    eid = _snake(new["id"].rpartition(".")[2])
    _set(m, ("id", eid), ("name", _scalar(new["name"])))
    return f"{offering_id}.{eid}", offering, "editions", m, True


def _alias_scope(group: _Group, eid: str) -> str:
    """The scope an alias is unique in: the group's (a class's lineage, a vendor), and an
    offering for an edition."""
    return eid.rpartition(".")[0] if group.section == "edition" else group.scope


def _apply(doc: _Doc, tax: Taxonomy, c: _Cluster, reply: dict[str, Any],
           owned: dict[tuple[str, str], str]
           ) -> tuple[str, bool, list[str], Callable[[], None]]:
    """Apply one reply to the document: (entry id, created, aliases added, undo). A "new"
    whose id the group already has merges into that entry (created false)."""
    group = c.group
    if reply["action"] == "new":
        eid, owner, key, entry, named = _new_entry(doc, group, reply["new"])
        if eid not in {e.id for e in group.entries}:
            scope = _alias_scope(group, eid)
            texts = [m.text for m in c.members if (scope, m.text) not in owned]
            if texts:
                _add_aliases(entry, texts)  # after the name / class / definition
            if group.section == "offering":
                entry["editions"] = _flow_seq([])
            entry["proposed"] = True
            return eid, True, texts, _append(owner, key, entry, named=named)
    else:
        eid = reply["id"]
    scope = _alias_scope(group, eid)
    texts = [m.text for m in c.members if (scope, m.text) not in owned]
    return eid, False, texts, _add_aliases(doc.entry(group, eid, tax), texts)


# ---- the long tail --------------------------------------------------------------------------

def _load_tail(path: Path) -> dict[str, Any]:
    old: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        old = loaded if isinstance(loaded, dict) else {}
    return {"source": None, **{k: list(old.get(k) or []) for k in KINDS}}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _tail_row(group: _Group, s: _Stat) -> dict[str, Any]:
    row: dict[str, Any] = {"text": s.text, "count": s.count, "vendor_count": len(s.vendors)}
    if group.section == "object_type":
        row["class"] = group.scope
    elif group.section in ("offering", "edition"):
        row |= {"field": group.section, "vendor": group.vendor}
    return row


# ---- propose --------------------------------------------------------------------------------

def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


async def propose(conn: psycopg.AsyncConnection, *, client: Any, embedder: _Embedder,
                  settings: VendorkbSettings, kind: Kind, distance_threshold: float,
                  out_path: str | Path, max_usd: float = 2.0,
                  limit: int | None = None) -> ProposeReport:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, got {kind!r}")
    path = Path(out_path)
    tail_path = long_tail_path(path)
    report = ProposeReport(kind=kind, out_path=str(path), long_tail_path=str(tail_path))
    # Before any paid work: a missing or broken file, or a model with no price, is refused.
    text = path.read_text(encoding="utf-8")
    tax = validate(parse_doc(text.encode("utf-8"), path))
    doc = _Doc(text)
    price = price_per_mtok(settings.extract_model, settings)

    groups = _Groups(tax)
    await _collect(conn, kind, groups, report)
    items = [(g, s) for g in groups.groups.values() for s in g.stats.values()]
    report.strings = len(items)
    clusterable = sorted((it for it in items if it[1].count >= 2 or len(it[1].vendors) >= 2),
                         key=lambda it: (-it[1].count, -len(it[1].vendors), it[1].text))
    long_tail = [it for it in items if it[1].count < 2 and len(it[1].vendors) < 2]
    report.long_tail_prefiltered = len(long_tail)
    if len(clusterable) > MAX_CLUSTER_STRINGS:
        report.long_tail_capped = len(clusterable) - MAX_CLUSTER_STRINGS
        log.warning("taxonomy propose: %d clusterable strings; clustering the top %d, the"
                    " rest go to the long tail", len(clusterable), MAX_CLUSTER_STRINGS)
        long_tail.extend(clusterable[MAX_CLUSTER_STRINGS:])
        clusterable = clusterable[:MAX_CLUSTER_STRINGS]
    del items
    report.clustered = len(clusterable)

    texts = list(dict.fromkeys(s.text for _, s in clusterable))
    vectors = (_unit(np.asarray(await embedder.embed(texts, input_type="document"),
                                dtype=np.float32)) if texts else np.zeros((0, 0)))
    row_of = {t: i for i, t in enumerate(texts)}
    by_group: dict[tuple[Section, str], list[_Stat]] = {}
    for g, s in clusterable:
        by_group.setdefault((g.section, g.scope), []).append(s)
    todo: list[tuple[_Cluster, np.ndarray]] = []
    for key, stats in by_group.items():
        group = groups.groups[key]
        stats.sort(key=lambda s: s.text)
        member_vectors = vectors[[row_of[s.text] for s in stats]]
        for idx in cluster_strings([s.text for s in stats], member_vectors,
                                   distance_threshold=distance_threshold):
            report.clusters += 1
            members = [stats[i] for i in idx]
            if all(m.text in group.covered for m in members):
                report.already_covered += 1
            else:
                centroid = _unit(member_vectors[idx].mean(axis=0))
                todo.append((_Cluster(group, members), centroid))
    del vectors
    report.eligible = report.already_covered + len(todo)
    todo.sort(key=lambda t: (-sum(m.count for m in t[0].members), t[0].members[0].text))
    if limit is not None and len(todo) > limit:
        report.skipped_by_limit = len(todo) - limit
        todo = todo[:max(0, limit)]

    # The candidates: the group's entries nearest each cluster's centroid.
    entry_texts = list(dict.fromkeys(e.embed_text for c, _ in todo for e in c.group.entries))
    if entry_texts:
        entry_vectors = _unit(np.asarray(
            await embedder.embed(entry_texts, input_type="document"), dtype=np.float32))
        entry_row = {t: i for i, t in enumerate(entry_texts)}
        for c, centroid in todo:
            entries = c.group.entries
            if entries:
                sims = entry_vectors[[entry_row[e.embed_text] for e in entries]] @ centroid
                order = sorted(range(len(entries)), key=lambda i: (-float(sims[i]), i))
                c.candidates = [entries[i] for i in order[:TOP_CANDIDATES]]
    clusters = [c for c, _ in todo]
    del todo

    systems = {s: _system(s, tax) for s in {c.group.section for c in clusters}}
    spent, in_flight, stop = 0.0, 0.0, False
    fatal: Exception | None = None
    sem = asyncio.Semaphore(max(1, settings.extract_concurrency))

    async def one(c: _Cluster) -> dict[str, Any] | None:
        nonlocal spent, in_flight, stop, fatal
        first = c.members[0].text
        async with sem:
            user = _user_message(c)
            system = systems[c.group.section]
            estimate = cost_usd((len(system) + len(user)) // CHARS_PER_TOKEN + 1, MAX_TOKENS,
                                batch=False, price=price)
            if stop or spent + in_flight + estimate > max_usd:
                if not stop and fatal is None:
                    report.stopped_by_budget = True
                    log.warning("taxonomy propose: budget $%.2f reached (spent $%.4f, in"
                                " flight $%.4f); stopping", max_usd, spent, in_flight)
                stop = True
                report.unproposed += 1
                return None
            in_flight += estimate
            try:
                request = _request(system, reply_schema(c.group.section), user, settings)
                text, error, cost = await _call(client, request, settings, price)
            except (anthropic.BadRequestError, openai.BadRequestError) as e:
                stop, fatal = True, fatal or e
                report.invalid += 1
                report.errors.append(f"{first}: {e}")
                return None
            except Exception as e:  # API errors and any odd provider shape: this cluster only
                report.invalid += 1
                report.errors.append(f"{first}: {type(e).__name__}: {e}")
                log.warning("taxonomy propose: call failed for %r: %s", first, e)
                return None
            finally:
                in_flight -= estimate
            spent += cost
            try:
                if error is not None:
                    raise ValueError(error)
                return _parse(text, c.group)
            except ValueError as e:
                report.invalid += 1
                report.errors.append(f"{first}: {e}")
                log.warning("taxonomy propose: unusable reply for %r: %s", first, e)
                return None

    replies = await asyncio.gather(*(one(c) for c in clusters))
    report.spent_usd = spent

    # Merge in cluster order (biggest first); each change must keep the file valid.
    owned: dict[tuple[str, str], str] = {}
    for g in groups.groups.values():
        for e in g.entries:
            for a in (normalize(e.name), *e.aliases):
                owned.setdefault((_alias_scope(g, e.id), a), e.id)
    for c, reply in zip(clusters, replies):
        if reply is None:
            continue
        if reply["action"] == "none":
            report.none += 1
            continue
        first = c.members[0].text
        try:
            eid, created, added, undo = _apply(doc, tax, c, reply, owned)
        except Exception as e:  # nothing is changed before _apply's last step
            report.invalid += 1
            report.errors.append(f"{first}: {e}")
            continue
        try:
            validate(doc.data)
        except TaxonomyError as e:
            undo()
            report.invalid += 1
            report.errors.append(f"{first}: {'; '.join(e.problems)}")
            log.warning("taxonomy propose: %r would break the taxonomy: %s", first, e)
            continue
        scope = _alias_scope(c.group, eid)
        for a in added:
            owned.setdefault((scope, a), eid)
        if created:
            name = str(reply["new"]["name"])
            owned.setdefault((scope, normalize(name)), eid)
            report.added += 1
            c.group.entries.append(_Entry(eid, name, "", tuple(added)))
        else:
            report.aliased += 1
            report.merged_into_existing += int(reply["action"] == "new")
        report.proposed += 1

    out = doc.dump()
    validate(parse_doc(out.encode("utf-8"), path))  # as the loader will read it
    tail = sorted(long_tail, key=lambda it: (it[0].scope, it[0].section, -it[1].count,
                                             -len(it[1].vendors), it[1].text))
    tail_doc = _load_tail(tail_path)
    tail_doc["source"] = path.name
    tail_doc[kind] = [_tail_row(g, s) for g, s in tail]
    report.long_tail = len(tail)
    _write(path, out)
    _write(tail_path, yaml.safe_dump(tail_doc, sort_keys=False, allow_unicode=True, width=100))
    log.info("taxonomy propose: %s", report)
    if fatal is not None:
        raise fatal
    return report
