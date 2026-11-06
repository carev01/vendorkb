"""Propose taxonomy entries from the extracted claims (spec §5, DESIGN.md §5.4).

``propose`` reads the distinct normalized ``capability_raw`` (kind "feature") or
``target_raw`` (kind "target") strings of ``raw_claims`` with their claim and vendor counts,
embeds them (input_type document), clusters them (agglomerative, cosine, average linkage),
and asks the extraction model (``extract_backend`` / ``extract_model``, strict JSON schema)
for one ``{id, name, category, definition, aliases}`` per cluster. Then it merges the result
into the YAML at ``out_path`` (created when missing, ``version: 1``; it holds only
``version``, ``features`` and ``targets``):

- Only strings with >= 2 claims or >= 2 vendors are embedded and clustered, at most
  ``MAX_CLUSTER_STRINGS`` of them (the top by claim count, then vendor count): clustering
  memory is quadratic. Every other string goes to the long tail unembedded. The long tail
  is written to a separate file next to ``out_path`` (``long_tail_path``), under the kind,
  as ``{text, count, vendor_count}`` (replaced on every run of that kind); the mapper still
  reaches those strings by nearest neighbour.
- ``limit`` caps the clusters sent (biggest first), for a smoke run.
- A cluster whose every string is already an alias or name in the file is not sent again.
- Every cluster string becomes an alias of its entry (the mapper's exact match); an alias
  another entry already owns stays with that entry. A proposed id that already exists merges
  its aliases into that entry.
- Target ids are dotted (``aws.rds``); ``parent`` is the id minus its last segment, and any
  missing ancestor is added as a bare entry for the reviewer to fill in (category
  Platforms at the top level, else its child's).
- An existing file is validated first (``v1_format.validate``); a hand-broken one is refused.
- ``support_count`` / ``vendor_count`` are recomputed for every entry of the kind from the
  strings its aliases and name cover.

The budget guard stops dispatching before a call whose estimate (prompt chars / 3 in, the
full ``MAX_TOKENS`` out, at the Haiku price), added to the spend so far and the estimates in
flight, would exceed ``max_usd``. Spend is the provider's ``usage.cost`` when reported, else
the token price. Clusters not sent are counted in ``unproposed``; the file still gets what
was proposed. A 400 stops dispatching and is raised after the file is written; any other
failure costs only its cluster (counted in ``invalid``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import anthropic
import numpy as np
import openai
import psycopg
import yaml

from vendorkb.claims.extract import _reported_cost, cost_usd
from vendorkb.config import VendorkbSettings
from vendorkb.taxonomy.cluster import cluster_strings, normalize
from vendorkb.taxonomy.v1_format import (CATEGORIES, FEATURE_ID_RE, TARGET_ID_RE, read_doc,
                                          validate)

log = logging.getLogger(__name__)

__all__ = ["CATEGORIES", "ProposeReport", "long_tail_path", "parent_of", "propose", "slug_id"]

Kind = Literal["feature", "target"]
# OpenRouter turns reasoning effort into a thinking budget (>= 1024 tokens on Anthropic
# models), so the reply needs headroom above it.
MAX_TOKENS = 2000
MAX_CLUSTER_STRINGS = 15_000  # ~1.8 GB of distance matrices
CHARS_PER_TOKEN = 3
MAX_MEMBERS_IN_PROMPT = 40
MAX_VENDORS_IN_PROMPT = 10

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "definition": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id", "name", "category", "definition", "aliases"],
    "additionalProperties": False,
}

_COMMON = """\
The categories are: Workloads (what is protected: databases, SaaS apps, VMs, file systems), \
Platforms (where it runs: hypervisors, clouds, operating systems), Storage targets (where \
backups go), Data protection (how backups are made, kept and copied), Security, Recovery, \
Management (operations, monitoring, automation, APIs).
- name: a short human-readable name.
- definition: one vendor-neutral sentence.
- aliases: other common phrasings of the same thing, lowercase, without vendor or product \
names; do not repeat the listed phrasings. May be empty.
Never use a vendor's product name as the entry. Return one entry for the whole cluster."""

SYSTEM_PROMPTS: dict[Kind, str] = {
    "feature": """\
You build a vendor-neutral taxonomy of backup product capabilities. You get a cluster of \
capability phrasings extracted from several vendors' documentation, and propose the one \
taxonomy feature they name.
- id: snake_case, lowercase ASCII, general and stable, e.g. "immutable_backups", \
"instant_vm_recovery", "file_level_restore", "max_retention_period".
""" + _COMMON,
    "target": """\
You build a vendor-neutral taxonomy of what backup products protect and where they store \
backups. You get a cluster of target phrasings (workloads, platforms, storage) extracted from \
several vendors' documentation, and propose the one taxonomy target they name.
- id: dotted lowercase, general to specific, each segment snake_case: the provider or family \
first, then the service, then the variant, e.g. "aws.rds", "aws.rds.postgresql", \
"microsoft365.exchange_online", "vmware.vsphere", "s3_compatible", "tape.lto".
""" + _COMMON,
}


class _Embedder(Protocol):
    async def embed(self, texts: list[str], *, input_type: Literal["document", "query"] = ...
                    ) -> list[list[float]]: ...


@dataclass
class _Stat:
    text: str
    count: int = 0
    vendors: set[str] = field(default_factory=set)


@dataclass
class ProposeReport:
    kind: str = ""
    out_path: str = ""
    long_tail_path: str = ""
    strings: int = 0
    clustered: int = 0              # strings embedded and clustered
    long_tail_prefiltered: int = 0  # 1 claim, 1 vendor: never embedded
    long_tail_capped: int = 0       # clusterable but past MAX_CLUSTER_STRINGS
    clusters: int = 0
    eligible: int = 0
    already_covered: int = 0
    proposed: int = 0
    skipped_by_limit: int = 0
    merged_into_existing: int = 0
    unproposed: int = 0  # eligible but not sent (budget stop or a 400)
    invalid: int = 0     # replies that failed validation or the call
    long_tail: int = 0   # strings in long_tail.<kind>
    ancestors_added: int = 0
    spent_usd: float = 0.0
    stopped_by_budget: bool = False
    errors: list[str] = field(default_factory=list)


def slug_id(raw: str, kind: Kind) -> str:
    """A valid feature (snake_case) or target (dotted) id from a model's proposal."""
    def seg(s: str, prefix: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        return f"{prefix}_{s}" if s[:1].isdigit() else s
    if kind == "feature":
        out = seg(raw, "f")
        regex = FEATURE_ID_RE
    else:
        out = ".".join(p for p in (seg(s, "t") for s in raw.split(".")) if p)
        regex = TARGET_ID_RE
    if not regex.match(out):
        raise ValueError(f"cannot make a {kind} id from {raw!r}")
    return out


def parent_of(target_id: str) -> str | None:
    head, sep, _ = target_id.rpartition(".")
    return head if sep else None


async def _strings(conn: psycopg.AsyncConnection, kind: Kind) -> list[_Stat]:
    col = "capability_raw" if kind == "feature" else "target_raw"
    cur = await conn.execute(f"SELECT {col}, vendor FROM raw_claims WHERE {col} IS NOT NULL")
    stats: dict[str, _Stat] = {}
    for text, vendor in await cur.fetchall():
        key = normalize(text)
        if key:
            st = stats.setdefault(key, _Stat(key))
            st.count += 1
            st.vendors.add(vendor)
    return sorted(stats.values(), key=lambda s: s.text)


def _user_message(kind: Kind, members: list[_Stat]) -> str:
    vendors = sorted({v for m in members for v in m.vendors})
    shown = sorted(members, key=lambda m: (-m.count, m.text))[:MAX_MEMBERS_IN_PROMPT]
    lines = [f"- {m.text} ({m.count})" for m in shown]
    if len(members) > len(shown):
        lines.append(f"- ... and {len(members) - len(shown)} more")
    return (f"Kind: {kind}\nVendors ({len(vendors)}): "
            f"{', '.join(vendors[:MAX_VENDORS_IN_PROMPT])}"
            f"{', ...' if len(vendors) > MAX_VENDORS_IN_PROMPT else ''}\n"
            f"Phrasings (claim count):\n" + "\n".join(lines))


def _request(kind: Kind, user: str, settings: VendorkbSettings) -> dict[str, Any]:
    system = SYSTEM_PROMPTS[kind]
    if settings.extract_backend == "openai":
        return {
            "model": settings.extract_model, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "taxonomy_entry", "strict": True, "schema": PROPOSAL_SCHEMA}},
            "extra_body": {"usage": {"include": True}, "reasoning": {"effort": "low"}},
        }
    return {
        "model": settings.extract_model, "max_tokens": MAX_TOKENS, "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": "low",
                          "format": {"type": "json_schema", "schema": PROPOSAL_SCHEMA}},
    }


def _estimate(kind: Kind, user: str) -> float:
    chars = len(SYSTEM_PROMPTS[kind]) + len(user)
    return cost_usd(chars // CHARS_PER_TOKEN + 1, MAX_TOKENS, batch=False)


async def _call(client: Any, request: dict[str, Any], settings: VendorkbSettings
                ) -> tuple[str | None, str | None, float]:
    """(reply text, error, cost) for one proposal call."""
    if settings.extract_backend == "openai":
        resp = await client.chat.completions.create(**request)
        usage = getattr(resp, "usage", None)
        cost = _reported_cost(usage)
        if cost is None:
            cost = cost_usd(int(getattr(usage, "prompt_tokens", 0) or 0),
                            int(getattr(usage, "completion_tokens", 0) or 0), batch=False)
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
                    int(getattr(usage, "output_tokens", 0) or 0), batch=False)
    if getattr(msg, "stop_reason", None) in ("refusal", "max_tokens"):
        return None, str(msg.stop_reason), cost
    text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
    return text, None if text is not None else "no-text", cost


def _parse(text: str | None, kind: Kind) -> dict[str, Any]:
    """The validated proposal, or a ValueError naming the problem."""
    if text is None:
        raise ValueError("no reply text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid-json: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("invalid-shape")
    if data.get("category") not in CATEGORIES:
        raise ValueError(f"category {data.get('category')!r}")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("no name")
    aliases = data.get("aliases")
    if not isinstance(aliases, list):
        raise ValueError("aliases not a list")
    definition = data.get("definition")
    return {"id": slug_id(str(data.get("id") or ""), kind), "name": name.strip(),
            "category": data["category"],
            "definition": definition.strip() if isinstance(definition, str) else "",
            "aliases": [normalize(a) for a in aliases if isinstance(a, str) and normalize(a)]}


def long_tail_path(out_path: str | Path) -> Path:
    """``taxonomy/taxonomy.yaml`` -> ``taxonomy/long_tail.yaml``; ``x/t.yaml`` ->
    ``x/t.long_tail.yaml``."""
    out = Path(out_path)
    return out.with_name("long_tail.yaml" if out.stem == "taxonomy"
                         else f"{out.stem}.long_tail.yaml")


def _load_doc(path: Path) -> dict[str, Any]:
    """The taxonomy at ``path`` (validated), or a fresh ``version: 1`` one."""
    if not path.exists():
        return {"version": 1, "features": [], "targets": []}
    old = read_doc(path)
    old.setdefault("version", 1)
    validate(old)
    return {"version": old["version"], "features": list(old.get("features") or []),
            "targets": list(old.get("targets") or [])}


def _load_tail(path: Path) -> dict[str, Any]:
    old = read_doc(path) if path.exists() else {}
    return {"source": None, "feature": list(old.get("feature") or []),
            "target": list(old.get("target") or [])}


def _covered(entries: list[dict[str, Any]]) -> dict[str, str]:
    """normalized alias / name / id -> entry id."""
    index: dict[str, str] = {}
    for e in entries:
        for a in e.get("aliases") or []:
            index.setdefault(normalize(str(a)), e["id"])
    for e in entries:
        index.setdefault(normalize(str(e.get("name") or "")), e["id"])
        index.setdefault(normalize(str(e["id"]).replace("_", " ")), e["id"])
    index.pop("", None)
    return index


def _new_entry(kind: Kind, p: dict[str, Any], aliases: list[str]) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": p["id"], "name": p["name"], "category": p["category"]}
    if kind == "target":
        entry["parent"] = parent_of(p["id"])
    entry |= {"definition": p["definition"], "aliases": aliases, "support_count": 0,
              "vendor_count": 0}
    return entry


def _merge(kind: Kind, entries: list[dict[str, Any]], proposals: list[tuple[list[_Stat], dict]],
           report: ProposeReport) -> None:
    by_id = {e["id"]: e for e in entries}
    owner = _covered(entries)
    for members, p in proposals:
        target = by_id.get(p["id"])
        if target is not None:
            report.merged_into_existing += 1
        wanted = [m.text for m in members] + p["aliases"]
        aliases = [a for a in dict.fromkeys(wanted)
                   if owner.get(a) in (None, p["id"])]
        if target is None:
            target = _new_entry(kind, p, [])
            entries.append(target)
            by_id[p["id"]] = target
        target["aliases"] = list(dict.fromkeys([*(target.get("aliases") or []), *aliases]))
        for a in aliases:
            owner.setdefault(a, p["id"])
        report.proposed += 1
    if kind == "target":
        for e in list(entries):
            child, node = e, parent_of(e["id"])
            while node is not None and node not in by_id:
                category = child["category"] if parent_of(node) else "Platforms"
                anc = _new_entry(kind, {"id": node, "name": node, "category": category,
                                        "definition": "", "aliases": []}, [])
                entries.append(anc)
                by_id[node] = anc
                report.ancestors_added += 1
                child, node = anc, parent_of(node)


def _recount(entries: list[dict[str, Any]], stats: list[_Stat]) -> None:
    by_text = {s.text: s for s in stats}
    for e in entries:
        keys = {normalize(str(a)) for a in e.get("aliases") or []}
        keys.add(normalize(str(e.get("name") or "")))
        covered = [by_text[k] for k in keys if k in by_text]
        e["support_count"] = sum(s.count for s in covered)
        e["vendor_count"] = len({v for s in covered for v in s.vendors})


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
                   encoding="utf-8")
    os.replace(tmp, path)


async def propose(conn: psycopg.AsyncConnection, *, client: Any, embedder: _Embedder,
                  settings: VendorkbSettings, kind: Kind, distance_threshold: float,
                  out_path: str | Path, max_usd: float = 2.0,
                  limit: int | None = None) -> ProposeReport:
    if kind not in ("feature", "target"):
        raise ValueError(f"kind must be 'feature' or 'target', got {kind!r}")
    path = Path(out_path)
    tail_path = long_tail_path(path)
    report = ProposeReport(kind=kind, out_path=str(path), long_tail_path=str(tail_path))
    doc = _load_doc(path)  # before any paid work: a broken file is refused up front
    stats = await _strings(conn, kind)
    report.strings = len(stats)
    clusterable = sorted((s for s in stats if s.count >= 2 or len(s.vendors) >= 2),
                         key=lambda s: (-s.count, -len(s.vendors), s.text))
    long_tail: list[_Stat] = [s for s in stats if s.count < 2 and len(s.vendors) < 2]
    report.long_tail_prefiltered = len(long_tail)
    if len(clusterable) > MAX_CLUSTER_STRINGS:
        report.long_tail_capped = len(clusterable) - MAX_CLUSTER_STRINGS
        log.warning("taxonomy propose: %d clusterable strings; clustering the top %d, the"
                    " rest go to the long tail", len(clusterable), MAX_CLUSTER_STRINGS)
        long_tail.extend(clusterable[MAX_CLUSTER_STRINGS:])
        clusterable = clusterable[:MAX_CLUSTER_STRINGS]
    del stats
    report.clustered = len(clusterable)
    texts = [s.text for s in clusterable]
    vectors = (np.asarray(await embedder.embed(texts, input_type="document"), dtype=np.float32)
               if texts else np.zeros((0, 0), dtype=np.float32))
    groups = cluster_strings(texts, vectors, distance_threshold=distance_threshold)
    del vectors, texts
    report.clusters = len(groups)
    entries: list[dict[str, Any]] = doc[f"{kind}s"]
    covered = _covered(entries)
    todo: list[list[_Stat]] = []
    for g in groups:
        members = [clusterable[i] for i in g]
        if all(m.text in covered for m in members):
            report.already_covered += 1
        else:
            todo.append(members)
    report.eligible = report.already_covered + len(todo)
    todo.sort(key=lambda ms: (-sum(m.count for m in ms), ms[0].text))
    if limit is not None and len(todo) > limit:
        report.skipped_by_limit = len(todo) - limit
        todo = todo[:max(0, limit)]

    spent, in_flight, stop = 0.0, 0.0, False
    fatal: Exception | None = None
    sem = asyncio.Semaphore(max(1, settings.extract_concurrency))

    async def one(members: list[_Stat]) -> dict[str, Any] | None:
        nonlocal spent, in_flight, stop, fatal
        async with sem:
            user = _user_message(kind, members)
            estimate = _estimate(kind, user)
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
                text, error, cost = await _call(client, _request(kind, user, settings),
                                                settings)
            except (anthropic.BadRequestError, openai.BadRequestError) as e:
                stop, fatal = True, fatal or e
                report.invalid += 1
                report.errors.append(f"{members[0].text}: {e}")
                return None
            except Exception as e:  # API errors and any odd provider shape: this cluster only
                report.invalid += 1
                report.errors.append(f"{members[0].text}: {type(e).__name__}: {e}")
                log.warning("taxonomy propose: call failed for %r: %s", members[0].text, e)
                return None
            finally:
                in_flight -= estimate
            spent += cost
            try:
                if error is not None:
                    raise ValueError(error)
                return _parse(text, kind)
            except ValueError as e:
                report.invalid += 1
                report.errors.append(f"{members[0].text}: {e}")
                log.warning("taxonomy propose: unusable reply for %r: %s", members[0].text, e)
                return None

    replies = await asyncio.gather(*(one(ms) for ms in todo))
    report.spent_usd = spent
    _merge(kind, entries, [(ms, p) for ms, p in zip(todo, replies) if p is not None], report)
    _recount(entries, clusterable + long_tail)
    entries.sort(key=lambda e: e["id"])
    tail = sorted(long_tail, key=lambda s: (-s.count, -len(s.vendors), s.text))
    tail_doc = _load_tail(tail_path)
    tail_doc["source"] = path.name
    tail_doc[kind] = [{"text": s.text, "count": s.count, "vendor_count": len(s.vendors)}
                      for s in tail]
    report.long_tail = len(tail)
    _write(path, doc)
    _write(tail_path, tail_doc)
    log.info("taxonomy propose: %s", report)
    if fatal is not None:
        raise fatal
    return report
