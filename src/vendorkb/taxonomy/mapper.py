"""Map the v3 claims to taxonomy v2 (spec §6.1), with LLM adjudication.

Only claims of the claims-v3 prompt family (``prompt_version LIKE 'claims-v3%'``) are read and
written; every other row keeps its values. By default only the claims not yet mapped under the
current taxonomy version are read (``mapped_version`` NULL or another version): a claim mapped
once keeps that mapping until the taxonomy changes, and ``remap`` maps every claim again.
``vendor`` limits a run to one vendor's claims: each run commits on its own, so a large backlog
mapped vendor by vendor loses at most one vendor's paid adjudications to a failure.
Strings are compared normalized (``cluster.normalize``), each distinct one resolved once:

1. **Target** (``target_raw``): an exact target alias (scope ``''``) is ``alias``. Otherwise the
   ``map_candidates`` targets nearest by cosine over their names and aliases (a target scores
   its best key; ``lookup_aliases`` are no keys) go to an adjudication call, which picks one (``llm``) or answers ``none``
   (``unmapped``). A claim with no target text has a NULL ``target_method`` unless its
   scope names a target (below).
2. **target_scope**, given the resolved target T, tried in this order: a target alias whose
   target has T as a proper ancestor (the claim's target becomes that child); a variant alias
   of T; a ``target_object_type`` alias of T; an ``object_type`` alias of each class in T's
   lineage, most specific first. A claim with no target text tries its scope as a target
   alias; one whose target text did not map leaves the scope unresolved (an unrelated target
   named in a scope is never taken as a child of T). ``class_id`` is the final target's class.
3. **Dimension** (``capability_raw``): an exact dimension alias is ``alias``. Otherwise the
   candidates are the nearest dimensions among those that apply to the final target's class
   (``aspects.dimension_applies``; every dimension when there is no target), adjudicated the
   same way. A ``coverage.*`` dimension with no object type is kept (object type NULL) and
   counted in ``coverage_without_object_type``: it does not count as coverage.
4. **Offering**: an offering alias within the claim's vendor (the raw vendor text, matched
   exactly). **Edition**: an edition alias within that offering. With no offering text, the one
   (offering, edition) among the vendor's offerings whose edition has the alias sets both (none
   when several do). An offering text that did not map leaves the edition unmapped.

Nearest neighbour alone never maps. One adjudication is made per distinct (kind, string,
candidate set) and cached for the run; it carries up to ``MAX_CONTEXTS`` of the claims using the
string (capability, target, target scope, vendor, evidence quote) as context -- the quote
measurably helps: on the 150 labelled non-alias claims, dimension precision 82.8% -> 88.6% and
wrong picks 21 -> 14 (a softer "none" rule found more dimensions but lost precision) -- and its
strict JSON reply is
``{"choice": <integer>}`` -- one fixed schema for every call (the candidates are numbered 1..k
in the prompt; 0 is none; anything else is ``llm_invalid``). Calls go to ``map_model`` on
``map_backend``, ``extract_concurrency`` at a time; each is estimated before it is sent (its
prompt plus ``MAP_MAX_TOKENS`` of output, priced by ``map_price``) and the run stops sending
once spent plus in flight plus the next estimate would pass ``max_usd``: what is left is
unmapped and counted in ``skipped_by_budget``. A rejected request (``BadRequestError``: a bad
model or schema) stops the run and raises with nothing written; a 400 that says "rate limit" is
retried by ``llm.call`` and, when it persists, fails that adjudication only. The alias and
candidate keys and the strings to adjudicate are embedded in one pass per run (in slices) and
kept as float32. A capability blank after normalization is unmapped without a call.

Every v3 claim read is then (re)written in one transaction and stamped with the taxonomy
version, so downstream builds use only mappings made under the current version. A claim whose
mapping waits on an adjudication that was skipped by the budget or failed (``llm_invalid``) is
written unmapped with no stamp, so the next run maps it again (``claims_to_retry``).
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Literal, Protocol

import anthropic
import numpy as np
import openai
import psycopg

from vendorkb.claims.extract import cost_usd, price_per_mtok
from vendorkb.config import VendorkbSettings
from vendorkb.taxonomy.aspects import dimension_applies
from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.llm import build_request, call, object_schema, rate_limited
from vendorkb.taxonomy.load import Taxonomy, current_version
from vendorkb.taxonomy.propose import CHARS_PER_TOKEN, PROMPT_FAMILY

log = logging.getLogger(__name__)

__all__ = ["MapReport", "map_claims"]

Kind = Literal["target", "dimension"]
MAP_MAX_TOKENS = 1024  # one short JSON reply; headroom for any thinking the effort allows
MAX_CONTEXTS = 3  # example claims shown per adjudication
QUOTE_CHARS = 300  # evidence quote kept per example claim
_NN_BLOCK = 1024  # query rows per similarity matmul
_EMBED_SLICE = 4096  # texts per embed call (the embedder batches further)
_NO_CLASS = "(none)"


class _Embedder(Protocol):
    async def embed(self, texts: list[str], *, input_type: Literal["document", "query"] = ...
                    ) -> list[list[float]]: ...


@dataclass
class MapReport:
    taxonomy_version: str = ""
    claims: int = 0
    claims_already_mapped: int = 0  # not read: mapped under this taxonomy version already
    claims_to_retry: int = 0  # written with no stamp (budget stop or failed adjudication)
    target_alias: int = 0
    target_llm: int = 0
    target_unmapped: int = 0
    target_none: int = 0  # no target text (product-wide unless the scope names a target)
    child_target: int = 0  # target_scope named a child target of T
    variant_target: int = 0  # the target text was a variant alias: its target and variant
    target_product: int = 0  # the target text named the claim's product: treated as no target
    scope_target: int = 0  # no T: target_scope named a target
    variant: int = 0
    object_type: int = 0
    scope_unmapped: int = 0  # a target_scope that resolved to nothing
    dimension_alias: int = 0
    dimension_llm: int = 0
    dimension_unmapped: int = 0
    coverage_without_object_type: int = 0
    offering_mapped: int = 0
    offering_unmapped: int = 0
    offering_product: int = 0  # the offering text was a product name of the vendor
    offering_none: int = 0  # no offering text (offering_from_edition among them)
    offering_from_edition: int = 0  # no offering text; set from a unique edition
    edition_mapped: int = 0
    edition_unmapped: int = 0
    edition_none: int = 0
    embedded_strings: int = 0
    adjudications: int = 0  # distinct (kind, string, candidate set) needing a call
    llm_calls: int = 0
    llm_none: int = 0
    llm_invalid: int = 0  # failed calls and replies outside the schema
    embeddings_cached: int = 0  # strings whose vector came from embedding_cache
    embeddings_new: int = 0  # strings embedded by the provider (and cached)
    decisions_reused: int = 0  # adjudications answered from map_decisions (no call)
    decisions_carried: int = 0  # same phrase, fewer or reordered candidates: earlier answer kept
    decisions_kept: int = 0  # target asked again, answered none: the earlier answer, still offered
    decisions_seeded: int = 0  # seed=True: decisions newly recorded from the stored mappings
    decisions_reasked: int = 0  # stored or earlier answer named a reask entity: asked again
    skipped_by_budget: int = 0  # adjudications not sent
    claims_skipped_by_budget: int = 0  # claims left unmapped by them
    stopped_by_budget: bool = False
    spent_usd: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class _Claim:
    claim_id: Any
    vendor: str
    capability_raw: str
    target_raw: str | None
    target_scope_raw: str | None
    evidence_quote: str  # whitespace collapsed, at most QUOTE_CHARS
    capability: str | None  # None: blank after normalization (never adjudicated)
    target: str | None
    target_scope: str | None
    offering: str | None
    edition: str | None


_SUFFIXES = (("ies", "y"), ("ing", ""), ("ed", ""), ("s", ""))


def fold_inflections(text: str) -> str:
    """``text`` (normalized) with each word's common inflection folded the same way for every
    form: "scanning", "scanned" and "scan" -> "scan"; "restores" and "restore" -> "restor";
    "policies" -> "policy". Only ever compared with other folded text."""
    words = []
    for w in text.split():
        if len(w) > 4 and not w.endswith("ss"):
            for suffix, repl in _SUFFIXES:
                if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                    w = w[:-len(suffix)] + repl
                    break
        if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
            w = w[:-1]  # scann -> scan
        if len(w) > 3 and w.endswith("e"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def _unique_index(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """key -> value for keys that map to one value only (an ambiguous key is left out)."""
    seen: dict[str, set] = {}
    for key, value in pairs:
        seen.setdefault(key, set()).add(value)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


@dataclass
class _Tax:
    """The loaded taxonomy v2, read once."""

    aliases: dict[tuple[str, str], dict[str, str]]  # (entity_type, scope) -> alias -> id
    search: dict[tuple[str, str], dict[str, str]]  # the same, candidate-search keys only
    target_parent: dict[str, str | None]
    target_class: dict[str, str]
    target_name: dict[str, str]
    class_parent: dict[str, str | None]
    dimensions: dict[str, tuple[str, str, list[str] | None]]  # id -> name, definition, applies
    vendor_offerings: dict[str, list[str]]
    _folded: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)
    _variants: dict[str, tuple[str, str]] | None = None

    def lookup(self, entity_type: str, scope: str, text: str | None) -> str | None:
        """The exact alias, else the one entry whose alias folds to the same inflections
        ("malware scanning" finds "malware scan"); None when absent or ambiguous."""
        if not text:
            return None
        table = self.aliases.get((entity_type, scope), {})
        hit = table.get(text)
        if hit is not None or not table:
            return hit
        key = (entity_type, scope)
        if key not in self._folded:
            self._folded[key] = _unique_index([(fold_inflections(a), e) for a, e in table.items()])
        return self._folded[key].get(fold_inflections(text))

    def variant_target(self, text: str | None) -> tuple[str, str] | None:
        """(target, variant) when ``text`` is the alias of exactly one variant of any target:
        "FSx for Windows File Server" as a target text is aws.fsx, variant windows."""
        if not text:
            return None
        if self._variants is None:
            pairs = [(a, (scope, v)) for (etype, scope), table in self.aliases.items()
                     if etype == "variant" for a, v in table.items()]
            self._variants = _unique_index(pairs + [(fold_inflections(a), tv) for a, tv in pairs])
        return self._variants.get(text) or self._variants.get(fold_inflections(text))

    def lineage(self, class_id: str) -> list[str]:
        """``class_id`` and its ancestors, most specific first."""
        out: list[str] = []
        node: str | None = class_id
        while node is not None and node not in out:
            out.append(node)
            node = self.class_parent.get(node)
        return out

    def is_ancestor(self, ancestor: str, target_id: str) -> bool:
        node = self.target_parent.get(target_id)
        seen: set[str] = set()
        while node is not None and node not in seen:
            if node == ancestor:
                return True
            seen.add(node)
            node = self.target_parent.get(node)
        return False

    def applicable(self, class_id: str | None) -> set[str]:
        if class_id is None:
            return set(self.dimensions)
        lineage = self.lineage(class_id)
        return {d for d, (_, _, applies) in self.dimensions.items()
                if dimension_applies(applies, lineage)}


async def _rows(conn: psycopg.AsyncConnection, sql: str) -> list[tuple]:
    cur = await conn.execute(sql)
    return list(await cur.fetchall())


async def _snapshot(conn: psycopg.AsyncConnection) -> _Tax:
    aliases: dict[tuple[str, str], dict[str, str]] = {}
    search: dict[tuple[str, str], dict[str, str]] = {}
    for alias, etype, scope, eid, is_key in await _rows(
            conn, "SELECT alias, entity_type, scope, entity_id, search FROM aliases"
                  " ORDER BY 1, 2, 3"):
        aliases.setdefault((etype, scope), {})[alias] = eid
        if is_key:
            search.setdefault((etype, scope), {})[alias] = eid
    targets = await _rows(conn, "SELECT target_id, parent_id, class_id, name FROM targets")
    vendor_offerings: dict[str, list[str]] = {}
    for oid, vendor in await _rows(conn, "SELECT offering_id, vendor FROM offerings ORDER BY 1"):
        vendor_offerings.setdefault(vendor, []).append(oid)
    return _Tax(
        aliases=aliases,
        search=search,
        target_parent={t[0]: t[1] for t in targets},
        target_class={t[0]: t[2] for t in targets},
        target_name={t[0]: t[3] for t in targets},
        class_parent=dict(await _rows(conn, "SELECT class_id, parent_id FROM classes")),
        dimensions={r[0]: (r[1], r[2], r[3]) for r in await _rows(
            conn, "SELECT dimension_id, name, definition, applies_to FROM dimensions")},
        vendor_offerings=vendor_offerings,
    )


def _with_dimensions(tax: _Tax, taxonomy: Taxonomy) -> _Tax:
    """A copy of ``tax`` with the dimensions of ``taxonomy`` (a validated YAML) in place of its
    own: the rows, aliases and search keys that ``load_taxonomy`` would store for them, built by
    the same ``alias_rows`` and ``lookup_only``. Everything else is kept and nothing is
    written, so a replay can rank candidates under a taxonomy that is not loaded."""
    lookup_only = taxonomy.lookup_only()
    aliases: dict[str, str] = {}
    search: dict[str, str] = {}
    for alias, etype, scope, eid in taxonomy.alias_rows():
        if etype == "dimension":
            aliases[alias] = eid
            if (alias, etype, scope) not in lookup_only:
                search[alias] = eid
    return dataclasses.replace(
        tax,
        aliases={**tax.aliases, ("dimension", ""): aliases},
        search={**tax.search, ("dimension", ""): search},
        dimensions={d.id: (d.name, d.definition,
                           None if d.applies_to is None else list(d.applies_to))
                    for d in taxonomy.dimensions},
        _folded={})  # the folded alias indexes were built from the old dimension aliases


async def _vendor_products(conn: psycopg.AsyncConnection) -> dict[str, set[str]]:
    """vendor -> the normalized names of its products (documents.product) and of itself."""
    out: dict[str, set[str]] = {}
    for vendor, product in await _rows(conn, "SELECT DISTINCT vendor, product FROM documents"):
        names = out.setdefault(vendor, {normalize(vendor)})
        if product:
            names.add(normalize(product))
    return out


def _norm(text: str | None) -> str | None:
    """The normalized text, None for NULL or blank."""
    return (normalize(text) or None) if text is not None else None


async def _claims(conn: psycopg.AsyncConnection, unmapped_at: str | None,
                  vendor: str | None = None) -> list[_Claim]:
    """The v3 claims; with ``unmapped_at``, only those not stamped with that version; with
    ``vendor``, only that vendor's."""
    cur = await conn.execute(
        "SELECT claim_id, vendor, capability_raw, target_raw, target_scope, offering, edition,"
        " evidence_quote FROM raw_claims WHERE prompt_version LIKE %(family)s"
        " AND (%(version)s::text IS NULL OR mapped_version IS DISTINCT FROM %(version)s)"
        " AND (%(vendor)s::text IS NULL OR vendor = %(vendor)s) ORDER BY claim_id",
        {"family": PROMPT_FAMILY, "version": unmapped_at, "vendor": vendor})
    return [_Claim(cid, vendor, cap, tgt, scope, " ".join((quote or "").split())[:QUOTE_CHARS],
                   _norm(cap), _norm(tgt), _norm(scope), _norm(off), _norm(ed))
            for cid, vendor, cap, tgt, scope, off, ed, quote in await cur.fetchall()]


# ---- nearest candidates ---------------------------------------------------------------------

def _unit(vectors: Any) -> np.ndarray:
    """Rows scaled to unit length, as float32."""
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


_CACHE_FETCH = 20_000  # cached vectors read per query (about 80 MB at 1,024 dimensions)
_CACHE_MAX_CHARS = 1_000  # longer strings are embedded every time, never cached


async def _embed_all(embedder: _Embedder, texts: list[str], *, conn: psycopg.AsyncConnection,
                     model: str, report: MapReport) -> np.ndarray:
    """Unit float32 vectors of ``texts``, written into one preallocated array. Vectors in
    embedding_cache (same ``model``, input type "document") are read back bit for bit; the rest
    are embedded in ``_EMBED_SLICE`` slices, each cached as soon as it arrives (peak memory: the
    array plus one slice)."""
    row = {t: i for i, t in enumerate(texts)}
    out: np.ndarray | None = None
    have = np.zeros(len(texts), dtype=bool)
    keys = [t for t in texts if len(t) <= _CACHE_MAX_CHARS]
    async with conn.cursor(binary=True) as cur:
        for start in range(0, len(keys), _CACHE_FETCH):
            await cur.execute(
                "SELECT text, vec FROM embedding_cache WHERE model = %s"
                " AND input_type = 'document' AND text = ANY(%s)",
                (model, keys[start:start + _CACHE_FETCH]))
            for text, vec in await cur.fetchall():
                v = np.frombuffer(vec, dtype="<f4")
                if out is None:
                    out = np.empty((len(texts), len(v)), dtype=np.float32)
                out[row[text]] = v
                have[row[text]] = True
    missing = [t for t in texts if not have[row[t]]]
    report.embeddings_cached += len(texts) - len(missing)
    report.embeddings_new += len(missing)
    for start in range(0, len(missing), _EMBED_SLICE):
        part = missing[start:start + _EMBED_SLICE]
        raw = np.asarray(await embedder.embed(part, input_type="document"), dtype=np.float32)
        if out is None:
            out = np.empty((len(texts), raw.shape[1]), dtype=np.float32)
        out[[row[t] for t in part]] = raw
        async with conn.transaction(), conn.cursor() as cur:  # one commit per slice
            await cur.executemany(
                "INSERT INTO embedding_cache (model, input_type, text, vec)"
                " VALUES (%s, 'document', %s, %s) ON CONFLICT DO NOTHING",
                [(model, t, v.astype("<f4").tobytes()) for t, v in zip(part, raw)
                 if len(t) <= _CACHE_MAX_CHARS])
    if out is None:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.where(norms == 0, 1.0, norms)
    return out


class _Keys:
    """An entity type's alias keys, grouped by entity: an entity scores its best key."""

    def __init__(self, index: dict[str, str]) -> None:
        pairs = sorted(index.items(), key=lambda kv: (kv[1], kv[0]))
        self.texts = [k for k, _ in pairs]
        self.entities: list[str] = []
        starts: list[int] = []
        for i, (_, eid) in enumerate(pairs):
            if not self.entities or self.entities[-1] != eid:
                self.entities.append(eid)
                starts.append(i)
        self.starts = np.asarray(starts, dtype=np.intp)
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def scores(self, queries: np.ndarray) -> np.ndarray:
        """(len(queries), len(entities)) best cosine per entity."""
        if not len(self.entities):
            return np.zeros((len(queries), 0), dtype=np.float32)
        return np.maximum.reduceat(queries @ self.vectors.T, self.starts, axis=1)


def _top(scores: np.ndarray, entities: list[str], allowed: np.ndarray | None, k: int
         ) -> list[str]:
    """The ``k`` best entities (ties by id), among the ``allowed`` mask when given."""
    if allowed is not None:
        scores = np.where(allowed, scores, -np.inf)
    k = min(k, len(entities) if allowed is None else int(allowed.sum()))
    if k <= 0:
        return []
    best = np.argpartition(-scores, k - 1)[:k]
    return [entities[i] for i in sorted(best.tolist(),
                                        key=lambda i: (-float(scores[i]), entities[i]))]


def _pin(cands: list[str], picks: list[str], valid: Callable[[str], bool], k: int
         ) -> tuple[str, ...]:
    """``cands`` plus the earlier picks it lacks (valid ones, at most ``k - 1``) in place of its
    lowest-ranked entries, so an answer chosen before stays on offer when a taxonomy change
    pushes it out: in a blind grade of such displaced questions the earlier answer was the
    better one about twice as often as the new one (2026-11-08)."""
    extra = [p for p in picks if p not in cands and valid(p)][:max(k - 1, 0)]
    if not cands or not extra:
        return tuple(cands)
    return tuple(cands[:max(k - len(extra), 0)] + extra if len(cands) + len(extra) > k
                 else cands + extra)


async def _earlier_picks(conn: psycopg.AsyncConnection, settings: VendorkbSettings,
                         phrases: list[str]) -> dict[tuple[str, str, str], list[str]]:
    """(kind, phrase, class or '') -> the entities chosen for that question before (same model
    family and prompt rules), latest first."""
    if not phrases:
        return {}
    cur = await conn.execute(
        "SELECT kind, phrase, coalesce(class_id, ''), choice FROM map_decisions"
        " WHERE model = %s AND rules = %s AND choice IS NOT NULL AND phrase = ANY(%s)"
        " GROUP BY 1, 2, 3, 4 ORDER BY max(decided_at) DESC",
        (model_family(settings.map_model), PROMPT_RULES, phrases))
    out: dict[tuple[str, str, str], list[str]] = {}
    for kind, phrase, class_id, choice in await cur.fetchall():
        out.setdefault((kind, phrase, class_id), []).append(choice)
    return out


# ---- adjudication ---------------------------------------------------------------------------

@dataclass
class _Adjudication:
    kind: Kind
    text: str
    candidates: tuple[str, ...]
    class_id: str | None = None  # dimension: the workload class the candidates apply to
    contexts: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    claims: int = 0
    choice: str | None = None
    skipped: bool = False
    failed: bool = False  # the call failed or its reply was unusable: no decision made
    decided: bool = False  # the model answered (a candidate or none)
    cached: bool = False  # answered from map_decisions
    reasked: bool = False  # its stored or earlier answer named a reask entity
    claim_ids: list[Any] = field(default_factory=list)

    def add_context(self, c: _Claim) -> None:
        self.claims += 1
        self.claim_ids.append(c.claim_id)
        ctx = (c.capability_raw, c.target_raw or "-", c.target_scope_raw or "-", c.vendor,
               c.evidence_quote or "-")
        if ctx not in self.contexts:
            self.contexts.append(ctx)


_SYSTEM: dict[Kind, str] = {
    "target": """\
You map the target of a claim extracted from backup-product documentation to one entry of a
workload taxonomy. The target is the workload, platform, service or storage the claim applies
to. You get the target phrase, up to three claims that use it (capability | target | target
scope | vendor | evidence quote) and the nearest taxonomy targets, numbered (number. id | name |
details).

Choose a candidate only when the phrase names that same workload (its name, an abbreviation or
another name for it). Answer "none" when the phrase names a broader or narrower workload than
every candidate, a different service of the same vendor, the backup product's own component,
or anything else. A close name is not enough: "none" is the right answer whenever you are not
sure. Reply with JSON {"choice": <the candidate's number>}, or {"choice": 0} for none.""",
    "dimension": """\
You map the capability of a claim extracted from backup-product documentation to one dimension
of a vendor-neutral comparison taxonomy. A dimension is one comparable aspect, defined by its
name and definition. You get the capability phrase, up to three claims that use it (capability
| target | target scope | vendor | evidence quote: the documentation sentence the claim comes
from), the workload class of the claim's target, and the nearest dimensions that apply to that
class, numbered (number. id | name | definition).

Choose the candidate whose definition covers what the capability states: the capability may be
narrower than the dimension (one value or one option of it), never about something else. A
coverage dimension (coverage.*) fits only a capability that says whether objects can be backed
up, restored, exported or compared. Answer "none" when no candidate's definition covers the
capability, or when you are not sure: a related topic is not enough. Reply with JSON
{"choice": <the candidate's number>}, or {"choice": 0} for none.""",
}


def _details(tax: _Tax, kind: Kind, eid: str) -> tuple[str, str]:
    if kind == "dimension":
        name, definition, _ = tax.dimensions[eid]
        return name, " ".join(definition.split())
    parent = tax.target_parent.get(eid)
    details = f"class {tax.target_class[eid]}" + (f", part of {parent}" if parent else "")
    return tax.target_name[eid], details


# A stored decision answers the same question: these prompt rules, the model family, the kind,
# the phrase, the workload class and the numbered candidates. The example claims are left out of
# the key, so a phrase's decision survives new claims using it.
PROMPT_RULES = hashlib.sha256(json.dumps(_SYSTEM, sort_keys=True).encode()).hexdigest()[:12]


def model_family(model: str) -> str:
    """One name per model across backends: anthropic/claude-haiku-5.5 = claude-haiku-5-5."""
    return model.lower().split(":", 1)[0].removeprefix("anthropic/").replace(".", "-")


def decision_key(a: _Adjudication, model: str) -> str:
    return hashlib.sha256(json.dumps([PROMPT_RULES, model_family(model), a.kind, a.text,
                                      a.class_id, list(a.candidates)]).encode()).hexdigest()


def _user_message(tax: _Tax, a: _Adjudication) -> str:
    lines = [f"Phrase: {a.text}", f"Kind: {a.kind}"]
    if a.kind == "dimension":
        lines.append(f"Workload class: {a.class_id or _NO_CLASS}")
    lines += ["", "Claims (capability | target | target scope | vendor | evidence quote):"]
    lines += [f"- {' | '.join(ctx)}" for ctx in a.contexts[:MAX_CONTEXTS]]
    lines += ["", "Candidates (number. id | name | details):"]
    lines += [f"{n}. {eid} | {' | '.join(_details(tax, a.kind, eid))}"
              for n, eid in enumerate(a.candidates, 1)]
    return "\n".join(lines)


# One fixed schema for every adjudication: the API compiles a grammar per distinct schema and
# rate-limits compilations, so the candidates are numbered in the prompt, never enumerated here.
REPLY_SCHEMA: dict[str, Any] = object_schema({"choice": {"type": "integer"}})


def _parse(text: str | None, candidates: tuple[str, ...]) -> str | None:
    """The chosen candidate id (``choice`` 1..k), None for 0, or a ValueError."""
    if text is None:
        raise ValueError("no reply text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid-json: {e}") from e
    choice = data.get("choice") if isinstance(data, dict) else None
    if isinstance(choice, bool) or not isinstance(choice, int) \
            or not 0 <= choice <= len(candidates):
        raise ValueError(f"choice {choice!r} is not 0..{len(candidates)}")
    return None if choice == 0 else candidates[choice - 1]


def map_price(settings: VendorkbSettings) -> tuple[float, float]:
    """(input, output) USD per million tokens for ``map_model``: the price table when it lists
    the model, else the EXTRACT_PRICE_* override (which is meant for the extraction model and
    must not underprice a listed map model), else a ValueError."""
    try:
        return price_per_mtok(settings.map_model)
    except ValueError:
        return price_per_mtok(settings.map_model, settings)


class _Adjudicator:
    """Sends adjudications under one budget for the whole run."""

    def __init__(self, tax: _Tax, client: Any, settings: VendorkbSettings, max_usd: float,
                 report: MapReport, conn: psycopg.AsyncConnection,
                 reask: frozenset[str] = frozenset()) -> None:
        self.tax, self.client, self.settings, self.conn = tax, client, settings, conn
        self.max_usd, self.report, self.reask = max_usd, report, reask
        self.price = map_price(settings)
        self.spent = self.in_flight = 0.0
        self.stop = False
        self.fatal: Exception | None = None
        self.sem = asyncio.Semaphore(max(1, settings.extract_concurrency))

    async def run(self, items: list[_Adjudication]) -> None:
        self.report.adjudications += len(items)
        model = self.settings.map_model
        keyed = {decision_key(a, model): a for a in items}
        cur = await self.conn.execute("SELECT key, choice FROM map_decisions WHERE key = ANY(%s)",
                                      (list(keyed),))
        for key, choice in await cur.fetchall():
            a = keyed[key]
            if choice in self.reask:
                a.reasked = True
            elif choice is None or choice in a.candidates:
                a.choice, a.cached = choice, True
                self.report.decisions_reused += 1
        await self._carry([a for a in items if not a.cached])
        self.report.decisions_reasked += sum(a.reasked for a in items)
        try:
            await asyncio.gather(*(self._one(a) for a in items if not a.cached))
            await self._keep([a for a in items if a.kind == "target" and a.choice is None
                              and (a.cached or a.decided)])
        finally:  # paid decisions are kept even when the run stops
            await self._store([(k, a) for k, a in keyed.items() if a.decided], "llm",
                              replace=True)
        self.report.spent_usd = self.spent
        if self.fatal is not None:
            raise self.fatal

    async def seed(self, items: list[_Adjudication], current: dict[Any, tuple]) -> None:
        """Record, without a call, the decision the stored mappings of each adjudication's claims
        imply: the candidate (method llm) or none (method unmapped) that most of them carry,
        when more than half agree. ``current``: claim_id -> (target_id, target_method,
        dimension_id, dimension_method) for the claims mapped under the current version."""
        model = self.settings.map_model
        rows = []
        for a in items:
            votes: dict[str | None, int] = {}
            for cid in a.claim_ids:
                cur = current.get(cid)
                if cur is None:
                    continue
                chosen, method = (cur[0], cur[1]) if a.kind == "target" else (cur[2], cur[3])
                if method == "llm" and chosen in a.candidates:
                    votes[chosen] = votes.get(chosen, 0) + 1
                elif method == "unmapped":
                    votes[None] = votes.get(None, 0) + 1
            if votes:
                choice, n = max(votes.items(), key=lambda kv: kv[1])
                if n * 2 > sum(votes.values()):
                    a.choice, a.decided = choice, True
                    rows.append((decision_key(a, model), a))
        self.report.decisions_seeded += await self._store(rows, "seed")

    async def _carry(self, items: list[_Adjudication]) -> None:
        """Stability: a question whose candidates changed keeps the latest earlier answer for the
        same phrase and class (same model family and prompt rules) when nothing new is offered
        (its candidates are a subset of the earlier ones) and that answer is still among them,
        or was none. A new candidate, or an answer that dropped out, means asking again. This
        keeps repeated runs from flipping between equally plausible answers."""
        if not items:
            return
        cur = await self.conn.execute(
            "SELECT DISTINCT ON (kind, phrase, class_id) kind, phrase, class_id, candidates, choice"
            " FROM map_decisions WHERE model = %s AND rules = %s"
            " AND (kind, phrase, coalesce(class_id, '')) IN"
            "  (SELECT * FROM unnest(%s::text[], %s::text[], %s::text[]))"
            " ORDER BY kind, phrase, class_id, decided_at DESC",
            (model_family(self.settings.map_model), PROMPT_RULES, [a.kind for a in items],
             [a.text for a in items], [a.class_id or "" for a in items]))
        earlier = {(k, p, c): (cands, choice) for k, p, c, cands, choice in await cur.fetchall()}
        for a in items:
            prev = earlier.get((a.kind, a.text, a.class_id))
            if prev is None:
                continue
            cands, choice = prev
            if choice in self.reask:
                a.reasked = True
            elif set(a.candidates) <= set(cands) and (choice is None or choice in a.candidates):
                a.choice, a.cached = choice, True
                self.report.decisions_carried += 1

    async def _keep(self, items: list[_Adjudication]) -> None:
        """A target question asked again, because new candidates were offered, keeps the answer
        of the decision before it when it now answers none and that answer is still offered:
        re-asked target questions answered none for obvious matches ("rhel" -> linux) far more
        often than they dropped a wrong pick (taxonomy round 2, 2026-11-08). Dimensions are
        left alone: there a re-asked none was the better answer as often as not. The kept answer
        is what gets stored; a none stored by an older run is read the same way."""
        if not items:
            return
        model = self.settings.map_model
        keys = [decision_key(a, model) for a in items]
        cur = await self.conn.execute(
            "SELECT q.key, (SELECT m.choice FROM map_decisions m WHERE m.kind = 'target'"
            "  AND m.phrase = q.phrase AND m.model = %s AND m.rules = %s AND m.key <> q.key"
            "  AND m.decided_at < coalesce((SELECT d.decided_at FROM map_decisions d"
            "   WHERE d.key = q.key), now())"
            "  ORDER BY m.decided_at DESC LIMIT 1)"
            " FROM unnest(%s::text[], %s::text[]) AS q(key, phrase)",
            (model_family(model), PROMPT_RULES, keys, [a.text for a in items]))
        earlier: dict[str, str | None] = dict(await cur.fetchall())
        for key, a in zip(keys, items):
            choice = earlier.get(key)
            if choice is not None and choice in a.candidates and choice not in self.reask:
                a.choice = choice
                self.report.decisions_kept += 1

    async def _store(self, rows: list[tuple[str, _Adjudication]], source: str, *,
                     replace: bool = False) -> int:
        """Insert decisions; returns how many rows were written. A stored decision for the same
        key is left alone, or with ``replace`` (a question asked again, see ``reask``)
        replaced."""
        if not rows:
            return 0
        family = model_family(self.settings.map_model)
        conflict = ("DO UPDATE SET candidates = EXCLUDED.candidates, choice = EXCLUDED.choice,"
                    " model = EXCLUDED.model, source = EXCLUDED.source, rules = EXCLUDED.rules,"
                    " decided_at = now()" if replace else "DO NOTHING")
        async with self.conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO map_decisions (key, kind, phrase, class_id, candidates, choice,"
                " model, source, rules) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                f" ON CONFLICT (key) {conflict}",
                [(k, a.kind, a.text, a.class_id, list(a.candidates), a.choice, family, source,
                  PROMPT_RULES) for k, a in rows])
            return max(cur.rowcount, 0)  # executemany: the sum over the statements

    async def _one(self, a: _Adjudication) -> None:
        report, settings = self.report, self.settings
        async with self.sem:
            system, user = _SYSTEM[a.kind], _user_message(self.tax, a)
            estimate = cost_usd((len(system) + len(user)) // CHARS_PER_TOKEN + 1,
                                MAP_MAX_TOKENS, batch=False, price=self.price)
            if self.stop or self.spent + self.in_flight + estimate > self.max_usd:
                if not self.stop and self.fatal is None:
                    report.stopped_by_budget = True
                    log.warning("taxonomy map: budget $%.4f reached (spent $%.4f, in flight"
                                " $%.4f); the rest stay unmapped", self.max_usd, self.spent,
                                self.in_flight)
                self.stop = True
                if self.fatal is None:
                    a.skipped = True
                    report.skipped_by_budget += 1
                return
            self.in_flight += estimate
            try:
                request = build_request(
                    backend=settings.map_backend, model=settings.map_model,
                    max_tokens=MAP_MAX_TOKENS, system=system, user=user,
                    schema=REPLY_SCHEMA, name="taxonomy_map")
                text, error, cost = await call(self.client, request,
                                               backend=settings.map_backend, price=self.price)
            except (anthropic.BadRequestError, openai.BadRequestError) as e:
                if rate_limited(e):  # still rate-limited after call()'s retries: this string
                    a.failed = True
                    report.llm_invalid += 1
                    report.errors.append(f"{a.kind} {a.text!r}: {e}")
                    log.warning("taxonomy map: rate-limited for %s %r: %s", a.kind, a.text, e)
                    return
                self.stop, self.fatal = True, self.fatal or e
                return
            except Exception as e:  # API errors and odd provider shapes: this string only
                self.spent += estimate  # it may have been billed: count it against the cap
                a.failed = True
                report.llm_invalid += 1
                report.errors.append(f"{a.kind} {a.text!r}: {type(e).__name__}: {e}")
                log.warning("taxonomy map: call failed for %s %r: %s", a.kind, a.text, e)
                return
            finally:
                self.in_flight -= estimate
            self.spent += cost
            report.llm_calls += 1
            try:
                if error is not None:
                    raise ValueError(error)
                a.choice = _parse(text, a.candidates)
                a.decided = True
            except ValueError as e:
                a.failed = True
                report.llm_invalid += 1
                report.errors.append(f"{a.kind} {a.text!r}: {e}")
                log.warning("taxonomy map: unusable reply for %s %r: %s", a.kind, a.text, e)
                return
            if a.choice is None:
                report.llm_none += 1


# ---- the per-claim resolution ---------------------------------------------------------------

@dataclass
class _Mapped:
    target_id: str | None = None
    target_method: str | None = None
    class_id: str | None = None
    variant_id: str | None = None
    object_type_id: str | None = None
    dimension_id: str | None = None
    dimension_method: str = "unmapped"
    offering_id: str | None = None
    edition_id: str | None = None
    skipped: bool = False
    failed: bool = False


def _resolve_scope(tax: _Tax, c: _Claim, m: _Mapped, report: MapReport) -> None:
    scope = c.target_scope
    if m.target_id is None:
        # Only a claim with no target text takes its scope as the target: a target text that
        # did not map leaves the scope unresolved.
        hit = tax.lookup("target", "", scope) if c.target is None else None
        if hit is not None:
            m.target_id, m.target_method = hit, "alias"
            report.scope_target += 1
        elif scope is not None:
            report.scope_unmapped += 1
        return
    if scope is None:
        return
    t = m.target_id
    child = tax.lookup("target", "", scope)
    if child is not None and tax.is_ancestor(t, child):
        m.target_id = child
        report.child_target += 1
        return
    variant = tax.lookup("variant", t, scope)
    if variant is not None and m.variant_id is None:
        m.variant_id = variant
        report.variant += 1
        return
    object_type = tax.lookup("target_object_type", t, scope) or next(
        (o for cid in tax.lineage(tax.target_class[t])
         if (o := tax.lookup("object_type", cid, scope)) is not None), None)
    if object_type is not None:
        m.object_type_id = object_type
        report.object_type += 1
    else:
        report.scope_unmapped += 1


def _resolve_offering(tax: _Tax, c: _Claim, m: _Mapped, report: MapReport,
                      products: frozenset[str] | set[str]) -> None:
    if c.offering is None:
        report.offering_none += 1
    else:
        m.offering_id = tax.lookup("offering", c.vendor, c.offering)
        if m.offering_id is not None:
            report.offering_mapped += 1
        elif c.offering in products:  # a product name, not a commercial offering
            report.offering_product += 1
        else:
            report.offering_unmapped += 1
    if c.edition is None:
        report.edition_none += 1
        return
    if m.offering_id is not None:
        m.edition_id = tax.lookup("edition", m.offering_id, c.edition)
    elif c.offering is None:  # an offering text that did not map: the edition stays free text
        hits = [(oid, e) for oid in tax.vendor_offerings.get(c.vendor, [])
                if (e := tax.lookup("edition", oid, c.edition)) is not None]
        if len(hits) == 1:  # one (offering, edition) among the vendor's offerings has the alias
            m.offering_id, m.edition_id = hits[0]
            report.offering_from_edition += 1
    if m.edition_id is not None:
        report.edition_mapped += 1
    else:
        report.edition_unmapped += 1


def _count(report: MapReport, prefix: str, method: str | None) -> None:
    name = f"{prefix}_{method}"
    setattr(report, name, getattr(report, name) + 1)


async def map_claims(conn: psycopg.AsyncConnection, *, embedder: _Embedder, client: Any,
                     settings: VendorkbSettings, max_usd: float, remap: bool = False,
                     vendor: str | None = None, seed: bool = False,
                     reask: frozenset[str] = frozenset()) -> MapReport:
    version = await current_version(conn)
    if version is None:
        raise RuntimeError("no taxonomy loaded: run `vendorkb taxonomy load PATH` first")
    report = MapReport(taxonomy_version=version)
    tax = await _snapshot(conn)
    # Built before any paid work: it refuses a model with no known price.
    adjudicator = _Adjudicator(tax, client, settings, max_usd, report, conn, reask)
    k = max(1, settings.map_candidates)
    # seed reads every claim mapped under this version and records their decisions
    claims = await _claims(conn, None if remap or seed else version, vendor)
    report.claims = len(claims)
    # A target text that names the claim's own product (and no taxonomy target) is about the
    # product itself, not a workload: it is treated as no target text.
    products = await _vendor_products(conn)
    for i, c in enumerate(claims):
        if (c.target is not None and c.target in products.get(c.vendor, ())
                and tax.lookup("target", "", c.target) is None
                and tax.variant_target(c.target) is None):
            claims[i] = dataclasses.replace(c, target=None)
            report.target_product += 1
    current: dict[Any, tuple] = {}
    if seed:
        cur = await conn.execute(
            "SELECT claim_id, target_id, target_method, dimension_id, dimension_method"
            " FROM raw_claims WHERE prompt_version LIKE %s AND mapped_version = %s",
            (PROMPT_FAMILY, version))
        current = {r[0]: tuple(r[1:]) for r in await cur.fetchall()}
    if not remap:
        cur = await conn.execute(
            "SELECT count(*) FROM raw_claims WHERE prompt_version LIKE %s"
            " AND mapped_version = %s AND (%s::text IS NULL OR vendor = %s)",
            (PROMPT_FAMILY, version, vendor, vendor))
        report.claims_already_mapped = (await cur.fetchone())[0]  # type: ignore[index]

    target_keys = _Keys(tax.search.get(("target", ""), {}))  # lookup aliases are no keys
    dimension_keys = _Keys(tax.search.get(("dimension", ""), {}))
    target_queries = sorted({c.target for c in claims if c.target is not None
                             and tax.variant_target(c.target) is None
                             and tax.lookup("target", "", c.target) is None})
    dimension_queries = sorted({c.capability for c in claims if c.capability is not None
                                and tax.lookup("dimension", "", c.capability) is None})
    query_rows: dict[str, int] = {}
    query_vectors = np.zeros((0, 0), dtype=np.float32)
    if target_queries or dimension_queries:
        texts = list(dict.fromkeys(
            [*target_keys.texts, *dimension_keys.texts, *target_queries, *dimension_queries]))
        vectors = await _embed_all(  # the cache keys on the dimension too
            embedder, texts, conn=conn, model=f"{settings.embed_model}@{settings.embed_dim}",
            report=report)
        report.embedded_strings = len(texts)
        row = {t: i for i, t in enumerate(texts)}
        target_keys.vectors = vectors[[row[t] for t in target_keys.texts]]
        dimension_keys.vectors = vectors[[row[t] for t in dimension_keys.texts]]
        queries = list(dict.fromkeys([*target_queries, *dimension_queries]))
        query_vectors = vectors[[row[q] for q in queries]]
        query_rows = {q: i for i, q in enumerate(queries)}
        del vectors

    def candidates(keys: _Keys, texts: list[str], allowed: dict[str, list[np.ndarray | None]]
                   ) -> dict[tuple[str, int], list[str]]:
        """Top candidates per (text, position in ``allowed[text]``), in blocks."""
        out: dict[tuple[str, int], list[str]] = {}
        for start in range(0, len(texts), _NN_BLOCK):
            block = texts[start:start + _NN_BLOCK]
            scores = keys.scores(query_vectors[[query_rows[t] for t in block]])
            for text, text_scores in zip(block, scores):
                for i, allow in enumerate(allowed[text]):
                    out[(text, i)] = _top(text_scores, keys.entities, allow, k)
        return out

    picks = await _earlier_picks(conn, settings,
                                 list(dict.fromkeys([*target_queries, *dimension_queries])))

    # 1. Targets.
    mapped = {c.claim_id: _Mapped() for c in claims}
    target_cands = candidates(target_keys, target_queries, {t: [None] for t in target_queries})
    target_adj = {t: _Adjudication("target", t, _pin(target_cands[(t, 0)],
                                                     picks.get(("target", t, ""), []),
                                                     tax.target_class.__contains__, k))
                  for t in target_queries if target_cands[(t, 0)]}
    for c in claims:
        if c.target in target_adj:
            target_adj[c.target].add_context(c)
    if seed:
        await adjudicator.seed(list(target_adj.values()), current)
    else:
        await adjudicator.run(list(target_adj.values()))

    for c in claims:
        m = mapped[c.claim_id]
        if c.target is not None:
            hit = tax.lookup("target", "", c.target)
            variant = tax.variant_target(c.target) if hit is None else None
            if hit is not None:
                m.target_id, m.target_method = hit, "alias"
            elif variant is not None:
                (m.target_id, m.variant_id), m.target_method = variant, "alias"
                report.variant_target += 1
            else:
                a = target_adj.get(c.target)
                m.target_id = a.choice if a is not None else None
                m.target_method = "llm" if m.target_id is not None else "unmapped"
                m.skipped = a is not None and a.skipped
                m.failed = a is not None and a.failed
        _resolve_scope(tax, c, m, report)
        if m.target_id is not None:
            m.class_id = tax.target_class[m.target_id]
        _count(report, "target", m.target_method if c.target is not None else "none")
        _resolve_offering(tax, c, m, report, products.get(c.vendor, frozenset()))

    # 3. Dimensions, the candidates drawn from the dimensions that apply to the target's class.
    classes_of: dict[str, list[str | None]] = {}
    for c in claims:
        if c.capability is not None and tax.lookup("dimension", "", c.capability) is None:
            seen = classes_of.setdefault(c.capability, [])
            if mapped[c.claim_id].class_id not in seen:
                seen.append(mapped[c.claim_id].class_id)
    applicable: dict[str | None, np.ndarray] = {}  # class -> mask over dimension entities
    # What may be pinned: a lookup_only dimension (no search key) is never offered.
    allowed_ids: dict[str | None, set[str]] = {}
    searchable = set(dimension_keys.entities)
    for class_ids in classes_of.values():
        for cid in class_ids:
            if cid not in applicable:
                allowed_ids[cid] = allowed = tax.applicable(cid) & searchable
                applicable[cid] = np.asarray([e in allowed for e in dimension_keys.entities],
                                             dtype=bool)
    dim_cands = candidates(dimension_keys, dimension_queries,
                           {t: [applicable[cid] for cid in classes_of[t]]
                            for t in dimension_queries})
    dim_adj: dict[tuple[str, tuple[str, ...]], _Adjudication] = {}
    adj_of: dict[tuple[str, str | None], _Adjudication | None] = {}
    for text in dimension_queries:
        for i, cid in enumerate(classes_of[text]):
            cands = _pin(dim_cands[(text, i)], picks.get(("dimension", text, cid or ""), []),
                         allowed_ids[cid].__contains__, k)
            if not cands:
                adj_of[(text, cid)] = None
                continue
            a = dim_adj.setdefault((text, cands), _Adjudication("dimension", text, cands, cid))
            adj_of[(text, cid)] = a
    for c in claims:
        a = (adj_of.get((c.capability, mapped[c.claim_id].class_id))
             if c.capability is not None else None)
        if a is not None:
            a.add_context(c)
    if seed:  # decisions recorded; the claims themselves are left as they are
        await adjudicator.seed(list(dim_adj.values()), current)
        log.info("taxonomy map (seed): %s", report)
        return report
    await adjudicator.run(list(dim_adj.values()))

    rows = []
    for c in claims:
        m = mapped[c.claim_id]
        hit = tax.lookup("dimension", "", c.capability)
        if hit is not None:
            m.dimension_id, m.dimension_method = hit, "alias"
        else:
            a = adj_of.get((c.capability, m.class_id)) if c.capability is not None else None
            if a is not None and a.choice is not None:
                m.dimension_id, m.dimension_method = a.choice, "llm"
            m.skipped = m.skipped or (a is not None and a.skipped)
            m.failed = m.failed or (a is not None and a.failed)
        _count(report, "dimension", m.dimension_method)
        if (m.dimension_id or "").startswith("coverage.") and m.object_type_id is None:
            report.coverage_without_object_type += 1
        report.claims_skipped_by_budget += int(m.skipped)
        done = not (m.skipped or m.failed)
        report.claims_to_retry += int(not done)
        rows.append((c.claim_id, m.dimension_id, m.dimension_method, m.class_id, m.target_id,
                     m.target_method, m.variant_id, m.object_type_id, m.offering_id,
                     m.edition_id, done))

    async with conn.transaction():
        await conn.execute(
            "CREATE TEMP TABLE _claim_map (claim_id uuid PRIMARY KEY, dimension_id text,"
            " dimension_method text, class_id text, target_id text, target_method text,"
            " variant_id text, object_type_id text, offering_id text, edition_id text,"
            " done boolean) ON COMMIT DROP")
        async with conn.cursor() as cur:
            async with cur.copy("COPY _claim_map FROM STDIN") as copy:
                for r in rows:
                    await copy.write_row(r)
        await conn.execute(
            "UPDATE raw_claims r SET dimension_id=m.dimension_id,"
            " dimension_method=m.dimension_method, class_id=m.class_id,"
            " target_id=m.target_id, target_method=m.target_method, target_conf=NULL,"
            " variant_id=m.variant_id, object_type_id=m.object_type_id,"
            " offering_id=m.offering_id, edition_id=m.edition_id,"
            " mapped_version=CASE WHEN m.done THEN %s END"
            " FROM _claim_map m WHERE r.claim_id = m.claim_id AND r.prompt_version LIKE %s",
            (version, PROMPT_FAMILY))
    log.info("taxonomy map: %s", report)
    return report
