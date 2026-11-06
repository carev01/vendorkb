"""Map the v3 claims to taxonomy v2 (spec §6.1), with LLM adjudication.

Only claims of the claims-v3 prompt family (``prompt_version LIKE 'claims-v3%'``) are read and
written; every other row keeps its values. Strings are compared normalized
(``cluster.normalize``), each distinct one resolved once:

1. **Target** (``target_raw``): an exact target alias (scope ``''``) is ``alias``. Otherwise the
   ``map_candidates`` targets nearest by cosine over their names and aliases (a target scores
   its best key) go to an adjudication call, which picks one (``llm``) or answers ``none``
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
string (capability, target, target scope, vendor) as context, and its strict JSON reply is
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
version, so downstream builds use only mappings made under the current version.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
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
from vendorkb.taxonomy.load import current_version
from vendorkb.taxonomy.propose import CHARS_PER_TOKEN, PROMPT_FAMILY

log = logging.getLogger(__name__)

__all__ = ["MapReport", "map_claims"]

Kind = Literal["target", "dimension"]
MAP_MAX_TOKENS = 1024  # one short JSON reply; headroom for any thinking the effort allows
MAX_CONTEXTS = 3  # example claims shown per adjudication
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
    target_alias: int = 0
    target_llm: int = 0
    target_unmapped: int = 0
    target_none: int = 0  # no target text (product-wide unless the scope names a target)
    child_target: int = 0  # target_scope named a child target of T
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
    capability: str | None  # None: blank after normalization (never adjudicated)
    target: str | None
    target_scope: str | None
    offering: str | None
    edition: str | None


@dataclass
class _Tax:
    """The loaded taxonomy v2, read once."""

    aliases: dict[tuple[str, str], dict[str, str]]  # (entity_type, scope) -> alias -> id
    target_parent: dict[str, str | None]
    target_class: dict[str, str]
    target_name: dict[str, str]
    class_parent: dict[str, str | None]
    dimensions: dict[str, tuple[str, str, list[str] | None]]  # id -> name, definition, applies
    vendor_offerings: dict[str, list[str]]

    def lookup(self, entity_type: str, scope: str, text: str | None) -> str | None:
        if not text:
            return None
        return self.aliases.get((entity_type, scope), {}).get(text)

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
    for alias, etype, scope, eid in await _rows(
            conn, "SELECT alias, entity_type, scope, entity_id FROM aliases ORDER BY 1, 2, 3"):
        aliases.setdefault((etype, scope), {})[alias] = eid
    targets = await _rows(conn, "SELECT target_id, parent_id, class_id, name FROM targets")
    vendor_offerings: dict[str, list[str]] = {}
    for oid, vendor in await _rows(conn, "SELECT offering_id, vendor FROM offerings ORDER BY 1"):
        vendor_offerings.setdefault(vendor, []).append(oid)
    return _Tax(
        aliases=aliases,
        target_parent={t[0]: t[1] for t in targets},
        target_class={t[0]: t[2] for t in targets},
        target_name={t[0]: t[3] for t in targets},
        class_parent=dict(await _rows(conn, "SELECT class_id, parent_id FROM classes")),
        dimensions={r[0]: (r[1], r[2], r[3]) for r in await _rows(
            conn, "SELECT dimension_id, name, definition, applies_to FROM dimensions")},
        vendor_offerings=vendor_offerings,
    )


def _norm(text: str | None) -> str | None:
    """The normalized text, None for NULL or blank."""
    return (normalize(text) or None) if text is not None else None


async def _claims(conn: psycopg.AsyncConnection) -> list[_Claim]:
    cur = await conn.execute(
        "SELECT claim_id, vendor, capability_raw, target_raw, target_scope, offering, edition"
        " FROM raw_claims WHERE prompt_version LIKE %s ORDER BY claim_id", (PROMPT_FAMILY,))
    return [_Claim(cid, vendor, cap, tgt, scope, _norm(cap), _norm(tgt), _norm(scope),
                   _norm(off), _norm(ed))
            for cid, vendor, cap, tgt, scope, off, ed in await cur.fetchall()]


# ---- nearest candidates ---------------------------------------------------------------------

def _unit(vectors: Any) -> np.ndarray:
    """Rows scaled to unit length, as float32."""
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


async def _embed_all(embedder: _Embedder, texts: list[str]) -> np.ndarray:
    """Unit float32 vectors of ``texts``, embedded in ``_EMBED_SLICE`` slices into one
    preallocated array (peak memory: the array plus one slice of Python floats)."""
    out: np.ndarray | None = None
    for start in range(0, len(texts), _EMBED_SLICE):
        part = _unit(await embedder.embed(texts[start:start + _EMBED_SLICE],
                                          input_type="document"))
        if out is None:
            out = np.empty((len(texts), part.shape[1]), dtype=np.float32)
        out[start:start + len(part)] = part
    return out if out is not None else np.zeros((0, 0), dtype=np.float32)


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


# ---- adjudication ---------------------------------------------------------------------------

@dataclass
class _Adjudication:
    kind: Kind
    text: str
    candidates: tuple[str, ...]
    class_id: str | None = None  # dimension: the workload class the candidates apply to
    contexts: list[tuple[str, str, str, str]] = field(default_factory=list)
    claims: int = 0
    choice: str | None = None
    skipped: bool = False

    def add_context(self, c: _Claim) -> None:
        self.claims += 1
        ctx = (c.capability_raw, c.target_raw or "-", c.target_scope_raw or "-", c.vendor)
        if ctx not in self.contexts:
            self.contexts.append(ctx)


_SYSTEM: dict[Kind, str] = {
    "target": """\
You map the target of a claim extracted from backup-product documentation to one entry of a
workload taxonomy. The target is the workload, platform, service or storage the claim applies
to. You get the target phrase, up to three claims that use it (capability | target | target
scope | vendor) and the nearest taxonomy targets, numbered (number. id | name | details).

Choose a candidate only when the phrase names that same workload (its name, an abbreviation or
another name for it). Answer "none" when the phrase names a broader or narrower workload than
every candidate, a different service of the same vendor, the backup product's own component,
or anything else. A close name is not enough: "none" is the right answer whenever you are not
sure. Reply with JSON {"choice": <the candidate's number>}, or {"choice": 0} for none.""",
    "dimension": """\
You map the capability of a claim extracted from backup-product documentation to one dimension
of a vendor-neutral comparison taxonomy. A dimension is one comparable aspect, defined by its
name and definition. You get the capability phrase, up to three claims that use it (capability
| target | target scope | vendor), the workload class of the claim's target, and the nearest
dimensions that apply to that class, numbered (number. id | name | definition).

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


def _user_message(tax: _Tax, a: _Adjudication) -> str:
    lines = [f"Phrase: {a.text}", f"Kind: {a.kind}"]
    if a.kind == "dimension":
        lines.append(f"Workload class: {a.class_id or _NO_CLASS}")
    lines += ["", "Claims (capability | target | target scope | vendor):"]
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
                 report: MapReport) -> None:
        self.tax, self.client, self.settings = tax, client, settings
        self.max_usd, self.report = max_usd, report
        self.price = map_price(settings)
        self.spent = self.in_flight = 0.0
        self.stop = False
        self.fatal: Exception | None = None
        self.sem = asyncio.Semaphore(max(1, settings.extract_concurrency))

    async def run(self, items: list[_Adjudication]) -> None:
        self.report.adjudications += len(items)
        await asyncio.gather(*(self._one(a) for a in items))
        self.report.spent_usd = self.spent
        if self.fatal is not None:
            raise self.fatal

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
                    report.llm_invalid += 1
                    report.errors.append(f"{a.kind} {a.text!r}: {e}")
                    log.warning("taxonomy map: rate-limited for %s %r: %s", a.kind, a.text, e)
                    return
                self.stop, self.fatal = True, self.fatal or e
                return
            except Exception as e:  # API errors and odd provider shapes: this string only
                self.spent += estimate  # it may have been billed: count it against the cap
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
            except ValueError as e:
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
    if variant is not None:
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


def _resolve_offering(tax: _Tax, c: _Claim, m: _Mapped, report: MapReport) -> None:
    if c.offering is None:
        report.offering_none += 1
    else:
        m.offering_id = tax.lookup("offering", c.vendor, c.offering)
        if m.offering_id is not None:
            report.offering_mapped += 1
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
                     settings: VendorkbSettings, max_usd: float) -> MapReport:
    version = await current_version(conn)
    if version is None:
        raise RuntimeError("no taxonomy loaded: run `vendorkb taxonomy load PATH` first")
    report = MapReport(taxonomy_version=version)
    tax = await _snapshot(conn)
    # Built before any paid work: it refuses a model with no known price.
    adjudicator = _Adjudicator(tax, client, settings, max_usd, report)
    k = max(1, settings.map_candidates)
    claims = await _claims(conn)
    report.claims = len(claims)

    target_keys = _Keys(tax.aliases.get(("target", ""), {}))
    dimension_keys = _Keys(tax.aliases.get(("dimension", ""), {}))
    target_queries = sorted({c.target for c in claims if c.target is not None
                             and tax.lookup("target", "", c.target) is None})
    dimension_queries = sorted({c.capability for c in claims if c.capability is not None
                                and tax.lookup("dimension", "", c.capability) is None})
    query_rows: dict[str, int] = {}
    query_vectors = np.zeros((0, 0), dtype=np.float32)
    if target_queries or dimension_queries:
        texts = list(dict.fromkeys(
            [*target_keys.texts, *dimension_keys.texts, *target_queries, *dimension_queries]))
        vectors = await _embed_all(embedder, texts)
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

    # 1. Targets.
    mapped = {c.claim_id: _Mapped() for c in claims}
    target_cands = candidates(target_keys, target_queries, {t: [None] for t in target_queries})
    target_adj = {t: _Adjudication("target", t, tuple(target_cands[(t, 0)]))
                  for t in target_queries if target_cands[(t, 0)]}
    for c in claims:
        if c.target in target_adj:
            target_adj[c.target].add_context(c)
    await adjudicator.run(list(target_adj.values()))

    for c in claims:
        m = mapped[c.claim_id]
        if c.target is not None:
            hit = tax.lookup("target", "", c.target)
            if hit is not None:
                m.target_id, m.target_method = hit, "alias"
            else:
                a = target_adj.get(c.target)
                m.target_id = a.choice if a is not None else None
                m.target_method = "llm" if m.target_id is not None else "unmapped"
                m.skipped = a is not None and a.skipped
        _resolve_scope(tax, c, m, report)
        if m.target_id is not None:
            m.class_id = tax.target_class[m.target_id]
        _count(report, "target", m.target_method if c.target is not None else "none")
        _resolve_offering(tax, c, m, report)

    # 3. Dimensions, the candidates drawn from the dimensions that apply to the target's class.
    classes_of: dict[str, list[str | None]] = {}
    for c in claims:
        if c.capability is not None and tax.lookup("dimension", "", c.capability) is None:
            seen = classes_of.setdefault(c.capability, [])
            if mapped[c.claim_id].class_id not in seen:
                seen.append(mapped[c.claim_id].class_id)
    applicable: dict[str | None, np.ndarray] = {}  # class -> mask over dimension entities
    for class_ids in classes_of.values():
        for cid in class_ids:
            if cid not in applicable:
                allowed = tax.applicable(cid)
                applicable[cid] = np.asarray([e in allowed for e in dimension_keys.entities],
                                             dtype=bool)
    dim_cands = candidates(dimension_keys, dimension_queries,
                           {t: [applicable[cid] for cid in classes_of[t]]
                            for t in dimension_queries})
    dim_adj: dict[tuple[str, tuple[str, ...]], _Adjudication] = {}
    adj_of: dict[tuple[str, str | None], _Adjudication | None] = {}
    for text in dimension_queries:
        for i, cid in enumerate(classes_of[text]):
            cands = tuple(dim_cands[(text, i)])
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
        _count(report, "dimension", m.dimension_method)
        if (m.dimension_id or "").startswith("coverage.") and m.object_type_id is None:
            report.coverage_without_object_type += 1
        report.claims_skipped_by_budget += int(m.skipped)
        rows.append((c.claim_id, m.dimension_id, m.dimension_method, m.class_id, m.target_id,
                     m.target_method, m.variant_id, m.object_type_id, m.offering_id,
                     m.edition_id))

    async with conn.transaction():
        await conn.execute(
            "CREATE TEMP TABLE _claim_map (claim_id uuid PRIMARY KEY, dimension_id text,"
            " dimension_method text, class_id text, target_id text, target_method text,"
            " variant_id text, object_type_id text, offering_id text, edition_id text)"
            " ON COMMIT DROP")
        async with conn.cursor() as cur:
            async with cur.copy("COPY _claim_map FROM STDIN") as copy:
                for r in rows:
                    await copy.write_row(r)
        await conn.execute(
            "UPDATE raw_claims r SET dimension_id=m.dimension_id,"
            " dimension_method=m.dimension_method, class_id=m.class_id,"
            " target_id=m.target_id, target_method=m.target_method, target_conf=NULL,"
            " variant_id=m.variant_id, object_type_id=m.object_type_id,"
            " offering_id=m.offering_id, edition_id=m.edition_id, mapped_version=%s"
            " FROM _claim_map m WHERE r.claim_id = m.claim_id AND r.prompt_version LIKE %s",
            (version, PROMPT_FAMILY))
    log.info("taxonomy map: %s", report)
    return report
