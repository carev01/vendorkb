"""The claim support check (BACKLOG 36): does a claim's evidence quote show that the vendor's
product has the capability? The evidence gate only checks that the quote is verbatim.

In scope: claims of the v3 family mapped under the current taxonomy version, with support
``supported`` or ``partial``, on a dimension of an aspect of any loaded topic. Two steps, each
writing ``claim_support`` as its replies arrive:

1. the checker (``SUPPORT_MODEL`` at ``SUPPORT_EFFORT``) asks each in-scope claim with no verdict;
   a claim it calls unsupported is *flagged*;
2. the reviewer (``SUPPORT_REVIEW_MODEL`` at ``SUPPORT_REVIEW_EFFORT``) re-asks every flagged
   claim; a claim it calls unsupported is *rejected*, one it calls supported is kept.

``capabilities build`` leaves out the rejected claims only: a flagged claim stays until its
review, so a wrong flag never costs a fact, and a vendor whose facts the reviewer rejects all
loses its "yes". A verdict is per claim, so nothing depends on the topics or the other claims
at build time. (Calibration on 1,078 gold-labelled facts: the checker flags 41% of the
unsupported facts and 2.6% of the supported ones; the reviewer kept all 26 wrongly flagged facts
and confirmed 34 of the 41 unsupported ones.)

Both steps run ``support_concurrency`` calls at once on the support backend (``SUPPORT_BACKEND``:
the Anthropic API by default, or an OpenAI-compatible endpoint). Each call is estimated before it
is sent, with ``MAX_TOKENS`` of output (the worst case); a step stops once spent plus in flight
plus the next estimate would pass its budget. A failed or unusable call writes no verdict: the
claim is asked again by the next run. A claim deleted while it was being asked is skipped
(``vanished``). A non-rate-limit 400, a 401, 402, 403 or 404 (a bad key, no credit, an unknown
model), or a TypeError (the Anthropic SDK's "no key" error) stops the step: nothing else would
succeed either. A stopped check still lets the review run; the first such error is raised after
both.
Spec and calibration: docs/superpowers/specs/2026-11-09-claim-support-check-design.md.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai
import psycopg

from vendorkb.claims.extract import PROMPT_FAMILY, cost_usd, price_per_mtok
from vendorkb.config import VendorkbSettings
from vendorkb.taxonomy.llm import build_request, call, object_schema, rate_limited
from vendorkb.taxonomy.load import current_version
from vendorkb.topics import load_topics

log = logging.getLogger(__name__)

LOCK_KEY = 0x766B65  # single-flight across support runs (sync 0x766B62 ... propose 0x766B64)
CHECK_PROMPT_VERSION = "support-v2"
REVIEW_PROMPT_VERSION = "review-v1"
# Room for the thinking the effort allows (the calibration's replies averaged about 100 thinking
# tokens for Haiku at high, 40 for Opus at low); a reply cut off at it is unusable, asked again
# later. Each call is estimated at this much output, so a budget bounds the worst case.
MAX_TOKENS = 4000
# Statuses no retry can fix: a bad key, no credit, no permission, an unknown model.
FATAL_STATUS = frozenset({401, 402, 403, 404})
CHARS_PER_TOKEN = 4

CHECK_SYSTEM = (
    "You check one capability claim extracted from a backup vendor's documentation: does the "
    "evidence quote show that THIS vendor's product (the vendor, product and page given) has or "
    "supports the stated capability?\n"
    "Answer true when the quote states it or plainly implies it for this product. That includes "
    "a quote that gives a requirement, version, limit or condition of the capability (\"X "
    "requires version 8.1 or later\", \"X is supported for the following storage vendors\"), "
    "that tells how to configure, use or manage the capability in this product (\"You can update "
    "the IdP configuration\"), or that lists what the capability supports. For a 'partial' claim "
    "a stated limit still counts.\n"
    "Answer false when the quote is generic advice or explains a concept without saying this "
    "product does it; describes another product, a third party's own feature, or a feature of "
    "the protected application rather than of this product; or concerns something other than "
    "the stated capability.\n"
    'Reply with JSON {"supported": true} or {"supported": false}.')
REVIEW_SYSTEM = (
    "You review one capability claim extracted from a backup vendor's documentation. Decide one "
    "thing: does the evidence quote show that this vendor's product has or supports the stated "
    "capability? Use only the claim's fields and no outside knowledge.\n"
    "Answer true when the quote states it, or plainly implies it, for this vendor's product. "
    "These count: a requirement, version, limit or condition of the capability; how to "
    "configure, use, manage or troubleshoot it in this product; a list of what it supports, or "
    "a setting value of it; for a 'partial' claim, a stated limit. The capability may belong to "
    "another product of the same vendor. A product that creates or configures immutability, "
    "locking or retention on a storage system counts.\n"
    "Answer false when the quote is generic advice or a concept explanation that does not say "
    "this product does it; when the capability belongs to another vendor's product, a cloud "
    "service, a storage system or the protected application and this product merely uses or "
    "reports on it; when the quote is about something other than the stated capability; or when "
    "it does not mention the capability at all.\n"
    "The target is context only: judge the capability. When unsure, give the answer a careful "
    "reader of the quote alone would give.\n"
    'Reply with JSON {"supported": true} or {"supported": false}.')
REPLY_SCHEMA = object_schema({"supported": {"type": "boolean"}})


@dataclass(frozen=True)
class ClaimRow:
    claim_id: str
    vendor: str  # the page's, as the citation shows it
    product: str | None
    title: str
    capability: str
    target: str | None
    support: str
    quote: str


@dataclass
class StepReport:
    claims: int = 0  # asked for (in scope and due)
    calls: int = 0  # replies received
    supported: int = 0
    unsupported: int = 0
    invalid: int = 0  # failed or unusable: no verdict written, asked again next run
    vanished: int = 0  # the claim was deleted while it was being asked
    skipped_by_budget: int = 0
    stopped_by_budget: bool = False
    spent_usd: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class SupportReport:
    taxonomy_version: str = ""
    locked: bool = False  # another support run holds the lock; nothing done
    check: StepReport = field(default_factory=StepReport)
    review: StepReport = field(default_factory=StepReport)
    flagged: int = 0  # after the run: flagged claims waiting for review (kept until then)
    rejected: int = 0  # after the run: claims the reviewer rejected (left out of capabilities)


def checker_id(model: str, effort: str, prompt_version: str) -> str:
    return f"{model}@{effort}/{prompt_version}"


def user_message(row: ClaimRow) -> str:
    return (f"Vendor: {row.vendor}\nProduct: {row.product or '(none)'}\n"
            f"Page title: {row.title}\nCapability: {row.capability}\n"
            f"Target: {row.target or '(none)'}\nSupport: {row.support}\n"
            f"Evidence quote: {row.quote}")


def parse_verdict(text: str | None) -> bool:
    """The reply's ``supported``; ValueError for anything else."""
    try:
        value = json.loads(text or "")["supported"]
    except (ValueError, TypeError, KeyError) as e:
        raise ValueError(f"unusable reply {text!r:.80}") from e
    if not isinstance(value, bool):
        raise ValueError(f"supported is not a boolean: {value!r:.40}")
    return value


async def topic_dimensions(conn: psycopg.AsyncConnection) -> list[str]:
    """The dimensions of every aspect of every loaded topic: the check's scope."""
    return sorted({d for t in await load_topics(conn) for a in t.aspects for d in a.dimensions})


_ROW = ("r.claim_id::text, d.vendor, d.product, coalesce(d.title, ''), r.capability_raw,"
        " r.target_raw, r.support, r.evidence_quote")
_SOURCE = (" FROM raw_claims r JOIN chunks ch ON ch.chunk_id = r.chunk_id"
           " JOIN documents d ON d.doc_id = ch.doc_id"
           " WHERE r.prompt_version LIKE %(family)s AND r.mapped_version = %(ver)s"
           " AND r.support IN ('supported', 'partial')"
           " AND r.dimension_id = ANY(%(dims)s::text[])")


async def due_for_check(conn: psycopg.AsyncConnection, version: str, *,
                        limit: int | None = None) -> list[ClaimRow]:
    """The in-scope claims with no verdict yet, by claim id."""
    dims = await topic_dimensions(conn)
    if not dims:
        return []
    cur = await conn.execute(
        f"SELECT {_ROW}{_SOURCE} AND NOT EXISTS (SELECT 1 FROM claim_support s"
        " WHERE s.claim_id = r.claim_id) ORDER BY r.claim_id"
        + (" LIMIT %(limit)s" if limit is not None else ""),
        {"family": PROMPT_FAMILY, "ver": version, "dims": dims, "limit": limit})
    return [ClaimRow(*row) for row in await cur.fetchall()]


async def flagged_ids(conn: psycopg.AsyncConnection) -> list[str]:
    """The claims the checker flagged that nobody has reviewed yet, by claim id."""
    cur = await conn.execute("SELECT claim_id::text FROM claim_support WHERE NOT supported"
                             " AND review_supported IS NULL ORDER BY claim_id")
    return [row[0] for row in await cur.fetchall()]


async def rejected_count(conn: psycopg.AsyncConnection) -> int:
    cur = await conn.execute("SELECT count(*) FROM claim_support WHERE NOT review_supported")
    return int((await cur.fetchone())[0])  # type: ignore[index]


# ``capabilities build``'s filter on raw_claims ``r``: leave out the claims the reviewer rejected.
NOT_REJECTED = ("NOT EXISTS (SELECT 1 FROM claim_support s WHERE s.claim_id = r.claim_id"
                " AND s.review_supported = false)")


async def _has_verdict(conn: psycopg.AsyncConnection, claim_id: str) -> bool:
    cur = await conn.execute("SELECT 1 FROM claim_support WHERE claim_id = %s", (claim_id,))
    return await cur.fetchone() is not None


async def _rows(conn: psycopg.AsyncConnection, ids: list[str]) -> list[ClaimRow]:
    cur = await conn.execute(
        f"SELECT {_ROW} FROM raw_claims r JOIN chunks ch ON ch.chunk_id = r.chunk_id"
        " JOIN documents d ON d.doc_id = ch.doc_id WHERE r.claim_id = ANY(%s::uuid[])"
        " ORDER BY r.claim_id", (ids,))
    return [ClaimRow(*row) for row in await cur.fetchall()]


class _Asker:
    """One step's calls under one budget: ``record`` stores each verdict as it arrives."""

    def __init__(self, *, client: Any, settings: VendorkbSettings, model: str, effort: str,
                 system: str, max_usd: float, report: StepReport, label: str,
                 record: Any) -> None:
        self.client, self.settings, self.model, self.effort = client, settings, model, effort
        self.system, self.max_usd, self.report, self.label = system, max_usd, report, label
        self.record = record
        try:
            self.price = price_per_mtok(model)  # never the EXTRACT_PRICE_* override
        except ValueError:
            raise ValueError(f"no price known for support model {model!r}: add it to"
                             " claims.extract.PRICES_PER_MTOK") from None
        self.spent = self.in_flight = 0.0
        self.stop = False
        self.fatal: Exception | None = None
        self.sem = asyncio.Semaphore(max(1, settings.support_concurrency))

    async def run(self, rows: list[ClaimRow]) -> None:
        self.report.claims += len(rows)
        try:
            await asyncio.gather(*(self._one(row) for row in rows))
        finally:
            self.report.spent_usd += self.spent
        if self.fatal is not None:
            raise self.fatal

    async def _one(self, row: ClaimRow) -> None:
        report, backend = self.report, self.settings.support_backend
        async with self.sem:
            user = user_message(row)
            estimate = cost_usd((len(self.system) + len(user)) // CHARS_PER_TOKEN + 1,
                                MAX_TOKENS, batch=False, price=self.price)
            if self.stop or self.spent + self.in_flight + estimate > self.max_usd:
                if not self.stop and self.fatal is None:
                    report.stopped_by_budget = True
                    log.warning("support %s: budget $%.4f reached (spent $%.4f, in flight"
                                " $%.4f); the rest wait for the next run", self.label,
                                self.max_usd, self.spent, self.in_flight)
                self.stop = True
                if self.fatal is None:
                    report.skipped_by_budget += 1
                return
            self.in_flight += estimate
            try:
                request = build_request(backend=backend, model=self.model,
                                        max_tokens=MAX_TOKENS, system=self.system, user=user,
                                        schema=REPLY_SCHEMA, name="support_verdict",
                                        effort=self.effort)
                text, error, cost = await call(self.client, request, backend=backend,
                                               price=self.price)
            except (anthropic.BadRequestError, openai.BadRequestError) as e:
                if rate_limited(e):  # still rate-limited after call()'s retries: this claim
                    report.invalid += 1
                    report.errors.append(f"{row.claim_id}: {e}")
                    return
                self.stop, self.fatal = True, self.fatal or e
                return
            except Exception as e:  # API errors and odd provider shapes: this claim only
                self.spent += estimate  # it may have been billed: count it against the cap
                if isinstance(e, TypeError) or getattr(e, "status_code", None) in FATAL_STATUS:
                    self.stop, self.fatal = True, self.fatal or e
                    return
                report.invalid += 1
                report.errors.append(f"{row.claim_id}: {type(e).__name__}: {e}")
                log.warning("support %s: call failed for %s: %s", self.label, row.claim_id, e)
                return
            finally:
                self.in_flight -= estimate
            self.spent += cost
            report.calls += 1
            try:
                if error is not None:
                    raise ValueError(error)
                verdict = parse_verdict(text)
            except ValueError as e:
                report.invalid += 1
                report.errors.append(f"{row.claim_id}: {e}")
                log.warning("support %s: unusable reply for %s: %s", self.label, row.claim_id,
                            e)
                return
            try:
                written = await self.record(row.claim_id, verdict, cost)
            except psycopg.errors.ForeignKeyViolation:
                written = False
            except Exception as e:  # the database failed: stop the step, keep what is stored
                self.stop, self.fatal = True, self.fatal or e
                return
            if not written:
                report.vanished += 1
            elif verdict:
                report.supported += 1
            else:
                report.unsupported += 1


async def check(conn: psycopg.AsyncConnection, version: str, *, client: Any,
                settings: VendorkbSettings, max_usd: float, report: StepReport,
                limit: int | None = None) -> None:
    """Ask the checker about every in-scope claim with no verdict (at most ``limit``)."""
    rows = await due_for_check(conn, version, limit=limit)
    checker = checker_id(settings.support_model, settings.support_effort,
                         CHECK_PROMPT_VERSION)

    async def record(claim_id: str, verdict: bool, cost: float) -> bool:
        cur = await conn.execute(
            "INSERT INTO claim_support (claim_id, supported, checker, cost_usd)"
            " SELECT claim_id, %s, %s, %s FROM raw_claims WHERE claim_id = %s"
            " ON CONFLICT (claim_id) DO NOTHING", (verdict, checker, cost, claim_id))
        return cur.rowcount > 0 or await _has_verdict(conn, claim_id)

    asker = _Asker(client=client, settings=settings, model=settings.support_model,
                   effort=settings.support_effort, system=CHECK_SYSTEM, max_usd=max_usd,
                   report=report, label="check", record=record)
    await asker.run(rows)


async def review(conn: psycopg.AsyncConnection, *, client: Any, settings: VendorkbSettings,
                 max_usd: float, report: StepReport, limit: int | None = None) -> None:
    """Ask the reviewer about every flagged claim (at most ``limit``)."""
    ids = await flagged_ids(conn)
    rows = await _rows(conn, ids[:limit] if limit is not None else ids)
    reviewer = checker_id(settings.support_review_model, settings.support_review_effort,
                          REVIEW_PROMPT_VERSION)

    async def record(claim_id: str, verdict: bool, cost: float) -> bool:
        cur = await conn.execute(
            "UPDATE claim_support SET review_supported = %s, reviewer = %s,"
            " reviewed_at = now(), cost_usd = cost_usd + %s WHERE claim_id = %s",
            (verdict, reviewer, cost, claim_id))
        return cur.rowcount > 0

    asker = _Asker(client=client, settings=settings, model=settings.support_review_model,
                   effort=settings.support_review_effort, system=REVIEW_SYSTEM,
                   max_usd=max_usd, report=report, label="review", record=record)
    await asker.run(rows)


async def run(conn: psycopg.AsyncConnection, *, client: Any, settings: VendorkbSettings,
              check_max_usd: float, review_max_usd: float,
              limit: int | None = None) -> SupportReport:
    """Check, then review (see the module docstring), under one advisory lock."""
    report = SupportReport()
    version = await current_version(conn)
    if version is None:
        return report
    report.taxonomy_version = version
    cur = await conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
    if not (await cur.fetchone())[0]:  # type: ignore[index]
        report.locked = True
        log.warning("support: another support run holds the lock; nothing done")
        return report
    fatal: Exception | None = None
    try:
        try:
            await check(conn, version, client=client, settings=settings,
                        max_usd=check_max_usd, report=report.check, limit=limit)
        except Exception as e:  # the review uses another model: it may still work
            fatal = e
        try:
            await review(conn, client=client, settings=settings, max_usd=review_max_usd,
                         report=report.review, limit=limit)
        except Exception as e:
            fatal = fatal or e
        report.flagged = len(await flagged_ids(conn))
        report.rejected = await rejected_count(conn)
    finally:
        await conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
    if fatal is not None:
        raise fatal
    for step in (report.check, report.review):
        step.errors = step.errors[:20]
    log.info("support: %s", report)
    return report
