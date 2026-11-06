"""Claim extraction: one Claude Haiku 5.5 call per chunk, every claim gated by its quote.

``run`` loads the chunks, skips those already done for (model, prompt version), checks the
budget, then extracts either synchronously (``client.messages.create``, the smoke-test path)
or through the Message Batches API (half price; results matched by ``custom_id`` = chunk_id).
Each chunk's result is written in one transaction: its ``extractions`` row plus the claims
the evidence gate accepted (``raw_claims``) or refused (``rejected_claims``).

A chunk whose call fails, is refused, is cut off at ``max_tokens``, returns unusable JSON, or
errors or expires inside a batch is recorded ``errored`` with the reason and retried by the
next run; it is never counted as done with zero claims.

``client`` is an ``anthropic.AsyncAnthropic`` or a fake with the same surface:
``messages.create(**params)``, ``messages.batches.create(requests=)``,
``messages.batches.retrieve(id)`` and ``await messages.batches.results(id)`` (an async
iterator of results with ``custom_id`` and ``result.type``/``result.message``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import anthropic
import psycopg
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from vendorkb.claims import gate
from vendorkb.config import VendorkbSettings

log = logging.getLogger(__name__)

PROMPT_VERSION = "claims-v1"
MAX_TOKENS = 2000
SUPPORT_VALUES = ("supported", "partial", "not_supported", "unclear")

# Haiku 5.5 list prices (USD per million tokens); the Batches API halves both.
PRICE_IN_PER_MTOK = 0.10
PRICE_OUT_PER_MTOK = 0.50
BATCH_DISCOUNT = 0.5
# The pre-submit estimate: prompt characters / 4, plus this allowance for the reply.
EXPECTED_OUTPUT_TOKENS = 400
# Batches API caps: 100,000 requests and 256 MB per batch (bytes kept well under).
MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 200 * 1024 * 1024

_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "capability": {"type": "string"},
                    "target": _NULLABLE_STRING,
                    "support": {"type": "string", "enum": list(SUPPORT_VALUES)},
                    "conditions": _NULLABLE_STRING,
                    "evidence_quote": {"type": "string"},
                },
                "required": ["capability", "target", "support", "conditions",
                             "evidence_quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You extract capability claims from one chunk of a backup vendor's product documentation.

A capability is something the product can do: a feature, a supported workload or platform, a \
storage target, a limit, or a security control. Each claim has:
- capability: a short, normalized phrase for the capability, e.g. "immutable backups", \
"instant VM recovery", "object lock", "file-level restore". Lowercase, no product names.
- target: the workload, platform or storage the capability applies to, as a short phrase \
(e.g. "Amazon RDS", "Microsoft 365", "VMware vSphere", "S3-compatible storage"), or null when \
the capability is general.
- support: "supported" when the text says the product does it; "partial" when it does with \
notable restrictions or only in some editions or versions; "not_supported" ONLY when the text \
explicitly says it is not supported or not possible; "unclear" when the text mentions the \
capability but does not settle whether it is supported. Never infer "not_supported" from \
silence.
- conditions: prerequisites, limits, editions, versions or licensing the text attaches to the \
claim, as a short phrase, or null.
- evidence_quote: a verbatim span copied exactly from the chunk text that supports the \
claim, at most about 300 characters. Copy it character for character; do not paraphrase, \
shorten with ellipses, or join separate sentences.

The claims are about the vendor and product named before the chunk. Extract only what the \
chunk states, one claim per distinct capability and target. Skip navigation text, \
boilerplate, and steps that do not state a capability. If the chunk states no capability, \
return {"claims": []}.\
"""


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    doc_id: str
    vendor: str
    product: str | None
    version: str | None
    title: str | None
    section_path: list[str]
    text: str


@dataclass
class ExtractReport:
    chunks: int = 0
    done: int = 0
    skipped: int = 0
    errored: int = 0
    claims_accepted: int = 0
    claims_rejected: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0
    stopped_by_budget: bool = False


@dataclass
class _Outcome:
    status: Literal["done", "errored"]
    error: str | None = None
    claims: list[Any] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def cost_usd(input_tokens: int, output_tokens: int, *, batch: bool) -> float:
    cost = (input_tokens * PRICE_IN_PER_MTOK + output_tokens * PRICE_OUT_PER_MTOK) / 1e6
    return cost * BATCH_DISCOUNT if batch else cost


def build_request(chunk: ChunkRow, settings: VendorkbSettings) -> dict[str, Any]:
    """Messages API params for one chunk (the text already carries its title / trail)."""
    header = [f"Vendor: {chunk.vendor}", f"Product: {chunk.product or 'unknown'}"]
    if chunk.version:
        header.append(f"Version: {chunk.version}")
    content = "\n".join(header) + "\n\n<chunk>\n" + chunk.text + "\n</chunk>"
    return {
        "model": settings.extract_model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": "low",
                          "format": {"type": "json_schema", "schema": CLAIM_SCHEMA}},
    }


def _estimate(request: dict[str, Any], *, batch: bool) -> float:
    chars = len(request["system"]) + sum(len(m["content"]) for m in request["messages"])
    return cost_usd(chars // 4, EXPECTED_OUTPUT_TOKENS, batch=batch)


def _usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage", None)
    return (int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0))


def _parse(message: Any) -> _Outcome:
    tin, tout = _usage(message)
    stop = getattr(message, "stop_reason", None)
    if stop in ("refusal", "max_tokens"):
        return _Outcome("errored", stop, input_tokens=tin, output_tokens=tout)
    text = next((b.text for b in message.content if getattr(b, "type", None) == "text"), None)
    if text is None:
        return _Outcome("errored", "no-text", input_tokens=tin, output_tokens=tout)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return _Outcome("errored", f"invalid-json: {e}", input_tokens=tin, output_tokens=tout)
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return _Outcome("errored", "invalid-shape", input_tokens=tin, output_tokens=tout)
    return _Outcome("done", claims=data["claims"], input_tokens=tin, output_tokens=tout)


def _judge(claim: Any, text: str) -> tuple[bool, str]:
    """Validate one claim's fields, then gate its evidence quote against the chunk."""
    if not isinstance(claim, dict):
        return False, "invalid-claim"
    if not isinstance(claim.get("capability"), str) or not claim["capability"].strip():
        return False, "invalid-claim:capability"
    if claim.get("support") not in SUPPORT_VALUES:
        return False, "invalid-claim:support"
    for key in ("target", "conditions"):
        if claim.get(key) is not None and not isinstance(claim[key], str):
            return False, f"invalid-claim:{key}"
    quote = claim.get("evidence_quote")
    return gate.check(quote if isinstance(quote, str) else "", text)


def _field(claim: Any, key: str) -> str | None:
    value = claim.get(key) if isinstance(claim, dict) else None
    return value if isinstance(value, str) else (None if value is None else str(value))


async def _record(conn: psycopg.AsyncConnection, chunk: ChunkRow, outcome: _Outcome, *,
                  model: str, batch: bool, report: ExtractReport) -> float:
    """Write one chunk's result in one transaction; returns its cost.

    Tokens and cost accumulate across attempts, so a retried chunk keeps what its failed
    attempts spent (the budget counts it).
    """
    cost = cost_usd(outcome.input_tokens, outcome.output_tokens, batch=batch)
    judged = [(c, *_judge(c, chunk.text)) for c in outcome.claims]
    accepted = [(c, why) for c, ok, why in judged if ok]
    rejected = [(c, why) for c, ok, why in judged if not ok]
    async with conn.transaction():
        cur = await conn.execute(
            "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
            " status, error, n_claims, input_tokens, output_tokens, cost_usd)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (chunk_id, extractor_model, prompt_version) DO UPDATE SET"
            " status=EXCLUDED.status, error=EXCLUDED.error, n_claims=EXCLUDED.n_claims,"
            " input_tokens=extractions.input_tokens + EXCLUDED.input_tokens,"
            " output_tokens=extractions.output_tokens + EXCLUDED.output_tokens,"
            " cost_usd=extractions.cost_usd + EXCLUDED.cost_usd, created_at=now()"
            " RETURNING extraction_id",
            (uuid.uuid4(), chunk.chunk_id, model, PROMPT_VERSION, outcome.status,
             outcome.error, len(accepted), outcome.input_tokens, outcome.output_tokens, cost))
        extraction_id = (await cur.fetchone())[0]  # type: ignore[index]
        await conn.execute("DELETE FROM raw_claims WHERE extraction_id=%s", (extraction_id,))
        await conn.execute("DELETE FROM rejected_claims WHERE extraction_id=%s",
                           (extraction_id,))
        meta = (extraction_id, chunk.chunk_id, chunk.vendor, chunk.product, chunk.version)
        async with conn.cursor() as c:
            if accepted:
                await c.executemany(
                    "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor, product,"
                    " version, capability_raw, target_raw, support, conditions, evidence_quote,"
                    " gate, extractor_model, prompt_version)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(uuid.uuid4(), *meta, cl["capability"].strip(), _field(cl, "target"),
                      cl["support"], _field(cl, "conditions"), cl["evidence_quote"], why,
                      model, PROMPT_VERSION) for cl, why in accepted])
            if rejected:
                await c.executemany(
                    "INSERT INTO rejected_claims (claim_id, extraction_id, chunk_id, vendor,"
                    " product, version, capability_raw, target_raw, support, conditions,"
                    " evidence_quote, reason, extractor_model, prompt_version)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(uuid.uuid4(), *meta, _field(cl, "capability"), _field(cl, "target"),
                      _field(cl, "support"), _field(cl, "conditions"),
                      _field(cl, "evidence_quote"), why, model, PROMPT_VERSION)
                     for cl, why in rejected])
    report.input_tokens += outcome.input_tokens
    report.output_tokens += outcome.output_tokens
    report.est_cost_usd += cost
    if outcome.status == "done":
        report.done += 1
        report.claims_accepted += len(accepted)
        report.claims_rejected += len(rejected)
    else:
        report.errored += 1
        log.warning("extract: chunk %s errored: %s", chunk.chunk_id, outcome.error)
    return cost


async def _load_chunks(conn: psycopg.AsyncConnection, ids: list[str]) -> dict[str, ChunkRow]:
    cur = await conn.execute(
        "SELECT c.chunk_id::text, c.doc_id::text, d.vendor, d.product, d.version, d.title,"
        " c.section_path, c.text FROM chunks c JOIN documents d USING (doc_id)"
        " WHERE c.chunk_id = ANY(%s::uuid[])", (ids,))
    return {row[0]: ChunkRow(*row) for row in await cur.fetchall()}


async def _done_ids(conn: psycopg.AsyncConnection, ids: list[str], model: str) -> set[str]:
    cur = await conn.execute(
        "SELECT chunk_id::text FROM extractions WHERE chunk_id = ANY(%s::uuid[])"
        " AND extractor_model=%s AND prompt_version=%s AND status='done'",
        (ids, model, PROMPT_VERSION))
    return {row[0] for row in await cur.fetchall()}


async def _spent(conn: psycopg.AsyncConnection) -> float:
    cur = await conn.execute("SELECT coalesce(sum(cost_usd), 0) FROM extractions")
    return float((await cur.fetchone())[0])  # type: ignore[index]


def _api_error(e: Exception) -> _Outcome:
    return _Outcome("errored", f"{type(e).__name__}: {e}"[:500])


async def _run_sync(conn: psycopg.AsyncConnection, pending: list[ChunkRow], *, client: Any,
                    settings: VendorkbSettings, spent: float, report: ExtractReport) -> None:
    for chunk in pending:
        request = build_request(chunk, settings)
        if spent + _estimate(request, batch=False) > settings.extract_budget_usd:
            report.stopped_by_budget = True
            log.warning("extract: budget $%.2f reached (spent $%.4f); stopping",
                        settings.extract_budget_usd, spent)
            return
        try:
            message = await client.messages.create(**request)
        except anthropic.RateLimitError as e:
            outcome = _api_error(e)
        except anthropic.APIStatusError as e:
            outcome = _api_error(e)
        except anthropic.APIConnectionError as e:
            outcome = _api_error(e)
        else:
            outcome = _parse(message)
        spent += await _record(conn, chunk, outcome, model=settings.extract_model, batch=False,
                               report=report)


def _groups(items: list[tuple[ChunkRow, dict[str, Any]]]
            ) -> list[list[tuple[ChunkRow, dict[str, Any]]]]:
    """Split requests under the Batches API's count and size caps."""
    groups: list[list[tuple[ChunkRow, dict[str, Any]]]] = [[]]
    size = 0
    for item in items:
        n = len(json.dumps(item[1])) + 100
        if groups[-1] and (len(groups[-1]) >= MAX_BATCH_REQUESTS or size + n > MAX_BATCH_BYTES):
            groups.append([])
            size = 0
        groups[-1].append(item)
        size += n
    return [g for g in groups if g]


async def _run_batch(conn: psycopg.AsyncConnection, pending: list[ChunkRow], *, client: Any,
                     settings: VendorkbSettings, spent: float, report: ExtractReport,
                     poll_seconds: float) -> None:
    items: list[tuple[ChunkRow, dict[str, Any]]] = []
    estimate = spent
    for chunk in pending:
        request = build_request(chunk, settings)
        estimate += _estimate(request, batch=True)
        if estimate > settings.extract_budget_usd:
            report.stopped_by_budget = True
            log.warning("extract: budget $%.2f admits %d of %d chunks (spent $%.4f)",
                        settings.extract_budget_usd, len(items), len(pending), spent)
            break
        items.append((chunk, request))
    for group in _groups(items):
        by_id = {chunk.chunk_id: chunk for chunk, _ in group}
        batch = await client.messages.batches.create(requests=[
            Request(custom_id=chunk.chunk_id,
                    params=cast(MessageCreateParamsNonStreaming, request))
            for chunk, request in group])
        log.info("extract: submitted batch %s (%d requests)", batch.id, len(group))
        while (await client.messages.batches.retrieve(batch.id)).processing_status != "ended":
            await asyncio.sleep(poll_seconds)
        seen: set[str] = set()
        async for result in await client.messages.batches.results(batch.id):
            row = by_id.get(result.custom_id)
            if row is None or result.custom_id in seen:
                log.warning("extract: batch %s: unexpected custom_id %s", batch.id,
                            result.custom_id)
                continue
            seen.add(result.custom_id)
            kind = result.result.type
            outcome = (_parse(result.result.message) if kind == "succeeded"
                       else _Outcome("errored", f"batch:{kind}"))
            await _record(conn, row, outcome, model=settings.extract_model, batch=True,
                          report=report)
        for chunk_id in by_id.keys() - seen:
            await _record(conn, by_id[chunk_id], _Outcome("errored", "batch:no-result"),
                          model=settings.extract_model, batch=True, report=report)


async def run(conn: psycopg.AsyncConnection, chunk_ids: list[str], *, client: Any,
              settings: VendorkbSettings, mode: Literal["sync", "batch"],
              poll_seconds: float = 30) -> ExtractReport:
    ids = list(dict.fromkeys(str(uuid.UUID(str(i))) for i in chunk_ids))
    report = ExtractReport(chunks=len(ids))
    chunks = await _load_chunks(conn, ids)
    missing = [i for i in ids if i not in chunks]
    if missing:
        log.warning("extract: %d chunk ids not in the database (skipped), e.g. %s",
                    len(missing), missing[0])
    done = await _done_ids(conn, ids, settings.extract_model)
    pending = [chunks[i] for i in ids if i in chunks and i not in done]
    report.skipped = len(ids) - len(pending)
    if not pending:
        return report
    spent = await _spent(conn)
    if mode == "sync":
        await _run_sync(conn, pending, client=client, settings=settings, spent=spent,
                        report=report)
    elif mode == "batch":
        await _run_batch(conn, pending, client=client, settings=settings, spent=spent,
                         report=report, poll_seconds=poll_seconds)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return report
