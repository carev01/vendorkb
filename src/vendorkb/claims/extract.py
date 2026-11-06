"""Claim extraction: one Claude Haiku 5.5 call per chunk, every claim gated by its quote.

``run`` takes a session advisory lock (one run at a time; a second exits with
``locked=True``), loads the chunks, skips those done or failed for (model, prompt version),
checks the budget, then extracts either synchronously (``client.messages.create``, the
smoke-test path) or through the Message Batches API (half price; results matched by
``custom_id`` = chunk_id). Each chunk's result is written in one transaction: its
``extractions`` row plus the claims the evidence gate accepted (``raw_claims``, with the
matched chunk span as ``evidence_quote``) or refused (``rejected_claims``).

Paid work is ledgered in ``extract_batches`` (no foreign key to chunks, so sync deleting
chunks mid-run never erases spend): a batch row is written right after
``batches.create`` succeeds, before polling, and a batch-mode run first resumes every batch
not yet ``processed``. Sync-mode spend goes on a ``sync-<uuid>`` row. The budget's spend so
far is the ledger's sum; the pre-submit estimate reserves ``extract_max_tokens`` of output per
request, so it bounds the worst case.

A chunk whose call fails, is refused, is cut off at ``max_tokens``, returns unusable JSON, or
errors or expires inside a batch is recorded ``errored`` with the reason and retried by the
next run; after ``MAX_ATTEMPTS`` it is ``failed`` and skipped. It is never counted as done
with zero claims. A result for a chunk that sync has since deleted is skipped
(``skipped_deleted``).

``settings.extract_backend`` picks the client. "anthropic": an ``anthropic.AsyncAnthropic``
(sync or batch mode). "openai": an ``openai.AsyncOpenAI`` on any OpenAI-compatible endpoint
(OpenRouter's ``anthropic/claude-haiku-5.5:batch``, say), sync mode only, with strict
``json_schema`` output and the cost the provider reports (``usage.cost``; the token price as
fallback). Sync mode runs ``extract_concurrency`` calls at once and writes each result as it
completes; the budget check counts the estimates of the calls in flight.

The anthropic ``client`` is an ``anthropic.AsyncAnthropic`` or a fake with the same surface:
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
import openai
import psycopg
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from vendorkb.claims import gate
from vendorkb.config import VendorkbSettings

log = logging.getLogger(__name__)

PROMPT_VERSION = "claims-v1"
MAX_TOKENS = 2000  # the default of ``settings.extract_max_tokens``
# Errors that mean the reply was cut off at max_tokens ("length": OpenAI-compatible
# finish_reason; "max_tokens": Anthropic stop_reason). ``run(retry_truncated=True)`` resets
# such chunks so a re-run with a larger EXTRACT_MAX_TOKENS extracts them again.
TRUNCATION_ERRORS = ("length", "max_tokens")
MAX_ATTEMPTS = 3  # an errored chunk is 'failed' (never retried) after this many attempts
SUPPORT_VALUES = ("supported", "partial", "not_supported", "unclear")
LOCK_KEY = 0x766B63  # single-flight across extract runs (sync uses 0x766B62)
SYNC_LEDGER_PREFIX = "sync-"

# Haiku 5.5 list prices (USD per million tokens); the Batches API halves both.
PRICE_IN_PER_MTOK = 0.10
PRICE_OUT_PER_MTOK = 0.50
BATCH_DISCOUNT = 0.5
# The pre-submit estimate: prompt characters / 3 (generous for the tokenizer) in, and the
# full extract_max_tokens out (thinking is billed as output), so the cap holds in the worst case.
CHARS_PER_TOKEN = 3
# Batches API caps: 100,000 requests and 256 MB per batch (bytes kept well under).
MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 200 * 1024 * 1024
# Polling a batch survives transient errors (connection, 429, 5xx) with backoff.
MAX_POLL_FAILURES = 20
MAX_POLL_BACKOFF_SECONDS = 600.0

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
"instant VM recovery", "object lock", "file-level restore", "maximum retention period". \
Lowercase, no product names.
- target: the workload, platform or storage the capability applies to, as a short phrase \
(e.g. "Amazon RDS", "Microsoft 365", "VMware vSphere", "S3-compatible storage"), or null when \
the capability is general.
- support: "supported" when the text says the product does it; "partial" when it does with \
notable restrictions or only in some editions or versions; "not_supported" ONLY when the text \
explicitly says it is not supported or not possible; "unclear" when the text mentions the \
capability but does not settle whether it is supported. Never infer "not_supported" from \
silence.
- conditions: prerequisites, limits, editions, versions or licensing the text attaches to the \
claim, as a short phrase, or null. Put numeric limits here: "maximum retention period" with \
conditions "up to 7 years".
- evidence_quote: the full sentence, or the full table row including its row label, copied \
exactly from the chunk text, that states the capability; it must contain the words naming the \
capability and the target. Never quote a bare cell such as "Yes" or "Supported". At most \
about 300 characters. Copy it character for character; do not paraphrase, shorten with \
ellipses, or join separate sentences.

The Vendor, Product and Version lines before the chunk are context, not chunk text: never \
quote them. The claims are about that vendor and product. Extract only what the chunk \
states, one claim per distinct capability and target. Skip navigation text, boilerplate, and \
steps that do not state a capability. If the chunk states no capability, return \
{"claims": []}.\
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
    skipped: int = 0          # already done
    skipped_failed: int = 0   # already failed MAX_ATTEMPTS times
    skipped_deleted: int = 0  # not (or no longer) in chunks
    errored: int = 0          # errored this run (includes those that became failed)
    failed: int = 0           # reached MAX_ATTEMPTS this run
    claims_accepted: int = 0
    claims_rejected: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0  # cost of the results this run stored (incl. resumed batches)
    resumed_batches: int = 0
    reset_truncated: int = 0  # truncated chunks reset by retry_truncated
    stopped_by_budget: bool = False
    locked: bool = False


@dataclass
class _Outcome:
    status: Literal["done", "errored"]
    error: str | None = None
    claims: list[Any] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None  # as the provider reported it; None: priced from the tokens


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
        "max_tokens": settings.extract_max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": "low",
                          "format": {"type": "json_schema", "schema": CLAIM_SCHEMA}},
    }


def _cost(outcome: _Outcome, *, batch: bool) -> float:
    if outcome.cost is not None:
        return outcome.cost
    return cost_usd(outcome.input_tokens, outcome.output_tokens, batch=batch)


def build_openai_request(chunk: ChunkRow, settings: VendorkbSettings) -> dict[str, Any]:
    """Chat Completions params for the OpenAI-compatible backend (same prompt and schema)."""
    req = build_request(chunk, settings)
    return {
        "model": settings.extract_model,
        "max_tokens": settings.extract_max_tokens,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *req["messages"]],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "claims", "strict": True, "schema": CLAIM_SCHEMA}},
        "extra_body": {"usage": {"include": True}, "reasoning": {"effort": "low"}},
    }


def _estimate(request: dict[str, Any], *, batch: bool) -> float:
    chars = len(request["system"]) + sum(len(m["content"]) for m in request["messages"])
    return cost_usd(chars // CHARS_PER_TOKEN + 1, request["max_tokens"], batch=batch)


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
    return _parse_text(text, tin, tout)


def _parse_text(text: str | None, tin: int, tout: int, cost: float | None = None) -> _Outcome:
    if text is None:
        return _Outcome("errored", "no-text", input_tokens=tin, output_tokens=tout, cost=cost)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return _Outcome("errored", f"invalid-json: {e}", input_tokens=tin, output_tokens=tout,
                        cost=cost)
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return _Outcome("errored", "invalid-shape", input_tokens=tin, output_tokens=tout,
                        cost=cost)
    return _Outcome("done", claims=data["claims"], input_tokens=tin, output_tokens=tout,
                    cost=cost)


def _reported_cost(usage: Any) -> float | None:
    """``usage.cost`` (USD) when the provider reports it (OpenRouter with usage.include)."""
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = (getattr(usage, "model_extra", None) or {}).get("cost")
    try:
        return None if cost is None else float(cost)
    except (TypeError, ValueError):
        return None


def _parse_openai(response: Any) -> _Outcome:
    usage = getattr(response, "usage", None)
    tin = int(getattr(usage, "prompt_tokens", 0) or 0)
    tout = int(getattr(usage, "completion_tokens", 0) or 0)
    cost = _reported_cost(usage)
    choices = getattr(response, "choices", None) or []
    if not choices:
        return _Outcome("errored", "no-choices", input_tokens=tin, output_tokens=tout,
                        cost=cost)
    choice = choices[0]
    if choice.finish_reason == "length":
        return _Outcome("errored", "length", input_tokens=tin, output_tokens=tout, cost=cost)
    if getattr(choice.message, "refusal", None):
        return _Outcome("errored", "refusal", input_tokens=tin, output_tokens=tout, cost=cost)
    return _parse_text(choice.message.content, tin, tout, cost)


def _judge(claim: Any, text: str) -> gate.GateResult:
    """Validate one claim's fields, then gate its evidence quote against the chunk."""
    if not isinstance(claim, dict):
        return gate.GateResult(False, "invalid-claim")
    if not isinstance(claim.get("capability"), str) or not claim["capability"].strip():
        return gate.GateResult(False, "invalid-claim:capability")
    if claim.get("support") not in SUPPORT_VALUES:
        return gate.GateResult(False, "invalid-claim:support")
    for key in ("target", "conditions"):
        if claim.get(key) is not None and not isinstance(claim[key], str):
            return gate.GateResult(False, f"invalid-claim:{key}")
    quote = claim.get("evidence_quote")
    return gate.match(quote if isinstance(quote, str) else "", text)


def _field(claim: Any, key: str) -> str | None:
    value = claim.get(key) if isinstance(claim, dict) else None
    return value if isinstance(value, str) else (None if value is None else str(value))


async def _record(conn: psycopg.AsyncConnection, chunk_id: str, chunk: ChunkRow | None,
                  outcome: _Outcome, *, model: str, ledger_id: str, batch: bool,
                  report: ExtractReport) -> None:
    """Write one chunk's result in one transaction.

    The result's tokens and cost always count (they were paid for); in sync mode they are
    added to the run's ledger row in the same transaction. The chunk row is locked against
    deletion (FOR KEY SHARE) for the write; a chunk that is gone is skipped. Only a new or
    'errored' extraction row is (re)written, and never twice by the same batch, so a resumed
    batch is idempotent and 'done' / 'failed' rows are final.
    """
    cost = _cost(outcome, batch=batch)
    report.input_tokens += outcome.input_tokens
    report.output_tokens += outcome.output_tokens
    report.est_cost_usd += cost
    async with conn.transaction():
        if not batch:
            await conn.execute(
                "UPDATE extract_batches SET actual_cost_usd = coalesce(actual_cost_usd, 0) + %s,"
                " chunk_ids = array_append(chunk_ids, %s) WHERE batch_id=%s",
                (cost, chunk_id, ledger_id))
        cur = await conn.execute(
            "SELECT 1 FROM chunks WHERE chunk_id=%s FOR KEY SHARE", (chunk_id,))
        if chunk is None or await cur.fetchone() is None:
            report.skipped_deleted += 1
            log.warning("extract: chunk %s no longer exists; result skipped", chunk_id)
            return
        judged = [(c, _judge(c, chunk.text)) for c in outcome.claims]
        accepted = [(c, r) for c, r in judged if r.ok]
        rejected = [(c, r) for c, r in judged if not r.ok]
        cur = await conn.execute(
            "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
            " status, error, attempts, last_batch_id, n_claims, input_tokens, output_tokens,"
            " cost_usd) VALUES (%s,%s,%s,%s,%s,%s,1,%s,%s,%s,%s,%s)"
            " ON CONFLICT (chunk_id, extractor_model, prompt_version) DO UPDATE SET"
            " status = CASE WHEN EXCLUDED.status = 'errored'"
            "   AND extractions.attempts + 1 >= %s THEN 'failed' ELSE EXCLUDED.status END,"
            " error=EXCLUDED.error, n_claims=EXCLUDED.n_claims,"
            " attempts=extractions.attempts + 1, last_batch_id=EXCLUDED.last_batch_id,"
            " input_tokens=extractions.input_tokens + EXCLUDED.input_tokens,"
            " output_tokens=extractions.output_tokens + EXCLUDED.output_tokens,"
            " cost_usd=extractions.cost_usd + EXCLUDED.cost_usd, created_at=now()"
            " WHERE extractions.status = 'errored'"
            "   AND extractions.last_batch_id IS DISTINCT FROM EXCLUDED.last_batch_id"
            " RETURNING extraction_id, status",
            (uuid.uuid4(), chunk_id, model, PROMPT_VERSION, outcome.status, outcome.error,
             ledger_id, len(accepted), outcome.input_tokens, outcome.output_tokens, cost,
             MAX_ATTEMPTS))
        row = await cur.fetchone()
        if row is None:  # already applied by this batch, or final ('done' / 'failed')
            log.info("extract: chunk %s already recorded; result not reapplied", chunk_id)
            return
        extraction_id, status = row
        meta = (extraction_id, chunk.chunk_id, chunk.vendor, chunk.product, chunk.version)
        async with conn.cursor() as c:
            if accepted:
                await c.executemany(
                    "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor, product,"
                    " version, capability_raw, target_raw, support, conditions, evidence_quote,"
                    " model_quote, gate, extractor_model, prompt_version)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(uuid.uuid4(), *meta, cl["capability"].strip(), _field(cl, "target"),
                      cl["support"], _field(cl, "conditions"), r.span,
                      cl["evidence_quote"], r.reason, model, PROMPT_VERSION)
                     for cl, r in accepted])
            if rejected:
                await c.executemany(
                    "INSERT INTO rejected_claims (claim_id, extraction_id, chunk_id, vendor,"
                    " product, version, capability_raw, target_raw, support, conditions,"
                    " evidence_quote, model_quote, reason, extractor_model, prompt_version)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(uuid.uuid4(), *meta, _field(cl, "capability"), _field(cl, "target"),
                      _field(cl, "support"), _field(cl, "conditions"), None,
                      _field(cl, "evidence_quote"), r.reason, model, PROMPT_VERSION)
                     for cl, r in rejected])
    if status == "done":
        report.done += 1
        report.claims_accepted += len(accepted)
        report.claims_rejected += len(rejected)
    else:
        report.errored += 1
        if status == "failed":
            report.failed += 1
        log.warning("extract: chunk %s %s: %s", chunk_id, status, outcome.error)


async def _load_chunks(conn: psycopg.AsyncConnection, ids: list[str]) -> dict[str, ChunkRow]:
    cur = await conn.execute(
        "SELECT c.chunk_id::text, c.doc_id::text, d.vendor, d.product, d.version, d.title,"
        " c.section_path, c.text FROM chunks c JOIN documents d USING (doc_id)"
        " WHERE c.chunk_id = ANY(%s::uuid[])", (ids,))
    return {row[0]: ChunkRow(*row) for row in await cur.fetchall()}


async def _final_ids(conn: psycopg.AsyncConnection, ids: list[str],
                     model: str) -> dict[str, str]:
    """chunk_id -> 'done' | 'failed' for the chunks that must not be submitted again."""
    cur = await conn.execute(
        "SELECT chunk_id::text, status FROM extractions WHERE chunk_id = ANY(%s::uuid[])"
        " AND extractor_model=%s AND prompt_version=%s AND status IN ('done', 'failed')",
        (ids, model, PROMPT_VERSION))
    return {cid: status for cid, status in await cur.fetchall()}


async def _reset_truncated(conn: psycopg.AsyncConnection, ids: list[str], model: str) -> int:
    cur = await conn.execute(
        "UPDATE extractions SET status='errored', attempts=0 WHERE chunk_id = ANY(%s::uuid[])"
        " AND extractor_model=%s AND prompt_version=%s AND status IN ('errored', 'failed')"
        " AND error = ANY(%s)", (ids, model, PROMPT_VERSION, list(TRUNCATION_ERRORS)))
    log.info("extract: reset %d truncated chunks for a retry", cur.rowcount)
    return cur.rowcount


async def _spent(conn: psycopg.AsyncConnection) -> float:
    cur = await conn.execute(
        "SELECT coalesce(sum(coalesce(actual_cost_usd, est_cost_usd)), 0) FROM extract_batches")
    return float((await cur.fetchone())[0])  # type: ignore[index]


async def _call(client: Any, chunk: ChunkRow, settings: VendorkbSettings
                ) -> tuple[_Outcome, Exception | None]:
    """One extraction call; returns the outcome and, for a 400, the error to stop the run on
    (a request-shape bug fails every chunk the same way). Transient errors (429, 5xx,
    connection) are retried by the client (``max_retries``) before they land here."""
    try:
        if settings.extract_backend == "openai":
            return _parse_openai(await client.chat.completions.create(
                **build_openai_request(chunk, settings))), None
        return _parse(await client.messages.create(**build_request(chunk, settings))), None
    except (anthropic.BadRequestError, openai.BadRequestError) as e:
        return _api_error(e), e
    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError,
            openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
        return _api_error(e), None


async def _run_sync(conn: psycopg.AsyncConnection, pending: list[ChunkRow], *, client: Any,
                    settings: VendorkbSettings, spent: float, report: ExtractReport) -> None:
    """Up to ``extract_concurrency`` calls at once, each result written as it completes.

    Before each dispatch, spend so far plus the estimates of the calls in flight plus this
    chunk's estimate must fit the budget. Each chunk's cost goes onto the run's ledger row in
    the same transaction as its result, so the ledger is right after a crash. Writes share
    the one connection, so they are serialized. On a 400 or an unexpected error nothing more
    is dispatched; the calls in flight finish and are recorded, then the error is raised.
    """
    ledger_id = f"{SYNC_LEDGER_PREFIX}{uuid.uuid4()}"
    await conn.execute(
        "INSERT INTO extract_batches (batch_id, extractor_model, prompt_version, chunk_ids,"
        " est_cost_usd, actual_cost_usd, status) VALUES (%s,%s,%s,'{}',0,0,'submitted')",
        (ledger_id, settings.extract_model, PROMPT_VERSION))
    slots = asyncio.Semaphore(max(1, settings.extract_concurrency))
    db = asyncio.Lock()
    in_flight = 0.0
    errors: list[BaseException] = []
    tasks: list[asyncio.Task[None]] = []

    async def work(chunk: ChunkRow, estimate: float) -> None:
        nonlocal spent, in_flight
        try:
            outcome, fatal = await _call(client, chunk, settings)
            async with db:
                await _record(conn, chunk.chunk_id, chunk, outcome,
                              model=settings.extract_model, ledger_id=ledger_id, batch=False,
                              report=report)
            spent += _cost(outcome, batch=False)
            if fatal is not None:
                errors.append(fatal)
        except Exception as e:  # re-raised by the dispatcher once the others finish
            errors.append(e)
        finally:
            in_flight -= estimate
            slots.release()

    try:
        for chunk in pending:
            await slots.acquire()
            if errors:
                slots.release()
                break
            estimate = _estimate(build_request(chunk, settings), batch=False)
            if spent + in_flight + estimate > settings.extract_budget_usd:
                slots.release()
                report.stopped_by_budget = True
                log.warning("extract: budget $%.2f reached (spent $%.4f, in flight $%.4f);"
                            " stopping", settings.extract_budget_usd, spent, in_flight)
                break
            in_flight += estimate
            tasks.append(asyncio.create_task(work(chunk, estimate)))
        await asyncio.gather(*tasks)
    finally:
        await conn.execute("UPDATE extract_batches SET status='processed', ended_at=now()"
                           " WHERE batch_id=%s", (ledger_id,))
    if errors:
        raise errors[0]


def _api_error(e: Exception) -> _Outcome:
    return _Outcome("errored", f"{type(e).__name__}: {e}"[:500])


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


def _transient(e: Exception) -> bool:
    if isinstance(e, anthropic.APIConnectionError):
        return True
    return isinstance(e, anthropic.APIStatusError) and (
        e.status_code == 429 or e.status_code >= 500)


async def _wait_ended(client: Any, batch_id: str, poll_seconds: float) -> None:
    """Poll until the batch has ended, riding out transient errors with backoff."""
    failures = 0
    while True:
        try:
            status = (await client.messages.batches.retrieve(batch_id)).processing_status
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            if not _transient(e) or failures >= MAX_POLL_FAILURES:
                raise
            failures += 1
            delay = min(poll_seconds * 2 ** failures, MAX_POLL_BACKOFF_SECONDS)
            log.warning("extract: polling batch %s failed (%s); retry %d in %.0fs",
                        batch_id, type(e).__name__, failures, delay)
            await asyncio.sleep(delay)
            continue
        failures = 0
        if status == "ended":
            return
        await asyncio.sleep(poll_seconds)


def _batch_error(result: Any) -> str:
    kind = result.type
    if kind != "errored":
        return f"batch:{kind}"
    inner = getattr(getattr(result, "error", None), "error", None)
    etype = inner.get("type") if isinstance(inner, dict) else getattr(inner, "type", None)
    return f"batch:errored:{etype}" if etype else "batch:errored"


async def _process_batch(conn: psycopg.AsyncConnection, batch_id: str, chunk_ids: list[str],
                         *, client: Any, settings: VendorkbSettings, report: ExtractReport,
                         poll_seconds: float) -> None:
    """Wait for a batch, store every result, then mark it processed with its actual cost."""
    await _wait_ended(client, batch_id, poll_seconds)
    expected = set(chunk_ids)
    rows = await _load_chunks(conn, chunk_ids)
    seen: set[str] = set()
    actual = 0.0
    async for result in await client.messages.batches.results(batch_id):
        cid = result.custom_id
        if cid not in expected or cid in seen:
            log.warning("extract: batch %s: unexpected custom_id %s", batch_id, cid)
            continue
        seen.add(cid)
        outcome = (_parse(result.result.message) if result.result.type == "succeeded"
                   else _Outcome("errored", _batch_error(result.result)))
        actual += _cost(outcome, batch=True)
        await _record(conn, cid, rows.get(cid), outcome, model=settings.extract_model,
                      ledger_id=batch_id, batch=True, report=report)
    for cid in sorted(expected - seen):
        await _record(conn, cid, rows.get(cid), _Outcome("errored", "batch:no-result"),
                      model=settings.extract_model, ledger_id=batch_id, batch=True,
                      report=report)
    await conn.execute(
        "UPDATE extract_batches SET status='processed', actual_cost_usd=%s, ended_at=now()"
        " WHERE batch_id=%s", (actual, batch_id))
    log.info("extract: batch %s processed (%d results, $%.4f)", batch_id, len(seen), actual)


async def _resume_batches(conn: psycopg.AsyncConnection, *, client: Any,
                          settings: VendorkbSettings, report: ExtractReport,
                          poll_seconds: float) -> None:
    cur = await conn.execute(
        "SELECT batch_id, chunk_ids FROM extract_batches WHERE status='submitted'"
        " AND batch_id NOT LIKE %s AND extractor_model=%s AND prompt_version=%s"
        " ORDER BY submitted_at", (SYNC_LEDGER_PREFIX + "%", settings.extract_model,
                                   PROMPT_VERSION))
    for batch_id, chunk_ids in await cur.fetchall():
        log.info("extract: resuming batch %s (%d requests)", batch_id, len(chunk_ids))
        await _process_batch(conn, batch_id, list(chunk_ids), client=client,
                             settings=settings, report=report, poll_seconds=poll_seconds)
        report.resumed_batches += 1


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
        batch = await client.messages.batches.create(requests=[
            Request(custom_id=chunk.chunk_id,
                    params=cast(MessageCreateParamsNonStreaming, request))
            for chunk, request in group])
        chunk_ids = [chunk.chunk_id for chunk, _ in group]
        await conn.execute(
            "INSERT INTO extract_batches (batch_id, extractor_model, prompt_version, chunk_ids,"
            " est_cost_usd, status) VALUES (%s,%s,%s,%s,%s,'submitted')",
            (batch.id, settings.extract_model, PROMPT_VERSION, chunk_ids,
             sum(_estimate(r, batch=True) for _, r in group)))
        log.info("extract: submitted batch %s (%d requests)", batch.id, len(group))
        await _process_batch(conn, batch.id, chunk_ids, client=client, settings=settings,
                             report=report, poll_seconds=poll_seconds)


async def run(conn: psycopg.AsyncConnection, chunk_ids: list[str], *, client: Any,
              settings: VendorkbSettings, mode: Literal["sync", "batch"],
              poll_seconds: float = 30, retry_truncated: bool = False) -> ExtractReport:
    """Extract ``chunk_ids`` (see the module docstring). ``retry_truncated``: first, under the
    lock, reset the attempt counter of those chunks whose last error was a truncation
    (``TRUNCATION_ERRORS``; 'errored' or 'failed') so they are extracted again."""
    if mode not in ("sync", "batch"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "batch" and settings.extract_backend != "anthropic":
        raise ValueError("batch mode needs the anthropic backend; for the openai backend use"
                         " sync mode (OpenRouter's ':batch' model is already half price)")
    ids = list(dict.fromkeys(str(uuid.UUID(str(i))) for i in chunk_ids))
    report = ExtractReport(chunks=len(ids))
    cur = await conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
    if not (await cur.fetchone())[0]:  # type: ignore[index]
        report.locked = True
        log.warning("extract: another extract run holds the lock; nothing done")
        return report
    try:
        if retry_truncated:
            report.reset_truncated = await _reset_truncated(conn, ids, settings.extract_model)
        if mode == "batch":
            await _resume_batches(conn, client=client, settings=settings, report=report,
                                  poll_seconds=poll_seconds)
        chunks = await _load_chunks(conn, ids)
        final = await _final_ids(conn, ids, settings.extract_model)
        missing = [i for i in ids if i not in chunks and i not in final]
        if missing:
            log.warning("extract: %d chunk ids not in the database (skipped), e.g. %s",
                        len(missing), missing[0])
        report.skipped_deleted += len(missing)
        report.skipped = sum(1 for s in final.values() if s == "done")
        report.skipped_failed = sum(1 for s in final.values() if s == "failed")
        pending = [chunks[i] for i in ids if i in chunks and i not in final]
        if not pending:
            return report
        spent = await _spent(conn)
        if mode == "sync":
            await _run_sync(conn, pending, client=client, settings=settings, spent=spent,
                            report=report)
        else:
            await _run_batch(conn, pending, client=client, settings=settings, spent=spent,
                             report=report, poll_seconds=poll_seconds)
    finally:
        await conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
    return report
