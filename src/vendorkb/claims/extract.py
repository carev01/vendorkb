"""Claim extraction: one Claude Haiku 5.5 call per chunk, every claim gated by its quote.

``run`` takes a session advisory lock (one run at a time; a second exits with
``locked=True``), loads the chunks, skips those done or failed for (model, prompt version),
checks the budget, then extracts either synchronously (``client.messages.create``, the
smoke-test path) or through the Message Batches API (half price; results matched by
``custom_id`` = chunk_id). Each chunk's result is written in one transaction: its
``extractions`` row plus the claims the evidence gate accepted (``raw_claims``, with the
matched chunk span as ``evidence_quote``) or refused (``rejected_claims``).

Paid work is ledgered in ``extract_batches`` (no foreign key to chunks, so sync deleting
chunks mid-run never erases spend): a batch row is written under a ``pending-<uuid>`` id before
``batches.create`` and renamed to the provider's batch id once it returns, so a crash between
the two leaves the spend counted. A batch-mode run first resumes every batch not yet
``processed``; a ``pending-`` row (submitted or not, nobody knows) is never polled: its chunks
are held back, never submitted again, until a person checks the provider's batch list and
fixes or deletes the row (``unconfirmed_batches`` / ``held_chunks`` in the report). Sync-mode spend goes on a ``sync-<uuid>`` row. The budget's spend so
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
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypeGuard, cast

import anthropic
import openai
import psycopg
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from vendorkb.claims import gate
from vendorkb.config import VendorkbSettings

log = logging.getLogger(__name__)

# Each prompt revision bumps the minor version (v3 = first precision round, v3.1 = after round 1).
PROMPT_VERSION = "claims-v3.1"
PROMPT_FAMILY = "claims-v3%"  # claims-v3, claims-v3.1, ...: every revision of the v3 schema
MAX_TOKENS = 2000  # the default of ``settings.extract_max_tokens``
# Errors that mean the reply was cut off at max_tokens ("length": OpenAI-compatible
# finish_reason; "max_tokens": Anthropic stop_reason). ``run(retry_truncated=True)`` resets
# such chunks so a re-run with a larger EXTRACT_MAX_TOKENS extracts them again.
TRUNCATION_ERRORS = ("length", "max_tokens")
MAX_ATTEMPTS = 3  # an errored chunk is 'failed' (never retried) after this many attempts
SUPPORT_VALUES = ("supported", "partial", "not_supported", "unclear")
KIND_VALUES = ("capability", "licensing", "service")
AVAILABILITY_VALUES = ("generally_available", "preview", "early_access", "limited_availability",
                       "planned", "deprecated", "unstated")
DEFAULT_STATE_VALUES = ("on", "off")
BOUND_VALUES = ("max", "min", "exact")
LOCK_KEY = 0x766B63  # single-flight across extract runs (sync uses 0x766B62)
SYNC_LEDGER_PREFIX = "sync-"
PENDING_LEDGER_PREFIX = "pending-"  # a batch row written before its batches.create returned

# List prices (USD per million tokens, input / output); the Batches API halves both. Haiku's
# is the rate for prompts of 100K tokens or fewer, which every chunk prompt is. A model not
# listed here is refused by ``run`` (its spend could not be capped) unless
# EXTRACT_PRICE_IN_PER_MTOK / EXTRACT_PRICE_OUT_PER_MTOK are set.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),  # the same model through OpenRouter
    "claude-opus-5-5": (4.00, 20.00),
}
BATCH_DISCOUNT = 0.5
# Prompt caching (the system prompt is a cache breakpoint): a cache write costs 1.25x the input
# price (5-minute TTL) and a cache read a per-model fraction of it; the batch discount applies
# on top. A model not listed reads at DEFAULT_CACHE_READ_MULTIPLIER.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIERS: dict[str, float] = {
    "claude-haiku-5-5": 0.1,
    "claude-opus-5-5": 0.05,
}
DEFAULT_CACHE_READ_MULTIPLIER = 0.1
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


def _enum(values: tuple[str, ...]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _nullable_enum(values: tuple[str, ...]) -> dict[str, Any]:
    return {"anyOf": [_enum(values), {"type": "null"}]}


# One claim's fields, in the order the model writes them (and ``validate_claim`` checks them).
_CLAIM_PROPERTIES: dict[str, Any] = {
    "kind": _enum(KIND_VALUES),
    "capability": {"type": "string"},
    "target": _NULLABLE_STRING,
    "target_scope": _NULLABLE_STRING,
    "support": _enum(SUPPORT_VALUES),
    "offering": _NULLABLE_STRING,
    "edition": _NULLABLE_STRING,
    "availability": _enum(AVAILABILITY_VALUES),
    "default_state": _nullable_enum(DEFAULT_STATE_VALUES),
    "value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    "unit": _NULLABLE_STRING,
    "bound": _nullable_enum(BOUND_VALUES),
    "conditions": _NULLABLE_STRING,
    "evidence_quote": {"type": "string"},
}

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _CLAIM_PROPERTIES,
                "required": list(_CLAIM_PROPERTIES),
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_ENUMS: dict[str, tuple[tuple[str, ...], bool]] = {  # field -> (values, nullable)
    "kind": (KIND_VALUES, False),
    "support": (SUPPORT_VALUES, False),
    "availability": (AVAILABILITY_VALUES, False),
    "default_state": (DEFAULT_STATE_VALUES, True),
    "bound": (BOUND_VALUES, True),
}


def _is_number(value: Any) -> TypeGuard[int | float]:
    """A finite int or float (never a bool, inf, NaN, or an int too large for a float)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int beyond float range
        return False


def validate_claim(claim: dict[str, Any]) -> str | None:
    """``"schema:<field>"`` for the first field (in schema order) that is missing or invalid.

    Every field is required, as in ``CLAIM_SCHEMA``; ``capability`` must be non-blank. The
    evidence quote's content is the gate's job, not this check's.
    """
    for key, spec in _CLAIM_PROPERTIES.items():
        if key not in claim:
            return f"schema:{key}"
        value = claim[key]
        if key in _ENUMS:
            values, nullable = _ENUMS[key]
            ok = value in values or (nullable and value is None)
        elif key == "value":
            ok = value is None or _is_number(value)
        elif spec == _NULLABLE_STRING:
            ok = value is None or isinstance(value, str)
        else:  # capability, evidence_quote
            ok = isinstance(value, str) and (key != "capability" or bool(value.strip()))
        if not ok:
            return f"schema:{key}"
    return None


SYSTEM_PROMPT = """\
You extract claims from one chunk of a backup vendor's product documentation.

A claim is one fact the chunk states about the product. Each claim has:
- kind: one of
  - "capability": what the product itself does, or explicitly does not do: a feature, a \
supported workload or platform, a storage target, a limit, or a security control.
  - "licensing": the license metric (per user, per VM, per TB), what an edition or license \
includes, free tiers or editions, license portability.
  - "service": technical support terms, certifications and compliance attestations, release \
cadence, marketplace availability.
- capability: a short, normalized phrase for the fact, e.g. "immutable backups", "instant VM \
recovery", "object lock", "file-level restore", "maximum retention period", "per-user \
licensing", "soc 2 type ii attestation". Lowercase, no product names.
- target: the workload, platform, service or storage the claim applies to, as a short phrase \
(e.g. "Amazon RDS", "Microsoft 365", "VMware vSphere", "S3-compatible storage"), or null when \
the claim is general. Never the product's own components (its proxy, agent, appliance, \
gateway, console or a version number): name what they protect or connect to, or use null. \
It is never a standard, protocol, framework, specification or file format that a feature \
implements or supports (e.g. a consent framework, an encryption standard): name the workload, \
platform, service or storage, or use null, and mention the standard in capability or \
conditions if needed.
- target_scope: the object type or variant within the target, e.g. "users", "conditional \
access policies", "sandbox orgs", "FSx for Lustre", "RDS for Oracle", or null when the claim \
covers the whole target. Give one claim per object type: a sentence that lists six object types \
gives six claims, each with one target_scope and the same evidence_quote. Never an operation \
or capability (e.g. "file-level restore"), a setting, one of the product's own components, or \
a storage location of the product.
- support: one of
  - "supported": the text says the product does it. A capability is still "supported" when \
it needs prerequisites, setup steps, permissions, an API or service enabled, a license or a \
minimum version, or is off by default and can be turned on. Record prerequisites in \
conditions, a license tier in edition, and a stated default in default_state. A licensing or \
service fact the text states is "supported".
  - "partial": the capability itself is restricted: it works for only part of the target \
(some object types, some versions of the target, some regions or platforms), or the text names \
a functional limitation of it. Preview, beta, early access or "limited to some customers" go \
in availability, and an edition limit goes in edition; neither makes a capability partial. A \
limit that the target itself implies is not partial either (clipboard control for the Windows \
clipboard is Windows-only by nature): "partial" needs the capability to be restricted relative \
to its stated target. A prerequisite or a default setting is never a reason for "partial".
  - "not_supported": ONLY when the text explicitly says it is not supported or not possible. \
Never infer it from silence.
  - "unclear": the text names the capability but gives nothing showing the product does it. \
Text that explains how to configure, use, size, manage or monitor the capability shows that \
it does: that is "supported".
- offering: a separately named product, service or deployment form of the vendor that the \
text says the claim applies to, only when narrower than the Product line, e.g. "Veeam Data \
Cloud" (versus the self-managed software), "self-managed", "Express", or a hardware appliance \
sold as its own deployment form (e.g. "Grid appliance"); otherwise null. It is never a \
feature or backup method (e.g. snapshots, IntelliSnap), a console or component (e.g. a \
management console, proxy, gateway), or a cloud, region or platform (those go in target or \
conditions). Never the product on the Product line, an edition or a version.
- edition: the edition or license tier the text attaches to this claim, e.g. "Premium", \
"Advanced and above", only when the text says so; otherwise null. A license metric such as \
"per user" is not an edition.
- availability: "generally_available", "preview", "early_access", "limited_availability", \
"planned" or "deprecated", only when the text says so; otherwise "unstated". A plain \
present-tense statement ("Backups of Amazon RDS are supported") is "unstated", not \
"generally_available".
- default_state: "on" or "off" only when the text states the default outright ("enabled by \
default", "disabled by default", "on by default", "is off unless ..."); otherwise null. "You can \
enable compression" or "you must enable X first" says nothing about the default: null. A \
default selection in a list or form ("all items are selected by default") is not the \
capability's default_state.
- value: the number for a limit or a measure, only when the evidence_quote states it \
explicitly, e.g. 99 for "retention of up to 99 years", 10 for "an RPO of 10 minutes"; \
otherwise null. Never a version, build or step number, and never a number you work out.
- unit: the unit of value, e.g. "years", "minutes", "regions", "records per job"; null when \
value is null.
- bound: "max", "min" or "exact" for value; null when value is null. "Up to 99 years" is \
"max". "As often as every 15 minutes" is "min" (the smallest interval), not "max". A plain \
number is "exact".
- conditions: everything else the text attaches to this claim, as a short phrase, or null: \
platforms ("VMware and Hyper-V only"), prerequisites, permissions, a minimum version, a limit \
that value cannot hold, a dependency on another product or service that the text states \
outright, or the reason for a "not_supported". What the capability lets you configure or \
customize (its options) is not a condition. Never infer a dependency the text does not \
state, and do not repeat what the other fields hold. Only conditions the text ties to this \
claim; do not carry over conditions from other sentences about other things.
- evidence_quote: the full sentence, or the full table row including its row label, copied \
exactly from the chunk text, that states the claim; it must contain the words naming the \
capability and the target, and the number when value is set. Never quote a bare cell such as \
"Yes" or "Supported". At most about 300 characters. Copy it character for character; do not \
paraphrase, shorten with ellipses, or join separate sentences.

Never extract these:
- procedures and steps: how to install, configure or troubleshoot, unless a sentence states \
what the product can do;
- installation and sizing requirements of the product's own components (disk space, memory, \
CPU, ports), unless the sentence states a limit of what the product can protect or store;
- actions the user performs in another system (a workaround in Salesforce, a setting in \
Azure), even when they are needed for the product to work;
- known issues, bugs, fixed defects and odd behaviour (data converted on restore, a job that \
fails in some case). A sentence explicitly saying something is not supported is still a \
"not_supported" claim;
- prices, discounts and billing amounts: a licensing claim says what a license covers or how \
it is counted, never a price;
- announcements about documentation, blueprints, guides or training being available;
- version and build tables that list component releases, not what the product protects;
- changelog or release-note lines that only name a new setting, directive, parameter or option \
without saying what the product does with it (e.g. "Add AllowIncompleteJobs Job directive").

The Vendor, Product and Version lines before the chunk are context, not chunk text: never \
quote them. The claims are about that vendor and product. Extract only what the chunk \
states, one claim per distinct fact, target and object type. Skip navigation text and \
boilerplate. If the chunk states no claim, return {"claims": []}.\
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
    unconfirmed_batches: int = 0  # pending- ledger rows: the submit outcome is unknown
    held_chunks: int = 0      # requested chunks in those rows, never submitted again
    reset_truncated: int = 0  # truncated chunks reset by retry_truncated
    stopped_by_budget: bool = False
    locked: bool = False


@dataclass
class _Outcome:
    status: Literal["done", "errored"]
    error: str | None = None
    claims: list[Any] = field(default_factory=list)
    input_tokens: int = 0        # uncached input
    output_tokens: int = 0
    cost: float | None = None  # as the provider reported it; None: priced from the tokens
    cache_write_tokens: int = 0  # Anthropic usage.cache_creation_input_tokens
    cache_read_tokens: int = 0   # Anthropic usage.cache_read_input_tokens

    @property
    def total_input_tokens(self) -> int:
        """Every input token: uncached, written to the cache and read from it."""
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


def _price_key(model: str) -> str:
    """'anthropic/claude-opus-5.5:batch' (OpenRouter) -> 'claude-opus-5-5'."""
    return model.split("/")[-1].split(":")[0].replace(".", "-")


def price_per_mtok(model: str, settings: VendorkbSettings | None = None) -> tuple[float, float]:
    """(input, output) USD per million tokens; an explicit settings override wins."""
    if settings is not None and settings.extract_price_in_per_mtok > 0 \
            and settings.extract_price_out_per_mtok > 0:
        return settings.extract_price_in_per_mtok, settings.extract_price_out_per_mtok
    try:
        return PRICES_PER_MTOK[_price_key(model)]
    except KeyError:
        raise ValueError(f"no price known for model {model!r}: set EXTRACT_PRICE_IN_PER_MTOK"
                         " and EXTRACT_PRICE_OUT_PER_MTOK") from None


def cache_read_multiplier(model: str) -> float:
    """The cache-read price as a fraction of the input price."""
    return CACHE_READ_MULTIPLIERS.get(_price_key(model), DEFAULT_CACHE_READ_MULTIPLIER)


def cost_usd(input_tokens: int, output_tokens: int, *, batch: bool,
             price: tuple[float, float] = PRICES_PER_MTOK["claude-haiku-5-5"],
             cache_write_tokens: int = 0, cache_read_tokens: int = 0,
             cache_read_mult: float = DEFAULT_CACHE_READ_MULTIPLIER) -> float:
    """USD for one call; ``input_tokens`` is the uncached input only."""
    effective_in = (input_tokens + cache_write_tokens * CACHE_WRITE_MULTIPLIER
                    + cache_read_tokens * cache_read_mult)
    cost = (effective_in * price[0] + output_tokens * price[1]) / 1e6
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
        # One cache breakpoint on the (static) system prompt: every chunk after the first
        # reads it from the cache.
        "system": [{"type": "text", "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": settings.extract_effort,
                          "format": {"type": "json_schema", "schema": CLAIM_SCHEMA}},
    }


def _cost(outcome: _Outcome, *, batch: bool, settings: VendorkbSettings) -> float:
    if outcome.cost is not None:
        return outcome.cost
    return cost_usd(outcome.input_tokens, outcome.output_tokens, batch=batch,
                    price=price_per_mtok(settings.extract_model, settings),
                    cache_write_tokens=outcome.cache_write_tokens,
                    cache_read_tokens=outcome.cache_read_tokens,
                    cache_read_mult=cache_read_multiplier(settings.extract_model))


def build_openai_request(chunk: ChunkRow, settings: VendorkbSettings) -> dict[str, Any]:
    """Chat Completions params for the OpenAI-compatible backend (same prompt and schema)."""
    req = build_request(chunk, settings)
    return {
        "model": settings.extract_model,
        "max_tokens": settings.extract_max_tokens,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *req["messages"]],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "claims", "strict": True, "schema": CLAIM_SCHEMA}},
        "extra_body": {"usage": {"include": True}, "reasoning": {"effort": settings.extract_effort}},
    }


def _estimate(request: dict[str, Any], *, batch: bool, settings: VendorkbSettings) -> float:
    """Worst case: the whole prompt priced as plain (uncached) input, full max_tokens out."""
    system = request["system"]
    chars = (len(system) if isinstance(system, str) else sum(len(b["text"]) for b in system)) \
        + sum(len(m["content"]) for m in request["messages"])
    return cost_usd(chars // CHARS_PER_TOKEN + 1, request["max_tokens"], batch=batch,
                    price=price_per_mtok(request["model"], settings))


def _usage(message: Any) -> tuple[int, int, int, int]:
    """(uncached input, output, cache write, cache read) tokens of an Anthropic message."""
    usage = getattr(message, "usage", None)

    def tokens(key: str) -> int:
        return int(getattr(usage, key, 0) or 0)
    return (tokens("input_tokens"), tokens("output_tokens"),
            tokens("cache_creation_input_tokens"), tokens("cache_read_input_tokens"))


def _parse(message: Any) -> _Outcome:
    tin, tout, cache_write, cache_read = _usage(message)
    stop = getattr(message, "stop_reason", None)
    if stop in ("refusal", "max_tokens"):
        outcome = _Outcome("errored", stop, input_tokens=tin, output_tokens=tout)
    else:
        text = next((b.text for b in message.content
                     if getattr(b, "type", None) == "text"), None)
        outcome = _parse_text(text, tin, tout)
    outcome.cache_write_tokens, outcome.cache_read_tokens = cache_write, cache_read
    return outcome


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
    """Validate one claim's fields (``validate_claim``), then gate its evidence quote."""
    if not isinstance(claim, dict):
        return gate.GateResult(False, "schema:claim")
    invalid = validate_claim(claim)
    if invalid is not None:
        return gate.GateResult(False, invalid)
    return gate.match(claim["evidence_quote"], text)


def _no_nul(text: str) -> str:
    """``text`` without NUL bytes, which a Postgres text column cannot hold."""
    return text.replace("\x00", "")


def _clean(claim: Any) -> Any:
    """A claim with NUL removed from every string value: validated, gated and stored clean."""
    if not isinstance(claim, dict):
        return claim
    return {k: _no_nul(v) if isinstance(v, str) else v for k, v in claim.items()}


def _field(claim: Any, key: str) -> str | None:
    value = claim.get(key) if isinstance(claim, dict) else None
    return None if value is None else _no_nul(value if isinstance(value, str) else str(value))


def _number(claim: Any) -> float | None:
    """``value`` when it is a number (a rejected claim's may not be), else None."""
    value = claim.get("value") if isinstance(claim, dict) else None
    return float(value) if _is_number(value) else None


# The v3 qualifier columns, in the same order in raw_claims and rejected_claims inserts.
_V3_TEXT = ("kind", "target_scope", "offering", "edition", "availability", "default_state")
_V3_TAIL = ("unit", "bound")
V3_COLUMNS = ", ".join((*_V3_TEXT, "value", *_V3_TAIL))


def _v3_values(claim: Any) -> tuple[str | float | None, ...]:
    """The v3 column values; a null ``value`` stores ``unit`` and ``bound`` as NULL too."""
    no_value = isinstance(claim, dict) and claim.get("value") is None
    return (*(_field(claim, k) for k in _V3_TEXT), _number(claim),
            *(None if no_value else _field(claim, k) for k in _V3_TAIL))


async def _record(conn: psycopg.AsyncConnection, chunk_id: str, chunk: ChunkRow | None,
                  outcome: _Outcome, *, settings: VendorkbSettings, ledger_id: str,
                  batch: bool, report: ExtractReport) -> None:
    """Write one chunk's result in one transaction.

    The result's tokens and cost always count (they were paid for); in sync mode they are
    added to the run's ledger row in the same transaction. The chunk row is locked against
    deletion (FOR KEY SHARE) for the write; a chunk that is gone is skipped. The claims are
    gated against the text the chunk holds under that lock, not ``chunk.text``: a re-sync may
    have reused the chunk with new text since ``chunk`` was loaded, and a quote no longer in
    it is rejected. Only a new or 'errored' extraction row is (re)written, and never twice by
    the same batch, so a resumed batch is idempotent and 'done' / 'failed' rows are final.
    """
    model = settings.extract_model
    cost = _cost(outcome, batch=batch, settings=settings)
    report.input_tokens += outcome.total_input_tokens
    report.output_tokens += outcome.output_tokens
    report.est_cost_usd += cost
    async with conn.transaction():
        if not batch:
            await conn.execute(
                "UPDATE extract_batches SET actual_cost_usd = coalesce(actual_cost_usd, 0) + %s,"
                " chunk_ids = array_append(chunk_ids, %s) WHERE batch_id=%s",
                (cost, chunk_id, ledger_id))
        cur = await conn.execute(
            "SELECT text FROM chunks WHERE chunk_id=%s FOR KEY SHARE", (chunk_id,))
        locked = await cur.fetchone()
        if chunk is None or locked is None:
            report.skipped_deleted += 1
            log.warning("extract: chunk %s no longer exists; result skipped", chunk_id)
            return
        judged = [(c, _judge(c, locked[0])) for c in map(_clean, outcome.claims)]
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
            (uuid.uuid4(), chunk_id, model, PROMPT_VERSION, outcome.status,
             None if outcome.error is None else _no_nul(outcome.error),
             ledger_id, len(accepted), outcome.total_input_tokens, outcome.output_tokens, cost,
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
                    f" model_quote, gate, extractor_model, prompt_version, {V3_COLUMNS})"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                    " %s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(uuid.uuid4(), *meta, cl["capability"].strip(), _field(cl, "target"),
                      cl["support"], _field(cl, "conditions"), r.span,
                      cl["evidence_quote"], _no_nul(r.reason), model, PROMPT_VERSION,
                      *_v3_values(cl))
                     for cl, r in accepted])
            if rejected:
                await c.executemany(
                    "INSERT INTO rejected_claims (claim_id, extraction_id, chunk_id, vendor,"
                    " product, version, capability_raw, target_raw, support, conditions,"
                    " evidence_quote, model_quote, reason, extractor_model, prompt_version,"
                    f" {V3_COLUMNS})"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                    " %s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(uuid.uuid4(), *meta, _field(cl, "capability"), _field(cl, "target"),
                      _field(cl, "support"), _field(cl, "conditions"), None,
                      _field(cl, "evidence_quote"), _no_nul(r.reason), model, PROMPT_VERSION,
                      *_v3_values(cl))
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


async def unextracted_ids(conn: psycopg.AsyncConnection, *, vendor: str | None = None,
                          limit: int | None = None) -> list[str]:
    """Chunks with no final ('done' or 'failed') extraction of the v3 prompt family by any
    model: new and re-synced chunks, and those still 'errored' (``run`` retries them). A chunk
    another model already extracted (the Opus teacher set, say) is not extracted again. A
    superseded chunk (an earlier version of a page, kept for its claims) is never extracted."""
    cur = await conn.execute(
        "SELECT c.chunk_id::text FROM chunks c JOIN documents d USING (doc_id)"
        " WHERE c.superseded_at IS NULL AND NOT EXISTS (SELECT 1 FROM extractions e"
        " WHERE e.chunk_id = c.chunk_id"
        " AND e.prompt_version LIKE %(family)s AND e.status IN ('done', 'failed'))"
        " AND (%(vendor)s::text IS NULL OR d.vendor = %(vendor)s)"
        " ORDER BY c.chunk_id LIMIT %(limit)s",
        {"family": PROMPT_FAMILY, "vendor": vendor, "limit": limit})
    return [r[0] for r in await cur.fetchall()]


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
    is_openai = settings.extract_backend == "openai"
    try:
        reply = (await client.chat.completions.create(**build_openai_request(chunk, settings))
                 if is_openai else
                 await client.messages.create(**build_request(chunk, settings)))
    except (anthropic.BadRequestError, openai.BadRequestError) as e:
        return _api_error(e), e
    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError,
            openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
        return _api_error(e), None
    try:
        return (_parse_openai(reply) if is_openai else _parse(reply)), None
    except Exception as e:  # noqa: BLE001 - a paid reply of an unexpected shape: its cost is
        # recorded with the chunk errored, and the error still stops the run
        return _unparsed(reply, is_openai, e), e


def _unparsed(reply: Any, is_openai: bool, error: Exception) -> _Outcome:
    """An errored outcome carrying whatever usage a reply the parser choked on reports."""
    outcome = _Outcome("errored", f"parse:{type(error).__name__}")
    usage = getattr(reply, "usage", None)
    if is_openai:
        outcome.input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        outcome.output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        outcome.cost = _reported_cost(usage)
    elif usage is not None:
        (outcome.input_tokens, outcome.output_tokens, outcome.cache_write_tokens,
         outcome.cache_read_tokens) = _usage(reply)
    return outcome


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
                              settings=settings, ledger_id=ledger_id, batch=False,
                              report=report)
            spent += _cost(outcome, batch=False, settings=settings)
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
            estimate = _estimate(build_request(chunk, settings), batch=False, settings=settings)
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
        actual += _cost(outcome, batch=True, settings=settings)
        await _record(conn, cid, rows.get(cid), outcome, settings=settings,
                      ledger_id=batch_id, batch=True, report=report)
    for cid in sorted(expected - seen):
        await _record(conn, cid, rows.get(cid), _Outcome("errored", "batch:no-result"),
                      settings=settings, ledger_id=batch_id, batch=True,
                      report=report)
    await conn.execute(
        "UPDATE extract_batches SET status='processed', actual_cost_usd=%s, ended_at=now()"
        " WHERE batch_id=%s", (actual, batch_id))
    log.info("extract: batch %s processed (%d results, $%.4f)", batch_id, len(seen), actual)


async def _held_by_unconfirmed(conn: psycopg.AsyncConnection, settings: VendorkbSettings,
                               report: ExtractReport) -> set[str]:
    """The chunk ids of ``pending-`` batch rows: a run crashed between writing the row and
    learning the batch id, so the batch may exist (and be paid for) at the provider."""
    cur = await conn.execute(
        "SELECT batch_id, chunk_ids FROM extract_batches WHERE status='submitted'"
        " AND batch_id LIKE %s AND extractor_model=%s AND prompt_version=%s",
        (PENDING_LEDGER_PREFIX + "%", settings.extract_model, PROMPT_VERSION))
    rows = await cur.fetchall()
    for batch_id, chunk_ids in rows:
        log.warning("extract: batch row %s (%d chunks) never learned its batch id: check the"
                    " provider's batch list, then set its batch_id or delete it; its chunks"
                    " are held back", batch_id, len(chunk_ids))
    report.unconfirmed_batches = len(rows)
    return {str(c) for _, chunk_ids in rows for c in chunk_ids}


async def _resume_batches(conn: psycopg.AsyncConnection, *, client: Any,
                          settings: VendorkbSettings, report: ExtractReport,
                          poll_seconds: float) -> None:
    cur = await conn.execute(
        "SELECT batch_id, chunk_ids FROM extract_batches WHERE status='submitted'"
        " AND batch_id NOT LIKE %s AND batch_id NOT LIKE %s AND extractor_model=%s"
        " AND prompt_version=%s ORDER BY submitted_at",
        (SYNC_LEDGER_PREFIX + "%", PENDING_LEDGER_PREFIX + "%", settings.extract_model,
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
        estimate += _estimate(request, batch=True, settings=settings)
        if estimate > settings.extract_budget_usd:
            report.stopped_by_budget = True
            log.warning("extract: budget $%.2f admits %d of %d chunks (spent $%.4f)",
                        settings.extract_budget_usd, len(items), len(pending), spent)
            break
        items.append((chunk, request))
    for group in _groups(items):
        chunk_ids = [chunk.chunk_id for chunk, _ in group]
        pending_id = f"{PENDING_LEDGER_PREFIX}{uuid.uuid4()}"
        await conn.execute(  # ledger first: a crash after the submit must not lose the spend
            "INSERT INTO extract_batches (batch_id, extractor_model, prompt_version, chunk_ids,"
            " est_cost_usd, status) VALUES (%s,%s,%s,%s,%s,'submitted')",
            (pending_id, settings.extract_model, PROMPT_VERSION, chunk_ids,
             sum(_estimate(r, batch=True, settings=settings) for _, r in group)))
        try:
            batch = await client.messages.batches.create(requests=[
                Request(custom_id=chunk.chunk_id,
                        params=cast(MessageCreateParamsNonStreaming, request))
                for chunk, request in group])
        except anthropic.APIStatusError as e:
            if e.status_code < 500:  # refused outright: no batch exists, nothing is owed
                await conn.execute("DELETE FROM extract_batches WHERE batch_id=%s",
                                   (pending_id,))
            raise  # a 5xx or a dropped connection may still have created it: the row stays
        await conn.execute("UPDATE extract_batches SET batch_id=%s WHERE batch_id=%s",
                           (batch.id, pending_id))
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
    price_per_mtok(settings.extract_model, settings)  # refuse a model whose spend can't be capped
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
        held = await _held_by_unconfirmed(conn, settings, report)
        chunks = await _load_chunks(conn, ids)
        final = await _final_ids(conn, ids, settings.extract_model)
        missing = [i for i in ids if i not in chunks and i not in final]
        if missing:
            log.warning("extract: %d chunk ids not in the database (skipped), e.g. %s",
                        len(missing), missing[0])
        report.skipped_deleted += len(missing)
        report.skipped = sum(1 for s in final.values() if s == "done")
        report.skipped_failed = sum(1 for s in final.values() if s == "failed")
        report.held_chunks = sum(1 for i in ids if i in held and i not in final)
        pending = [chunks[i] for i in ids if i in chunks and i not in final and i not in held]
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
