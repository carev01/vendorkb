"""Usage metrics of the MCP server, served on their own port (``metrics_port``), never as a route
of the MCP app.

Every tool call counts in ``vendorkb_mcp_requests_total`` (tool, key, client, outcome) and times
in ``vendorkb_mcp_request_seconds`` (tool, client); ``answer_question`` adds its tokens, cost (at
the configured prices) and the vendors the question named. ``client`` is ``eval`` when the
request carries ``X-Vendorkb-Client: eval`` (the agent evals send it, any case), else ``user``:
``BearerTokenGate`` sets ``client_kind`` for the request. No label carries question or answer
text or a key's token: tools, key labels, clients and outcomes are fixed sets, vendors come from
scope detection's known names. The pipeline snapshot (``metrics_db.PipelineCollector``) is
registered on the same registry by ``start``.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from vendorkb.config import VendorkbSettings
from vendorkb.metrics_db import PipelineCollector

# the request's client: "eval" (X-Vendorkb-Client: eval) or "user"
client_kind: ContextVar[str] = ContextVar("vendorkb_client_kind", default="user")

SECONDS_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160)
OUTCOMES = ("ok", "error", "abstained")


def client_of(header: bytes | str | None) -> str:
    """The client an ``X-Vendorkb-Client`` header value names: ``eval`` or ``user``."""
    if isinstance(header, bytes):
        header = header.decode("latin-1")
    return "eval" if (header or "").strip().lower() == "eval" else "user"


@dataclass
class Observation:
    """One tool call; the handler may set ``outcome`` to ``abstained``."""
    outcome: str = "ok"


REGISTRY: CollectorRegistry
PIPELINE: PipelineCollector
_requests: Counter
_seconds: Histogram
_tokens: Counter
_cost: Counter
_vendors: Counter
_build_info: Gauge
_started = False


def _build() -> None:
    """A fresh registry with every usage metric on it (and a fresh pipeline collector)."""
    global REGISTRY, PIPELINE, _requests, _seconds, _tokens, _cost, _vendors, _build_info
    global _started
    REGISTRY = CollectorRegistry(auto_describe=True)
    PIPELINE = PipelineCollector()
    _started = False
    _requests = Counter("vendorkb_mcp_requests", "MCP tool calls",
                        ["tool", "key", "client", "outcome"], registry=REGISTRY)
    _seconds = Histogram("vendorkb_mcp_request_seconds", "MCP tool call duration",
                         ["tool", "client"], buckets=SECONDS_BUCKETS, registry=REGISTRY)
    _tokens = Counter("vendorkb_answer_tokens", "Answer model tokens of answer_question",
                      ["kind", "client"], registry=REGISTRY)
    _cost = Counter("vendorkb_answer_cost_usd", "Answer model cost of answer_question (USD, at "
                    "the configured prices)", ["client"], registry=REGISTRY)
    _vendors = Counter("vendorkb_questions_by_vendor", "answer_question calls per vendor named",
                       ["vendor", "client"], registry=REGISTRY)
    _build_info = Gauge("vendorkb_build_info", "The running vendorkb version", ["version"],
                        registry=REGISTRY)


_build()


def reset_for_tests() -> None:
    """Start over with a fresh registry (tests)."""
    _build()


def set_build_info(version: str) -> None:
    _build_info.clear()
    _build_info.labels(version or "unknown").set(1)


@asynccontextmanager
async def observe(tool: str, key: str) -> AsyncIterator[Observation]:
    """Count and time one tool call; an exception (cancellation too) makes it an error."""
    obs = Observation()
    client = client_kind.get()
    started = time.perf_counter()
    try:
        yield obs
    except BaseException:
        obs.outcome = "error"
        raise
    finally:
        outcome = obs.outcome if obs.outcome in OUTCOMES else "ok"
        _requests.labels(tool, key, client, outcome).inc()
        _seconds.labels(tool, client).observe(time.perf_counter() - started)


def record_answer(out: dict[str, Any], settings: VendorkbSettings | None) -> None:
    """Tokens, cost (only at a price above 0) and the named vendors of one answer; ``settings``
    None counts no cost."""
    client = client_kind.get()
    usage = out.get("usage") or {}
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    if prompt:
        _tokens.labels("prompt", client).inc(prompt)
    if completion:
        _tokens.labels("completion", client).inc(completion)
    if settings is not None:
        cost = (prompt * settings.answer_price_in_per_mtok
                + completion * settings.answer_price_out_per_mtok) / 1e6
        if cost > 0:
            _cost.labels(client).inc(cost)
    for vendor in (out.get("scope") or {}).get("vendors") or []:
        _vendors.labels(str(vendor), client).inc()


def start(settings: VendorkbSettings) -> None:
    """Serve the registry, with the pipeline collector on it, on ``settings.metrics_port``;
    nothing at port 0 or when already started. OSError when the port is busy: the bind comes
    first, so a failed one leaves the collector unregistered."""
    global _started
    if not settings.metrics_port or _started:
        return
    start_http_server(settings.metrics_port, registry=REGISTRY)
    REGISTRY.register(PIPELINE)
    _started = True
