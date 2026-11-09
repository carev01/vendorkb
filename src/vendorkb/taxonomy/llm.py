"""Strict-JSON model calls shared by ``taxonomy propose`` and ``taxonomy map``.

Both backends take the same strict JSON schema (every property required,
``additionalProperties: false``, nullable through ``anyOf``): ``anthropic`` through
``output_config.format``, ``openai`` (any OpenAI-compatible endpoint, e.g. OpenRouter) through
``response_format``. ``call`` returns the reply text, an error label for a reply that cannot be
used (cut off, refused, empty), and the call's cost: the provider's reported cost when it gives
one, else priced from the token counts. A 400 whose message says "rate limit" (Anthropic's
grammar-compilation limit is a 400, not a 429, so the SDK does not retry it) is retried up to
``RATE_LIMIT_RETRIES`` times with exponential back-off, then re-raised; any other 400 is
raised at once.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

import anthropic
import openai

from vendorkb.claims.extract import _reported_cost, cost_usd

log = logging.getLogger(__name__)

RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BACKOFF_S = 15.0  # doubled on each retry

Backend = Literal["anthropic", "openai"]


def object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    """A strict JSON-schema object: every property required, no others allowed."""
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def build_request(*, backend: Backend, model: str, max_tokens: int, system: str, user: str,
                  schema: dict[str, Any], name: str, effort: str = "low") -> dict[str, Any]:
    """The request for one strict-JSON call; ``name`` names the schema (openai only)."""
    if backend == "openai":
        return {
            "model": model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": schema}},
            "extra_body": {"usage": {"include": True}, "reasoning": {"effort": effort}},
        }
    return {
        "model": model, "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    }


def rate_limited(e: BaseException) -> bool:
    """A 400 that is a rate limit (retryable), not a bad request."""
    return isinstance(e, anthropic.BadRequestError | openai.BadRequestError) \
        and "rate limit" in str(e).lower()


async def _create(create: Any, request: dict[str, Any]) -> Any:
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return await create(**request)
        except (anthropic.BadRequestError, openai.BadRequestError) as e:
            if not rate_limited(e) or attempt == RATE_LIMIT_RETRIES:
                raise
            delay = RATE_LIMIT_BACKOFF_S * 2 ** attempt
            log.warning("rate-limited 400 (%s); retry %d/%d in %.0f s", e, attempt + 1,
                        RATE_LIMIT_RETRIES, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def call(client: Any, request: dict[str, Any], *, backend: Backend,
               price: tuple[float, float]) -> tuple[str | None, str | None, float]:
    """(reply text, error, cost) for one call built by ``build_request``."""
    if backend == "openai":
        resp = await _create(client.chat.completions.create, request)
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
    msg = await _create(client.messages.create, request)
    usage = getattr(msg, "usage", None)
    cost = cost_usd(int(getattr(usage, "input_tokens", 0) or 0),
                    int(getattr(usage, "output_tokens", 0) or 0), batch=False, price=price)
    if getattr(msg, "stop_reason", None) in ("refusal", "max_tokens"):
        return None, str(msg.stop_reason), cost
    text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
    return text, None if text is not None else "no-text", cost
