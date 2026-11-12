"""The MCP server over streamable HTTP (``vendorkb mcp``): POST /mcp behind its keys (each a
bearer token, or the same value as ``X-API-Key``), GET /healthz for probes. Stateless: every
request stands alone, so replicas need no affinity.

``MCP_BEARER_TOKEN`` is required (the app refuses to start without one). With
``MCP_HIDDEN_VENDORS`` set, that key's tools read the database as ``MCP_RESTRICTED_DSN``'s role,
whose row-level security hides those vendors (``access``), behind a ``GuardedBackend``; the
start-up check proves the role hides them, or the server does not start.
``MCP_FULL_ACCESS_TOKEN`` (optional) is a second key with no vendor hidden. Each key has its own
MCP server and backend, so nothing (the evidence cache included) is shared between them.
``mcp_allowed_hosts`` is the Host-header allow-list that guards against DNS rebinding;
``host:*`` allows any port.
"""
from __future__ import annotations

import hmac
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from vendorkb.access import check_restricted, parse_vendors
from vendorkb.config import VendorkbSettings
from vendorkb.mcp_backend import Backend, DbBackend, GuardedBackend
from vendorkb.mcp_server import build_server

log = logging.getLogger(__name__)


class BearerTokenGate:
    """401 (``WWW-Authenticate: Bearer``) unless the request carries one of the keys, exactly,
    as ``Authorization: Bearer <token>`` or as ``X-API-Key: <token>`` (Copilot Studio's API-key
    authentication); either header matching is enough. The request goes to that key's app.
    Constant-time comparisons, every key compared."""

    def __init__(self, apps: Mapping[str, ASGIApp]) -> None:
        if not apps or not all(apps):
            raise ValueError("BearerTokenGate needs non-empty tokens")
        self._keys = [(f"Bearer {t}".encode(), t.encode(), app) for t, app in apps.items()]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        headers = dict(scope.get("headers") or [])
        auth, api_key = headers.get(b"authorization", b""), headers.get(b"x-api-key", b"")
        chosen = None
        for bearer, key, app in self._keys:
            matched = hmac.compare_digest(auth, bearer) | hmac.compare_digest(api_key, key)
            if matched and chosen is None:
                chosen = app
        if chosen is None:
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"www-authenticate", b"Bearer"),
                                    (b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"Unauthorized"})
            return
        await chosen(scope, receive, send)


@dataclass(frozen=True)
class Key:
    """One MCP key: its token, the vendors hidden from it and the DSN its tools read through."""
    label: str
    token: str
    hidden: tuple[str, ...]
    dsn: str


def mcp_keys(settings: VendorkbSettings) -> list[Key]:
    """The keys of the settings; ValueError on a missing or contradictory one."""
    if not settings.mcp_bearer_token:
        raise ValueError("MCP_BEARER_TOKEN is required: the MCP endpoint never runs open")
    hidden = tuple(parse_vendors(settings.mcp_hidden_vendors))
    if hidden and not settings.mcp_restricted_dsn:
        raise ValueError("MCP_HIDDEN_VENDORS needs MCP_RESTRICTED_DSN: a role that hides them")
    # a restricted DSN with no vendor to hide is unused: it is set up before the vendors are
    # named (docs/deploy/k3s.md §7c)
    keys = [Key("restricted" if hidden else "main", settings.mcp_bearer_token, hidden,
                settings.mcp_restricted_dsn if hidden else settings.vendorkb_dsn)]
    if settings.mcp_full_access_token:
        if settings.mcp_full_access_token == settings.mcp_bearer_token:
            raise ValueError("MCP_FULL_ACCESS_TOKEN must differ from MCP_BEARER_TOKEN")
        keys.append(Key("full", settings.mcp_full_access_token, (), settings.vendorkb_dsn))
    return keys


async def _health(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def build_app(settings: VendorkbSettings,
              backends: dict[str, Backend] | None = None) -> Starlette:
    """The app; ``backends`` (by key label, tests) replaces the database backends. A key's
    backend is guarded when it hides vendors."""
    keys = mcp_keys(settings)
    hosts = [h.strip() for h in settings.mcp_allowed_hosts.split(",") if h.strip()]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts,
        allowed_origins=[f"https://{h}" for h in hosts if not h.endswith(":*")])
    servers, apps = [], {}
    for key in keys:
        inner = (backends or {}).get(key.label) or DbBackend(
            settings.model_copy(update={"vendorkb_dsn": key.dsn}))
        server = build_server(GuardedBackend(inner, list(key.hidden)) if key.hidden else inner,
                              label=key.label)
        servers.append(server)
        apps[key.token] = server.streamable_http_app(
            streamable_http_path="/mcp", stateless_http=True, transport_security=security)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        for key in keys:
            if key.hidden and backends is None:  # fails closed: the server does not start
                role = await check_restricted(key.dsn, key.hidden)
                log.info("MCP key %s: role %s hides %s", key.label, role, ", ".join(key.hidden))
        async with AsyncExitStack() as stack:
            for server in servers:
                await stack.enter_async_context(server.session_manager.run())
            yield

    return Starlette(routes=[Route("/healthz", _health),
                             Route("/mcp", endpoint=BearerTokenGate(apps))],
                     lifespan=lifespan)
