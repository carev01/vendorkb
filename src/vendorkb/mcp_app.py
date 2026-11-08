"""The MCP server over streamable HTTP (``vendorkb mcp``): POST /mcp behind a bearer token,
GET /healthz for probes. Stateless: every request stands alone, so replicas need no affinity.

The token is required (the app refuses to start without one). ``mcp_allowed_hosts`` is the
Host-header allow-list that guards against DNS rebinding; ``host:*`` allows any port.
"""
from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from vendorkb.config import VendorkbSettings
from vendorkb.mcp_backend import Backend, DbBackend
from vendorkb.mcp_server import build_server


class BearerTokenGate:
    """401 (``WWW-Authenticate: Bearer``) unless the request carries exactly
    ``Authorization: Bearer <token>``; constant-time comparison."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        if not token:
            raise ValueError("BearerTokenGate needs a non-empty token")
        self.app, self._expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            got = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(got, self._expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"www-authenticate", b"Bearer"),
                                        (b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await self.app(scope, receive, send)


async def _health(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def build_app(settings: VendorkbSettings, backend: Backend | None = None) -> Starlette:
    if not settings.mcp_bearer_token:
        raise ValueError("MCP_BEARER_TOKEN is required: the MCP endpoint never runs open")
    server = build_server(backend or DbBackend(settings))
    hosts = [h.strip() for h in settings.mcp_allowed_hosts.split(",") if h.strip()]
    mcp_app = server.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=hosts,
            allowed_origins=[f"https://{h}" for h in hosts if not h.endswith(":*")]))

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with server.session_manager.run():
            yield

    return Starlette(routes=[Route("/healthz", _health),
                             Route("/mcp", endpoint=BearerTokenGate(mcp_app,
                                                                    settings.mcp_bearer_token))],
                     lifespan=lifespan)
