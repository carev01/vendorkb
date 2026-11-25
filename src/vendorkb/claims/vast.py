"""The extractor on a Vast.ai serverless endpoint (deploy/vast): wake it, and keep bad hosts out.

The endpoint scales to zero. Its OpenAI-compatible proxy (``https://openai.vast.ai/<name>``,
no ``/v1``; the account's Vast API key as the Bearer token) answers 504 after about a minute
while no worker is ready, so ``wake`` sends a one-token chat request until one is answered.

Vast recruits a worker on the first request but does not move off a bad host: a worker whose
model load fails (the template's watchdog reports a slow or failed download as a failed load)
is rebooted, stopped and restarted on the same machine. While waking, ``wake`` therefore
checks the endpoint's workers between requests:

- a worker in ``error`` or ``rebooting`` is destroyed and its machine added to the worker
  group's ``machine_id notin [...]`` filter (the list lives in the worker group, so it lasts
  across runs, capped at ``MAX_EXCLUDED``);
- a worker not ready ``worker_ready_s`` after it was first seen is destroyed, its machine not
  excluded (a stopped worker whose GPU another renter took looks the same).

Destroying a worker makes the endpoint recruit another for the requests still waiting.

Scaled to zero, Vast only stops a worker (``cold_workers`` 0 notwithstanding): a stopped worker
keeps billing for its disk (about $0.03 an hour at 60 GB) and its GPU may be rented by someone
else before the next wake. ``release`` destroys the endpoint's workers once a refresh is done
with them; the next wake recruits a fresh one (a cold start of a few minutes).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

PROXY = "https://openai.vast.ai"
ROUTE = "https://run.vast.ai"
CONSOLE = "https://console.vast.ai/api/v0"
READY = frozenset({"idle", "busy", "ready", "running", "active"})
FAILED = frozenset({"error", "rebooting"})
WAITING = frozenset({502, 503, 504, 520, 522, 524})  # no worker ready yet: ask again
MAX_EXCLUDED = 50


def proxy_url(endpoint: str) -> str:
    """The OpenAI client's base URL for an endpoint (it appends /chat/completions)."""
    return f"{PROXY}/{endpoint}"


@dataclass
class WakeReport:
    seconds: float = 0.0
    attempts: int = 0
    destroyed: list[int] = field(default_factory=list)  # worker (instance) ids
    excluded: list[int] = field(default_factory=list)  # machine ids added to the filter


class VastError(RuntimeError):
    pass


class Vast:
    """The few Vast calls ``wake`` needs, on one httpx client (``transport`` for tests)."""

    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 timeout_s: float = 30.0) -> None:
        self._http = httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"},
                                       timeout=timeout_s, transport=transport)

    async def __aenter__(self) -> Vast:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._http.aclose()

    async def _json(self, method: str, url: str, **kw: Any) -> Any:
        r = await self._http.request(method, url, **kw)
        if r.status_code >= 400:
            raise VastError(f"{method} {url}: HTTP {r.status_code} {r.text[:200]}")
        return r.json() if r.content else {}

    async def workergroup(self, endpoint: str) -> dict[str, Any]:
        data = await self._json("GET", f"{CONSOLE}/autojobs/")
        groups = data.get("results", data) if isinstance(data, dict) else data
        for g in groups:
            if g.get("endpoint_name") == endpoint:
                return dict(g)
        raise VastError(f"no Vast worker group for endpoint {endpoint!r}")

    async def workers(self, endpoint_id: int) -> list[dict[str, Any]]:
        data = await self._json("POST", f"{ROUTE}/get_endpoint_workers/",
                                json={"id": endpoint_id})
        return list(data) if isinstance(data, list) else []

    async def machine_id(self, instance_id: int) -> int | None:
        data = await self._json("GET", f"{CONSOLE}/instances/{instance_id}/",
                                params={"owner": "me"})
        inst = data.get("instances") or {}
        mid = inst.get("machine_id") if isinstance(inst, dict) else None
        return int(mid) if mid is not None else None

    async def destroy(self, instance_id: int) -> None:
        await self._json("DELETE", f"{CONSOLE}/instances/{instance_id}/", json={})

    async def exclude(self, group: dict[str, Any], search_params: str,
                      machine_ids: set[int]) -> list[int]:
        """Add ``machine_ids`` to the worker group's exclusion; returns the full list."""
        current = (group.get("search_query") or {}).get("machine_id") or {}
        excluded = [int(m) for m in current.get("notin") or []]
        if "neq" in current:
            excluded.append(int(current["neq"]))
        merged = list(dict.fromkeys([*excluded, *sorted(machine_ids)]))[-MAX_EXCLUDED:]
        query = f"{search_params} machine_id notin [{','.join(map(str, merged))}]"
        await self._json("PUT", f"{CONSOLE}/autojobs/{group['id']}/", json={
            "client_id": "me", "autojob_id": group["id"], "endpoint_id": group["endpoint_id"],
            "search_params": f"{query} verified=True rentable=True rented=False"})
        group["search_query"] = {**(group.get("search_query") or {}),
                                 "machine_id": {"notin": [str(m) for m in merged]}}
        return merged


async def _probe(http: httpx.AsyncClient, url: str, model: str) -> int | None:
    """One one-token chat request: its HTTP status, or None when it timed out / failed."""
    try:
        r = await http.post(url, json={"model": model, "max_tokens": 1,
                                       "messages": [{"role": "user", "content": "ready?"}]})
    except httpx.HTTPError as e:
        log.info("vast: no answer yet (%s)", type(e).__name__)
        return None
    return r.status_code


async def _supervise(vast: Vast, group: dict[str, Any], search_params: str,
                     first_seen: dict[int, float], worker_ready_s: float,
                     report: WakeReport) -> None:
    now = time.monotonic()
    bad: set[int] = set()
    for w in await vast.workers(int(group["endpoint_id"])):
        wid, status = int(w["id"]), str(w.get("status", ""))
        first_seen.setdefault(wid, now)
        if status in READY or wid in report.destroyed:
            continue
        failed = status in FAILED
        if not failed and now - first_seen[wid] < worker_ready_s:
            continue
        machine = await vast.machine_id(wid) if failed else None
        log.warning("vast: destroying worker %d (%s, %.0f s)%s", wid, status,
                    now - first_seen[wid], f", excluding machine {machine}" if machine else "")
        await vast.destroy(wid)
        report.destroyed.append(wid)
        if machine is not None:
            bad.add(machine)
    if bad:
        await vast.exclude(group, search_params, bad)
        report.excluded.extend(sorted(bad))


async def release(endpoint: str, api_key: str, *,
                  transport: httpx.AsyncBaseTransport | None = None) -> list[int]:
    """Destroy every worker of the endpoint (call it only once no request is in flight);
    returns their ids."""
    async with Vast(api_key, transport=transport) as vast:
        group = await vast.workergroup(endpoint)
        ids = [int(w["id"]) for w in await vast.workers(int(group["endpoint_id"]))]
        for wid in ids:
            await vast.destroy(wid)
    log.info("vast: released endpoint %s (destroyed workers %s)", endpoint, ids)
    return ids


async def wake(endpoint: str, api_key: str, *, model: str, search_params: str,
               timeout_s: float, worker_ready_s: float = 900.0, poll_s: float = 15.0,
               probe_timeout_s: float = 90.0,
               transport: httpx.AsyncBaseTransport | None = None) -> WakeReport:
    """Send one-token requests to the endpoint until one is answered, replacing bad workers
    meanwhile; TimeoutError after ``timeout_s``. A 4xx other than 408/429 is a configuration
    error (a wrong key, endpoint name or path) and raises at once."""
    report = WakeReport()
    url = proxy_url(endpoint) + "/chat/completions"
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    first_seen: dict[int, float] = {}
    async with Vast(api_key, transport=transport) as vast, httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=probe_timeout_s,
            transport=transport) as http:
        group = await vast.workergroup(endpoint)
        while True:
            report.attempts += 1
            status = await _probe(http, url, model)
            if status == 200:
                report.seconds = time.monotonic() - t0
                log.info("vast: endpoint %s ready after %.0f s (%s)", endpoint, report.seconds,
                         report)
                return report
            if status is not None and status not in WAITING and status not in (408, 429):
                raise VastError(f"{url}: HTTP {status} (check VAST_ENDPOINT and the key)")
            try:
                await _supervise(vast, group, search_params, first_seen, worker_ready_s, report)
            except (VastError, httpx.HTTPError) as e:  # supervision is best effort
                log.warning("vast: could not check the workers: %s", e)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"the Vast endpoint {endpoint} did not answer in "
                                   f"{timeout_s:.0f} s ({report})")
            await asyncio.sleep(poll_s)
