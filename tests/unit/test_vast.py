"""claims.vast.wake against a fake Vast API (proxy, route, console)."""
import json

import httpx
import pytest

from vendorkb.claims import vast

GROUP = {"id": 7, "endpoint_id": 44, "endpoint_name": "ex",
         "search_query": {"machine_id": {"notin": ["111"]}}}


class FakeVast:
    """Answers the proxy with ``proxy`` statuses in turn; workers from ``workers`` in turn."""

    def __init__(self, proxy, workers=(), machines=None, workers_status=200):
        self.proxy = list(proxy)
        self.workers = list(workers)
        self.machines = machines or {}
        self.workers_status = workers_status
        self.calls = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, url, body, request.headers.get("authorization")))
        if url.startswith(vast.PROXY):
            status = self.proxy.pop(0) if len(self.proxy) > 1 else self.proxy[0]
            return httpx.Response(status, json={"choices": []} if status == 200 else None)
        if url == f"{vast.CONSOLE}/autojobs/":
            return httpx.Response(200, json={"results": [dict(GROUP), {"id": 8,
                                                         "endpoint_name": "other"}]})
        if url == f"{vast.ROUTE}/get_endpoint_workers/":
            if self.workers_status != 200:
                return httpx.Response(self.workers_status)
            return httpx.Response(200, json=self.workers.pop(0) if len(self.workers) > 1
                                  else (self.workers[0] if self.workers else []))
        if request.method == "GET" and "/instances/" in url:
            wid = int(url.split("/instances/")[1].split("/")[0])
            return httpx.Response(200, json={"instances": {"machine_id": self.machines[wid]}})
        if request.method in ("DELETE", "PUT"):
            return httpx.Response(200, json={"success": True})
        return httpx.Response(404)

    def of(self, method, fragment=""):
        return [c for c in self.calls if c[0] == method and fragment in c[1]]


async def wake(fake, **kw):
    kw = {"model": "m", "search_params": "num_gpus=1", "timeout_s": 60, "poll_s": 0, **kw}
    return await vast.wake("ex", "vk", transport=httpx.MockTransport(fake), **kw)


async def test_retries_through_the_cold_start_and_sends_the_vast_key():
    fake = FakeVast([504, 504, 200], workers=[[{"id": 1, "status": "loading"}]])
    report = await wake(fake)
    probes = fake.of("POST", vast.PROXY)
    assert len(probes) == report.attempts == 3
    assert probes[0][1] == "https://openai.vast.ai/ex/chat/completions"  # no /v1
    assert probes[0][2]["max_tokens"] == 1 and probes[0][2]["model"] == "m"
    assert {c[3] for c in fake.calls} == {"Bearer vk"}
    assert report.destroyed == [] and not fake.of("DELETE")


async def test_a_failed_worker_is_destroyed_and_its_machine_excluded():
    fake = FakeVast([504, 504, 200],
                    workers=[[{"id": 5, "status": "rebooting"}], [{"id": 6, "status": "idle"}]],
                    machines={5: 222})
    report = await wake(fake)
    assert report.destroyed == [5] and report.excluded == [222]
    assert [c[1] for c in fake.of("DELETE")] == [f"{vast.CONSOLE}/instances/5/"]
    (put,) = fake.of("PUT")
    assert put[1] == f"{vast.CONSOLE}/autojobs/7/"
    assert put[2]["endpoint_id"] == 44
    # the earlier exclusion is kept, the new machine added, after the base filter
    assert put[2]["search_params"].startswith("num_gpus=1 machine_id notin [111,222]")


async def test_a_worker_that_never_gets_ready_is_destroyed_but_not_excluded():
    fake = FakeVast([504, 200], workers=[[{"id": 9, "status": "model_loading"}]])
    report = await wake(fake, worker_ready_s=0)
    assert report.destroyed == [9] and report.excluded == []
    assert not fake.of("PUT") and not fake.of("GET", "/instances/")


async def test_a_ready_worker_is_left_alone_and_a_destroyed_one_not_destroyed_twice():
    fake = FakeVast([504, 504, 504, 200],
                    workers=[[{"id": 3, "status": "error"}], [{"id": 3, "status": "error"},
                                                              {"id": 4, "status": "idle"}]],
                    machines={3: 333})
    report = await wake(fake, worker_ready_s=0)
    assert report.destroyed == [3]


async def test_supervision_failures_do_not_stop_the_wake():
    fake = FakeVast([504, 200], workers_status=500)
    report = await wake(fake)
    assert report.attempts == 2


@pytest.mark.parametrize("status", [401, 403, 404])
async def test_a_configuration_error_fails_at_once(status):
    fake = FakeVast([status])
    with pytest.raises(vast.VastError, match=f"HTTP {status}"):
        await wake(fake)
    assert len(fake.of("POST", vast.PROXY)) == 1


async def test_times_out():
    with pytest.raises(TimeoutError, match="did not answer"):
        await wake(FakeVast([504]), timeout_s=0)


async def test_an_unknown_endpoint_is_an_error():
    fake = FakeVast([200])
    with pytest.raises(vast.VastError, match="no Vast worker group"):
        await vast.wake("missing", "vk", model="m", search_params="x", timeout_s=1,
                        transport=httpx.MockTransport(fake))


async def test_release_destroys_every_worker_of_the_endpoint():
    fake = FakeVast([200], workers=[[{"id": 5, "status": "idle"}, {"id": 6, "status": "stopped"}]])
    ids = await vast.release("ex", "vk", transport=httpx.MockTransport(fake))
    assert ids == [5, 6]
    assert [c[1] for c in fake.of("DELETE")] == [f"{vast.CONSOLE}/instances/5/",
                                                f"{vast.CONSOLE}/instances/6/"]
    assert fake.of("POST", vast.ROUTE)[0][2] == {"id": 44}  # this endpoint's workers
    assert not fake.of("POST", vast.PROXY)  # never wakes it


async def test_release_of_an_empty_endpoint_destroys_nothing():
    fake = FakeVast([200])
    assert await vast.release("ex", "vk", transport=httpx.MockTransport(fake)) == []
    assert not fake.of("DELETE")
