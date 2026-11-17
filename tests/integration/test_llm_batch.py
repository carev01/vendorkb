"""llm_batch (the Message Batches runner for map and support) against a real Postgres."""
import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from vendorkb import llm_batch
from vendorkb.db import connect, init_schema

PRICE = (1.0, 5.0)  # USD per MTok in / out


def _msg(text, *, stop_reason="end_turn", tokens=(100, 10)):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]))


class Batches:
    """``outcome(custom_id)`` -> ("succeeded", message) | ("errored",) | ("expired",) | None
    (no result). Batches end when ``ended`` holds their id."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.created: list[list] = []
        self.ended: set[str] = set()
        self.refuse: Exception | None = None

    async def create(self, *, requests):
        if self.refuse is not None:
            raise self.refuse
        self.created.append(list(requests))
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}")

    async def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended" if batch_id in self.ended
                               else "in_progress")

    async def results(self, batch_id):
        requests = self.created[int(batch_id.split("_")[1]) - 1]

        async def gen():
            for r in requests:
                out = self.outcome(r["custom_id"])
                if out is None:
                    continue
                result = SimpleNamespace(type=out[0])
                if out[0] == "succeeded":
                    result.message = out[1]
                yield SimpleNamespace(custom_id=r["custom_id"], result=result)
        return gen()


def _client(batches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _item(cid, *, max_tokens=100):
    return llm_batch.Item(cid, {"model": "m", "max_tokens": max_tokens, "system": "s",
                                "messages": [{"role": "user", "content": f"about {cid}"}]},
                          {"phrase": cid})


@pytest.fixture
async def conn(vkb_dsn):
    c = await connect(vkb_dsn)
    await c.execute("DROP TABLE IF EXISTS llm_batches, llm_batch_items CASCADE")
    await init_schema(c, embed_dim=8)
    yield c
    await c.close()


async def _items(conn):
    cur = await conn.execute("SELECT custom_id, batch_id, attempts, status, payload, last_error"
                             " FROM llm_batch_items ORDER BY custom_id")
    return {r[0]: r[1:] for r in await cur.fetchall()}


async def test_submit_then_collect_stores_results_once_at_the_batch_price(conn):
    stored = []

    async def handle(r):
        stored.append((r.custom_id, r.payload["phrase"], r.text, r.cost))
        return None

    batches = Batches(lambda cid: ("succeeded", _msg(json.dumps({"id": cid}))))
    client, report = _client(batches), llm_batch.BatchReport()
    await llm_batch.submit(conn, client, kind="k", model="m", items=[_item("a"), _item("b")],
                           price=PRICE, max_usd=1.0, report=report)
    assert (report.submitted, report.batches) == (2, 1)
    assert [r["custom_id"] for r in batches.created[0]] == ["a", "b"]
    (row,) = await (await conn.execute("SELECT batch_id, kind, custom_ids, status, est_cost_usd"
                                       " FROM llm_batches")).fetchall()
    assert row[:4] == ("msgbatch_1", "k", ["a", "b"], "submitted") and row[4] > 0
    assert await llm_batch.held_ids(conn, "k") == {"a", "b"}
    assert await llm_batch.in_flight(conn, ["k"]) == 1

    await llm_batch.collect(conn, client, kind="k", price=PRICE, handle=handle, report=report)
    assert stored == [] and report.collected == 0  # not ended yet: nothing happens
    batches.ended.add("msgbatch_1")
    await llm_batch.collect(conn, client, kind="k", price=PRICE, handle=handle, report=report)
    # 100 in at $1/MTok + 10 out at $5/MTok, halved
    cost = (100 * 1.0 + 10 * 5.0) / 1e6 * 0.5
    assert stored == [("a", "a", '{"id": "a"}', cost), ("b", "b", '{"id": "b"}', cost)]
    assert (report.collected, report.used, report.spent_usd) == (1, 2, pytest.approx(2 * cost))
    assert {k: v[2] for k, v in (await _items(conn)).items()} == {"a": "done", "b": "done"}
    assert await llm_batch.held_ids(conn, "k") == set()
    await llm_batch.collect(conn, client, kind="k", price=PRICE, handle=handle, report=report)
    assert len(stored) == 2  # processed once


async def test_unusable_results_are_due_again_until_the_attempt_cap(conn):
    outcomes = {"a": ("errored",), "b": ("expired",), "c": None,  # no result at all
                "d": ("succeeded", _msg("x", stop_reason="max_tokens")),
                "e": ("succeeded", _msg("not json"))}

    async def handle(r):
        return "unusable reply"  # e's parser refuses it

    batches = Batches(lambda cid: outcomes[cid])
    client = _client(batches)
    for attempt in range(1, llm_batch.MAX_ATTEMPTS + 1):
        report = llm_batch.BatchReport()
        due = [cid for cid in outcomes if cid not in await llm_batch.held_ids(conn, "k")]
        assert due == list(outcomes)
        await llm_batch.submit(conn, client, kind="k", model="m",
                               items=[_item(c) for c in due], price=PRICE, max_usd=1.0,
                               report=report)
        batches.ended.add(f"msgbatch_{attempt}")
        await llm_batch.collect(conn, client, kind="k", price=PRICE, handle=handle,
                                report=report)
        assert report.errored == 5
        assert report.failed == (5 if attempt == llm_batch.MAX_ATTEMPTS else 0)
    rows = await _items(conn)
    assert {k: (v[1], v[2]) for k, v in rows.items()} == {
        c: (llm_batch.MAX_ATTEMPTS, "failed") for c in outcomes}
    assert rows["a"][4] == "batch:errored" and rows["b"][4] == "batch:expired"
    assert rows["c"][4] == "batch:no-result" and rows["d"][4] == "max_tokens"
    assert rows["e"][4] == "unusable reply"
    assert await llm_batch.held_ids(conn, "k") == set(outcomes)  # failed: never asked again
    assert await llm_batch.reset_failed(conn, "k") == 5
    assert await llm_batch.held_ids(conn, "k") == set()


async def test_the_budget_reserves_each_request_at_its_full_output(conn):
    one = llm_batch.reserve_usd(_item("a", max_tokens=1000).params, PRICE)
    assert one == pytest.approx(((len(json.dumps("s")) + len(json.dumps(
        [{"role": "user", "content": "about a"}]))) // 3 + 1 + 1000 * 5) / 1e6 * 0.5)
    report = llm_batch.BatchReport()
    batches = Batches(lambda cid: None)
    await llm_batch.submit(conn, _client(batches), kind="k", model="m",
                           items=[_item(c, max_tokens=1000) for c in "abc"], price=PRICE,
                           max_usd=one * 2.5, report=report)
    assert [r["custom_id"] for r in batches.created[0]] == ["a", "b"]
    assert (report.submitted, report.skipped_by_budget) == (2, 1)
    assert report.reserved_usd == pytest.approx(one * 2)


async def test_a_refused_create_leaves_no_row_and_a_crash_leaves_a_held_pending_row(conn):
    batches = Batches(lambda cid: None)
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches")
    batches.refuse = anthropic.BadRequestError(
        "bad", response=httpx.Response(400, request=request), body=None)
    with pytest.raises(anthropic.BadRequestError):
        await llm_batch.submit(conn, _client(batches), kind="k", model="m", items=[_item("a")],
                               price=PRICE, max_usd=1.0, report=llm_batch.BatchReport())
    assert await (await conn.execute("SELECT count(*) FROM llm_batches")).fetchone() == (0,)

    batches.refuse = anthropic.APIConnectionError(request=request)  # it may exist
    with pytest.raises(anthropic.APIConnectionError):
        await llm_batch.submit(conn, _client(batches), kind="k", model="m", items=[_item("a")],
                               price=PRICE, max_usd=1.0, report=llm_batch.BatchReport())
    assert await llm_batch.held_ids(conn, "k") == {"a"}  # held: never asked twice
    report = llm_batch.BatchReport()

    async def handle(r):
        raise AssertionError("a pending row has no batch to collect")

    await llm_batch.collect(conn, _client(batches), kind="k", price=PRICE, handle=handle,
                            report=report)
    assert report.unconfirmed == 1 and report.collected == 0


async def test_kinds_are_separate_and_wait_ended_polls_until_every_batch_ended(conn):
    batches = Batches(lambda cid: None)
    client = _client(batches)
    for kind in ("k1", "k2"):
        await llm_batch.submit(conn, client, kind=kind, model="m", items=[_item("a")],
                               price=PRICE, max_usd=1.0, report=llm_batch.BatchReport())
    assert await llm_batch.held_ids(conn, "k1") == {"a"} == await llm_batch.held_ids(conn, "k2")
    polls = []
    real_retrieve = batches.retrieve

    async def retrieve(batch_id):
        polls.append(batch_id)
        if len(polls) >= 3:
            batches.ended.update({"msgbatch_1", "msgbatch_2"})
        return await real_retrieve(batch_id)

    batches.retrieve = retrieve
    await llm_batch.wait_ended(conn, client, ["k1", "k2"], poll_seconds=0)
    assert len(polls) >= 3
