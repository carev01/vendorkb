import json
import uuid
from types import SimpleNamespace

import anthropic
import httpx
from pgvector import Vector

from vendorkb.claims import extract
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema

TEXT = ("Protecting Databases / Amazon RDS\n\n"
        "Backups of Amazon RDS for PostgreSQL instances are supported with immutable "
        "snapshots. Cross-region copy requires the Enterprise edition.")

GOOD = {"capability": "immutable snapshots", "target": "Amazon RDS for PostgreSQL",
        "support": "supported", "conditions": None,
        "evidence_quote": "Backups of Amazon RDS for PostgreSQL instances are supported"}
PARTIAL = {"capability": "cross-region copy", "target": None, "support": "partial",
           "conditions": "Enterprise edition",
           "evidence_quote": "cross-region   copy requires the ENTERPRISE edition."}
INVENTED = {"capability": "tape backup", "target": None, "support": "supported",
            "conditions": None, "evidence_quote": "Tape backup is fully supported for LTO-9."}


def _settings(**kw) -> VendorkbSettings:
    return VendorkbSettings(_env_file=None, **kw)


def _message(payload, *, stop_reason="end_turn", tokens=(1000, 200)):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]))


class FakeBatches:
    def __init__(self, outcomes):
        self.outcomes = outcomes  # chunk_id -> ("succeeded", message) | ("errored",) | ...
        self.created: list[list] = []
        self.retrieves = 0
        self.retrieve_errors: list[Exception] = []  # raised, in order, before any status
        self.on_retrieve = None  # optional async hook run on each successful retrieve
        self.extra_results: list[tuple[str, str]] = []  # (custom_id, kind) appended
        self.results_fail_after: int | None = None  # next results() dies after N items

    async def create(self, *, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id=f"batch_{len(self.created)}", processing_status="in_progress")

    async def retrieve(self, message_batch_id):
        if self.retrieve_errors:
            raise self.retrieve_errors.pop(0)
        if self.on_retrieve is not None:
            await self.on_retrieve()
        self.retrieves += 1
        status = "ended" if self.retrieves % 2 == 0 else "in_progress"
        return SimpleNamespace(id=message_batch_id, processing_status=status)

    async def results(self, message_batch_id):
        requests = self.created[int(message_batch_id.split("_")[1]) - 1]
        pairs = [(r["custom_id"], self.outcomes[r["custom_id"]]) for r in reversed(requests)]
        pairs += [(cid, (kind,)) for cid, kind in self.extra_results]
        fail_after, self.results_fail_after = self.results_fail_after, None

        async def gen():
            for n, (custom_id, (kind, *rest)) in enumerate(pairs):  # any order: keyed by id
                if n == fail_after:
                    raise RuntimeError("connection dropped mid-results")
                result = SimpleNamespace(type=kind)
                if kind == "succeeded":
                    result.message = rest[0]
                elif kind == "errored":
                    result.error = SimpleNamespace(
                        type="error", error=SimpleNamespace(type="invalid_request_error"))
                yield SimpleNamespace(custom_id=custom_id, result=result)
        return gen()


class FakeMessages:
    def __init__(self, replies=None, batches=None):
        self.replies = replies or {}  # chunk text marker -> message | Exception
        self.calls: list[dict] = []
        self.batches = batches or FakeBatches({})

    async def create(self, **params):
        self.calls.append(params)
        body = params["messages"][0]["content"]
        for marker, reply in self.replies.items():
            if marker in body:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        return _message({"claims": []})


class FakeClient:
    def __init__(self, **kw):
        self.messages = FakeMessages(**kw)


async def _fresh(conn) -> None:
    await conn.execute("DROP TABLE IF EXISTS capabilities, raw_claims, rejected_claims, "
                       "extractions, extract_batches, chunks, documents CASCADE")
    await init_schema(conn, embed_dim=768)


async def _doc(conn, *, vendor="rubrik", product="Rubrik Security Cloud",
               texts=(TEXT,)) -> list[str]:
    doc_id = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s,%s,%s,'S','Doc','http://u',%s, now())",
        (doc_id, vendor, product, doc_id))
    ids = []
    for n, text in enumerate(texts):
        cid = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
            " embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (cid, doc_id, ["Protecting Databases", "Amazon RDS"], n, text, 50,
             Vector([0.1] * 768)))
        ids.append(cid)
    return ids


async def _rows(conn, sql, *args):
    cur = await conn.execute(sql, args)
    return await cur.fetchall()


def test_schema_and_request_shape():
    item = extract.CLAIM_SCHEMA["properties"]["claims"]["items"]
    assert extract.CLAIM_SCHEMA["additionalProperties"] is False
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"]) == {
        "capability", "target", "support", "conditions", "evidence_quote"}
    assert item["properties"]["support"]["enum"] == [
        "supported", "partial", "not_supported", "unclear"]
    chunk = extract.ChunkRow(chunk_id="c1", doc_id="d1", vendor="rubrik", product=None,
                             version=None, title="T", section_path=["A"], text=TEXT)
    req = extract.build_request(chunk, _settings())
    assert req["model"] == "claude-haiku-5-5" and req["max_tokens"] == 2000
    assert req["system"] == extract.SYSTEM_PROMPT
    assert req["output_config"] == {
        "effort": "low", "format": {"type": "json_schema", "schema": extract.CLAIM_SCHEMA}}
    assert "thinking" not in req and "temperature" not in req
    assert req["messages"][0]["role"] == "user" and TEXT in req["messages"][0]["content"]
    assert "rubrik" in req["messages"][0]["content"]
    assert extract.PROMPT_VERSION == "claims-v1"


async def test_sync_writes_accepted_and_rejected_with_document_metadata(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
        client = FakeClient(replies={"Amazon RDS": _message(
            {"claims": [GOOD, PARTIAL, INVENTED]}, tokens=(1000, 200))})
        report = await extract.run(conn, [cid], client=client, settings=_settings(),
                                   mode="sync")
        assert (report.chunks, report.done, report.skipped, report.errored) == (1, 1, 0, 0)
        assert (report.claims_accepted, report.claims_rejected) == (2, 1)
        assert (report.input_tokens, report.output_tokens) == (1000, 200)
        assert abs(report.est_cost_usd - (1000 * 0.10 + 200 * 0.50) / 1e6) < 1e-12
        accepted = await _rows(
            conn, "SELECT vendor, product, version, capability_raw, target_raw, support,"
            " conditions, extractor_model, prompt_version FROM raw_claims"
            " ORDER BY capability_raw")
        assert accepted == [
            ("rubrik", "Rubrik Security Cloud", None, "cross-region copy", None, "partial",
             "Enterprise edition", "claude-haiku-5-5", "claims-v1"),
            ("rubrik", "Rubrik Security Cloud", None, "immutable snapshots",
             "Amazon RDS for PostgreSQL", "supported", None, "claude-haiku-5-5",
             "claims-v1")]
        quotes = await _rows(conn, "SELECT evidence_quote, model_quote, gate FROM raw_claims"
                                   " ORDER BY capability_raw")
        assert quotes == [
            ("Cross-region copy requires the Enterprise edition.", PARTIAL["evidence_quote"],
             "exact"),
            (GOOD["evidence_quote"], GOOD["evidence_quote"], "exact")]
        rejected = await _rows(conn, "SELECT capability_raw, vendor, reason"
                                     " FROM rejected_claims")
        assert len(rejected) == 1
        assert rejected[0][:2] == ("tape backup", "rubrik")
        assert rejected[0][2].startswith("no-match:")
        [ext] = await _rows(conn, "SELECT status, n_claims, input_tokens, output_tokens"
                                  " FROM extractions WHERE chunk_id=%s", cid)
        assert ext == ("done", 2, 1000, 200)
        # every claim links to its extraction
        linked = await _rows(conn, "SELECT count(*) FROM raw_claims r JOIN extractions e"
                                   " USING (extraction_id) WHERE e.chunk_id=%s", cid)
        assert linked == [(2,)]
    finally:
        await conn.close()


async def test_null_product_is_kept(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn, vendor="druva", product=None)
        client = FakeClient(replies={"Amazon RDS": _message({"claims": [GOOD, INVENTED]})})
        report = await extract.run(conn, [cid], client=client, settings=_settings(),
                                   mode="sync")
        assert report.done == 1
        assert await _rows(conn, "SELECT vendor, product FROM raw_claims") == [("druva", None)]
        assert await _rows(conn, "SELECT vendor, product FROM rejected_claims") == [
            ("druva", None)]
        assert "druva" in client.messages.calls[0]["messages"][0]["content"]
    finally:
        await conn.close()


async def test_rerun_skips_done_chunks(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
        client = FakeClient(replies={"Amazon RDS": _message({"claims": [GOOD]})})
        await extract.run(conn, [cid], client=client, settings=_settings(), mode="sync")
        again = await extract.run(conn, [cid], client=client, settings=_settings(),
                                  mode="sync")
        assert (again.done, again.skipped, again.claims_accepted) == (0, 1, 0)
        assert len(client.messages.calls) == 1
        assert await _rows(conn, "SELECT count(*) FROM raw_claims") == [(1,)]
        # a chunk with no capability content is done too, with zero claims
        [empty] = await _doc(conn, texts=("Nothing to see here.",))
        first = await extract.run(conn, [empty], client=client, settings=_settings(),
                                  mode="sync")
        assert (first.done, first.claims_accepted) == (1, 0)
        second = await extract.run(conn, [empty], client=client, settings=_settings(),
                                   mode="sync")
        assert second.skipped == 1 and len(client.messages.calls) == 2
    finally:
        await conn.close()


async def test_budget_refusal_submits_nothing(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ids = await _doc(conn, texts=(TEXT, TEXT + " more"))
        client = FakeClient()
        report = await extract.run(conn, ids, client=client,
                                   settings=_settings(extract_budget_usd=1e-9), mode="sync")
        assert report.stopped_by_budget is True
        assert report.done == 0 and client.messages.calls == []
        batch = await extract.run(conn, ids, client=client,
                                  settings=_settings(extract_budget_usd=1e-9), mode="batch")
        assert batch.stopped_by_budget is True and client.messages.batches.created == []
        assert await _rows(conn, "SELECT count(*) FROM extractions") == [(0,)]
    finally:
        await conn.close()


async def test_budget_counts_spend_so_far(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ids = await _doc(conn, texts=(TEXT, TEXT + " two", TEXT + " three"))
        # each call costs 1M * $0.10/MTok = $0.10: after one, spend plus the next chunk's
        # estimate (a fraction of a cent) exceeds a $0.1001 budget
        client = FakeClient(replies={"Amazon RDS": _message({"claims": []},
                                                            tokens=(1_000_000, 0))})
        report = await extract.run(conn, ids, client=client,
                                   settings=_settings(extract_budget_usd=0.1001, extract_concurrency=1), mode="sync")
        assert report.done == 1 and report.stopped_by_budget is True
        assert len(client.messages.calls) == 1
        again = await extract.run(conn, ids, client=client,
                                  settings=_settings(extract_budget_usd=0.1001, extract_concurrency=1), mode="sync")
        assert again.done == 0 and again.stopped_by_budget is True
    finally:
        await conn.close()


async def test_batch_partial_failure_is_retried_next_run(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ok, bad, late = await _doc(conn, texts=(TEXT, TEXT + " B", TEXT + " C"))
        batches = FakeBatches({ok: ("succeeded", _message({"claims": [GOOD]},
                                                           tokens=(1000, 100))),
                               bad: ("errored",), late: ("expired",)})
        client = FakeClient(batches=batches)
        report = await extract.run(conn, [ok, bad, late], client=client, settings=_settings(),
                                   mode="batch", poll_seconds=0)
        assert (report.done, report.errored, report.claims_accepted) == (1, 2, 1)
        assert abs(report.est_cost_usd - (1000 * 0.10 + 100 * 0.50) / 2e6) < 1e-12
        [requests] = batches.created
        assert {r["custom_id"] for r in requests} == {ok, bad, late}
        assert requests[0]["params"]["model"] == "claude-haiku-5-5"
        status = dict(await _rows(conn, "SELECT chunk_id::text, status FROM extractions"))
        assert status == {ok: "done", bad: "errored", late: "errored"}
        errors = dict(await _rows(conn, "SELECT chunk_id::text, error FROM extractions"))
        assert errors[bad] == "batch:errored:invalid_request_error"
        assert errors[late] == "batch:expired"
        [ledger] = await _rows(conn, "SELECT status, actual_cost_usd FROM extract_batches")
        assert ledger[0] == "processed" and abs(ledger[1] - report.est_cost_usd) < 1e-12
        assert await _rows(conn, "SELECT n_claims FROM extractions WHERE status='errored'") == [
            (0,), (0,)]

        batches.outcomes = {bad: ("succeeded", _message({"claims": [GOOD]})),
                            late: ("succeeded", _message({"claims": []}))}
        again = await extract.run(conn, [ok, bad, late], client=client, settings=_settings(),
                                  mode="batch", poll_seconds=0)
        assert (again.skipped, again.done, again.errored) == (1, 2, 0)
        assert {r["custom_id"] for r in batches.created[1]} == {bad, late}
        status = dict(await _rows(conn, "SELECT chunk_id::text, status FROM extractions"))
        assert status == {ok: "done", bad: "done", late: "done"}
        assert await _rows(conn, "SELECT count(*) FROM extractions") == [(3,)]
    finally:
        await conn.close()


async def test_refusal_invalid_json_and_api_error_are_errored(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        refused, garbage, cut, failing = await _doc(
            conn, texts=(TEXT + " #refuse", TEXT + " #garbage", TEXT + " #cut", TEXT + " #fail"))
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        api_error = anthropic.APIConnectionError(request=request)
        client = FakeClient(replies={
            "#refuse": _message({"claims": []}, stop_reason="refusal", tokens=(500, 5)),
            "#garbage": _message("{not json"),
            "#cut": _message('{"claims": [', stop_reason="max_tokens"),
            "#fail": api_error})
        report = await extract.run(conn, [refused, garbage, cut, failing], client=client,
                                   settings=_settings(), mode="sync")
        assert (report.done, report.errored) == (0, 4)
        rows = dict(await _rows(conn, "SELECT chunk_id::text, error FROM extractions"
                                      " WHERE status='errored'"))
        assert rows[refused] == "refusal"
        assert rows[garbage].startswith("invalid-json")
        assert rows[cut] == "max_tokens"
        assert rows[failing].startswith("APIConnectionError")
        assert report.input_tokens >= 500  # a refusal's usage is still paid for
        assert await _rows(conn, "SELECT count(*) FROM raw_claims") == [(0,)]
    finally:
        await conn.close()


async def test_cli_extract_prints_report(vkb_dsn, monkeypatch, capsys, tmp_path):
    import asyncio

    from vendorkb import cli

    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
    finally:
        await conn.close()
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text(f"{cid}\n\n")
    client = FakeClient(replies={"Amazon RDS": _message({"claims": [GOOD]})})
    monkeypatch.setattr(cli, "_extract_client", lambda settings: client)
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)
    code = await asyncio.to_thread(cli.main, ["extract", "--ids-file", str(ids_file),
                                              "--mode", "sync", "--limit", "5"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["done"] == 1 and out["claims_accepted"] == 1


def _conn_error():
    return anthropic.APIConnectionError(
        request=httpx.Request("GET", "https://api.anthropic.com/v1/messages/batches/x"))


async def test_second_concurrent_run_is_locked_out(vkb_dsn):
    conn = await connect(vkb_dsn)
    holder = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
        await holder.execute("SELECT pg_advisory_lock(%s)", (extract.LOCK_KEY,))
        client = FakeClient()
        report = await extract.run(conn, [cid], client=client, settings=_settings(),
                                   mode="sync")
        assert report.locked is True and report.done == 0 and client.messages.calls == []
        await holder.execute("SELECT pg_advisory_unlock(%s)", (extract.LOCK_KEY,))
        report = await extract.run(conn, [cid], client=client, settings=_settings(),
                                   mode="sync")
        assert report.locked is False and report.done == 1
    finally:
        await holder.close()
        await conn.close()


async def test_crash_while_polling_resumes_the_paid_batch(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        a, b = await _doc(conn, texts=(TEXT, TEXT + " B"))
        batches = FakeBatches({a: ("succeeded", _message({"claims": [GOOD]})),
                               b: ("succeeded", _message({"claims": []}))})
        batches.retrieve_errors = [RuntimeError("pod evicted")]
        client = FakeClient(batches=batches)
        try:
            await extract.run(conn, [a, b], client=client, settings=_settings(), mode="batch",
                              poll_seconds=0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the poll failure should have ended the run")
        [(bid, status, est)] = await _rows(
            conn, "SELECT batch_id, status, est_cost_usd FROM extract_batches")
        assert (bid, status) == ("batch_1", "submitted") and est > 0
        assert await _rows(conn, "SELECT count(*) FROM extractions") == [(0,)]

        report = await extract.run(conn, [a, b], client=client, settings=_settings(),
                                   mode="batch", poll_seconds=0)
        assert report.resumed_batches == 1 and len(batches.created) == 1  # no resubmission
        assert (report.done, report.claims_accepted) == (2, 1)
        assert await _rows(conn, "SELECT status FROM extract_batches") == [("processed",)]
        # a further run finds nothing to resume and nothing to do
        again = await extract.run(conn, [a, b], client=client, settings=_settings(),
                                  mode="batch", poll_seconds=0)
        assert (again.resumed_batches, again.skipped, len(batches.created)) == (0, 2, 1)
    finally:
        await conn.close()


async def test_transient_poll_errors_are_retried(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [a] = await _doc(conn)
        batches = FakeBatches({a: ("succeeded", _message({"claims": [GOOD]}))})
        batches.retrieve_errors = [_conn_error(), _conn_error()]
        client = FakeClient(batches=batches)
        report = await extract.run(conn, [a], client=client, settings=_settings(),
                                   mode="batch", poll_seconds=0)
        assert report.done == 1 and batches.retrieve_errors == []
    finally:
        await conn.close()


async def test_chunk_deleted_mid_batch_is_skipped_and_spend_kept(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [a] = await _doc(conn)
        [gone] = await _doc(conn, texts=(TEXT + " doomed",))
        batches = FakeBatches({a: ("succeeded", _message({"claims": [GOOD]})),
                               gone: ("succeeded", _message({"claims": [GOOD]},
                                                            tokens=(1_000_000, 0)))})

        async def resync():  # sync re-chunks the document while the batch is in flight
            await conn.execute("DELETE FROM documents WHERE doc_id ="
                               " (SELECT doc_id FROM chunks WHERE chunk_id=%s)", (gone,))
        batches.on_retrieve = resync
        client = FakeClient(batches=batches)
        report = await extract.run(conn, [a, gone], client=client, settings=_settings(),
                                   mode="batch", poll_seconds=0)
        assert (report.done, report.skipped_deleted, report.errored) == (1, 1, 0)
        # the deleted chunk's result was paid for: the ledger keeps it
        [(actual,)] = await _rows(conn, "SELECT actual_cost_usd FROM extract_batches")
        assert actual >= 0.05
        assert await extract._spent(conn) == actual
    finally:
        await conn.close()


async def test_spend_survives_chunk_deletion(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [a] = await _doc(conn)
        client = FakeClient(replies={"Amazon RDS": _message({"claims": [GOOD]},
                                                            tokens=(1_000_000, 0))})
        await extract.run(conn, [a], client=client, settings=_settings(), mode="sync")
        await conn.execute("DELETE FROM documents")
        assert await _rows(conn, "SELECT count(*) FROM extractions") == [(0,)]
        assert abs(await extract._spent(conn) - 0.10) < 1e-9
    finally:
        await conn.close()


async def test_chunk_errored_three_times_is_failed_and_skipped(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [a] = await _doc(conn)
        client = FakeClient(replies={"Amazon RDS": _message({"claims": []},
                                                            stop_reason="max_tokens")})
        for n in range(1, 4):
            report = await extract.run(conn, [a], client=client, settings=_settings(),
                                       mode="sync")
            assert report.errored == 1 and report.failed == (1 if n == 3 else 0)
        assert await _rows(conn, "SELECT status, attempts FROM extractions") == [("failed", 3)]
        report = await extract.run(conn, [a], client=client, settings=_settings(), mode="sync")
        assert report.skipped_failed == 1 and len(client.messages.calls) == 3
    finally:
        await conn.close()


async def test_canceled_unknown_and_duplicate_results(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        a, c = await _doc(conn, texts=(TEXT, TEXT + " C"))
        batches = FakeBatches({a: ("succeeded", _message({"claims": [GOOD]})),
                               c: ("canceled",)})
        batches.extra_results = [(str(uuid.uuid4()), "errored"), (a, "errored")]
        client = FakeClient(batches=batches)
        report = await extract.run(conn, [a, c], client=client, settings=_settings(),
                                   mode="batch", poll_seconds=0)
        assert (report.done, report.errored) == (1, 1)
        status = dict(await _rows(conn, "SELECT chunk_id::text, status FROM extractions"))
        assert status == {a: "done", c: "errored"}  # the duplicate did not overwrite a
    finally:
        await conn.close()


async def test_requests_split_across_batches(vkb_dsn, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ids = await _doc(conn, texts=(TEXT, TEXT + " 2", TEXT + " 3"))
        monkeypatch.setattr(extract, "MAX_BATCH_REQUESTS", 2)
        batches = FakeBatches({i: ("succeeded", _message({"claims": []})) for i in ids})
        client = FakeClient(batches=batches)
        report = await extract.run(conn, ids, client=client, settings=_settings(),
                                   mode="batch", poll_seconds=0)
        assert [len(b) for b in batches.created] == [2, 1] and report.done == 3
        assert await _rows(conn, "SELECT count(*) FROM extract_batches"
                                 " WHERE status='processed'") == [(2,)]
    finally:
        await conn.close()


async def test_bad_request_stops_a_sync_run(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ids = await _doc(conn, texts=(TEXT, TEXT + " 2"))
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        bad = anthropic.BadRequestError(
            "bad schema", response=httpx.Response(400, request=request), body=None)
        client = FakeClient(replies={"Amazon RDS": bad})
        try:
            await extract.run(conn, ids, client=client,
                              settings=_settings(extract_concurrency=1), mode="sync")
        except anthropic.BadRequestError:
            pass
        else:
            raise AssertionError("a 400 should stop the run")
        assert len(client.messages.calls) == 1
        assert await _rows(conn, "SELECT status FROM extractions") == [("errored",)]
        assert await _rows(conn, "SELECT status FROM extract_batches") == [("processed",)]
    finally:
        await conn.close()


async def test_resume_after_crash_mid_results_applies_each_result_once(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        a, b = await _doc(conn, texts=(TEXT, TEXT + " B"))
        # results come in reverse request order: b (errored) first, then a
        batches = FakeBatches({a: ("succeeded", _message({"claims": [GOOD]})),
                               b: ("errored",)})
        batches.results_fail_after = 1
        client = FakeClient(batches=batches)
        try:
            await extract.run(conn, [a, b], client=client, settings=_settings(), mode="batch",
                              poll_seconds=0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the results failure should have ended the run")
        assert await _rows(conn, "SELECT status, attempts FROM extractions") == [("errored", 1)]
        report = await extract.run(conn, [a, b], client=client, settings=_settings(),
                                   mode="batch", poll_seconds=0)
        assert report.resumed_batches == 1
        rows = dict(await _rows(conn, "SELECT chunk_id::text, attempts FROM extractions"))
        # batch_1's errored result for b counts once (not twice); the run then resubmits b
        # in batch_2, which errors again: 2 attempts in all
        assert rows == {a: 1, b: 2} and len(batches.created) == 2
    finally:
        await conn.close()



def _chat(payload, *, finish="stop", tokens=(1000, 200), cost=None, refusal=None):
    content = payload if payload is None or isinstance(payload, str) else json.dumps(payload)
    usage = SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1])
    if cost is not None:
        usage.cost = cost
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
            content=content, refusal=refusal))], usage=usage)


class FakeCompletions:
    def __init__(self, replies, delay=0.0, crash_on=None):
        self.replies = replies  # chunk text marker -> response | Exception
        self.calls: list[dict] = []
        self.delay = delay
        self.crash_on = crash_on  # 1-based call number that raises RuntimeError
        self.active = 0
        self.max_active = 0

    async def create(self, **params):
        import asyncio
        self.calls.append(params)
        if self.crash_on is not None and len(self.calls) == self.crash_on:
            raise RuntimeError("pod evicted")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        body = params["messages"][-1]["content"]
        for marker, reply in self.replies.items():
            if marker in body:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        return _chat({"claims": []}, cost=0.0)


class FakeOpenAI:
    def __init__(self, replies=None, **kw):
        self.chat = SimpleNamespace(completions=FakeCompletions(replies or {}, **kw))


def _oa(**kw) -> VendorkbSettings:
    return _settings(extract_backend="openai",
                     extract_model="anthropic/claude-haiku-5.5:batch", **kw)


def test_openai_request_shape():
    chunk = extract.ChunkRow(chunk_id="c1", doc_id="d1", vendor="rubrik", product=None,
                             version=None, title="T", section_path=["A"], text=TEXT)
    req = extract.build_openai_request(chunk, _oa())
    assert req["model"] == "anthropic/claude-haiku-5.5:batch" and req["max_tokens"] == 2000
    assert req["messages"][0] == {"role": "system", "content": extract.SYSTEM_PROMPT}
    assert req["messages"][1]["role"] == "user" and TEXT in req["messages"][1]["content"]
    assert req["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "claims", "strict": True, "schema": extract.CLAIM_SCHEMA}}
    assert req["extra_body"] == {"usage": {"include": True}, "reasoning": {"effort": "low"}}


async def test_openai_success_uses_reported_cost(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
        client = FakeOpenAI({"Amazon RDS": _chat({"claims": [GOOD, INVENTED]}, cost=0.0123)})
        report = await extract.run(conn, [cid], client=client, settings=_oa(), mode="sync")
        assert (report.done, report.claims_accepted, report.claims_rejected) == (1, 1, 1)
        assert abs(report.est_cost_usd - 0.0123) < 1e-12
        assert await _rows(conn, "SELECT extractor_model, input_tokens, output_tokens"
                                 " FROM extractions") == [
            ("anthropic/claude-haiku-5.5:batch", 1000, 200)]
        assert abs(await extract._spent(conn) - 0.0123) < 1e-9
    finally:
        await conn.close()


async def test_openai_cost_in_model_extra_is_used():
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, model_extra={"cost": 0.5})
    assert extract._reported_cost(usage) == 0.5


async def test_openai_missing_cost_falls_back_to_the_token_price(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
        client = FakeOpenAI({"Amazon RDS": _chat({"claims": [GOOD]}, tokens=(1000, 200))})
        report = await extract.run(conn, [cid], client=client, settings=_oa(), mode="sync")
        assert report.done == 1
        expected = (1000 * 0.10 + 200 * 0.50) / 1e6
        assert abs(report.est_cost_usd - expected) < 1e-12
        assert abs(await extract._spent(conn) - expected) < 1e-12
    finally:
        await conn.close()


async def test_openai_length_refusal_and_empty_are_errored(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        cut, refused, empty, bad = await _doc(
            conn, texts=(TEXT + " #cut", TEXT + " #refuse", TEXT + " #empty", TEXT + " #bad"))
        client = FakeOpenAI({
            "#cut": _chat('{"claims": [', finish="length", cost=0.001),
            "#refuse": _chat(None, refusal="I can't help with that"),
            "#empty": _chat(None),
            "#bad": _chat("{nope")})
        report = await extract.run(conn, [cut, refused, empty, bad], client=client,
                                   settings=_oa(), mode="sync")
        assert (report.done, report.errored) == (0, 4)
        rows = dict(await _rows(conn, "SELECT chunk_id::text, error FROM extractions"))
        assert rows[cut] == "length" and rows[refused] == "refusal"
        assert rows[empty] == "no-text" and rows[bad].startswith("invalid-json")
    finally:
        await conn.close()


async def test_openai_runs_concurrently_and_stops_at_the_budget(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ids = await _doc(conn, texts=tuple(f"{TEXT} n{i}" for i in range(20)))
        client = FakeOpenAI({"Amazon RDS": _chat({"claims": [GOOD]}, cost=0.01)}, delay=0.01)
        report = await extract.run(conn, ids, client=client,
                                   settings=_oa(extract_budget_usd=0.05, extract_concurrency=4),
                                   mode="sync")
        completions = client.chat.completions
        assert completions.max_active > 1  # calls overlapped
        assert report.stopped_by_budget is True and 1 <= report.done < 20
        spent = await extract._spent(conn)
        assert abs(spent - 0.01 * report.done) < 1e-9
        # in flight is bounded by the concurrency, so the overshoot is too
        assert spent <= 0.05 + 4 * 0.01
        [(n_ids,)] = await _rows(conn, "SELECT cardinality(chunk_ids) FROM extract_batches")
        assert n_ids == report.done
    finally:
        await conn.close()


async def test_openai_ledger_survives_a_crash(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ids = await _doc(conn, texts=tuple(f"{TEXT} n{i}" for i in range(5)))
        client = FakeOpenAI({"Amazon RDS": _chat({"claims": []}, cost=0.01)}, crash_on=3)
        try:
            await extract.run(conn, ids, client=client,
                              settings=_oa(extract_concurrency=1), mode="sync")
        except RuntimeError:
            pass
        else:
            raise AssertionError("the crash should end the run")
        assert len(client.chat.completions.calls) == 3  # nothing dispatched after the crash
        assert abs(await extract._spent(conn) - 0.02) < 1e-9
        assert await _rows(conn, "SELECT count(*) FROM extractions WHERE status='done'") == [
            (2,)]
        # the next run picks up where it stopped, and the ledger keeps adding up
        client.chat.completions.crash_on = None
        report = await extract.run(conn, ids, client=client,
                                   settings=_oa(extract_concurrency=1), mode="sync")
        assert (report.skipped, report.done) == (2, 3)
        assert abs(await extract._spent(conn) - 0.05) < 1e-9
    finally:
        await conn.close()


async def test_openai_backend_refuses_batch_mode(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
        try:
            await extract.run(conn, [cid], client=FakeOpenAI(), settings=_oa(), mode="batch")
        except ValueError as e:
            assert "sync" in str(e)
        else:
            raise AssertionError("batch mode needs the anthropic backend")
    finally:
        await conn.close()


async def test_cli_extract_openai_backend(vkb_dsn, monkeypatch, capsys, tmp_path):
    import asyncio

    from vendorkb import cli

    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
    finally:
        await conn.close()
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text(f"{cid}\n")
    client = FakeOpenAI({"Amazon RDS": _chat({"claims": [GOOD]}, cost=0.001)})
    seen = {}

    def build(settings):
        seen["backend"] = settings.extract_backend
        return client
    monkeypatch.setattr(cli, "_extract_client", build)
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)
    code = await asyncio.to_thread(cli.main, ["extract", "--ids-file", str(ids_file),
                                              "--backend", "openai"])
    assert code == 0 and seen["backend"] == "openai"
    out = json.loads(capsys.readouterr().out)
    assert out["done"] == 1 and out["claims_accepted"] == 1


def test_cli_builds_openai_client_from_settings():
    from openai import AsyncOpenAI

    from vendorkb import cli

    client = cli._extract_client(_oa(extract_base_url="https://openrouter.ai/api/v1",
                                     extract_api_key="k"))
    assert isinstance(client, AsyncOpenAI)
    assert str(client.base_url).startswith("https://openrouter.ai/api/v1")
    assert client.max_retries == cli.EXTRACT_MAX_RETRIES
