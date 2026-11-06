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

    async def create(self, *, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id=f"batch_{len(self.created)}", processing_status="in_progress")

    async def retrieve(self, message_batch_id):
        self.retrieves += 1
        status = "ended" if self.retrieves % 2 == 0 else "in_progress"
        return SimpleNamespace(id=message_batch_id, processing_status=status)

    async def results(self, message_batch_id):
        requests = self.created[int(message_batch_id.split("_")[1]) - 1]

        async def gen():
            for req in reversed(requests):  # any order: the runner keys by custom_id
                kind, *rest = self.outcomes[req["custom_id"]]
                result = SimpleNamespace(type=kind)
                if kind == "succeeded":
                    result.message = rest[0]
                elif kind == "errored":
                    result.error = SimpleNamespace(type="error", error={"type": "api_error"})
                yield SimpleNamespace(custom_id=req["custom_id"], result=result)
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
                       "extractions, chunks, documents CASCADE")
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
                                   settings=_settings(extract_budget_usd=0.1001), mode="sync")
        assert report.done == 1 and report.stopped_by_budget is True
        assert len(client.messages.calls) == 1
        again = await extract.run(conn, ids, client=client,
                                  settings=_settings(extract_budget_usd=0.1001), mode="sync")
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
    monkeypatch.setattr(cli, "_anthropic_client", lambda settings: client)
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)
    code = await asyncio.to_thread(cli.main, ["extract", "--ids-file", str(ids_file),
                                              "--mode", "sync", "--limit", "5"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["done"] == 1 and out["claims_accepted"] == 1
