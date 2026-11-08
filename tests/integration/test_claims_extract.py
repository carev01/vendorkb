import json
import uuid
from types import SimpleNamespace

import anthropic
import httpx
import psycopg
import pytest
from pgvector import Vector

from vendorkb.claims import extract
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema

TEXT = ("Protecting Databases / Amazon RDS\n\n"
        "Backups of Amazon RDS for PostgreSQL instances are supported with immutable "
        "snapshots. Cross-region copy requires the Enterprise edition.")

# The v3 qualifiers, all unstated: every claim in a reply carries every field.
V3 = {"kind": "capability", "target_scope": None, "offering": None, "edition": None,
      "availability": "unstated", "default_state": None, "value": None, "unit": None,
      "bound": None}
GOOD = {**V3, "capability": "immutable snapshots", "target": "Amazon RDS for PostgreSQL",
        "support": "supported", "conditions": None,
        "evidence_quote": "Backups of Amazon RDS for PostgreSQL instances are supported"}
PARTIAL = {**V3, "capability": "cross-region copy", "target": None, "support": "partial",
           "conditions": "Enterprise edition",
           "evidence_quote": "cross-region   copy requires the ENTERPRISE edition."}
INVENTED = {**V3, "capability": "tape backup", "target": None, "support": "supported",
            "conditions": None, "evidence_quote": "Tape backup is fully supported for LTO-9."}


def _settings(**kw) -> VendorkbSettings:
    return VendorkbSettings(_env_file=None, **kw)


def _message(payload, *, stop_reason="end_turn", tokens=(1000, 200), cache=(0, 0)):
    """``cache`` = (cache_creation_input_tokens, cache_read_input_tokens)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1],
                              cache_creation_input_tokens=cache[0],
                              cache_read_input_tokens=cache[1]))


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
    assert set(item["required"]) == set(item["properties"])
    assert item["properties"]["support"]["enum"] == [
        "supported", "partial", "not_supported", "unclear"]
    chunk = extract.ChunkRow(chunk_id="c1", doc_id="d1", vendor="rubrik", product=None,
                             version=None, title="T", section_path=["A"], text=TEXT)
    req = extract.build_request(chunk, _settings())
    assert req["model"] == "claude-haiku-5-5" and req["max_tokens"] == 2000
    assert req["system"] == [{"type": "text", "text": extract.SYSTEM_PROMPT,
                              "cache_control": {"type": "ephemeral"}}]
    assert req["output_config"] == {
        "effort": "low", "format": {"type": "json_schema", "schema": extract.CLAIM_SCHEMA}}
    assert "thinking" not in req and "temperature" not in req
    assert req["messages"][0]["role"] == "user" and TEXT in req["messages"][0]["content"]
    assert "rubrik" in req["messages"][0]["content"]
    assert extract.PROMPT_VERSION == "claims-v3.1"



def test_schema_has_the_v3_fields():
    item = extract.CLAIM_SCHEMA["properties"]["claims"]["items"]
    assert item["required"] == [
        "kind", "capability", "target", "target_scope", "support", "offering", "edition",
        "availability", "default_state", "value", "unit", "bound", "conditions",
        "evidence_quote"]
    assert item["additionalProperties"] is False
    props = item["properties"]
    assert props["kind"] == {"type": "string", "enum": ["capability", "licensing", "service"]}
    assert props["availability"]["enum"] == list(extract.AVAILABILITY_VALUES)
    assert "unstated" in extract.AVAILABILITY_VALUES
    for key, values in (("default_state", ["on", "off"]), ("bound", ["max", "min", "exact"])):
        assert props[key] == {"anyOf": [{"type": "string", "enum": values},
                                        {"type": "null"}]}
    assert props["value"] == {"anyOf": [{"type": "number"}, {"type": "null"}]}


def test_validate_claim_names_the_first_invalid_field():
    assert extract.validate_claim(GOOD) is None
    assert extract.validate_claim({**GOOD, "kind": "pricing"}) == "schema:kind"
    assert extract.validate_claim({**GOOD, "capability": " "}) == "schema:capability"
    assert extract.validate_claim({**GOOD, "availability": None}) == "schema:availability"
    assert extract.validate_claim({**GOOD, "default_state": "enabled"}) == \
        "schema:default_state"
    assert extract.validate_claim({**GOOD, "value": True}) == "schema:value"
    assert extract.validate_claim({**GOOD, "value": 1.5, "bound": "up to"}) == "schema:bound"
    assert extract.validate_claim({**GOOD, "unit": 7}) == "schema:unit"
    missing = dict(GOOD)
    del missing["target_scope"]
    assert extract.validate_claim(missing) == "schema:target_scope"
    # a value that is not a finite float is invalid, and checking it never raises
    for bad in (float("inf"), float("-inf"), float("nan"), 10**400):
        assert extract.validate_claim({**GOOD, "value": bad}) == "schema:value", bad
        assert extract._number({"value": bad}) is None
    assert extract._number({"value": 10**15}) == 1e15
    # the first invalid field in schema order wins
    assert extract.validate_claim({**GOOD, "kind": None, "value": "99"}) == "schema:kind"


def test_prompt_states_the_v3_rules():
    prompt = extract.SYSTEM_PROMPT
    for phrase in ("licensing", "service", "one claim per object type", "never a price",
                   "neither makes a capability partial", "never a feature or backup method",
                   "installation and sizing requirements",
                   "a hardware appliance sold as its own deployment form",
                   "states the default outright", "default selection in a list or form",
                   "only name a new setting", "never a standard, protocol",
                   "the target itself implies", "(its options) is not a condition"):
        assert phrase in prompt
    assert "requires" not in prompt
    for field in extract.CLAIM_SCHEMA["properties"]["claims"]["items"]["required"]:
        assert f"- {field}:" in prompt


async def test_v3_claim_fields_are_stored(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
        claim = {**GOOD, "kind": "licensing", "target_scope": "users",
                 "offering": "self-managed", "edition": "Premium",
                 "availability": "preview", "default_state": "off", "value": 99,
                 "unit": "years", "bound": "max"}
        client = FakeClient(replies={"Amazon RDS": _message({"claims": [claim]})})
        report = await extract.run(conn, [cid], client=client, settings=_settings(),
                                   mode="sync")
        assert (report.claims_accepted, report.claims_rejected) == (1, 0)
        rows = await _rows(
            conn, "SELECT kind, target_scope, offering, edition, availability, default_state,"
            " value, unit, bound, prompt_version FROM raw_claims")
        assert rows == [("licensing", "users", "self-managed", "Premium", "preview", "off",
                         99.0, "years", "max", extract.PROMPT_VERSION)]
    finally:
        await conn.close()



async def test_unit_and_bound_without_a_value_are_stored_null(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
        claim = {**GOOD, "value": None, "unit": "years", "bound": "max"}
        client = FakeClient(replies={"Amazon RDS": _message({"claims": [claim]})})
        report = await extract.run(conn, [cid], client=client, settings=_settings(),
                                   mode="sync")
        assert (report.claims_accepted, report.claims_rejected) == (1, 0)
        assert await _rows(conn, "SELECT value, unit, bound FROM raw_claims") == [
            (None, None, None)]
    finally:
        await conn.close()


async def test_nul_bytes_in_model_strings_are_stripped_before_storage(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [ok, err] = await _doc(conn, texts=(TEXT, TEXT + " second"))
        nul_good = {**GOOD, "capability": "immutable\x00 snapshots",
                    "conditions": "\x00PostgreSQL only",
                    "evidence_quote": "Backups of Amazon RDS for\x00 PostgreSQL instances are"
                                      " supported"}
        nul_invented = {**INVENTED, "target_scope": "\x00", "evidence_quote": "Tape\x00 backup is fully supported for LTO-9."}
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        client = FakeClient(replies={
            "second": anthropic.APIConnectionError(message="conn\x00 lost", request=request),
            "Amazon RDS": _message({"claims": [nul_good, nul_invented]})})
        report = await extract.run(conn, [ok, err], client=client, settings=_settings(),
                                   mode="sync")
        assert (report.done, report.errored) == (1, 1)
        # the cleaned quote matches the chunk, so the claim passes the gate
        assert await _rows(conn, "SELECT capability_raw, conditions, model_quote, evidence_quote,"
                                 " gate FROM raw_claims") == [
            ("immutable snapshots", "PostgreSQL only", GOOD["evidence_quote"],
             GOOD["evidence_quote"], "exact")]
        [(model_quote, scope, reason)] = await _rows(
            conn, "SELECT model_quote, target_scope, reason FROM rejected_claims")
        assert (model_quote, scope) == (INVENTED["evidence_quote"], "")
        assert "\x00" not in reason and reason.startswith("no-match")
        [(error,)] = await _rows(conn, "SELECT error FROM extractions WHERE chunk_id=%s", err)
        assert error == "APIConnectionError: conn lost"
    finally:
        await conn.close()

async def test_invalid_qualifier_rejects_only_that_claim(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)
        bad = {**GOOD, "value": "99", "unit": "years", "bound": "max", "edition": "Premium"}
        client = FakeClient(replies={"Amazon RDS": _message({"claims": [bad, PARTIAL]})})
        report = await extract.run(conn, [cid], client=client, settings=_settings(),
                                   mode="sync")
        assert (report.claims_accepted, report.claims_rejected) == (1, 1)
        assert await _rows(conn, "SELECT capability_raw, kind, availability"
                                 " FROM raw_claims") == [
            ("cross-region copy", "capability", "unstated")]
        assert await _rows(
            conn, "SELECT capability_raw, reason, evidence_quote, model_quote, kind, value,"
            " unit, bound, edition FROM rejected_claims") == [
            ("immutable snapshots", "schema:value", None, GOOD["evidence_quote"],
             "capability", None, "years", "max", "Premium")]
    finally:
        await conn.close()


def test_cache_writes_and_reads_are_priced_per_model():
    opus = _settings(extract_model="claude-opus-5-5")
    write = extract._parse(_message({"claims": []}, tokens=(100, 200), cache=(2000, 0)))
    read = extract._parse(_message({"claims": []}, tokens=(100, 200), cache=(0, 2000)))
    assert (write.cache_write_tokens, write.cache_read_tokens) == (2000, 0)
    assert (read.input_tokens, read.cache_read_tokens) == (100, 2000)
    assert write.total_input_tokens == read.total_input_tokens == 2100
    # opus-5-5 at $4 / $20: writes 1.25x input, reads 0.05x input
    # write: (100*4 + 2000*4*1.25 + 200*20) / 1e6; read: (100*4 + 2000*4*0.05 + 200*20) / 1e6
    assert extract._cost(write, batch=False, settings=opus) == pytest.approx(0.0144, abs=1e-12)
    assert extract._cost(read, batch=False, settings=opus) == pytest.approx(0.0048, abs=1e-12)
    assert extract._cost(read, batch=True, settings=opus) == pytest.approx(0.0024, abs=1e-12)
    # haiku-5-5 at $0.10 / $0.50 reads at 0.1x; an unlisted model defaults to 0.1x
    haiku = extract._cost(read, batch=False, settings=_settings())
    assert haiku == pytest.approx((100 * 0.10 + 2000 * 0.10 * 0.1 + 200 * 0.50) / 1e6,
                                  abs=1e-15)
    assert extract.cache_read_multiplier("anthropic/claude-opus-5.5:batch") == 0.05
    assert extract.cache_read_multiplier("some-other-model") == 0.1



def test_usage_without_cache_fields_is_priced_as_plain_input():
    opus = _settings(extract_model="claude-opus-5-5")
    plain = (100 * 4.00 + 200 * 20.00) / 1e6
    absent = SimpleNamespace(input_tokens=100, output_tokens=200)
    none = SimpleNamespace(input_tokens=100, output_tokens=200,
                           cache_creation_input_tokens=None, cache_read_input_tokens=None)
    for usage in (absent, none):
        msg = SimpleNamespace(content=[SimpleNamespace(type="text", text='{"claims": []}')],
                              stop_reason="end_turn", usage=usage)
        outcome = extract._parse(msg)
        assert outcome.status == "done"
        assert (outcome.cache_write_tokens, outcome.cache_read_tokens) == (0, 0)
        assert outcome.total_input_tokens == 100
        assert extract._cost(outcome, batch=False, settings=opus) == pytest.approx(plain,
                                                                                    abs=1e-12)

async def test_sync_run_counts_and_prices_cached_input(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ids = await _doc(conn, texts=(TEXT + " first", TEXT + " second"))
        client = FakeClient(replies={
            "first": _message({"claims": []}, tokens=(100, 200), cache=(2000, 0)),
            "second": _message({"claims": []}, tokens=(100, 200), cache=(0, 2000))})
        report = await extract.run(conn, ids, client=client, mode="sync",
                                   settings=_settings(extract_model="claude-opus-5-5"))
        assert report.done == 2
        assert report.input_tokens == 4200  # uncached + cache writes + cache reads
        assert report.est_cost_usd == pytest.approx(0.0144 + 0.0048, abs=1e-12)
        assert await _rows(conn, "SELECT sum(input_tokens)::int FROM extractions") == [(4200,)]
    finally:
        await conn.close()

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
             "Enterprise edition", "claude-haiku-5-5", extract.PROMPT_VERSION),
            ("rubrik", "Rubrik Security Cloud", None, "immutable snapshots",
             "Amazon RDS for PostgreSQL", "supported", None, "claude-haiku-5-5",
             extract.PROMPT_VERSION)]
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


async def test_claims_are_gated_against_the_text_the_chunk_holds_when_recorded(vkb_dsn):
    """A re-sync can reuse a chunk with new text while the model call is in flight: a quote
    that is no longer in the text must not be stored as evidence."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [cid] = await _doc(conn)

        class Resyncing(FakeMessages):
            async def create(self, **params):
                await conn.execute(
                    "UPDATE chunks SET text = replace(text, 'Backups of Amazon RDS for"
                    " PostgreSQL instances are supported with immutable snapshots.',"
                    " 'Snapshots of Aurora clusters are listed.') WHERE chunk_id = %s", (cid,))
                return await super().create(**params)

        client = FakeClient()
        client.messages = Resyncing(replies={"Amazon RDS": _message(
            {"claims": [GOOD, PARTIAL]})})
        report = await extract.run(conn, [cid], client=client, settings=_settings(),
                                   mode="sync")
        assert (report.done, report.claims_accepted, report.claims_rejected) == (1, 1, 1)
        assert [r[0] for r in await _rows(conn, "SELECT capability_raw FROM raw_claims")] == [
            "cross-region copy"]
        [(reason,)] = await _rows(conn, "SELECT reason FROM rejected_claims")
        assert reason.startswith("no-match:")
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


async def _extracted(conn, chunk_id, *, model="m", prompt=extract.PROMPT_VERSION,
                     status="done"):
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s,%s,%s,%s,%s)", (str(uuid.uuid4()), chunk_id, model, prompt, status))


async def test_unextracted_ids_are_chunks_with_no_final_v3_extraction(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        new, by_teacher, earlier_v3, errored, failed, old_family = await _doc(
            conn, texts=("a", "b", "c", "d", "e", "f"))
        [other_vendor] = await _doc(conn, vendor="cohesity", product="DataProtect")
        await _extracted(conn, by_teacher, model="claude-opus-5-5")  # another model: done
        await _extracted(conn, earlier_v3, prompt="claims-v3")  # an earlier v3 revision
        await _extracted(conn, errored, status="errored")  # the next run retries it
        await _extracted(conn, failed, status="failed")  # given up after MAX_ATTEMPTS
        await _extracted(conn, old_family, prompt="claims-v1")  # not the current family
        assert set(await extract.unextracted_ids(conn)) == {
            new, errored, old_family, other_vendor}
        assert set(await extract.unextracted_ids(conn, vendor="rubrik")) == {
            new, errored, old_family}
        assert len(await extract.unextracted_ids(conn, limit=2)) == 2
    finally:
        await conn.close()


async def test_unextracted_ids_skip_superseded_chunks(vkb_dsn):
    """An earlier version of a page is kept for its claims; it is never extracted again."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        superseded, current = await _doc(conn, texts=("earlier version", "current version"))
        await conn.execute("UPDATE chunks SET superseded_at = now() WHERE chunk_id = %s",
                           (superseded,))
        assert await extract.unextracted_ids(conn) == [current]
    finally:
        await conn.close()


async def test_cli_extract_unextracted(vkb_dsn, monkeypatch, capsys):
    import asyncio

    from vendorkb import cli

    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [new] = await _doc(conn)
        [done] = await _doc(conn, vendor="cohesity", product="DataProtect")
        await _extracted(conn, done, model="claude-opus-5-5")
    finally:
        await conn.close()
    client = FakeClient(replies={"Amazon RDS": _message({"claims": [GOOD]})})
    monkeypatch.setattr(cli, "_extract_client", lambda settings: client)
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)
    code = await asyncio.to_thread(cli.main, ["extract", "--unextracted", "--mode", "sync"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["chunks"], out["done"]) == (1, 1) and len(client.messages.calls) == 1


def test_cli_extract_needs_exactly_one_source(capsys):
    from vendorkb import cli

    for argv in (["extract"], ["extract", "--ids-file", "x", "--unextracted"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    with pytest.raises(SystemExit):  # --vendor narrows --unextracted only
        cli.main(["extract", "--ids-file", "x", "--vendor", "rubrik"])


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


async def test_crash_between_submit_and_ledger_holds_the_chunks(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        a, b = await _doc(conn, texts=(TEXT, TEXT + " B"))
        batches = FakeBatches({a: ("succeeded", _message({"claims": [GOOD]})),
                               b: ("succeeded", _message({"claims": []}))})
        real_create = batches.create

        async def create_then_crash(*, requests):
            await real_create(requests=requests)  # the provider has the (paid) batch
            raise RuntimeError("pod evicted before the reply was handled")
        batches.create = create_then_crash
        client = FakeClient(batches=batches)
        try:
            await extract.run(conn, [a, b], client=client, settings=_settings(), mode="batch",
                              poll_seconds=0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the crash should have ended the run")
        # the ledger row was written before the submit: its estimate counts, its chunks wait
        [(bid, status, est, held)] = await _rows(
            conn, "SELECT batch_id, status, est_cost_usd, chunk_ids FROM extract_batches")
        assert bid.startswith("pending-") and status == "submitted" and est > 0
        assert sorted(held) == sorted([a, b])
        batches.create = real_create
        report = await extract.run(conn, [a, b], client=client, settings=_settings(),
                                   mode="batch", poll_seconds=0)
        assert len(batches.created) == 1  # never submitted again
        assert (report.unconfirmed_batches, report.held_chunks, report.resumed_batches) == (1, 2, 0)
        assert await extract._spent(conn) == est
    finally:
        await conn.close()


async def test_a_refused_submit_leaves_no_ledger_row(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [a] = await _doc(conn)
        batches = FakeBatches({})

        async def refuse(*, requests):
            raise anthropic.BadRequestError(
                "bad", response=httpx.Response(400, request=httpx.Request(
                    "POST", "https://api.anthropic.com/v1/messages/batches")), body=None)
        batches.create = refuse
        try:
            await extract.run(conn, [a], client=FakeClient(batches=batches),
                              settings=_settings(), mode="batch", poll_seconds=0)
        except anthropic.BadRequestError:
            pass
        assert await _rows(conn, "SELECT count(*) FROM extract_batches") == [(0,)]
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


async def test_an_unparseable_paid_reply_is_recorded_with_its_cost(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [a] = await _doc(conn)
        odd = _chat({"claims": []}, cost=0.03)
        odd.choices[0].message = None  # a reply shape the parser does not expect
        client = FakeOpenAI({"Amazon RDS": odd})
        try:
            await extract.run(conn, [a], client=client, settings=_oa(), mode="sync")
        except AttributeError:
            pass
        else:
            raise AssertionError("an unexpected reply should still stop the run")
        assert abs(await extract._spent(conn) - 0.03) < 1e-9  # the paid call is ledgered
        [(status, error)] = await _rows(conn, "SELECT status, error FROM extractions")
        assert status == "errored" and error == "parse:AttributeError"
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


def test_max_tokens_follows_the_setting_on_both_backends():
    chunk = extract.ChunkRow(chunk_id="c1", doc_id="d1", vendor="rubrik", product=None,
                             version=None, title="T", section_path=["A"], text=TEXT)
    assert _settings().extract_max_tokens == 2000
    req = extract.build_request(chunk, _settings(extract_max_tokens=4000))
    assert req["max_tokens"] == 4000
    assert extract.build_openai_request(chunk, _oa(extract_max_tokens=4000))["max_tokens"] == 4000
    small = extract._estimate(extract.build_request(chunk, _settings()), batch=False,
                              settings=_settings())
    large = extract._estimate(req, batch=False, settings=_settings())
    assert abs((large - small) - 2000 * 0.50 / 1e6) < 1e-12  # reserves the larger output


def test_prices_follow_the_model_and_refuse_an_unknown_one():
    assert extract.price_per_mtok("claude-haiku-5-5") == (0.10, 0.50)
    assert extract.price_per_mtok("claude-opus-5-5") == (4.00, 20.00)
    assert extract.price_per_mtok("anthropic/claude-opus-5.5:batch") == (4.00, 20.00)
    with pytest.raises(ValueError, match="no price known"):
        extract.price_per_mtok("qwen/qwen3.5-9b")
    override = _settings(extract_price_in_per_mtok=0.2, extract_price_out_per_mtok=0.6)
    assert extract.price_per_mtok("qwen/qwen3.5-9b", override) == (0.2, 0.6)
    chunk = extract.ChunkRow(chunk_id="c1", doc_id="d1", vendor="rubrik", product=None,
                             version=None, title="T", section_path=["A"], text=TEXT)
    opus = _settings(extract_model="claude-opus-5-5")
    haiku_est = extract._estimate(extract.build_request(chunk, _settings()), batch=False,
                                  settings=_settings())
    opus_est = extract._estimate(extract.build_request(chunk, opus), batch=False, settings=opus)
    assert abs(opus_est / haiku_est - 40) < 1e-9


def test_effort_follows_the_setting_on_both_backends():
    chunk = extract.ChunkRow(chunk_id="c1", doc_id="d1", vendor="rubrik", product=None,
                             version=None, title="T", section_path=["A"], text=TEXT)
    assert extract.build_request(chunk, _settings(extract_effort="medium"))[
        "output_config"]["effort"] == "medium"
    assert extract.build_openai_request(chunk, _oa(extract_effort="high"))[
        "extra_body"]["reasoning"] == {"effort": "high"}


async def test_run_refuses_a_model_without_a_price(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        with pytest.raises(ValueError, match="no price known"):
            await extract.run(conn, [], client=None, mode="sync",
                              settings=_settings(extract_model="mystery-model"))
    finally:
        await conn.close()


async def _extraction(conn, cid, *, status, error, model, attempts=3) -> None:
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status, error, attempts, last_batch_id) VALUES (%s,%s,%s,%s,%s,%s,%s,'sync-old')",
        (uuid.uuid4(), cid, model, extract.PROMPT_VERSION, status, error, attempts))


async def test_retry_truncated_resets_only_length_failures(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        cut, cut_anth, cut_errored, bad, other_model, outside = await _doc(
            conn, texts=tuple(f"{TEXT} #{n}" for n in range(6)))
        model = _oa().extract_model
        await _extraction(conn, cut, status="failed", error="length", model=model)
        await _extraction(conn, cut_anth, status="failed", error="max_tokens", model=model)
        await _extraction(conn, cut_errored, status="errored", error="length", model=model,
                          attempts=1)
        await _extraction(conn, bad, status="failed", error="invalid-json: x", model=model)
        await _extraction(conn, other_model, status="failed", error="length", model="other")
        await _extraction(conn, outside, status="failed", error="length", model=model)
        client = FakeOpenAI({"Amazon RDS": _chat({"claims": [GOOD]}, cost=0.001)})
        ids = [cut, cut_anth, cut_errored, bad, other_model]
        report = await extract.run(conn, ids, client=client, settings=_oa(), mode="sync",
                                   retry_truncated=True)
        assert report.reset_truncated == 3
        assert (report.done, report.skipped_failed) == (4, 1)  # other_model: a fresh key
        rows = dict(await _rows(conn, "SELECT chunk_id::text, status FROM extractions"
                                      " WHERE extractor_model=%s", model))
        assert rows[cut] == rows[cut_anth] == rows[cut_errored] == "done"
        assert rows[bad] == "failed" and rows[outside] == "failed"  # not in the ids: untouched
        attempts = await _rows(conn, "SELECT attempts FROM extractions WHERE chunk_id=%s",
                               cut)
        assert attempts == [(1,)]  # counter reset, then this run's attempt

        # Without the flag nothing is reset.
        report = await extract.run(conn, [outside], client=client, settings=_oa(),
                                   mode="sync")
        assert (report.reset_truncated, report.skipped_failed) == (0, 1)
    finally:
        await conn.close()


async def _status_check(conn):
    cur = await conn.execute(
        "SELECT oid, pg_get_constraintdef(oid) FROM pg_constraint"
        " WHERE conname='extractions_status_check' AND conrelid='extractions'::regclass")
    return await cur.fetchall()


async def test_init_schema_keeps_the_status_check_and_indexes_the_cascades(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        [(oid, definition)] = await _status_check(conn)
        assert "failed" in definition
        await init_schema(conn, embed_dim=768)
        assert await _status_check(conn) == [(oid, definition)]  # not dropped and re-added
        # a database from before the 'failed' status gets the new check
        await conn.execute("ALTER TABLE extractions DROP CONSTRAINT extractions_status_check")
        await conn.execute("ALTER TABLE extractions ADD CONSTRAINT extractions_status_check"
                           " CHECK (status IN ('done', 'errored'))")
        await conn.execute("DELETE FROM taxonomy_meta WHERE key = 'schema_version'")  # an older database
        await init_schema(conn, embed_dim=768)
        [(_, definition)] = await _status_check(conn)
        assert "failed" in definition
        cur = await conn.execute("SELECT tablename, indexdef FROM pg_indexes"
                                 " WHERE tablename IN ('raw_claims', 'rejected_claims')")
        defs = await cur.fetchall()
        for table, column in (("raw_claims", "extraction_id"), ("raw_claims", "chunk_id"),
                              ("rejected_claims", "extraction_id"),
                              ("rejected_claims", "chunk_id")):
            assert any(t == table and f"({column})" in d for t, d in defs), (table, column)
    finally:
        await conn.close()


async def _v3_checks(conn):
    return await _rows(conn, "SELECT conname, oid FROM pg_constraint"
                             " WHERE conrelid = 'raw_claims'::regclass AND conname IN"
                             " ('raw_claims_kind_check', 'raw_claims_availability_check',"
                             " 'raw_claims_default_state_check', 'raw_claims_bound_check')"
                             " ORDER BY conname")


async def test_init_schema_upgrades_pre_v3_claims(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        for table in ("raw_claims", "rejected_claims"):
            await conn.execute(f"ALTER TABLE {table} DROP COLUMN kind, DROP COLUMN target_scope,"
                               " DROP COLUMN offering, DROP COLUMN edition,"
                               " DROP COLUMN availability, DROP COLUMN default_state,"
                               " DROP COLUMN value, DROP COLUMN unit, DROP COLUMN bound")
        _, ids = await _seed_regate(conn)  # claims-v1 rows, stored before v3
        await conn.execute("DELETE FROM taxonomy_meta WHERE key = 'schema_version'")  # an older database
        await init_schema(conn, embed_dim=768)
        assert await _rows(conn, "SELECT DISTINCT kind, availability, value FROM raw_claims") \
            == [("capability", None, None)]
        checks = await _v3_checks(conn)
        assert len(checks) == 4
        await init_schema(conn, embed_dim=768)
        assert await _v3_checks(conn) == checks  # not dropped and re-added
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute("UPDATE raw_claims SET bound = 'up to' WHERE claim_id = %s",
                               (ids["good"],))
    finally:
        await conn.close()


REGATE_TEXT = ("Agents\n\nInstant recovery is unsupported on Linux servers. Backups of Amazon "
               "RDS for PostgreSQL instances are supported with immutable snapshots.")


async def _seed_regate(conn) -> tuple[str, dict[str, str]]:
    [cid] = await _doc(conn, texts=(REGATE_TEXT,))
    ext = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status, n_claims) VALUES (%s,%s,'m','claims-v1','done',3)", (ext, cid))
    claims = {
        # accepted by the old gate: "supported" matched inside "unsupported", widened
        "flip": ("instant recovery", "supported", "unsupported on Linux servers",
                 "supported on Linux servers"),
        # stored before evidence_quote held the chunk span: re-derived verbatim
        "good": ("immutable snapshots", "supported",
                 "backups of Amazon RDS for PostgreSQL instances are supported",
                 "backups of Amazon RDS for PostgreSQL instances are supported"),
        # from before model_quote: the evidence quote is the model's text
        "legacy": ("instant recovery", "not_supported",
                   "Instant recovery is unsupported on Linux servers", None),
    }
    ids = {}
    for key, (cap, support, evidence, model_quote) in claims.items():
        ids[key] = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor, product,"
            " capability_raw, support, evidence_quote, model_quote, gate, extractor_model,"
            " prompt_version) VALUES (%s,%s,%s,'rubrik','RSC',%s,%s,%s,%s,'exact','m',"
            "'claims-v1')", (ids[key], ext, cid, cap, support, evidence, model_quote))
    return ext, ids


async def test_regate_moves_claims_the_current_gate_rejects(vkb_dsn):
    from vendorkb.claims import regate

    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ext, ids = await _seed_regate(conn)
        await conn.execute(
            "UPDATE raw_claims SET kind='licensing', target_scope='users', offering='Express',"
            " edition='Premium', availability='preview', default_state='off', value=99,"
            " unit='years', bound='max' WHERE claim_id=%s", (ids["flip"],))
        dry = await regate.regate(conn, batch_size=2, dry_run=True)
        assert (dry.checked, dry.kept, dry.moved, dry.updated) == (3, 2, 1, 1)
        assert await _rows(conn, "SELECT count(*) FROM raw_claims") == [(3,)]

        report = await regate.regate(conn, batch_size=2)
        assert (report.checked, report.kept, report.moved, report.updated) == (3, 2, 1, 1)
        kept = dict(await _rows(conn, "SELECT claim_id::text, evidence_quote FROM raw_claims"))
        assert set(kept) == {ids["good"], ids["legacy"]}
        # the span is re-derived from the model's quote (verbatim chunk text)
        assert kept[ids["good"]] == "Backups of Amazon RDS for PostgreSQL instances are supported"
        [(cid, reason, evidence, model_quote, cap)] = await _rows(
            conn, "SELECT claim_id::text, reason, evidence_quote, model_quote, capability_raw"
                  " FROM rejected_claims")
        assert cid == ids["flip"] and reason.startswith("regate:no-match")
        assert (evidence, model_quote, cap) == (None, "supported on Linux servers",
                                                "instant recovery")
        assert await _rows(conn, "SELECT kind, target_scope, offering, edition, availability,"
                                 " default_state, value, unit, bound FROM rejected_claims") == [
            ("licensing", "users", "Express", "Premium", "preview", "off", 99.0, "years",
             "max")]
        assert await _rows(conn, "SELECT n_claims FROM extractions WHERE extraction_id=%s",
                           ext) == [(2,)]
        again = await regate.regate(conn)
        assert (again.checked, again.moved, again.updated) == (2, 0, 0)
    finally:
        await conn.close()


async def test_cli_claims_regate(vkb_dsn, monkeypatch, capsys):
    import asyncio

    from vendorkb import cli

    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _seed_regate(conn)
    finally:
        await conn.close()
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)
    code = await asyncio.to_thread(cli.main, ["claims", "regate", "--batch-size", "1"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["checked"], out["moved"]) == (3, 1)
