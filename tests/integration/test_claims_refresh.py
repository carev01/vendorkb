"""``vendorkb refresh`` (claims.refresh) against a real Postgres, with fake clients.

Every phrase here is invented for the tests; none comes from vendor documentation."""
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from pgvector import Vector

from vendorkb.claims import extract, refresh
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import load

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
TEXT = "Okta protection\n\nNightly Okta jobs keep backed up objects such as users for a year."
CLAIM = {"kind": "capability", "capability": "backed up objects", "target": "Okta",
         "target_scope": "users", "support": "supported", "offering": None, "edition": None,
         "availability": "unstated", "default_state": None, "value": None, "unit": None,
         "bound": None, "conditions": None,
         "evidence_quote": "Nightly Okta jobs keep backed up objects such as users"}


@pytest.fixture(autouse=True)
def _test_dsn(_no_dotenv_file, vkb_dsn, monkeypatch):
    """The refresh records its task runs at ``settings.vendorkb_dsn``: point it at the test
    container, never at the default local database."""
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)


def _settings(**overrides) -> VendorkbSettings:
    return VendorkbSettings(_env_file=None, **overrides, extract_backend="openai",
                            extract_model="vendorkb-test", extract_price_in_per_mtok=1e-6,
                            extract_price_out_per_mtok=1e-6, map_backend="openai",
                            map_model="anthropic/claude-haiku-5.5")


class FakeExtract:
    """OpenAI-shaped: every chunk yields CLAIM."""

    def __init__(self):
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **params):
        self.calls.append(params)
        message = SimpleNamespace(content=json.dumps({"claims": [CLAIM]}), refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=message)],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100, cost=0.0))


class NoCalls:
    """A client or embedder that must not be used (the claim maps through aliases)."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **params):
        raise AssertionError("unexpected adjudication")

    async def embed(self, texts, *, input_type="document"):
        raise AssertionError("unexpected embedding")


class Ready:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    async def __call__(self):
        self.calls += 1
        if self.fail:
            raise TimeoutError("the extraction endpoint did not answer in time")


async def _fresh(conn) -> None:
    await conn.execute("DROP TABLE IF EXISTS capabilities, raw_claims, rejected_claims, "
                       "extractions, extract_batches, chunks, documents CASCADE")
    await init_schema(conn, embed_dim=8)
    await load.load_taxonomy(conn, FIXTURE)


async def _chunk(conn, *, fetched_at=NOW - timedelta(hours=3), text=TEXT) -> str:
    doc_id, chunk_id = str(uuid.uuid4()), str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s,'examplevendor','P','S','Doc','http://docs/x',%s,%s)",
        (doc_id, doc_id, fetched_at))
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s,%s,%s,0,%s,20,%s)",
        (chunk_id, doc_id, ["Okta protection"], text, Vector([0.1] * 8)))
    return chunk_id


async def _run(conn, client=None, ready=None, *, settle=60, limit=100):
    return await refresh.refresh(
        conn, settings=_settings(), extract_client=client or FakeExtract(),
        map_client=NoCalls(), support_client=NoCalls(), embedder=NoCalls(),
        ready=ready or Ready(),
        settle_minutes=settle, extract_limit=limit, map_max_usd=1.0, now=NOW)


async def _count(conn, sql) -> int:
    cur = await conn.execute(sql)
    return (await cur.fetchone())[0]


async def test_nothing_new_contacts_nothing(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        client, ready = FakeExtract(), Ready()
        report = await _run(conn, client, ready)
        assert (report.new_chunks, report.due) == (0, False)
        assert client.calls == [] and ready.calls == 0
        assert report.extract is None and report.map is None
        assert report.capabilities_rows is None
        await conn.execute("DELETE FROM task_runs")
        await _run(conn)
        assert await _count(conn, "SELECT count(*) FROM task_runs WHERE kind = 'refresh'") == 1
    finally:
        await conn.close()


async def test_new_chunks_skip_superseded_chunks(vkb_dsn):
    """An earlier version of a page, kept for its claims, never triggers a run."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        superseded = await _chunk(conn, fetched_at=NOW - timedelta(minutes=5))
        await conn.execute("UPDATE chunks SET superseded_at = now() WHERE chunk_id = %s",
                           (superseded,))
        assert await refresh.new_chunks(conn) == (0, None)
        client, ready = FakeExtract(), Ready()
        report = await _run(conn, client, ready)
        assert (report.new_chunks, report.due) == (0, False)
        assert client.calls == [] and ready.calls == 0

        await _chunk(conn, fetched_at=NOW - timedelta(hours=3))
        assert await refresh.new_chunks(conn) == (1, NOW - timedelta(hours=3))
    finally:
        await conn.close()


async def test_a_recent_burst_waits_for_the_feed_to_settle(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _chunk(conn, fetched_at=NOW - timedelta(hours=3))
        await _chunk(conn, fetched_at=NOW - timedelta(minutes=10))  # the feed is still busy
        client, ready = FakeExtract(), Ready()
        report = await _run(conn, client, ready, settle=60)
        assert (report.new_chunks, report.due) == (2, False)
        assert "settling" in report.reason
        assert client.calls == [] and ready.calls == 0
    finally:
        await conn.close()


async def test_a_backlog_that_fills_a_run_does_not_wait(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        for _ in range(3):
            await _chunk(conn, fetched_at=NOW - timedelta(minutes=5))
        client = FakeExtract()
        report = await _run(conn, client, settle=60, limit=2)
        assert report.due and report.extract["chunks"] == 2 and len(client.calls) == 2
    finally:
        await conn.close()


async def test_errored_chunks_never_trigger_a_run(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        cid = await _chunk(conn)
        await conn.execute(
            "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
            " status, error, attempts) VALUES (%s,%s,'vendorkb-test',%s,'errored','length',1)",
            (str(uuid.uuid4()), cid, extract.PROMPT_VERSION))
        client, ready = FakeExtract(), Ready()
        report = await _run(conn, client, ready)
        assert (report.new_chunks, report.due) == (0, False)
        assert client.calls == [] and ready.calls == 0
    finally:
        await conn.close()


async def test_an_endpoint_that_never_answers_fails_before_extracting(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _chunk(conn)
        client = FakeExtract()
        with pytest.raises(TimeoutError):
            await _run(conn, client, Ready(fail=True))
        assert client.calls == []
        assert await _count(conn, "SELECT count(*) FROM extractions") == 0
    finally:
        await conn.close()


async def test_a_due_run_extracts_maps_and_rebuilds_capabilities(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        new = await _chunk(conn)
        # an errored chunk rides along with the run that new data triggered
        errored = await _chunk(conn)
        await conn.execute(
            "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
            " status, error, attempts) VALUES (%s,%s,'vendorkb-test',%s,'errored','length',1)",
            (str(uuid.uuid4()), errored, extract.PROMPT_VERSION))
        client, ready = FakeExtract(), Ready()
        report = await _run(conn, client, ready)
        assert (report.new_chunks, report.due, ready.calls) == (1, True, 1)
        assert report.extract["done"] == 2 and len(client.calls) == 2
        assert report.map["claims"] == 2 and report.map["dimension_alias"] == 2
        assert report.capabilities_rows and report.capabilities_rows > 0
        version = await load.current_version(conn)
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE mapped_version IS"
                                  f" DISTINCT FROM '{version}'") == 0
        assert new and await _count(conn, "SELECT count(*) FROM capabilities") > 0

        again = await _run(conn, client, ready)  # nothing new: the endpoint is left alone
        assert (again.due, ready.calls, len(client.calls)) == (False, 1, 2)
    finally:
        await conn.close()


class SupportVerdicts:
    """OpenAI-shaped support checker: every claim is supported. Taxonomy-map adjudications
    must not reach it (the claims map through aliases)."""

    def __init__(self):
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **params):
        assert params["response_format"]["json_schema"]["name"] == "support_verdict"
        self.calls.append(params)
        message = SimpleNamespace(content=json.dumps({"supported": True}), refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=message)],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=20, cost=0.0002))


async def test_a_due_run_support_checks_the_new_claims_of_topic_dimensions(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
        doc["topics"] = [{"id": "everything", "name": "Everything", "triggers": ["everything"],
                          "aspects": [{"id": "all", "name": "All", "query": "All",
                                       "dimensions": [d["id"] for d in doc["dimensions"]]}]}]
        path = tmp_path / "taxonomy.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        await load.load_taxonomy(conn, path)
        await _chunk(conn)
        checker = SupportVerdicts()
        report = await refresh.refresh(
            conn, settings=_settings(support_backend="openai"), extract_client=FakeExtract(),
            map_client=NoCalls(), support_client=checker, embedder=NoCalls(), ready=Ready(), settle_minutes=60, extract_limit=100,
            map_max_usd=1.0, support_max_usd=1.0, support_review_max_usd=1.0, now=NOW)
        assert report.map["claims"] == 1 and report.map["llm_calls"] == 0
        assert report.support["check"]["calls"] == 1 and len(checker.calls) == 1
        assert report.support["review"]["claims"] == 0
        assert await _count(conn, "SELECT count(*) FROM claim_support WHERE supported") == 1
        assert report.capabilities_rows == 1
    finally:
        await conn.close()


async def test_a_failed_support_step_still_builds_what_was_mapped(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
        doc["topics"] = [{"id": "everything", "name": "Everything", "triggers": ["everything"],
                          "aspects": [{"id": "all", "name": "All", "query": "All",
                                       "dimensions": [d["id"] for d in doc["dimensions"]]}]}]
        path = tmp_path / "taxonomy.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        await load.load_taxonomy(conn, path)
        await _chunk(conn)

        class Unknown(SupportVerdicts):
            async def create(self, **params):
                error = RuntimeError("model not found")
                error.status_code = 404  # type: ignore[attr-defined]
                raise error

        report = await refresh.refresh(
            conn, settings=_settings(support_backend="openai"), extract_client=FakeExtract(),
            map_client=NoCalls(), support_client=Unknown(), embedder=NoCalls(), ready=Ready(), settle_minutes=60, extract_limit=100,
            map_max_usd=1.0, support_max_usd=1.0, support_review_max_usd=1.0, now=NOW)
        assert report.support == {"error": "RuntimeError: model not found"}
        assert report.capabilities_rows == 1
    finally:
        await conn.close()


async def test_rejections_stored_before_a_support_failure_are_built(vkb_dsn, tmp_path):
    """A due run whose extraction and mapping change nothing still builds when its reviewer
    rejected a claim before the support step failed."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
        doc["topics"] = [{"id": "everything", "name": "Everything", "triggers": ["everything"],
                          "aspects": [{"id": "all", "name": "All", "query": "All",
                                       "dimensions": [d["id"] for d in doc["dimensions"]]}]}]
        path = tmp_path / "taxonomy.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        await load.load_taxonomy(conn, path)
        await _chunk(conn)
        await _chunk(conn)
        first = await refresh.refresh(
            conn, settings=_settings(support_backend="openai"), extract_client=FakeExtract(),
            map_client=NoCalls(), support_client=SupportVerdicts(), embedder=NoCalls(),
            ready=Ready(), settle_minutes=60, extract_limit=100, map_max_usd=1.0,
            support_max_usd=1.0, support_review_max_usd=0.0, now=NOW)
        assert first.capabilities_rows == 2
        await conn.execute("UPDATE claim_support SET supported = false")  # both flagged
        # a new chunk whose extraction errors: the run is due but extracts and maps nothing
        await _chunk(conn, text="Unrelated text.")

        class Broken(FakeExtract):
            async def create(self, **params):  # an unusable reply: the chunk is errored
                message = SimpleNamespace(content="not json", refusal=None)
                return SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="stop", message=message)],
                    usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100, cost=0.0))

        class RejectThenFail(SupportVerdicts):
            async def create(self, **params):
                if self.calls:
                    error = RuntimeError("model not found")
                    error.status_code = 404  # type: ignore[attr-defined]
                    raise error
                self.calls.append(params)
                message = SimpleNamespace(content=json.dumps({"supported": False}),
                                          refusal=None)
                return SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="stop", message=message)],
                    usage=SimpleNamespace(prompt_tokens=300, completion_tokens=20, cost=0.002))

        # the review rejects the first flagged claim, then the second one hits a 404
        second = await refresh.refresh(
            conn, settings=_settings(support_backend="openai", support_concurrency=1),
            extract_client=Broken(), map_client=NoCalls(), support_client=RejectThenFail(),
            embedder=NoCalls(), ready=Ready(), settle_minutes=60, extract_limit=100,
            map_max_usd=1.0, support_max_usd=1.0, support_review_max_usd=1.0, now=NOW)
        assert second.extract["done"] == 0 and "error" in second.support
        assert await _count(conn, "SELECT count(*) FROM claim_support"
                                  " WHERE NOT review_supported") == 1
        assert second.capabilities_rows == 1  # the rejected claim left capabilities
    finally:
        await conn.close()


async def test_wait_until_ready_polls_until_the_endpoint_answers():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["authorization"]))
        return httpx.Response(200 if len(seen) >= 3 else 503)

    await refresh.wait_until_ready("http://serve.test/v1/", "k", timeout_s=60, poll_s=0,
                                   transport=httpx.MockTransport(handler))
    assert seen == [("http://serve.test/v1/models", "Bearer k")] * 3

    with pytest.raises(TimeoutError):
        await refresh.wait_until_ready("http://serve.test/v1", "k", timeout_s=0, poll_s=0,
                                       transport=httpx.MockTransport(
                                           lambda r: httpx.Response(503)))


# --- batch mode (BACKLOG 37) -------------------------------------------------------------

def _batch_settings() -> VendorkbSettings:
    return VendorkbSettings(_env_file=None, extract_backend="openai",
                            extract_model="vendorkb-test", extract_price_in_per_mtok=1e-6,
                            extract_price_out_per_mtok=1e-6, map_backend="anthropic",
                            map_model="claude-haiku-5-5", support_backend="anthropic",
                            support_model="claude-haiku-5-5",
                            support_review_model="claude-opus-5-5")


async def _with_topic(conn, tmp_path) -> None:
    doc = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    doc["topics"] = [{"id": "everything", "name": "Everything", "triggers": ["everything"],
                      "aspects": [{"id": "all", "name": "All", "query": "All",
                                   "dimensions": [d["id"] for d in doc["dimensions"]]}]}]
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    await load.load_taxonomy(conn, path)


async def test_batch_mode_publishes_new_facts_only_once_checked_and_reviewed(vkb_dsn, tmp_path):
    from tests.integration.test_claims_support import FakeBatchAPI

    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await conn.execute("DROP TABLE IF EXISTS llm_batches, llm_batch_items, claim_support"
                           " CASCADE")
        await init_schema(conn, embed_dim=8)
        await _with_topic(conn, tmp_path)
        await _chunk(conn)
        api = FakeBatchAPI({("check", "backed up objects"): False,  # flagged by the checker,
                            ("review", "backed up objects"): True},  # kept by the reviewer
                           auto_end=False)
        ready, extractor = Ready(), FakeExtract()

        async def run(extract_new):
            return await refresh.refresh(
                conn, settings=_batch_settings(), extract_client=extractor, map_client=NoCalls(),
                support_client=api, embedder=NoCalls(), ready=ready, settle_minutes=60,
                extract_limit=100, map_max_usd=1.0, support_max_usd=1.0,
                support_review_max_usd=1.0, batch=True, extract_new=extract_new, now=NOW)

        first = await run(True)  # the weekly run: extract, map (aliases), submit the check
        assert (first.mode, first.due, ready.calls, first.extract["done"]) == ("batch", True, 1, 1)
        assert first.support["check_batch"]["submitted"] == 1
        assert first.capabilities_deferred and first.capabilities_rows is None
        assert await _count(conn, "SELECT count(*) FROM capabilities") == 0

        hourly = await run(False)  # the check is open: due, but no extraction, no wake
        assert (hourly.due, hourly.open_batches, hourly.extract) == (True, 1, None)
        assert ready.calls == 1
        # nothing landed: nothing to build, nothing asked again
        assert hourly.capabilities_rows is None and len(api.created) == 1

        api.end()
        second = await run(False)  # the check lands (flagged): the review is submitted
        assert api.steps()[1] == ("review", ["backed up objects"])
        assert second.capabilities_deferred
        assert await _count(conn, "SELECT count(*) FROM capabilities") == 0

        api.end()
        third = await run(False)  # the review lands (kept): quiescent, the fact is published
        assert not third.capabilities_deferred and third.capabilities_rows == 1
        assert await _count(conn, "SELECT count(*) FROM capabilities") == 1

        idle = await run(False)  # nothing open, nothing new
        assert (idle.due, idle.reason) == (False, "no new chunks")
        assert len(extractor.calls) == 1 and ready.calls == 1
    finally:
        await conn.close()


async def test_the_collect_run_never_extracts_new_chunks(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await conn.execute("DROP TABLE IF EXISTS llm_batches, llm_batch_items CASCADE")
        await init_schema(conn, embed_dim=8)
        await _chunk(conn)
        ready = Ready()
        report = await refresh.refresh(
            conn, settings=_batch_settings(), extract_client=FakeExtract(),
            map_client=NoCalls(), support_client=NoCalls(), embedder=NoCalls(), ready=ready,
            settle_minutes=60, extract_limit=100, map_max_usd=1.0, batch=True,
            extract_new=False, now=NOW)
        assert (report.new_chunks, report.due, ready.calls) == (1, False, 0)
        assert report.reason == "extraction is left to the extracting run"
    finally:
        await conn.close()
