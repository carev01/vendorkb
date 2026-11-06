"""``taxonomy map`` (v2, with LLM adjudication) against a real Postgres, with a fake client and
embedder. The fixture taxonomy is extended here with a few synthetic entries.

Every phrase here is invented for the tests; none comes from vendor documentation."""
import hashlib
import json
import math
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
import yaml
from pgvector import Vector

from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import load, mapper

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
DIM = 32


def _axis(i: int, wobble: float = 0.0, j: int | None = None) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    if j is not None:
        v[j] = wobble
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _hashed(text: str) -> list[float]:
    """A deterministic pseudo-random unit vector: unrelated strings are near-orthogonal."""
    raw = hashlib.sha256(text.encode()).digest()
    v = [(b - 127.5) for b in raw[:DIM]]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


VECTORS = {
    "okta": _axis(0),
    "okta workforce": _axis(0, 0.05, 10),
    "skip automations during restore": _axis(1),
    "hold off workflows during restore": _axis(1, 0.05, 11),
    "longest retention": _axis(2),
    "keep copies for ages": _axis(2, 0.05, 12),
}


class FakeEmbedder:
    def __init__(self, table: dict[str, list[float]] | None = None):
        self.table = VECTORS if table is None else table
        self.calls: list[tuple[list[str], str]] = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append((list(texts), input_type))
        return [self.table.get(t, _hashed(t)) for t in texts]


def _phrase(user: str) -> str:
    return user.split("Phrase: ", 1)[1].split("\n", 1)[0]


def _kind(user: str) -> str:
    return user.split("Kind: ", 1)[1].split("\n", 1)[0]


def _candidates(user: str) -> list[str]:
    block = user.split("Candidates (id | name | details):\n", 1)[1]
    return [line[2:].split(" | ", 1)[0] for line in block.split("\n\n", 1)[0].splitlines()]


def _chat(payload, *, cost):
    message = SimpleNamespace(content=json.dumps(payload), refusal=None)
    usage = SimpleNamespace(prompt_tokens=400, completion_tokens=20, cost=cost)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)],
                           usage=usage)


def _pick(wanted: str):
    """Choose ``wanted`` when it is a candidate, else none."""
    return lambda candidates: wanted if wanted in candidates else "none"


REPLIES = {
    "okta workforce": _pick("okta"),
    "hold off workflows during restore": _pick("recovery.automation_bypass"),
    "keep copies for ages": _pick("retention.maximum_period"),
    "the dashboard thing": lambda candidates: "none",
    "mystery platform": lambda candidates: "none",
    "made up answer": lambda candidates: "not.a.candidate",
}


class FakeCompletions:
    def __init__(self, replies, cost):
        self.replies = replies
        self.cost = cost
        self.calls: list[dict] = []

    async def create(self, **params):
        self.calls.append(params)
        user = params["messages"][-1]["content"]
        choose = self.replies.get(_phrase(user))
        if choose is None:
            raise AssertionError(f"unexpected adjudication: {_phrase(user)!r}")
        return _chat({"choice": choose(_candidates(user))}, cost=self.cost)


class FakeOpenAI:
    def __init__(self, replies=None, cost=0.0005):
        self.chat = SimpleNamespace(completions=FakeCompletions(
            REPLIES if replies is None else replies, cost))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


def _settings(**kw) -> VendorkbSettings:
    return VendorkbSettings(_env_file=None, **{"map_backend": "openai",
                                               "map_model": "anthropic/claude-haiku-5.5"} | kw)


def _extended(tmp_path) -> Path:
    """The fixture plus FSx (with variants), Azure and Azure SQL, and a second offering of
    examplevendor that also has a Premium edition (edition id ``premium`` in both offerings;
    the alias "premium plus" only in the second)."""
    doc = yaml.safe_load(FIXTURE.read_text())
    doc["targets"] += [
        {"id": "aws.fsx", "name": "Amazon FSx", "class": "cloud", "aliases": ["FSx"],
         "variants": [{"id": "lustre", "name": "FSx for Lustre", "aliases": []},
                      {"id": "windows", "name": "FSx for Windows File Server",
                       "aliases": []}]},
        {"id": "azure", "name": "Microsoft Azure", "class": "cloud", "aliases": ["Azure"]},
        {"id": "azure.sql", "name": "Azure SQL Database", "class": "cloud.database",
         "aliases": []},
    ]
    doc["offerings"].append({
        "vendor": "examplevendor", "id": "examplevendor.onprem", "name": "Example On-Prem",
        "aliases": [], "editions": [{"id": "premium", "name": "Premium",
                                     "aliases": ["premium edition", "premium plus"]}]})
    out = tmp_path / "taxonomy.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


async def _fresh(conn, tmp_path) -> None:
    await conn.execute("DROP TABLE IF EXISTS raw_claims, rejected_claims, extractions, "
                       "extract_batches, chunks, documents CASCADE")
    await init_schema(conn, embed_dim=8)
    await load.load_taxonomy(conn, _extended(tmp_path))


async def _claim(conn, capability="some capability", target=None, *, vendor="examplevendor",
                 target_scope=None, offering=None, edition=None,
                 prompt_version="claims-v3.1") -> uuid.UUID:
    doc_id, chunk_id, ext_id, claim_id = (uuid.uuid4() for _ in range(4))
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s,%s,'P','S','Doc','http://docs/x',%s, now())",
        (doc_id, vendor, str(doc_id)))
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s,%s,%s,0,'synthetic text',10,%s)",
        (chunk_id, doc_id, ["A"], Vector([0.1] * 8)))
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s,%s,'m',%s,'done')", (ext_id, chunk_id, prompt_version))
    await conn.execute(
        "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor, product,"
        " capability_raw, target_raw, target_scope, offering, edition, support,"
        " evidence_quote, gate, extractor_model, prompt_version) VALUES"
        " (%s,%s,%s,%s,'P',%s,%s,%s,%s,%s,'supported','synthetic quote','exact','m',%s)",
        (claim_id, ext_id, chunk_id, vendor, capability, target, target_scope, offering,
         edition, prompt_version))
    return claim_id


COLUMNS = ("dimension_id", "dimension_method", "class_id", "target_id", "target_method",
           "variant_id", "object_type_id", "offering_id", "edition_id", "mapped_version")


async def _mapping(conn, claim_id) -> dict:
    cur = await conn.execute(f"SELECT {', '.join(COLUMNS)} FROM raw_claims WHERE claim_id=%s",
                             (claim_id,))
    return dict(zip(COLUMNS, await cur.fetchone()))


async def _run(conn, client=None, embedder=None, max_usd=1.0, **settings):
    return await mapper.map_claims(conn, embedder=embedder or FakeEmbedder(),
                                   client=client or FakeOpenAI(),
                                   settings=_settings(**settings), max_usd=max_usd)


async def test_alias_hits_make_no_llm_call(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        cid = await _claim(conn, "Backed  up objects", "Okta", target_scope="users",
                           offering="ECS", edition="premium edition")
        dim_only = await _claim(conn, "longest retention")  # no target: product-wide
        client, embedder = FakeOpenAI(), FakeEmbedder()
        report = await _run(conn, client, embedder)
        assert client.calls == [] and embedder.calls == []  # nothing to adjudicate
        version = await load.current_version(conn)
        assert await _mapping(conn, cid) == {
            "dimension_id": "coverage.backup", "dimension_method": "alias",
            "class_id": "saas.identity", "target_id": "okta", "target_method": "alias",
            "variant_id": None, "object_type_id": "user",
            "offering_id": "examplevendor.cloud_suite", "edition_id": "premium",
            "mapped_version": version}
        m = await _mapping(conn, dim_only)
        assert (m["dimension_id"], m["target_id"], m["target_method"], m["class_id"]) == (
            "retention.maximum_period", None, None, None)
        assert (report.claims, report.dimension_alias, report.target_alias,
                report.target_none, report.object_type, report.llm_calls) == (2, 2, 1, 1, 1, 0)
        assert report.spent_usd == 0.0 and report.taxonomy_version == version
    finally:
        await conn.close()


async def test_llm_choice_and_none(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        chosen = await _claim(conn, "keep copies for ages", "Okta Workforce")
        none = await _claim(conn, "the dashboard thing", "Okta")
        client = FakeOpenAI()
        report = await _run(conn, client)
        m = await _mapping(conn, chosen)
        assert (m["target_id"], m["target_method"], m["class_id"]) == (
            "okta", "llm", "saas.identity")
        assert (m["dimension_id"], m["dimension_method"]) == ("retention.maximum_period", "llm")
        m = await _mapping(conn, none)
        assert (m["dimension_id"], m["dimension_method"]) == (None, "unmapped")
        assert (m["target_id"], m["target_method"]) == ("okta", "alias")
        assert (report.target_llm, report.dimension_llm, report.dimension_unmapped) == (1, 1, 1)
        assert (report.llm_calls, report.llm_none, report.llm_invalid) == (3, 1, 0)
        assert abs(report.spent_usd - 0.0015) < 1e-9  # the reported costs, summed

        req = next(c for c in client.calls
                   if _phrase(c["messages"][-1]["content"]) == "okta workforce")
        assert req["model"] == "anthropic/claude-haiku-5.5"
        assert req["response_format"]["json_schema"]["strict"] is True
        schema = req["response_format"]["json_schema"]["schema"]
        assert schema["required"] == ["choice"] and schema["additionalProperties"] is False
        user = req["messages"][-1]["content"]
        candidates = _candidates(user)
        assert len(candidates) == 5 and candidates[0] == "okta"  # nearest first
        assert schema["properties"]["choice"]["enum"] == [*candidates, "none"]
        assert _kind(user) == "target"
        # the claim's capability, target, target scope and vendor are the context
        assert "keep copies for ages | Okta Workforce | - | examplevendor" in user
    finally:
        await conn.close()


async def test_target_scope_resolution(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        lustre = await _claim(conn, "longest retention", "Amazon FSx",
                              target_scope="FSx for Lustre")
        oracle = await _claim(conn, "longest retention", "RDS", target_scope="RDS for Oracle")
        oracle_from_aws = await _claim(conn, "longest retention", "AWS",
                                       target_scope="RDS for Oracle")
        users = await _claim(conn, "longest retention", "Okta", target_scope="Users")
        sign_on = await _claim(conn, "longest retention", "Okta",
                               target_scope="sign-on policies")
        no_target = await _claim(conn, "longest retention", None,
                                 target_scope="RDS for Oracle")
        unrelated = await _claim(conn, "longest retention", "AWS",
                                 target_scope="Azure SQL Database")
        same = await _claim(conn, "longest retention", "Salesforce", target_scope="SFDC")
        # a target text that did not map: the scope is not taken as the target
        unmapped_target = await _claim(conn, "longest retention", "Mystery Platform",
                                       target_scope="RDS for Oracle")
        report = await _run(conn)

        def picked(m):
            return m["target_id"], m["class_id"], m["variant_id"], m["object_type_id"]

        assert picked(await _mapping(conn, lustre)) == ("aws.fsx", "cloud", "lustre", None)
        assert picked(await _mapping(conn, oracle)) == (
            "aws.rds.oracle", "cloud.database", None, None)
        assert picked(await _mapping(conn, oracle_from_aws)) == (
            "aws.rds.oracle", "cloud.database", None, None)
        assert picked(await _mapping(conn, users)) == ("okta", "saas.identity", None, "user")
        assert picked(await _mapping(conn, sign_on)) == (
            "okta", "saas.identity", None, "access_policy")
        m = await _mapping(conn, no_target)
        assert picked(m) == ("aws.rds.oracle", "cloud.database", None, None)
        assert m["target_method"] == "alias"
        # Review Focus 5: an unrelated target named in the scope is never taken as a child
        assert picked(await _mapping(conn, unrelated)) == ("aws", "cloud", None, None)
        # the target itself again is not a child of itself
        assert picked(await _mapping(conn, same)) == ("salesforce", "saas.crm", None, None)
        m = await _mapping(conn, unmapped_target)
        assert picked(m) == (None, None, None, None) and m["target_method"] == "unmapped"
        assert (report.child_target, report.scope_target, report.variant,
                report.object_type, report.scope_unmapped) == (2, 1, 1, 2, 3)
    finally:
        await conn.close()


async def test_dimension_candidates_only_from_applicable_dimensions(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        phrase = "hold off workflows during restore"
        identity = await _claim(conn, phrase, "Okta")
        identity_again = await _claim(conn, phrase, "Entra")  # same class: cached
        crm = await _claim(conn, phrase, "Salesforce")
        cloud = await _claim(conn, phrase, "AWS")
        client = FakeOpenAI()
        report = await _run(conn, client)
        by_class = {}
        for c in client.calls:
            user = c["messages"][-1]["content"]
            assert _kind(user) == "dimension"
            by_class[user.split("Workload class: ", 1)[1].split("\n", 1)[0]] = _candidates(user)
        assert len(client.calls) == 3  # one per distinct candidate set
        assert by_class["saas.crm"][0] == "recovery.automation_bypass"
        assert "recovery.automation_bypass" not in by_class["saas.identity"]
        assert "recovery.automation_bypass" not in by_class["cloud"]
        assert "schedule.backup_frequency" not in by_class["cloud"]  # applies to saas only
        assert (await _mapping(conn, crm))["dimension_id"] == "recovery.automation_bypass"
        for cid in (identity, identity_again, cloud):
            assert (await _mapping(conn, cid))["dimension_method"] == "unmapped"
        assert report.adjudications == 3
    finally:
        await conn.close()


async def test_null_target_draws_from_every_dimension(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        cid = await _claim(conn, "hold off workflows during restore")
        client = FakeOpenAI()
        await _run(conn, client)
        user = client.calls[0]["messages"][-1]["content"]
        assert "Workload class: (none)" in user
        assert _candidates(user)[0] == "recovery.automation_bypass"
        assert (await _mapping(conn, cid))["dimension_id"] == "recovery.automation_bypass"
    finally:
        await conn.close()


async def test_coverage_needs_an_object_type(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        bare = await _claim(conn, "backed up objects", "Okta")
        with_type = await _claim(conn, "backed up objects", "Okta", target_scope="groups")
        report = await _run(conn)
        m = await _mapping(conn, bare)
        assert (m["dimension_id"], m["object_type_id"]) == ("coverage.backup", None)  # kept
        assert (await _mapping(conn, with_type))["object_type_id"] == "group"
        assert report.coverage_without_object_type == 1
    finally:
        await conn.close()


async def test_offering_and_edition_per_vendor(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        both = await _claim(conn, "longest retention", offering="the SaaS  offering",
                            edition="Premium edition")
        unique_edition = await _claim(conn, "longest retention", edition="standard edition")
        ambiguous = await _claim(conn, "longest retention", edition="premium")
        other_case = await _claim(conn, "longest retention", vendor="ExampleVendor",
                                  offering="ECS", edition="standard")
        unknown = await _claim(conn, "longest retention", offering="mystery cloud",
                               edition="gold")
        onprem = await _claim(conn, "longest retention", offering="Example On-Prem",
                              edition="premium")
        # an offering text that did not map: no vendor-wide edition search
        unmapped_offering = await _claim(conn, "longest retention", offering="mystery cloud",
                                         edition="standard edition")
        # edition id "premium" is in both offerings, the alias in one: unique by (offering,
        # edition), never ambiguous by edition id
        shared_id = await _claim(conn, "longest retention", edition="premium plus")
        report = await _run(conn)

        async def pair(cid):
            m = await _mapping(conn, cid)
            return m["offering_id"], m["edition_id"]

        assert await pair(both) == ("examplevendor.cloud_suite", "premium")
        assert await pair(unique_edition) == ("examplevendor.cloud_suite", "standard")
        assert await pair(ambiguous) == (None, None)
        assert await pair(other_case) == (None, None)  # vendor text matches exactly
        assert await pair(unknown) == (None, None)
        assert await pair(onprem) == ("examplevendor.onprem", "premium")
        assert await pair(unmapped_offering) == (None, None)
        assert await pair(shared_id) == ("examplevendor.onprem", "premium")
        assert (report.offering_mapped, report.offering_unmapped, report.offering_none,
                report.offering_from_edition) == (2, 3, 3, 2)
        assert (report.edition_mapped, report.edition_unmapped) == (4, 4)
    finally:
        await conn.close()


async def test_budget_stop_leaves_claims_unmapped(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        first = await _claim(conn, "keep copies for ages", "Okta")
        second = await _claim(conn, "the dashboard thing", "Okta Workforce")
        aliased = await _claim(conn, "longest retention", "Okta")
        client = FakeOpenAI(cost=0.0005)
        # one call fits (its estimate is ~0.0006), the next would pass the cap
        report = await _run(conn, client, max_usd=0.0009, extract_concurrency=1)
        assert len(client.calls) == 1 and report.stopped_by_budget
        assert report.skipped_by_budget == 2 and report.llm_calls == 1
        assert report.claims_skipped_by_budget == 2
        maps = [await _mapping(conn, c) for c in (first, second)]
        assert sum(m["dimension_method"] == "unmapped" or m["target_method"] == "unmapped"
                   for m in maps) == 2
        version = await load.current_version(conn)
        assert all(m["mapped_version"] == version for m in maps)  # stamped all the same
        assert (await _mapping(conn, aliased))["dimension_id"] == "retention.maximum_period"

        nothing = FakeOpenAI()
        report = await _run(conn, nothing, max_usd=0.0)
        # the target call and both dimension calls (one over every dimension now)
        assert nothing.calls == [] and report.skipped_by_budget == 3
        assert (await _mapping(conn, first))["dimension_method"] == "unmapped"
    finally:
        await conn.close()


async def test_invalid_reply_is_unmapped_and_counted(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        cid = await _claim(conn, "made up answer", "Okta")
        report = await _run(conn)
        assert (await _mapping(conn, cid))["dimension_method"] == "unmapped"
        assert report.llm_invalid == 1 and "made up answer" in report.errors[0]
    finally:
        await conn.close()


async def test_bad_request_writes_nothing(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        cid = await _claim(conn, "keep copies for ages", "Okta")

        class Broken:
            async def create(self, **params):
                raise openai.BadRequestError(
                    "bad model", response=httpx.Response(
                        400, request=httpx.Request("POST", "http://x")), body=None)

        client = SimpleNamespace(chat=SimpleNamespace(completions=Broken()))
        with pytest.raises(openai.BadRequestError):
            await _run(conn, client)
        assert (await _mapping(conn, cid))["mapped_version"] is None
    finally:
        await conn.close()


async def test_keys_embedded_once_as_float32(vkb_dsn, tmp_path, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        await _claim(conn, "keep copies for ages", "Okta Workforce")
        await _claim(conn, "the dashboard thing", "Okta Workforce", target_scope="nope")
        embedder = FakeEmbedder()
        seen = []
        real = mapper._unit

        def spy(vectors):
            out = real(vectors)
            seen.append(out.dtype)
            return out

        monkeypatch.setattr(mapper, "_unit", spy)
        report = await _run(conn, embedder=embedder)
        assert len(embedder.calls) == 1
        texts = embedder.calls[0][0]
        assert len(texts) == len(set(texts)) == report.embedded_strings
        assert "okta" in texts and "okta workforce" in texts and "the dashboard thing" in texts
        assert seen and all(str(d) == "float32" for d in seen)
    finally:
        await conn.close()


async def test_other_prompt_versions_untouched(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        old = [await _claim(conn, "longest retention", "Okta", prompt_version=v)
               for v in ("claims-v1", "claims-v2")]
        await conn.execute("UPDATE raw_claims SET target_id='legacy', target_method='nn',"
                           " mapped_version='1-old'")
        v3 = await _claim(conn, "longest retention", "Okta", prompt_version="claims-v3")
        report = await _run(conn)
        assert report.claims == 1
        for cid in old:
            m = await _mapping(conn, cid)
            assert (m["target_id"], m["target_method"], m["mapped_version"],
                    m["dimension_id"]) == ("legacy", "nn", "1-old", None)
        assert (await _mapping(conn, v3))["target_id"] == "okta"
    finally:
        await conn.close()


async def test_anthropic_backend_uses_the_map_model(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        await _claim(conn, "keep copies for ages", "Okta")
        calls = []

        class Messages:
            async def create(self, **params):
                calls.append(params)
                text = json.dumps({"choice": "retention.maximum_period"})
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=300, output_tokens=10))

        report = await mapper.map_claims(
            conn, embedder=FakeEmbedder(), client=SimpleNamespace(messages=Messages()),
            settings=VendorkbSettings(_env_file=None, extract_model="claude-opus-5-5"),
            max_usd=1.0)
        assert calls[0]["model"] == "claude-haiku-5-5"
        fmt = calls[0]["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["properties"]["choice"]["enum"][-1] == "none"
        assert report.dimension_llm == 1
        assert abs(report.spent_usd - (300 * 0.10 + 10 * 0.50) / 1e6) < 1e-12
    finally:
        await conn.close()


async def test_no_taxonomy_refuses(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=8)
        await conn.execute("DELETE FROM taxonomy_meta")
        with pytest.raises(RuntimeError, match="no taxonomy"):
            await _run(conn)
    finally:
        await conn.close()


async def test_schema_checks_are_idempotent(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=8)
        await init_schema(conn, embed_dim=8)
        cur = await conn.execute(
            "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint"
            " WHERE conrelid = 'raw_claims'::regclass AND conname LIKE '%method_check'")
        checks = dict(await cur.fetchall())
        assert "llm" in checks["raw_claims_target_method_check"]
        assert "nn" in checks["raw_claims_target_method_check"]
        assert "llm" in checks["raw_claims_dimension_method_check"]
    finally:
        await conn.close()


async def test_map_price_ignores_an_extract_override_for_a_listed_model(vkb_dsn, tmp_path):
    cheap = dict(extract_price_in_per_mtok=0.01, extract_price_out_per_mtok=0.02)
    assert mapper.map_price(_settings(**cheap)) == (0.10, 0.50)  # Haiku's table price
    unlisted = _settings(map_model="some/unlisted-model", **cheap)
    assert mapper.map_price(unlisted) == (0.01, 0.02)  # the override, for an unlisted model
    with pytest.raises(ValueError, match="no price"):
        mapper.map_price(_settings(map_model="some/unlisted-model"))
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        await _claim(conn, "keep copies for ages", "Okta")
        client = FakeOpenAI(cost=0.0001)
        # Haiku's estimate (~0.0006) passes 0.0003; the override's (~0.00002) would not
        report = await _run(conn, client, max_usd=0.0003, **cheap)
        assert client.calls == [] and report.skipped_by_budget == 1
    finally:
        await conn.close()


async def test_blank_capability_is_unmapped_without_a_call(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        cid = await _claim(conn, "   ", "Okta")
        client, embedder = FakeOpenAI(), FakeEmbedder()
        report = await _run(conn, client, embedder)
        assert client.calls == [] and embedder.calls == []
        m = await _mapping(conn, cid)
        assert (m["dimension_id"], m["dimension_method"], m["target_id"]) == (
            None, "unmapped", "okta")
        assert report.dimension_unmapped == 1
    finally:
        await conn.close()


async def test_embedding_in_slices_gives_the_same_mapping(vkb_dsn, tmp_path, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        cid = await _claim(conn, "keep copies for ages", "Okta Workforce")
        monkeypatch.setattr(mapper, "_EMBED_SLICE", 7)
        embedder = FakeEmbedder()
        report = await _run(conn, embedder=embedder)
        assert len(embedder.calls) == math.ceil(report.embedded_strings / 7) > 1
        assert all(len(texts) <= 7 for texts, _ in embedder.calls)
        m = await _mapping(conn, cid)
        assert (m["target_id"], m["dimension_id"]) == ("okta", "retention.maximum_period")
    finally:
        await conn.close()
