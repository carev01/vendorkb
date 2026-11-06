import hashlib
import json
import math
import uuid
from types import SimpleNamespace

import pytest
import yaml
from pgvector import Vector

from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import capabilities, load, mapper, propose

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


class FakeEmbedder:
    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.calls: list[tuple[list[str], str]] = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append((list(texts), input_type))
        return [self.table.get(t, _hashed(t)) for t in texts]


VECTORS = {
    "immutable backups": _axis(0),
    "immutable backup": _axis(0, 0.05, 5),
    "worm-protected backups": _axis(0, 0.3, 6),
    "instant vm recovery": _axis(1),
    "tape library export": _axis(2),
    "amazon rds": _axis(3),
    "aws rds": _axis(3, 0.05, 7),
    "vmware vsphere": _axis(4),
}


def _chat(payload, *, cost=0.001, tokens=(500, 100)):
    message = SimpleNamespace(content=json.dumps(payload), refusal=None)
    usage = SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1], cost=cost)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)],
                           usage=usage)


class FakeCompletions:
    def __init__(self, replies, cost=0.001):
        self.replies = replies  # marker in the user message -> proposal dict
        self.cost = cost
        self.calls: list[dict] = []

    async def create(self, **params):
        self.calls.append(params)
        body = params["messages"][-1]["content"]
        for marker, reply in self.replies.items():
            if marker in body:
                return _chat(reply, cost=self.cost)
        raise AssertionError(f"unexpected proposal request: {body}")


class FakeOpenAI:
    def __init__(self, replies, cost=0.001):
        self.chat = SimpleNamespace(completions=FakeCompletions(replies, cost))


PROPOSALS = {
    "immutable backup": {"id": "immutable_backups", "name": "Immutable backups",
                         "category": "Data protection",
                         "definition": "Backups that cannot be modified or deleted.",
                         "aliases": ["WORM backups", "immutable backup"]},
    "instant vm recovery": {"id": "Instant VM Recovery", "name": "Instant VM recovery",
                            "category": "Recovery",
                            "definition": "Run a VM directly from backup storage.",
                            "aliases": []},
    "amazon rds": {"id": "AWS.RDS", "name": "Amazon RDS", "category": "Workloads",
                   "definition": "Amazon's managed relational database service.",
                   "aliases": ["RDS"]},
}


def _settings(**kw) -> VendorkbSettings:
    return VendorkbSettings(_env_file=None, extract_backend="openai",
                            extract_model="anthropic/claude-haiku-5.5", **kw)


async def _fresh(conn) -> None:
    await conn.execute("DROP TABLE IF EXISTS capabilities, aliases, features, targets, "
                       "taxonomy_meta, raw_claims, rejected_claims, extractions, "
                       "extract_batches, chunks, documents CASCADE")
    await init_schema(conn, embed_dim=8)


async def _claim(conn, vendor, capability, target=None, *, support="supported",
                 product="P", url="http://docs/x") -> str:
    doc_id, chunk_id, ext_id, claim_id = (uuid.uuid4() for _ in range(4))
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s,%s,%s,'S','Doc',%s,%s, now())",
        (doc_id, vendor, product, url, str(doc_id)))
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s,%s,%s,0,%s,10,%s)",
        (chunk_id, doc_id, ["A"], f"{capability} text", Vector([0.1] * 8)))
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s,%s,'m','v','done')", (ext_id, chunk_id))
    await conn.execute(
        "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor, product, version,"
        " capability_raw, target_raw, support, conditions, evidence_quote, gate,"
        " extractor_model, prompt_version) VALUES (%s,%s,%s,%s,%s,NULL,%s,%s,%s,'cond',"
        " %s,'exact','m','v')",
        (claim_id, ext_id, chunk_id, vendor, product, capability, target, support,
         f"quote for {capability}"))
    return str(claim_id)


async def _seed_propose(conn) -> None:
    await _claim(conn, "rubrik", "Immutable  Backups", "Amazon RDS")
    await _claim(conn, "veeam", "immutable backup", "AWS RDS")
    await _claim(conn, "veeam", "Instant VM recovery", "VMware vSphere")
    await _claim(conn, "veeam", "instant vm recovery", None)
    await _claim(conn, "cohesity", "tape library export", None)


async def test_propose_features_writes_valid_yaml_and_merges(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _seed_propose(conn)
        out = tmp_path / "taxonomy.yaml"
        client = FakeOpenAI(PROPOSALS)
        embedder = FakeEmbedder(VECTORS)
        report = await propose.propose(conn, client=client, embedder=embedder,
                                       settings=_settings(), kind="feature",
                                       distance_threshold=0.2, out_path=out)
        assert embedder.calls and all(t == "document" for _, t in embedder.calls)
        assert (report.strings, report.clusters, report.proposed, report.long_tail) == (
            4, 3, 2, 1)
        assert not report.stopped_by_budget
        assert abs(report.spent_usd - 0.002) < 1e-9
        req = client.chat.completions.calls[0]
        assert req["model"] == "anthropic/claude-haiku-5.5"
        assert req["response_format"]["json_schema"]["strict"] is True
        schema = req["response_format"]["json_schema"]["schema"]
        assert set(schema["required"]) == {"id", "name", "category", "definition", "aliases"}
        assert schema["properties"]["category"]["enum"] == list(propose.CATEGORIES)

        doc = yaml.safe_load(out.read_text())
        assert doc["version"] == 1 and doc["targets"] == []
        feats = {f["id"]: f for f in doc["features"]}
        assert set(feats) == {"immutable_backups", "instant_vm_recovery"}
        imm = feats["immutable_backups"]
        assert imm["category"] == "Data protection"
        assert (imm["support_count"], imm["vendor_count"]) == (2, 2)
        assert set(imm["aliases"]) == {"immutable backups", "immutable backup",
                                       "worm backups"}
        ivr = feats["instant_vm_recovery"]
        assert (ivr["support_count"], ivr["vendor_count"]) == (2, 1)  # 2 claims, 1 vendor
        assert doc["long_tail"]["feature"] == [{"text": "tape library export", "count": 1}]

        # A re-run proposes nothing new: every member is already an alias.
        again = FakeOpenAI(PROPOSALS)
        report = await propose.propose(conn, client=again, embedder=embedder,
                                       settings=_settings(), kind="feature",
                                       distance_threshold=0.2, out_path=out)
        assert again.chat.completions.calls == [] and report.proposed == 0
        assert report.already_covered == 2
        assert yaml.safe_load(out.read_text())["features"] == doc["features"]

        # Targets merge into the same file and keep the features; ancestors are added.
        report = await propose.propose(conn, client=FakeOpenAI(PROPOSALS), embedder=embedder,
                                       settings=_settings(), kind="target",
                                       distance_threshold=0.2, out_path=out)
        doc2 = yaml.safe_load(out.read_text())
        assert doc2["features"] == doc["features"]
        tg = {t["id"]: t for t in doc2["targets"]}
        assert set(tg) == {"aws", "aws.rds"}
        assert tg["aws.rds"]["parent"] == "aws" and tg["aws"]["parent"] is None
        assert set(tg["aws.rds"]["aliases"]) == {"amazon rds", "aws rds", "rds"}
        assert doc2["long_tail"]["target"] == [{"text": "vmware vsphere", "count": 1}]
        assert doc2["long_tail"]["feature"] == doc["long_tail"]["feature"]
        # The file loads as-is.
        loaded = await load.load_taxonomy(conn, out)
        assert loaded["features"] == 2 and loaded["targets"] == 2
    finally:
        await conn.close()


async def test_propose_budget_guard_stops(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _seed_propose(conn)
        out = tmp_path / "t.yaml"
        client = FakeOpenAI(PROPOSALS, cost=1.0)
        report = await propose.propose(
            conn, client=client, embedder=FakeEmbedder(VECTORS),
            settings=_settings(extract_concurrency=1), kind="feature",
            distance_threshold=0.2, out_path=out, max_usd=0.5)
        assert report.stopped_by_budget
        assert (report.proposed, report.unproposed) == (1, 1)
        assert len(client.chat.completions.calls) == 1
        assert report.spent_usd == 1.0
        assert len(yaml.safe_load(out.read_text())["features"]) == 1

        zero = FakeOpenAI(PROPOSALS)
        report = await propose.propose(
            conn, client=zero, embedder=FakeEmbedder(VECTORS), settings=_settings(),
            kind="feature", distance_threshold=0.2, out_path=tmp_path / "z.yaml", max_usd=0.0)
        assert report.stopped_by_budget and zero.chat.completions.calls == []
    finally:
        await conn.close()


async def test_propose_anthropic_backend_request_shape(vkb_dsn, tmp_path):
    calls: list[dict] = []

    class Messages:
        async def create(self, **params):
            calls.append(params)
            text = json.dumps(PROPOSALS["immutable backup"])
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=400, output_tokens=80))

    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _claim(conn, "rubrik", "immutable backups")
        await _claim(conn, "veeam", "immutable backup")
        settings = VendorkbSettings(_env_file=None)
        report = await propose.propose(
            conn, client=SimpleNamespace(messages=Messages()),
            embedder=FakeEmbedder(VECTORS), settings=settings, kind="feature",
            distance_threshold=0.2, out_path=tmp_path / "a.yaml")
        assert report.proposed == 1
        assert calls[0]["model"] == "claude-haiku-5-5"
        assert calls[0]["output_config"]["format"]["type"] == "json_schema"
        assert abs(report.spent_usd - (400 * 0.10 + 80 * 0.50) / 1e6) < 1e-12
    finally:
        await conn.close()


TAXONOMY = {
    "version": 1,
    "features": [
        {"id": "immutable_backups", "name": "Immutable backups", "category": "Data protection",
         "definition": "d", "aliases": ["immutable backups", "WORM backups"],
         "support_count": 2, "vendor_count": 2},
        {"id": "instant_vm_recovery", "name": "Instant VM recovery", "category": "Recovery",
         "definition": "d", "aliases": [], "support_count": 2, "vendor_count": 1},
    ],
    "targets": [
        {"id": "aws", "name": "AWS", "category": "Platforms", "parent": None,
         "definition": "", "aliases": ["amazon web services"]},
        {"id": "aws.rds", "name": "Amazon RDS", "category": "Workloads", "parent": "aws",
         "definition": "", "aliases": ["amazon rds"]},
        {"id": "aws.s3", "name": "Amazon S3", "category": "Storage targets", "parent": "aws",
         "definition": "", "aliases": ["amazon s3"]},
    ],
    "long_tail": {"feature": [], "target": []},
}


def _write(path, doc) -> None:
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


async def _rows(conn, sql, *args):
    cur = await conn.execute(sql, args)
    return await cur.fetchall()


async def test_load_is_idempotent_and_replaces(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        path = tmp_path / "taxonomy.yaml"
        _write(path, TAXONOMY)
        first = await load.load_taxonomy(conn, path)
        second = await load.load_taxonomy(conn, path)
        assert first == second == {"version": "1", "features": 2, "targets": 3, "aliases": 5}
        assert await _rows(conn, "SELECT count(*) FROM features") == [(2,)]
        assert await _rows(conn, "SELECT target_id, parent_id, kind FROM targets"
                                 " ORDER BY target_id") == [
            ("aws", None, "platform"), ("aws.rds", "aws", "workload"),
            ("aws.s3", "aws", "storage")]
        assert ("worm backups", "feature", "immutable_backups") in await _rows(
            conn, "SELECT alias, entity_type, entity_id FROM aliases")
        assert await load.current_version(conn) == "1"

        doc = json.loads(json.dumps(TAXONOMY))
        doc["version"] = 2
        doc["features"] = doc["features"][:1]
        doc["targets"] = doc["targets"][:2]
        _write(path, doc)
        third = await load.load_taxonomy(conn, path)
        assert third == {"version": "2", "features": 1, "targets": 2, "aliases": 4}
        assert await _rows(conn, "SELECT feature_id FROM features") == [("immutable_backups",)]
        assert await load.current_version(conn) == "2"
    finally:
        await conn.close()


@pytest.mark.parametrize("mutate", [
    lambda d: d["targets"][1].update(parent="gcp"),  # dangling parent
    lambda d: d["features"][1]["aliases"].append("immutable backups"),  # alias on two
    lambda d: d["features"][0].update(category="Licensing"),  # not a seed category
    lambda d: d["features"][0].update(id="Bad Id"),
    lambda d: d.update(version="one"),
])
async def test_load_rejects_invalid_yaml(vkb_dsn, tmp_path, mutate):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc = json.loads(json.dumps(TAXONOMY))
        mutate(doc)
        path = tmp_path / "bad.yaml"
        _write(path, doc)
        with pytest.raises(ValueError):
            await load.load_taxonomy(conn, path)
        assert await _rows(conn, "SELECT count(*) FROM features") == [(0,)]
    finally:
        await conn.close()


async def _seed_mapped(conn, tmp_path) -> dict[str, str]:
    path = tmp_path / "taxonomy.yaml"
    _write(path, TAXONOMY)
    await load.load_taxonomy(conn, path)
    ids = {
        "alias": await _claim(conn, "rubrik", "Immutable  Backups", "Amazon  RDS",
                              url="http://docs/rubrik-rds"),
        "nn": await _claim(conn, "veeam", "WORM-protected backups", None,
                           support="partial"),
        "unmapped": await _claim(conn, "cohesity", "tape library export", "Amazon S3"),
        "s3": await _claim(conn, "veeam", "immutable backups", "amazon s3"),
        "t_unmapped": await _claim(conn, "commvault", "immutable backups", "VMware vSphere"),
    }
    return ids


async def test_mapper_alias_nearest_neighbour_and_unmapped(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ids = await _seed_mapped(conn, tmp_path)
        embedder = FakeEmbedder(VECTORS)
        report = await mapper.map_claims(conn, embedder=embedder, threshold=0.8)
        assert (report.claims, report.feature_alias, report.feature_nn,
                report.feature_unmapped) == (5, 3, 1, 1)
        assert (report.target_alias, report.target_nn, report.target_unmapped,
                report.target_none) == (3, 0, 1, 1)
        rows = {r[0]: r[1:] for r in await _rows(
            conn, "SELECT claim_id::text, feature_id, feature_method, feature_conf,"
                  " target_id, target_method, mapped_version FROM raw_claims")}
        assert rows[ids["alias"]] == ("immutable_backups", "alias", 1.0, "aws.rds", "alias",
                                      "1")
        fid, method, conf, tid, tmethod, _ = rows[ids["nn"]]
        assert (fid, method, tid, tmethod) == ("immutable_backups", "nn", None, None)
        assert 0.8 <= conf < 1.0
        assert rows[ids["unmapped"]][:3] == (None, "unmapped", None)
        assert rows[ids["unmapped"]][3:5] == ("aws.s3", "alias")
        assert rows[ids["t_unmapped"]][3:5] == (None, "unmapped")
    finally:
        await conn.close()


async def test_build_replaces_rows_and_query_includes_descendants(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        ids = await _seed_mapped(conn, tmp_path)
        await mapper.map_claims(conn, embedder=FakeEmbedder(VECTORS), threshold=0.8)
        assert await capabilities.build(conn) == 4  # every claim with a mapped feature
        assert await capabilities.build(conn) == 4  # replaced, never duplicated
        assert await _rows(conn, "SELECT count(*) FROM capabilities") == [(4,)]
        method = await _rows(conn, "SELECT mapping_method, mapping_conf FROM capabilities"
                                   " WHERE claim_id=%s", ids["alias"])
        assert method == [("alias", 1.0)]

        aws = await capabilities.query(conn, "immutable_backups", target="aws")
        assert {r.target_id for r in aws} == {"aws.rds", "aws.s3"}
        rds = await capabilities.query(conn, "immutable_backups", target="aws.rds")
        assert [(r.vendor, r.product, r.support, r.conditions, r.url) for r in rds] == [
            ("rubrik", "P", "supported", "cond", "http://docs/rubrik-rds")]
        assert rds[0].evidence_quote == "quote for Immutable  Backups" and rds[0].chunk_id
        assert await capabilities.query(conn, "immutable_backups", target="aws.r") == []
        # Aliases resolve too; filters narrow.
        assert len(await capabilities.query(conn, "WORM backups", target="Amazon RDS")) == 1
        veeam = await capabilities.query(conn, "immutable_backups", vendor="Veeam")
        assert {r.support for r in veeam} == {"partial", "supported"}
        partial = await capabilities.query(conn, "immutable_backups", support="partial")
        assert [r.vendor for r in partial] == ["veeam"]

        # A taxonomy change (version 2, S3 gone) replaces the rows after a re-map.
        doc = json.loads(json.dumps(TAXONOMY))
        doc["version"] = 2
        doc["targets"] = doc["targets"][:2]
        _write(tmp_path / "taxonomy.yaml", doc)
        await load.load_taxonomy(conn, tmp_path / "taxonomy.yaml")
        assert await capabilities.build(conn) == 0  # nothing mapped under version 2 yet
        await mapper.map_claims(conn, embedder=FakeEmbedder(VECTORS), threshold=0.8)
        assert await capabilities.build(conn) == 4
        assert await _rows(conn, "SELECT DISTINCT taxonomy_version FROM capabilities") == [
            ("2",)]
        assert {r.target_id for r in await capabilities.query(
            conn, "immutable_backups", target="aws")} == {"aws.rds"}
    finally:
        await conn.close()
