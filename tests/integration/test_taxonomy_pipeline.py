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
from vendorkb.taxonomy import propose, v1_format

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
    await _claim(conn, "rubrik", "immutable backups", "AWS RDS")
    await _claim(conn, "veeam", "immutable backup", "AWS RDS")
    await _claim(conn, "veeam", "Immutable Backup", "Amazon RDS")
    await _claim(conn, "veeam", "Instant VM recovery", "VMware vSphere")
    await _claim(conn, "veeam", "instant vm recovery", None)
    await _claim(conn, "cohesity", "tape library export", None)


async def test_propose_features_writes_valid_yaml_and_merges(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _seed_propose(conn)
        out = tmp_path / "taxonomy.yaml"
        # An old-format file: long_tail and unknown keys are dropped from the taxonomy.
        out.write_text(yaml.safe_dump({"version": 1, "features": [], "targets": [],
                                       "long_tail": {"feature": []}, "notes": "x"}))
        client = FakeOpenAI(PROPOSALS)
        embedder = FakeEmbedder(VECTORS)
        report = await propose.propose(conn, client=client, embedder=embedder,
                                       settings=_settings(), kind="feature",
                                       distance_threshold=0.2, out_path=out)
        embedded = [t for texts, _ in embedder.calls for t in texts]
        assert "tape library export" not in embedded  # a singleton is never embedded
        assert all(t == "document" for _, t in embedder.calls)
        assert (report.strings, report.clustered, report.clusters, report.proposed) == (
            4, 3, 2, 2)
        assert (report.long_tail, report.long_tail_prefiltered, report.long_tail_capped) == (
            1, 1, 0)
        assert not report.stopped_by_budget
        assert abs(report.spent_usd - 0.002) < 1e-9
        req = client.chat.completions.calls[0]
        assert req["model"] == "anthropic/claude-haiku-5.5"
        assert req["max_tokens"] == propose.MAX_TOKENS == 2000
        assert req["extra_body"]["reasoning"] == {"effort": "low"}
        assert req["response_format"]["json_schema"]["strict"] is True
        schema = req["response_format"]["json_schema"]["schema"]
        assert set(schema["required"]) == {"id", "name", "category", "definition", "aliases"}
        assert schema["properties"]["category"]["enum"] == list(propose.CATEGORIES)

        doc = yaml.safe_load(out.read_text())
        assert list(doc) == ["version", "features", "targets"]
        assert doc["version"] == 1 and doc["targets"] == []
        feats = {f["id"]: f for f in doc["features"]}
        assert set(feats) == {"immutable_backups", "instant_vm_recovery"}
        imm = feats["immutable_backups"]
        assert imm["category"] == "Data protection"
        assert (imm["support_count"], imm["vendor_count"]) == (4, 2)
        assert set(imm["aliases"]) == {"immutable backups", "immutable backup",
                                       "worm backups"}
        ivr = feats["instant_vm_recovery"]
        assert (ivr["support_count"], ivr["vendor_count"]) == (2, 1)  # 2 claims, 1 vendor
        tail_path = tmp_path / "long_tail.yaml"
        assert report.long_tail_path == str(tail_path)
        tail = yaml.safe_load(tail_path.read_text())
        assert tail["feature"] == [{"text": "tape library export", "count": 1,
                                    "vendor_count": 1}]

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
        assert tg["aws"]["category"] == "Platforms"  # a top-level ancestor
        assert set(tg["aws.rds"]["aliases"]) == {"amazon rds", "aws rds", "rds"}
        tail2 = yaml.safe_load(tail_path.read_text())
        assert tail2["target"] == [{"text": "vmware vsphere", "count": 1, "vendor_count": 1}]
        assert tail2["feature"] == tail["feature"]
        # The file passes the v1 validation propose applies to an existing file.
        _, features, targets = v1_format.validate(v1_format.read_doc(out))
        assert len(features) == 2 and len(targets) == 2
    finally:
        await conn.close()


async def test_propose_cap_limit_and_failures(vkb_dsn, tmp_path, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _seed_propose(conn)
        # The cap keeps the top strings by (count, vendor_count); the rest are long tail.
        monkeypatch.setattr(propose, "MAX_CLUSTER_STRINGS", 2)
        embedder = FakeEmbedder(VECTORS)
        report = await propose.propose(
            conn, client=FakeOpenAI(PROPOSALS), embedder=embedder, settings=_settings(),
            kind="feature", distance_threshold=0.2, out_path=tmp_path / "taxonomy.yaml",
            limit=1)
        assert (report.clustered, report.long_tail_capped, report.long_tail) == (2, 1, 2)
        assert sum(len(texts) for texts, _ in embedder.calls) == 2
        assert (report.proposed, report.skipped_by_limit) == (1, 0)  # one cluster of two
        monkeypatch.setattr(propose, "MAX_CLUSTER_STRINGS", 15_000)

        # --limit: only the first N clusters are sent.
        client = FakeOpenAI(PROPOSALS)
        report = await propose.propose(
            conn, client=client, embedder=embedder, settings=_settings(), kind="feature",
            distance_threshold=0.2, out_path=tmp_path / "l.yaml", limit=1)
        assert (report.proposed, report.skipped_by_limit) == (1, 1)
        assert len(client.chat.completions.calls) == 1 and not report.stopped_by_budget

        # An unexpected exception on one cluster costs that cluster only.
        broken = dict(PROPOSALS)
        client = FakeOpenAI(broken)

        async def flaky(**params):
            if "instant vm recovery" in params["messages"][-1]["content"]:
                raise RuntimeError("odd provider shape")
            return _chat(PROPOSALS["immutable backup"])
        client.chat.completions.create = flaky
        report = await propose.propose(
            conn, client=client, embedder=embedder, settings=_settings(), kind="feature",
            distance_threshold=0.2, out_path=tmp_path / "f.yaml")
        assert (report.proposed, report.invalid) == (1, 1)
        assert "odd provider shape" in report.errors[0]
        assert [f["id"] for f in yaml.safe_load((tmp_path / "f.yaml").read_text())[
            "features"]] == ["immutable_backups"]

        # A hand-broken file is refused before any call.
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"version": 1, "features": [{"name": "no id"}]}))
        client = FakeOpenAI(PROPOSALS)
        with pytest.raises(ValueError):
            await propose.propose(conn, client=client, embedder=embedder,
                                  settings=_settings(), kind="feature",
                                  distance_threshold=0.2, out_path=bad)
        assert client.chat.completions.calls == []
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
        await _claim(conn, "veeam", "immutable backups")
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
