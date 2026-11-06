"""``taxonomy propose`` (v2 aliases) against a real Postgres, with a fake client and embedder.

Every phrase here is invented for the tests; none comes from vendor documentation."""
import difflib
import hashlib
import json
import math
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pgvector import Vector

from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import load, propose

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


class FakeEmbedder:
    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.calls: list[tuple[list[str], str]] = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append((list(texts), input_type))
        return [self.table.get(t, _hashed(t)) for t in texts]


VECTORS = {
    # an existing dimension's embedding text ("name: definition") and a cluster beside it
    "Maximum retention period: The longest time a backup copy can be kept.": _axis(0),
    "longest backup retention": _axis(0, 0.05, 10),
    "keep backups for the longest time": _axis(0, 0.1, 11),
    "malware scanning of backups": _axis(1),
    "scan backups for malware": _axis(1, 0.05, 12),
    "per seat pricing": _axis(2),
    "price per seat": _axis(2, 0.05, 13),
    "the dashboard thing": _axis(3),
    "dashboard thing": _axis(3, 0.05, 14),
    "admin roles": _axis(4),
    "administrator roles": _axis(4, 0.05, 15),
    "sign in history": _axis(5),
    "sign-in history": _axis(5, 0.05, 16),
    "tenant wide settings": _axis(6),
    "tenant-wide settings": _axis(6, 0.05, 17),
    "example db cloud": _axis(7),
    "example db cloud service": _axis(7, 0.05, 18),
    "example cloud suite online": _axis(8),
    "example cloud suite service": _axis(8, 0.05, 19),
    "gold tier": _axis(9),
    "gold": _axis(9, 0.05, 20),
    "other vault": _axis(21),
    "other vault service": _axis(21, 0.05, 22),
}

REPLIES = {
    "longest backup retention": {"action": "alias", "id": "retention.maximum_period",
                                 "new": None},
    "malware scanning of backups": {"action": "new", "id": None, "new": {
        "family": "cyber", "id": "cyber.Malware Scan", "name": "Malware scanning",
        "definition": "Whether backups are checked for malicious software.",
        "applies_to": [], "value_unit": None}},
    "per seat pricing": {"action": "new", "id": None, "new": {
        "family": "pricing", "id": "per_seat", "name": "Per-seat price",
        "definition": "The price per seat.", "applies_to": [], "value_unit": "usd"}},
    "the dashboard thing": {"action": "none", "id": None, "new": None},
    "admin roles": {"action": "new", "id": None, "new": {"id": "role", "name": "Role"}},
    "sign in history": {"action": "alias", "id": "audit_log", "new": None},
    "tenant wide settings": {"action": "alias", "id": "tenant_config", "new": None},
    "unknown platform": {"action": "new", "id": None, "new": {
        "id": "unknown.platform", "name": "Unknown Platform", "class": "saas.identity"}},
    "example db cloud": {"action": "new", "id": None, "new": {
        "id": "example_db", "name": "Example DB", "class": "cloud.database"}},
    "example cloud suite online": {"action": "alias", "id": "examplevendor.cloud_suite",
                                   "new": None},
    "gold tier": {"action": "new", "id": None, "new": {
        "offering": "examplevendor.cloud_suite", "id": "gold", "name": "Gold"}},
    "other vault": {"action": "new", "id": None, "new": {"id": "Vault", "name": "Other Vault"}},
    "orphan edition": {"action": "new", "id": None, "new": {
        "offering": "othervendor.missing", "id": "orphan", "name": "Orphan"}},
}


def _chat(payload, *, cost=0.001, tokens=(500, 100)):
    message = SimpleNamespace(content=json.dumps(payload), refusal=None)
    usage = SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1], cost=cost)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)],
                           usage=usage)


class FakeCompletions:
    def __init__(self, replies, cost=0.001):
        self.replies = replies  # marker in the user message's phrasings -> reply dict
        self.cost = cost
        self.calls: list[dict] = []

    async def create(self, **params):
        self.calls.append(params)
        phrasings = _phrasings(params["messages"][-1]["content"])
        for marker, reply in self.replies.items():
            if marker in phrasings:
                return _chat(reply, cost=self.cost)
        raise AssertionError(f"unexpected proposal request: {phrasings}")


class FakeOpenAI:
    def __init__(self, replies, cost=0.001):
        self.chat = SimpleNamespace(completions=FakeCompletions(replies, cost))


def _phrasings(user: str) -> list[str]:
    """The cluster's phrasings listed in a user message (without their counts)."""
    block = user.split("Phrasings (claim count):\n", 1)[1].split("\n\n", 1)[0]
    return [line[2:].rsplit(" (", 1)[0] for line in block.splitlines()]


def _candidates(user: str) -> list[str]:
    block = user.split("Closest existing entries (id | name | definition):\n", 1)[1]
    return [line[2:].split(" | ", 1)[0] for line in block.split("\n\n", 1)[0].splitlines()]


def _settings(**kw) -> VendorkbSettings:
    return VendorkbSettings(_env_file=None, extract_backend="openai",
                            extract_model="anthropic/claude-haiku-5.5", **kw)


async def _fresh(conn) -> None:
    await conn.execute("DROP TABLE IF EXISTS raw_claims, rejected_claims, extractions, "
                       "extract_batches, chunks, documents CASCADE")
    await init_schema(conn, embed_dim=8)


async def _claim(conn, vendor, capability="some capability", target=None, *,
                 target_scope=None, offering=None, edition=None,
                 prompt_version="claims-v3.1") -> None:
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


async def _pair(conn, text, **kw) -> None:
    """Two claims from two vendors: clusterable."""
    await _claim(conn, "alpha", **{"capability": text} | kw)
    await _claim(conn, "beta", **{"capability": text} | kw)


async def _seed_dimensions(conn) -> None:
    for text in ("longest backup retention", "keep backups for the longest time",
                 "malware scanning of backups", "scan backups for malware",
                 "per seat pricing", "price per seat", "the dashboard thing",
                 "dashboard thing", "Max  Retention"):
        await _pair(conn, text)
    await _claim(conn, "alpha", "lonely phrase")  # 1 claim, 1 vendor: long tail
    await _claim(conn, "alpha", "old prompt phrase", prompt_version="claims-v2")
    await _claim(conn, "beta", "old prompt phrase", prompt_version="claims-v2")
    await _claim(conn, "beta", "first family phrase", prompt_version="claims-v3")  # v3 family


def _copy_fixture(tmp_path) -> Path:
    out = tmp_path / "taxonomy.yaml"
    shutil.copy(FIXTURE, out)
    return out


def _removed_lines(before: str, after: str) -> list[str]:
    return [line[2:] for line in difflib.ndiff(before.splitlines(), after.splitlines())
            if line.startswith("- ")]


def _validated(path: Path) -> load.Taxonomy:
    return load.validate(load.parse_doc(path.read_bytes(), path))


async def test_dimension_aliases_merge_in_place_and_new_entries_append(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _seed_dimensions(conn)
        out = _copy_fixture(tmp_path)
        before = out.read_text()
        client = FakeOpenAI(REPLIES)
        embedder = FakeEmbedder(VECTORS)
        report = await propose.propose(conn, client=client, embedder=embedder,
                                       settings=_settings(), kind="dimension",
                                       distance_threshold=0.2, out_path=out, max_usd=2.0,
                                       limit=None)
        embedded = [t for texts, _ in embedder.calls for t in texts]
        assert "old prompt phrase" not in embedded  # only claims-v3* rows are read
        assert "lonely phrase" not in embedded  # a singleton is never embedded
        assert all(t == "document" for _, t in embedder.calls)
        assert (report.strings, report.clustered, report.long_tail) == (11, 9, 2)
        assert (report.clusters, report.already_covered, report.eligible) == (5, 1, 5)
        assert (report.aliased, report.added, report.none, report.invalid) == (1, 1, 1, 1)
        assert report.proposed == 2 and not report.stopped_by_budget
        assert abs(report.spent_usd - 0.004) < 1e-9
        assert any("pricing" in e for e in report.errors)

        req = next(c for c in client.chat.completions.calls
                   if "longest backup retention" in _phrasings(c["messages"][-1]["content"]))
        assert req["model"] == "anthropic/claude-haiku-5.5"
        assert req["max_tokens"] == propose.MAX_TOKENS
        assert req["response_format"]["json_schema"]["strict"] is True
        schema = req["response_format"]["json_schema"]["schema"]
        assert schema["required"] == ["action", "id", "new"]
        assert schema["properties"]["action"]["enum"] == ["alias", "new", "none"]
        new = schema["properties"]["new"]["anyOf"][0]
        assert new["properties"]["family"]["enum"] == list(load.FAMILIES)
        user = req["messages"][-1]["content"]
        candidates = _candidates(user)
        assert len(candidates) == propose.TOP_CANDIDATES == 5
        assert candidates[0] == "retention.maximum_period"  # nearest by embedding first

        tax = _validated(out)  # the written file passes the loader's validation
        dims = {d.id: d for d in tax.dimensions}
        assert set(dims["retention.maximum_period"].aliases) == {
            "longest retention", "max retention", "longest backup retention",
            "keep backups for the longest time"}
        assert "cyber.malware_scan" in dims
        assert not any(d.startswith("pricing.") for d in dims)  # invalid: not written
        assert dims["cyber.malware_scan"].applies_to is None  # [] -> all
        assert set(dims["cyber.malware_scan"].aliases) == {"malware scanning of backups",
                                                           "scan backups for malware"}
        doc = yaml.safe_load(out.read_text())
        assert doc["dimensions"][-1]["id"] == "cyber.malware_scan"  # appended at the end
        assert doc["dimensions"][-1]["proposed"] is True
        assert [d["id"] for d in doc["dimensions"][:-1]] == [
            d["id"] for d in yaml.safe_load(before)["dimensions"]]  # order kept
        assert not any(d.get("proposed") for d in doc["dimensions"][:-1])
        # The review diff: only the merged alias line changes; comments and layout stay.
        assert _removed_lines(before, out.read_text()) == [
            "    aliases: [longest retention, max retention]"]
        assert out.read_text().startswith("# A small synthetic taxonomy v2 for tests.")

        tail = yaml.safe_load(propose.long_tail_path(out).read_text())
        assert tail["source"] == "taxonomy.yaml"
        assert tail["dimension"] == [
            {"text": "first family phrase", "count": 1, "vendor_count": 1},
            {"text": "lonely phrase", "count": 1, "vendor_count": 1}]

        # A re-run sends only the clusters that are still uncovered.
        again = FakeOpenAI(REPLIES)
        report = await propose.propose(conn, client=again, embedder=embedder,
                                       settings=_settings(), kind="dimension",
                                       distance_threshold=0.2, out_path=out, max_usd=2.0,
                                       limit=None)
        sent = {p for c in again.chat.completions.calls
                for p in _phrasings(c["messages"][-1]["content"])}
        assert sent == {"per seat pricing", "price per seat", "the dashboard thing",
                        "dashboard thing"}
        assert report.already_covered == 3
    finally:
        await conn.close()


async def test_object_types_per_resolved_class_and_targets(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _pair(conn, "c", target="Okta", target_scope="admin roles")
        await _pair(conn, "c", target="Okta", target_scope="administrator roles")
        await _pair(conn, "c", target="okta workforce identity", target_scope="sign in history")
        await _pair(conn, "c", target="Entra", target_scope="sign-in history")
        await _pair(conn, "c", target="Okta", target_scope="tenant wide settings")
        await _pair(conn, "c", target="Okta", target_scope="tenant-wide settings")
        await _pair(conn, "c", target="Unknown Platform", target_scope="widgets")
        await _pair(conn, "c", target="example db cloud")
        await _pair(conn, "c", target="Example DB Cloud Service")
        out = _copy_fixture(tmp_path)
        client = FakeOpenAI(REPLIES)
        report = await propose.propose(conn, client=client, embedder=FakeEmbedder(VECTORS),
                                       settings=_settings(), kind="object_type",
                                       distance_threshold=0.2, out_path=out, max_usd=2.0,
                                       limit=None)
        assert report.unresolved == 2  # two claims whose target names no target
        assert (report.aliased, report.added, report.invalid) == (2, 1, 0)
        users = [c["messages"][-1]["content"] for c in client.chat.completions.calls]
        okta = next(u for u in users if "admin roles" in _phrasings(u))
        assert "class saas.identity" in okta
        # the class's lineage, inherited object types included
        assert set(_candidates(okta)) == {"tenant_config", "user", "group", "access_policy",
                                          "audit_log"}
        tax = _validated(out)
        identity = {o.id: o for o in tax.object_types("saas.identity")}
        assert "sign in history" in identity["audit_log"].aliases
        assert "sign-in history" in identity["audit_log"].aliases
        assert "tenant wide settings" in identity["tenant_config"].aliases  # on the ancestor
        assert set(identity["role"].aliases) == {"admin roles", "administrator roles"}
        doc = yaml.safe_load(out.read_text())
        assert doc["classes"][1]["object_types"][-1] == {
            "id": "role", "name": "Role", "aliases": ["admin roles", "administrator roles"],
            "proposed": True}
        assert doc["classes"][0]["object_types"][0]["id"] == "tenant_config"

        # Targets: a new target whose parent is missing is invalid; a valid one appends.
        report = await propose.propose(conn, client=FakeOpenAI(REPLIES),
                                       embedder=FakeEmbedder(VECTORS), settings=_settings(),
                                       kind="target", distance_threshold=0.2, out_path=out,
                                       max_usd=2.0, limit=None)
        assert (report.added, report.invalid) == (1, 1)
        assert any("unknown.platform" in e for e in report.errors)
        doc = yaml.safe_load(out.read_text())
        assert doc["targets"][-1] == {
            "id": "example_db", "name": "Example DB", "class": "cloud.database",
            "aliases": ["example db cloud", "example db cloud service"], "proposed": True}
        _validated(out)
    finally:
        await conn.close()


async def test_offerings_and_editions_per_vendor(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        for text in ("Example Cloud Suite Online", "example cloud suite service"):
            for _ in range(2):
                await _claim(conn, "ExampleVendor", offering=text)
        for text in ("Gold tier", "gold", "premium edition"):
            for _ in range(2):
                await _claim(conn, "ExampleVendor", offering="ECS", edition=text)
        for text in ("Other Vault", "other vault service"):
            for _ in range(2):
                await _claim(conn, "othervendor", offering=text)
        for _ in range(2):
            await _claim(conn, "othervendor", edition="orphan edition")
        out = _copy_fixture(tmp_path)
        client = FakeOpenAI(REPLIES)
        report = await propose.propose(conn, client=client, embedder=FakeEmbedder(VECTORS),
                                       settings=_settings(), kind="offering",
                                       distance_threshold=0.2, out_path=out, max_usd=2.0,
                                       limit=None)
        # "ecs" and "premium edition" are already aliases of examplevendor's entries
        assert (report.aliased, report.added, report.invalid) == (1, 2, 1)
        assert any("othervendor.missing" in e for e in report.errors)
        users = [c["messages"][-1]["content"] for c in client.chat.completions.calls]
        gold = next(u for u in users if "gold tier" in _phrasings(u))
        assert "examplevendor.cloud_suite.premium" in _candidates(gold)
        assert "examplevendor.cloud_suite | Example Cloud Suite" in gold  # offerings listed
        tax = _validated(out)
        offerings = {f.id: f for f in tax.offerings}
        suite = offerings["examplevendor.cloud_suite"]
        assert {"example cloud suite online", "example cloud suite service"} <= set(
            suite.aliases)
        assert [e.id for e in suite.editions] == ["standard", "premium", "gold"]
        assert set(suite.editions[-1].aliases) == {"gold tier", "gold"}
        vault = offerings["othervendor.vault"]
        assert vault.vendor == "othervendor" and vault.name == "Other Vault"
        doc = yaml.safe_load(out.read_text())
        assert doc["offerings"][-1]["proposed"] is True
        assert doc["offerings"][0]["editions"][-1]["proposed"] is True
        tail = yaml.safe_load(propose.long_tail_path(out).read_text())
        assert tail["offering"] == []
    finally:
        await conn.close()


async def test_budget_guard_stops(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _seed_dimensions(conn)
        out = _copy_fixture(tmp_path)
        client = FakeOpenAI(REPLIES, cost=1.0)
        report = await propose.propose(
            conn, client=client, embedder=FakeEmbedder(VECTORS),
            settings=_settings(extract_concurrency=1), kind="dimension",
            distance_threshold=0.2, out_path=out, max_usd=0.5, limit=None)
        assert report.stopped_by_budget
        assert len(client.chat.completions.calls) == 1
        assert report.unproposed == 3 and report.spent_usd == 1.0
        _validated(out)

        zero = FakeOpenAI(REPLIES)
        report = await propose.propose(
            conn, client=zero, embedder=FakeEmbedder(VECTORS), settings=_settings(),
            kind="dimension", distance_threshold=0.2, out_path=_copy_fixture(tmp_path),
            max_usd=0.0, limit=None)
        assert report.stopped_by_budget and zero.chat.completions.calls == []
        assert report.unproposed == 4

        limited = FakeOpenAI(REPLIES)
        report = await propose.propose(
            conn, client=limited, embedder=FakeEmbedder(VECTORS), settings=_settings(),
            kind="dimension", distance_threshold=0.2, out_path=_copy_fixture(tmp_path),
            max_usd=2.0, limit=1)
        assert len(limited.chat.completions.calls) == 1 and report.skipped_by_limit == 3
    finally:
        await conn.close()


async def test_failures_and_refusals(vkb_dsn, tmp_path, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _seed_dimensions(conn)
        embedder = FakeEmbedder(VECTORS)

        # An unexpected exception on one cluster costs that cluster only.
        client = FakeOpenAI(REPLIES)
        real = client.chat.completions.create

        async def flaky(**params):
            if "malware scanning of backups" in _phrasings(params["messages"][-1]["content"]):
                raise RuntimeError("odd provider shape")
            return await real(**params)
        client.chat.completions.create = flaky
        out = _copy_fixture(tmp_path)
        report = await propose.propose(conn, client=client, embedder=embedder,
                                       settings=_settings(), kind="dimension",
                                       distance_threshold=0.2, out_path=out, max_usd=2.0,
                                       limit=None)
        assert report.invalid == 2 and report.added == 0
        assert any("odd provider shape" in e for e in report.errors)

        # An alias to an id the section does not have is invalid.
        bad = dict(REPLIES)
        bad["longest backup retention"] = {"action": "alias", "id": "retention.nope",
                                           "new": None}
        report = await propose.propose(conn, client=FakeOpenAI(bad), embedder=embedder,
                                       settings=_settings(), kind="dimension",
                                       distance_threshold=0.2, out_path=_copy_fixture(
                                           tmp_path), max_usd=2.0, limit=None)
        assert report.aliased == 0 and any("retention.nope" in e for e in report.errors)

        # A "new" whose id already exists merges into that entry.
        merge = dict(REPLIES)
        merge["malware scanning of backups"] = {"action": "new", "id": None, "new": {
            "family": "retention", "id": "maximum_period", "name": "Again",
            "definition": "Same as before.", "applies_to": [], "value_unit": None}}
        out = _copy_fixture(tmp_path)
        report = await propose.propose(conn, client=FakeOpenAI(merge), embedder=embedder,
                                       settings=_settings(), kind="dimension",
                                       distance_threshold=0.2, out_path=out, max_usd=2.0,
                                       limit=None)
        assert (report.merged_into_existing, report.added) == (1, 0)
        dims = {d.id: d for d in _validated(out).dimensions}
        assert "scan backups for malware" in dims["retention.maximum_period"].aliases
        assert dims["retention.maximum_period"].name == "Maximum retention period"

        # The cap keeps the top strings; the rest go to the long tail unembedded.
        monkeypatch.setattr(propose, "MAX_CLUSTER_STRINGS", 2)
        capped = FakeEmbedder(VECTORS)
        report = await propose.propose(conn, client=FakeOpenAI(REPLIES), embedder=capped,
                                       settings=_settings(), kind="dimension",
                                       distance_threshold=0.2, out_path=_copy_fixture(
                                           tmp_path), max_usd=0.0, limit=None)
        assert (report.clustered, report.long_tail_capped) == (2, 7)
        assert len(capped.calls[0][0]) == 2
        monkeypatch.setattr(propose, "MAX_CLUSTER_STRINGS", 15_000)

        # A hand-broken or missing file is refused before any call.
        broken = tmp_path / "broken.yaml"
        broken.write_text(FIXTURE.read_text().replace("version: 2", "version: 1"))
        client = FakeOpenAI(REPLIES)
        with pytest.raises(load.TaxonomyError):
            await propose.propose(conn, client=client, embedder=embedder,
                                  settings=_settings(), kind="dimension",
                                  distance_threshold=0.2, out_path=broken, max_usd=2.0,
                                  limit=None)
        with pytest.raises(FileNotFoundError):
            await propose.propose(conn, client=client, embedder=embedder,
                                  settings=_settings(), kind="dimension",
                                  distance_threshold=0.2, out_path=tmp_path / "none.yaml",
                                  max_usd=2.0, limit=None)
        assert client.chat.completions.calls == []
        with pytest.raises(ValueError):
            await propose.propose(conn, client=client, embedder=embedder,
                                  settings=_settings(), kind="feature",
                                  distance_threshold=0.2, out_path=out, max_usd=2.0,
                                  limit=None)
    finally:
        await conn.close()


async def test_anthropic_backend_request_shape(vkb_dsn, tmp_path):
    calls: list[dict] = []

    class Messages:
        async def create(self, **params):
            calls.append(params)
            text = json.dumps(REPLIES["longest backup retention"])
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=400, output_tokens=80))

    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await _pair(conn, "longest backup retention")
        settings = VendorkbSettings(_env_file=None)
        report = await propose.propose(
            conn, client=SimpleNamespace(messages=Messages()),
            embedder=FakeEmbedder(VECTORS), settings=settings, kind="dimension",
            distance_threshold=0.2, out_path=_copy_fixture(tmp_path), max_usd=2.0, limit=None)
        assert report.aliased == 1
        assert calls[0]["model"] == "claude-haiku-5-5"
        assert calls[0]["output_config"]["format"]["type"] == "json_schema"
        assert abs(report.spent_usd - (400 * 0.10 + 80 * 0.50) / 1e6) < 1e-12
    finally:
        await conn.close()
