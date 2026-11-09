import copy
import hashlib
import uuid
from pathlib import Path

import pytest
import yaml

from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import load

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
SHIPPED = Path(__file__).parent.parent.parent / "taxonomy" / "taxonomy.yaml"
V2_TABLES = ("dimensions", "classes", "object_types", "targets", "variants", "offerings",
             "editions", "aliases", "topics", "topic_aspects")
COUNTS = {"dimensions": 9, "classes": 5, "object_types": 9, "targets": 6, "variants": 2,
          "offerings": 1, "editions": 2, "aliases": 69, "topics": 0, "topic_aspects": 0}


def _doc() -> dict:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def _write(path: Path, doc: dict) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _topic() -> dict:
    """One topic over the fixture's dimensions, with mixed-case triggers and cues."""
    return {"id": "data_loss", "name": "Data loss protection",
            "triggers": ["Data  Loss", "lost data", "Data Loss"],
            "aspects": [
                {"id": "keeping", "name": "Keeping copies", "query": "How long copies are kept",
                 "cues": ["Retention", "keep  copies"],
                 "dimensions": ["retention.maximum_period", "schedule.backup_frequency"]},
                {"id": "getting_back", "name": "Getting data back",
                 "query": "Restoring objects, and writing them out of a backup",
                 "dimensions": ["coverage.restore", "coverage.export"]}]}


def _aspects(count: int) -> list[dict]:
    return [{"id": f"a{n}", "name": f"Aspect {n}", "query": f"aspect {n}",
             "dimensions": ["coverage.restore"]} for n in range(count)]


def _with_topic(change):
    """A ``mutate`` for the validation cases: add ``_topic()`` after ``change`` altered it."""
    def mutate(doc: dict) -> None:
        topic = _topic()
        change(topic)
        doc["topics"] = [topic]
    return mutate


async def _rows(conn, sql, *args):
    cur = await conn.execute(sql, args or None)
    return await cur.fetchall()


async def _counts(conn) -> dict[str, int]:
    return {t: (await _rows(conn, f"SELECT count(*) FROM {t}"))[0][0] for t in V2_TABLES}


async def _snapshot(conn) -> dict:
    snap = {t: sorted(map(repr, await _rows(conn, f"SELECT * FROM {t}"))) for t in V2_TABLES}
    snap["version"] = await load.current_version(conn)
    return snap


async def _fresh(conn) -> None:
    await conn.execute(f"DROP TABLE IF EXISTS capabilities, features, {', '.join(V2_TABLES)}"
                       " CASCADE")
    await init_schema(conn, embed_dim=768)
    await conn.execute("DELETE FROM taxonomy_meta")


async def test_valid_fixture_loads(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        report = await load.load_taxonomy(conn, FIXTURE)
        assert report["version"].startswith("2-") and len(report["version"]) == 10
        assert {k: report[k] for k in COUNTS} == COUNTS
        assert await _counts(conn) == COUNTS
        assert await load.current_version(conn) == report["version"]
        taxonomy = load.validate(load.parse_doc(FIXTURE.read_bytes(), FIXTURE))
        assert taxonomy.lineage("saas.identity") == ["saas.identity", "saas"]
        assert taxonomy.lineage("saas") == ["saas"]
        # parents come from the dotted id; applies_to NULL means all classes
        assert await _rows(conn, "SELECT target_id, parent_id, class_id FROM targets"
                                 " WHERE target_id LIKE 'aws%' ORDER BY target_id") == [
            ("aws", None, "cloud"), ("aws.rds", "aws", "cloud.database"),
            ("aws.rds.oracle", "aws.rds", "cloud.database")]
        assert await _rows(conn, "SELECT dimension_id, family, applies_to, value_unit, platform"
                                 " FROM dimensions WHERE dimension_id IN"
                                 " ('retention.maximum_period', 'recovery.automation_bypass',"
                                 " 'security.platform_hardening') ORDER BY dimension_id") == [
            ("recovery.automation_bypass", "recovery", ["saas.crm"], None, False),
            ("retention.maximum_period", "retention", None, "days", False),
            ("security.platform_hardening", "security", None, None, True)]
        # object types keep their YAML order within a class
        assert await _rows(conn, "SELECT object_type_id FROM object_types"
                                 " WHERE class_id='saas.identity' ORDER BY ordinal") == [
            ("user",), ("group",), ("access_policy",), ("audit_log",)]
        # aliases are normalized, names are aliases, and each pair has its scope
        aliases = set(await _rows(conn, "SELECT alias, entity_type, scope, entity_id"
                                        " FROM aliases"))
        assert {("okta workforce identity", "target", "", "okta"),
                ("okta", "target", "", "okta"),
                ("max retention", "dimension", "", "retention.maximum_period"),
                ("sign-on policies", "target_object_type", "okta", "access_policy"),
                ("system log", "target_object_type", "okta", "audit_log"),
                ("users", "object_type", "saas.identity", "user"),
                ("tenant configuration", "object_type", "saas", "tenant_config"),
                ("sandbox orgs", "variant", "salesforce", "sandbox_org"),
                ("ecs", "offering", "examplevendor", "examplevendor.cloud_suite"),
                ("premium edition", "edition", "examplevendor.cloud_suite", "premium"),
                } <= aliases
        assert all(a == " ".join(a.split()).lower() for a, *_ in aliases)
    finally:
        await conn.close()


def _broken_doc() -> dict:
    doc = _doc()
    dims = doc["dimensions"]
    dims[5]["aliases"].append("Max  Retention")             # duplicate dimension alias
    dims.append({"id": "telemetry.signal_count", "name": "Signal count",
                 "definition": "x", "aliases": [], "applies_to": "all"})  # unknown family
    doc["dimensions"] = [d for d in dims if d["id"] != "coverage.export"]  # missing required
    doc["targets"].append({"id": "jira", "name": "Jira", "class": "saas.work_management",
                           "aliases": []})                   # missing class
    doc["targets"][1]["object_aliases"]["record"] = ["log entries"]  # not in okta's lineage
    return doc


def test_every_problem_is_collected():
    with pytest.raises(load.TaxonomyError) as err:
        load.validate(_broken_doc())
    problems = err.value.problems
    assert len(problems) >= 5, problems
    text = "\n".join(problems)
    for needle in ("max retention", "telemetry", "saas.work_management", "record",
                   "coverage.export"):
        assert needle in text, (needle, problems)
    assert str(err.value).count("\n") >= 4  # the message lists every problem


async def test_every_problem_is_reported(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await load.load_taxonomy(conn, FIXTURE)
        before = await _snapshot(conn)
        with pytest.raises(load.TaxonomyError) as err:
            await load.load_taxonomy(conn, _write(tmp_path / "bad.yaml", _broken_doc()))
        assert len(err.value.problems) >= 5
        assert await _snapshot(conn) == before
    finally:
        await conn.close()


def test_child_class_cannot_redefine_parent_object_type():
    doc = _doc()
    doc["classes"][1]["object_types"].append(
        {"id": "tenant_config", "name": "Directory settings", "aliases": []})
    with pytest.raises(load.TaxonomyError) as err:
        load.validate(doc)
    assert any("tenant_config" in p and "saas.identity" in p for p in err.value.problems)


@pytest.mark.parametrize(("mutate", "needle"), [
    (lambda d: d.update(version=1), "version"),
    (lambda d: d["classes"].append({"id": "kubernetes.vm", "name": "K", "definition": "d",
                                    "object_types": []}), "kubernetes"),
    (lambda d: d["targets"].append({"id": "gcp.sql", "name": "G", "class": "cloud.database",
                                    "aliases": []}), "gcp"),
    (lambda d: d["dimensions"][4].update(applies_to=["saas.devops"]), "saas.devops"),
    (lambda d: d["dimensions"][4].update(platform="yes"), "platform"),
    (lambda d: d["dimensions"][4].update(requires=["coverage.backup"]), "requires"),
    (lambda d: d["targets"][0].update(aliases=["Okta"]), "okta"),
    (lambda d: d["targets"][2]["variants"].append({"id": "sales_cloud", "name": "Again",
                                                   "aliases": []}), "sales_cloud"),
    (lambda d: d["offerings"][0].update(id="cloud_suite"), "cloud_suite"),
    (lambda d: d["offerings"][0]["editions"][1].update(aliases=["Standard"]), "standard"),
    (lambda d: d["classes"][1]["object_types"][1].update(aliases=["tenant settings"]),
     "tenant settings"),
    (lambda d: d["targets"][1].update(object_aliases={"user": [["nested"]]}), "nested"),
    (lambda d: d["dimensions"][0].update(lookup_aliases=["longest retention"]),
     "longest retention"),
    (lambda d: d["targets"][0].update(lookup_aliases=[d["targets"][0]["name"]]),
     "both aliases and lookup"),
    (lambda d: d["targets"][2]["variants"][0].update(lookup_aliases=["x"]), "lookup_aliases"),
    (lambda d: d["dimensions"][0].update(lookup_only="yes"), "lookup_only"),
    (lambda d: d["targets"][0].update(lookup_only=True), "lookup_only"),
    (_with_topic(lambda t: t["aspects"][1].update(dimensions=["coverage.restore",
                                                              "security.nope"])),
     "unknown dimension 'security.nope'"),
    (_with_topic(lambda t: t["aspects"][0].update(dimensions=["Retention.Maximum_Period"])),
     "unknown dimension 'Retention.Maximum_Period'"),
    (_with_topic(lambda t: t["aspects"][1].update(id="keeping")), "duplicate aspect id"),
    (_with_topic(lambda t: t.update(triggers=[])), "triggers must be a non-empty list"),
    (_with_topic(lambda t: t.update(triggers=[" ", 7.5])), "triggers entry"),
    (_with_topic(lambda t: t.pop("triggers")), "triggers is required"),
    (_with_topic(lambda t: t.update(aspects=[])), "aspects must be a non-empty list"),
    (_with_topic(lambda t: t.update(aspects=_aspects(13))), "at most 12 aspects, got 13"),
    (_with_topic(lambda t: t.pop("aspects")), "aspects is required"),
    (_with_topic(lambda t: t.update(proposed=True)), "unknown key 'proposed'"),
    (_with_topic(lambda t: t["aspects"][0].update(weight=2)), "unknown key 'weight'"),
    (_with_topic(lambda t: t.update(id="Data-Loss")), "invalid topic id 'Data-Loss'"),
    (_with_topic(lambda t: t["aspects"][0].update(id="2nd")), "invalid aspect id '2nd'"),
    (_with_topic(lambda t: t["aspects"][0].update(dimensions=[])),
     "dimensions must be a non-empty list"),
    (_with_topic(lambda t: t["aspects"][0].pop("query")), "query is required"),
    (_with_topic(lambda t: t["aspects"][0].update(query="  ")), "query must be non-empty text"),
    (_with_topic(lambda t: t["aspects"][0].update(cues="retention")), "cues must be a list"),
    (lambda d: d.update(topics=[_topic(), _topic()]), "duplicate topic id 'data_loss'"),
    (lambda d: d.update(topics={"id": "data_loss"}), "topics: must be a list"),
])
def test_validation_rejects(mutate, needle):
    doc = copy.deepcopy(_doc())
    mutate(doc)
    with pytest.raises(load.TaxonomyError) as err:
        load.validate(doc)
    assert any(needle in p for p in err.value.problems), err.value.problems


def test_duplicate_yaml_key_is_refused():
    with pytest.raises(load.TaxonomyError, match="duplicate key"):
        load.parse_doc(b"version: 2\nversion: 2\n")


def test_invalid_yaml_is_a_taxonomy_error_naming_only_the_path():
    with pytest.raises(load.TaxonomyError) as info:
        load.parse_doc(b"version: 2\ndimensions: [unclosed SECRET-TEXT\n", "tax.yaml")
    assert info.value.problems == ["tax.yaml: not valid YAML"]
    assert info.value.__cause__ is None and info.value.__suppress_context__
    assert "SECRET" not in str(info.value)


async def test_v1_tables_are_dropped_once(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await conn.execute(f"DROP TABLE IF EXISTS capabilities, features, {', '.join(V2_TABLES)}"
                           ", raw_claims, rejected_claims, extractions, extract_batches,"
                           " chunks, documents CASCADE")
        await init_schema(conn, embed_dim=768)
        await conn.execute(f"DROP TABLE capabilities, {', '.join(V2_TABLES)} CASCADE")
        await conn.execute("""
            CREATE TABLE features (feature_id text PRIMARY KEY, name text NOT NULL,
                category text, definition text, parent_id text REFERENCES features);
            CREATE TABLE targets (target_id text PRIMARY KEY, name text NOT NULL, kind text,
                parent_id text REFERENCES targets);
            CREATE TABLE aliases (alias text NOT NULL, entity_type text NOT NULL
                CHECK (entity_type IN ('feature', 'target')), entity_id text NOT NULL,
                PRIMARY KEY (alias, entity_type));
            CREATE TABLE capabilities (cap_id uuid PRIMARY KEY, vendor text NOT NULL,
                feature_id text NOT NULL REFERENCES features, target_id text REFERENCES targets,
                support text NOT NULL, claim_id uuid NOT NULL REFERENCES raw_claims
                ON DELETE CASCADE, taxonomy_version text NOT NULL);
            INSERT INTO features VALUES ('immutable_backups', 'Immutable backups', 'Security',
                NULL, NULL);
            INSERT INTO targets VALUES ('aws', 'AWS', 'platform', NULL);
            INSERT INTO aliases VALUES ('aws', 'target', 'aws');
            INSERT INTO taxonomy_meta (key, value) VALUES ('taxonomy_version', '1-deadbeef')
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
        """)
        doc_id, chunk_id, ext_id, claim_id = (uuid.uuid4() for _ in range(4))
        await conn.execute("INSERT INTO documents (doc_id, vendor, content_hash, fetched_at)"
                           " VALUES (%s, 'v', 'h', now())", (doc_id,))
        await conn.execute("INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text,"
                           " token_count, embedding) VALUES (%s, %s, '{a}', 0, 't', 1, %s)",
                           (chunk_id, doc_id, [0.1] * 768))
        await conn.execute("INSERT INTO extractions (extraction_id, chunk_id, extractor_model,"
                           " prompt_version, status) VALUES (%s, %s, 'm', 'claims-v2', 'done')",
                           (ext_id, chunk_id))
        await conn.execute(
            "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor,"
            " capability_raw, support, evidence_quote, gate, extractor_model, prompt_version,"
            " feature_id, feature_method) VALUES (%s, %s, %s, 'v', 'immutable backups',"
            " 'supported', 't', 'exact', 'm', 'claims-v2', 'immutable_backups', 'alias')",
            (claim_id, ext_id, chunk_id))
        await conn.execute("INSERT INTO capabilities VALUES (%s, 'v', 'immutable_backups',"
                           " 'aws', 'supported', %s, '1-deadbeef')", (uuid.uuid4(), claim_id))

        await init_schema(conn, embed_dim=768)
        after_first = await _rows(conn, "SELECT table_name FROM information_schema.tables"
                                        " WHERE table_schema='public'")
        await init_schema(conn, embed_dim=768)
        tables = {r[0] for r in await _rows(conn, "SELECT table_name FROM"
                                                   " information_schema.tables"
                                                   " WHERE table_schema='public'")}
        assert tables == {r[0] for r in after_first}
        assert set(V2_TABLES) <= tables
        assert "features" not in tables
        # capabilities is recreated as v2, which has no feature_id: later runs keep it
        cap_columns = {r[0] for r in await _rows(conn, "SELECT column_name FROM"
                                                        " information_schema.columns"
                                                        " WHERE table_name='capabilities'")}
        assert "scope" in cap_columns and "feature_id" not in cap_columns
        columns = {r[0] for r in await _rows(conn, "SELECT column_name FROM"
                                                    " information_schema.columns"
                                                    " WHERE table_name='targets'")}
        assert "kind" not in columns and "class_id" in columns
        assert "scope" in {r[0] for r in await _rows(conn, "SELECT column_name FROM"
                                                           " information_schema.columns"
                                                           " WHERE table_name='aliases'")}
        assert await _counts(conn) == dict.fromkeys(V2_TABLES, 0)
        # the claim survives, with its v1 mapping columns; the stale v1 version is gone
        assert await _rows(conn, "SELECT claim_id, feature_id, feature_method FROM raw_claims"
                           ) == [(claim_id, "immutable_backups", "alias")]
        assert await load.current_version(conn) is None
        # and a v2 load works on the migrated database
        assert (await load.load_taxonomy(conn, FIXTURE))["targets"] == 6
    finally:
        await conn.close()


async def test_init_schema_keeps_a_loaded_v2_taxonomy(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await load.load_taxonomy(conn, FIXTURE)
        before = await _snapshot(conn)
        await init_schema(conn, embed_dim=768)
        assert await _snapshot(conn) == before
        assert await _counts(conn) == COUNTS and before["version"] is not None
    finally:
        await conn.close()


async def test_init_schema_drops_a_stray_v1_features_table(vkb_dsn):
    """A rolled-back (v1) image's schema run creates only ``features`` on a v2 database; the
    next v2 init_schema drops it and keeps the v2 taxonomy (the variants FK on targets would
    refuse a v1-drop of targets)."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        await load.load_taxonomy(conn, FIXTURE)
        before = await _snapshot(conn)
        await conn.execute("CREATE TABLE features (feature_id text PRIMARY KEY,"
                           " name text NOT NULL, category text, definition text,"
                           " parent_id text REFERENCES features)")
        await conn.execute("INSERT INTO features VALUES ('f', 'F', NULL, NULL, NULL)")
        await init_schema(conn, embed_dim=768)
        assert await _rows(conn, "SELECT to_regclass('features')") == [(None,)]
        assert await _snapshot(conn) == before
        assert await _counts(conn) == COUNTS and before["version"] is not None
    finally:
        await conn.close()


async def test_reload_replaces_everything(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        first = await load.load_taxonomy(conn, FIXTURE)
        doc = _doc()
        doc["dimensions"] = [d for d in doc["dimensions"]
                             if d["id"] != "retention.maximum_period"]
        doc["targets"][1]["object_aliases"].pop("audit_log")
        second = await load.load_taxonomy(conn, _write(tmp_path / "t.yaml", doc))
        assert second["version"] != first["version"] and second["version"].startswith("2-")
        assert await load.current_version(conn) == second["version"]
        assert second["dimensions"] == 8 and second["aliases"] == 69 - 3 - 1
        assert await _rows(conn, "SELECT * FROM dimensions"
                                 " WHERE dimension_id='retention.maximum_period'") == []
        assert await _rows(conn, "SELECT * FROM aliases"
                                 " WHERE entity_id='retention.maximum_period'") == []
        assert await _rows(conn, "SELECT * FROM aliases WHERE alias='system log'") == []
        assert await _counts(conn) == {**COUNTS, "dimensions": 8, "aliases": 65}
        # loading the same file again is a no-op
        assert await load.load_taxonomy(conn, _write(tmp_path / "t.yaml", doc)) == second
    finally:
        await conn.close()


def test_proposed_flag_is_accepted_on_reviewable_entries():
    doc = copy.deepcopy(_doc())
    doc["dimensions"][4]["proposed"] = True
    doc["targets"][0]["proposed"] = False
    doc["classes"][1]["object_types"][0]["proposed"] = True
    doc["offerings"][0]["proposed"] = True
    doc["offerings"][0]["editions"][0]["proposed"] = True
    taxonomy = load.validate(doc)
    assert len(taxonomy.dimensions) == 9  # a review marker only: nothing else changes


@pytest.mark.parametrize("mutate", [
    lambda d: d["dimensions"][4].update(proposed="yes"),
    lambda d: d["targets"][0].update(proposed=1),
    lambda d: d["classes"][1]["object_types"][0].update(proposed="true"),
    lambda d: d["offerings"][0].update(proposed=None),
])
def test_proposed_flag_must_be_boolean(mutate):
    doc = copy.deepcopy(_doc())
    mutate(doc)
    with pytest.raises(load.TaxonomyError) as err:
        load.validate(doc)
    assert any("proposed must be true or false" in p for p in err.value.problems), \
        err.value.problems


async def test_lookup_aliases_load_as_lookup_only(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc = _doc()
        retention = next(d for d in doc["dimensions"] if d["id"] == "retention.maximum_period")
        retention["lookup_aliases"] = ["Keep copies  FOREVER"]
        next(t for t in doc["targets"] if t["id"] == "okta")["lookup_aliases"] = ["okta classic"]
        loaded = await load.load_taxonomy(conn, _write(tmp_path / "t.yaml", doc))
        assert loaded["aliases"] == 69 + 2
        rows = await _rows(conn, "SELECT alias, entity_type, entity_id, search FROM aliases"
                                 " WHERE alias IN ('keep copies forever', 'okta classic',"
                                 " 'longest retention') ORDER BY 1")
        assert rows == [("keep copies forever", "dimension", "retention.maximum_period", False),
                        ("longest retention", "dimension", "retention.maximum_period", True),
                        ("okta classic", "target", "okta", False)]
    finally:
        await conn.close()


async def test_a_lookup_only_dimension_is_never_a_search_key(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc = _doc()
        retention = next(d for d in doc["dimensions"] if d["id"] == "retention.maximum_period")
        retention.update(lookup_only=True, lookup_aliases=["keep copies forever"])
        await load.load_taxonomy(conn, _write(tmp_path / "t.yaml", doc))
        rows = await _rows(conn, "SELECT alias, search FROM aliases"
                                 " WHERE entity_id = 'retention.maximum_period'"
                                 " ORDER BY alias COLLATE \"C\"")
        assert rows == [("keep copies forever", False), ("longest retention", False),
                        ("max retention", False), ("maximum retention period", False)]
    finally:
        await conn.close()


def test_topics_are_optional_and_parsed_in_order():
    assert load.validate(_doc()).topics == []
    doc = _doc()
    doc["topics"] = [_topic()]
    topic = load.validate(doc).topics[0]
    assert isinstance(topic, load.Topic)
    assert all(isinstance(a, load.Aspect) for a in topic.aspects)
    assert [a.id for a in topic.aspects] == ["keeping", "getting_back"]
    assert topic.triggers == ("data loss", "lost data")  # normalized, duplicates dropped
    assert topic.aspects[0].cues == ("retention", "keep copies")
    assert topic.aspects[1].cues == ()  # cues may be omitted
    assert topic.aspects[0].dimensions == ("retention.maximum_period",
                                           "schedule.backup_frequency")
    assert topic.aspects[1].query == "Restoring objects, and writing them out of a backup"


def test_a_topic_may_have_twelve_aspects():
    # the cap keeps a topic question inside its rerank budget at 4 sides
    doc = _doc()
    doc["topics"] = [{**_topic(), "aspects": _aspects(12)}]
    assert len(load.validate(doc).topics[0].aspects) == load.MAX_TOPIC_ASPECTS == 12


def test_topics_do_not_change_the_taxonomy_version():
    data = SHIPPED.read_bytes()
    version = load.version_string(2, data)
    # a cue edited, or an aspect added: the version stays (topics never change a mapping)
    cue = data.replace(b"cues: [anomal, unusual,", b"cues: [anomal, odd, unusual,")
    assert cue != data and load.version_string(2, cue) == version
    aspect = data + (b"  - id: extra\n    name: Extra\n    query: extra words\n"
                     b"    dimensions: [security.rbac]\n")
    assert load.version_string(2, aspect) == version
    # a dimension edited: the version changes
    dimension = data.replace(b"aliases: [entropy analysis,", b"aliases: [entropy score,", 1)
    assert dimension != data and load.version_string(2, dimension) != version


def test_only_the_topics_section_is_left_out_of_the_version():
    head, tail = b"version: 2\n", b"offerings:\n- id: a.b\n"
    topics = b"topics:\n- id: t\n  triggers: [x]\n# a comment in the section\n  aspects: []\n"
    version = load.version_string(2, head + topics + tail)
    assert version == load.version_string(2, head + tail) == load.version_string(
        2, head + tail + topics)
    assert version.startswith("2-") and len(version) == 10
    # what follows the section counts, as does a key that only starts with "topics"
    assert version != load.version_string(2, head + topics + b"offerings:\n- id: a.c\n")
    assert load.version_string(2, head + b"topics_extra: 1\n") != load.version_string(2, head)


def test_the_shipped_taxonomy_keeps_the_production_version():
    data = SHIPPED.read_bytes()
    cut = data[:data.index(b"\ntopics:") + 1]
    assert load.version_string(2, data) == f"2-{hashlib.sha256(cut).hexdigest()[:8]}"
    if load.parse_doc(data)["version"] == 2:
        # main's bytes before the topics were added: production's version. An edit of the
        # taxonomy outside its topics changes it, and this test with it, on purpose.
        assert load.version_string(2, data) == "2-c9fd3d55"


async def test_a_load_records_the_version_without_the_topics(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        plain = await load.load_taxonomy(conn, _write(tmp_path / "plain.yaml", _doc()))
        doc = _doc()
        doc["topics"] = [_topic()]
        with_topic = await load.load_taxonomy(conn, _write(tmp_path / "topic.yaml", doc))
        assert with_topic["topics"] == 1 and with_topic["version"] == plain["version"]
        assert await load.current_version(conn) == plain["version"]
    finally:
        await conn.close()


async def test_topics_load_into_their_tables(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc = _doc()
        doc["topics"] = [_topic()]
        loaded = await load.load_taxonomy(conn, _write(tmp_path / "t.yaml", doc))
        assert loaded["topics"] == 1 and loaded["topic_aspects"] == 2
        assert {k: loaded[k] for k in COUNTS} == {**COUNTS, "topics": 1, "topic_aspects": 2}
        assert await _rows(conn, "SELECT topic_id, name, triggers FROM topics") == [
            ("data_loss", "Data loss protection", ["data loss", "lost data"])]
        # aspects keep their YAML order; triggers and cues are stored lowercase
        assert await _rows(conn, "SELECT topic_id, aspect_id, ordinal, name, query, cues,"
                                 " dimensions FROM topic_aspects ORDER BY ordinal") == [
            ("data_loss", "keeping", 0, "Keeping copies", "How long copies are kept",
             ["retention", "keep copies"],
             ["retention.maximum_period", "schedule.backup_frequency"]),
            ("data_loss", "getting_back", 1, "Getting data back",
             "Restoring objects, and writing them out of a backup", [],
             ["coverage.restore", "coverage.export"])]
        # a load replaces the topics too: a file without any leaves none behind
        again = await load.load_taxonomy(conn, FIXTURE)
        assert again["topics"] == 0 and again["topic_aspects"] == 0
        assert await _counts(conn) == COUNTS
    finally:
        await conn.close()


async def test_init_schema_keeps_loaded_topics(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc = _doc()
        doc["topics"] = [_topic()]
        await load.load_taxonomy(conn, _write(tmp_path / "t.yaml", doc))
        before = await _snapshot(conn)
        # without the marker the DDL runs again: CREATE TABLE IF NOT EXISTS keeps the rows
        await conn.execute("DELETE FROM taxonomy_meta WHERE key = 'schema_version'")
        await init_schema(conn, embed_dim=768)
        assert await _snapshot(conn) == before
        assert len(await _rows(conn, "SELECT 1 FROM topic_aspects")) == 2
    finally:
        await conn.close()


def test_the_shipped_taxonomy_defines_the_ransomware_topic():
    shipped = Path(__file__).parent.parent.parent / "taxonomy" / "taxonomy.yaml"
    topics = load.validate(load.read_doc(shipped)).topics  # every aspect dimension exists
    ransomware = next(t for t in topics if t.id == "ransomware_protection")
    assert "ransomware" in ransomware.triggers and len(ransomware.aspects) == 8
