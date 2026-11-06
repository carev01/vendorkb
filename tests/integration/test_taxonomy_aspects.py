from pathlib import Path

import pytest

from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import load
from vendorkb.taxonomy.aspects import aspects

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
V2_TABLES = ("dimensions", "classes", "object_types", "targets", "variants", "offerings",
             "editions", "aliases")


@pytest.fixture
async def conn(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await conn.execute(f"DROP TABLE IF EXISTS capabilities, features, {', '.join(V2_TABLES)}"
                           " CASCADE")
        await init_schema(conn, embed_dim=768)
        await conn.execute("DELETE FROM taxonomy_meta")
        await load.load_taxonomy(conn, FIXTURE)
        yield conn
    finally:
        await conn.close()


def _ids(found, kind):
    return [a.dimension_id for a in found if a.kind == kind]


async def test_okta_gets_entra_aspects(conn):
    okta = await aspects(conn, target="okta")
    assert okta == await aspects(conn, target="entra_id")
    assert okta == await aspects(conn, cls="saas.identity")
    # the root's object types first, then saas.identity's in YAML order, each times the four
    # required coverage dimensions
    coverage = [(a.object_type_id, a.dimension_id) for a in okta if a.kind == "coverage"]
    assert [o for o, _ in coverage[::4]] == ["tenant_config", "user", "group", "access_policy",
                                              "audit_log"]
    assert [d for _, d in coverage[:4]] == ["coverage.backup", "coverage.restore",
                                            "coverage.export", "coverage.compare"]
    assert len(coverage) == 20
    # then the class's other dimensions in family order, then the platform ones last
    assert _ids(okta, "dimension") == ["coverage.attribute_depth", "schedule.backup_frequency",
                                       "retention.maximum_period"]
    assert _ids(okta, "platform") == ["security.platform_hardening"]
    assert [a.kind for a in okta] == ["coverage"] * 20 + ["dimension"] * 3 + ["platform"]
    retention = next(a for a in okta if a.dimension_id == "retention.maximum_period")
    assert (retention.family, retention.value_unit, retention.object_type_id) == (
        "retention", "days", None)
    assert retention.name == "Maximum retention period"


async def test_alias_resolves_target(conn):
    assert await aspects(conn, target="Okta Workforce Identity") == await aspects(
        conn, target="okta")
    assert await aspects(conn, target="  azure AD ") == await aspects(conn, target="entra_id")


async def test_class_only_dimension(conn):
    assert "recovery.automation_bypass" not in _ids(await aspects(conn, target="okta"),
                                                    "dimension")
    assert "recovery.automation_bypass" in _ids(await aspects(conn, target="salesforce"),
                                                "dimension")


async def test_parent_entry_applies_to_children(conn):
    assert "schedule.backup_frequency" in _ids(await aspects(conn, target="okta"), "dimension")
    assert "schedule.backup_frequency" in _ids(await aspects(conn, cls="saas"), "dimension")
    assert "schedule.backup_frequency" not in _ids(await aspects(conn, cls="cloud.database"),
                                                   "dimension")
    # the reverse direction: an entry for a child class does not reach its parent
    assert "recovery.automation_bypass" not in _ids(await aspects(conn, cls="saas"), "dimension")
    assert "recovery.automation_bypass" in _ids(await aspects(conn, cls="saas.crm"), "dimension")


async def test_class_without_object_types_has_no_coverage(conn):
    found = await aspects(conn, target="aws")
    assert not [a for a in found if a.kind == "coverage"]
    assert _ids(found, "platform") == ["security.platform_hardening"]


async def test_unknown_input_raises(conn):
    with pytest.raises(LookupError):
        await aspects(conn, target="nonexistent")
    with pytest.raises(LookupError):
        await aspects(conn, cls="saas.nonexistent")
    with pytest.raises(ValueError):
        await aspects(conn)
    with pytest.raises(ValueError):
        await aspects(conn, target="okta", cls="saas")


async def test_non_required_coverage_dimension_is_a_plain_dimension(conn):
    for found in (await aspects(conn, target="okta"), await aspects(conn, target="aws")):
        depth = [a for a in found if a.dimension_id == "coverage.attribute_depth"]
        assert [(a.kind, a.object_type_id, a.family) for a in depth] == [
            ("dimension", None, "coverage")]
        # first of the dimension group: coverage leads FAMILIES
        assert _ids(found, "dimension")[0] == "coverage.attribute_depth"


async def test_coverage_object_type_order_matches_taxonomy(conn):
    tax = load.validate(load.parse_doc(FIXTURE.read_bytes(), FIXTURE))
    for class_id in ("saas", "saas.identity", "saas.crm", "cloud", "cloud.database"):
        expected = [o.id for o in tax.object_types(class_id)]
        found = [a.object_type_id for a in await aspects(conn, cls=class_id)
                 if a.kind == "coverage" and a.dimension_id == "coverage.backup"]
        assert found == expected
