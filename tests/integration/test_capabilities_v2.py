"""``capabilities`` v2 build and query against a real Postgres, on the fixture taxonomy, with
claims inserted already mapped (no mapper, no LLM).

Every claim, quote and name here is invented for the tests."""
import dataclasses
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pgvector import Vector

from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import capabilities, load
from vendorkb.taxonomy.capabilities import CapFilter

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
OFFERING = "examplevendor.cloud_suite"


async def _fresh(conn) -> str:
    await conn.execute("DROP TABLE IF EXISTS capabilities, raw_claims, rejected_claims,"
                       " extractions, extract_batches, chunks, documents CASCADE")
    await init_schema(conn, embed_dim=8)
    await load.load_taxonomy(conn, FIXTURE)
    version = await load.current_version(conn)
    assert version is not None
    return version


async def _claim(conn, dimension: str | None, target: str | None = None, *,
                 vendor="examplevendor", product="P", class_id=None, variant=None,
                 object_type=None, offering=None, edition=None, support="supported",
                 value=None, unit=None, availability="unstated", kind="capability",
                 fetched_at=T0, prompt_version="claims-v3.1", mapped_version: str | None,
                 quote="synthetic quote", url="http://docs/x", title="Doc",
                 capability="some capability", target_text=None, target_scope=None,
                 conditions=None) -> uuid.UUID:
    doc_id, chunk_id, ext_id, claim_id = (uuid.uuid4() for _ in range(4))
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s,%s,%s,'S',%s,%s,%s,%s)",
        (doc_id, vendor, product, title, url, str(doc_id), fetched_at))
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s,%s,%s,0,'synthetic text',10,%s)",
        (chunk_id, doc_id, ["A"], Vector([0.1] * 8)))
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s,%s,'m',%s,'done')", (ext_id, chunk_id, prompt_version))
    await conn.execute(
        "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor, product, kind,"
        " capability_raw, target_raw, target_scope, conditions, support, availability, value,"
        " unit, evidence_quote, gate, extractor_model, prompt_version, dimension_id,"
        " dimension_method, class_id, target_id, variant_id, object_type_id, offering_id,"
        " edition_id, mapped_version)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'exact','m',%s,%s,"
        " 'alias',%s,%s,%s,%s,%s,%s,%s)",
        (claim_id, ext_id, chunk_id, vendor, product, kind, capability,
         target_text if target_text is not None else target, target_scope, conditions,
         support, availability, value, unit, quote, prompt_version, dimension, class_id,
         target, variant, object_type, offering, edition, mapped_version))
    return claim_id


async def _caps(conn) -> dict[uuid.UUID, dict]:
    cur = await conn.execute("SELECT * FROM capabilities ORDER BY claim_id")
    names = [d.name for d in cur.description]
    return {row[0]: dict(zip(names, row)) for row in await cur.fetchall()}


def _ids(facts) -> set[str]:
    return {f.claim_id for f in facts}


async def test_scope_assignment_and_selection(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        # an offering with no target and a workload dimension is product-wide (as query groups
        # it); an offering with a target, or on a platform dimension, is 'offering'
        offering_wide = await _claim(conn, "retention.maximum_period", offering=OFFERING,
                                     mapped_version=v)
        offering_row = await _claim(conn, "retention.maximum_period", "okta",
                                    class_id="saas.identity", offering=OFFERING,
                                    mapped_version=v)
        offering_platform = await _claim(conn, "security.platform_hardening",
                                         offering=OFFERING, mapped_version=v)
        product_wide = await _claim(conn, "retention.maximum_period", mapped_version=v)
        platform = await _claim(conn, "security.platform_hardening", mapped_version=v)
        targeted = await _claim(conn, "coverage.backup", "okta", class_id="saas.identity",
                                object_type="user", mapped_version=v)
        # not built: an older prompt, a stale mapping, and no dimension
        await _claim(conn, "coverage.backup", "okta", prompt_version="claims-v2",
                     mapped_version=v)
        await _claim(conn, "coverage.backup", "okta", mapped_version="2-stale000")
        await _claim(conn, None, "okta", mapped_version=v)

        assert await capabilities.build(conn) == 6
        caps = await _caps(conn)
        assert {k: r["scope"] for k, r in caps.items()} == {
            offering_wide: "product_wide", offering_row: "offering",
            offering_platform: "offering", product_wide: "product_wide", platform: "product",
            targeted: "product"}
        # the stored scope agrees with the group query sorts each row into
        res = await capabilities.query(conn, CapFilter(vendor="examplevendor"))
        assert {f.claim_id for f in res.product_wide} == {
            str(k) for k, r in caps.items() if r["scope"] == "product_wide"}
        assert all(f.scope == "product_wide" for f in res.product_wide)
        assert all(f.scope != "product_wide" for f in res.facts + res.product_level)
        row = caps[targeted]
        assert (row["vendor"], row["product"], row["dimension_id"], row["target_id"],
                row["class_id"], row["object_type_id"], row["support"], row["taxonomy_version"],
                row["doc_version"]) == ("examplevendor", "P", "coverage.backup", "okta",
                                        "saas.identity", "user", "supported", v, T0)
        # the claim's deletion (a document re-sync) cascades
        await conn.execute("DELETE FROM documents WHERE doc_id = (SELECT ch.doc_id FROM"
                           " chunks ch JOIN raw_claims r USING (chunk_id)"
                           " WHERE r.claim_id = %s)", (targeted,))
        assert targeted not in await _caps(conn)
    finally:
        await conn.close()


async def test_vendor_and_product_follow_a_retagged_document(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        cid = await _claim(conn, "retention.maximum_period", vendor="examplevendor",
                           product="Old Product", mapped_version=v)
        # the sync re-tags the document (store.update_metadata) without touching its chunks
        await conn.execute("UPDATE documents SET vendor='othervendor', product='New Product'")
        await capabilities.build(conn)
        row = (await _caps(conn))[cid]
        assert (row["vendor"], row["product"]) == ("othervendor", "New Product")
    finally:
        await conn.close()


async def test_product_wide_and_product_level_groups(vkb_dsn):
    """Review Focus 4: product-wide facts never land in ``facts``; facts with no offering
    land only in ``product_level``, and only when an offering is filtered."""
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        okta = await _claim(conn, "retention.maximum_period", "okta",
                            class_id="saas.identity", mapped_version=v, value=365, unit="days")
        okta_offering = await _claim(conn, "retention.maximum_period", "okta",
                                     class_id="saas.identity", offering=OFFERING,
                                     mapped_version=v)
        wide = await _claim(conn, "retention.maximum_period", mapped_version=v, value=36500,
                            unit="days")
        wide_offering = await _claim(conn, "retention.maximum_period", offering=OFFERING,
                                     mapped_version=v)
        wide_other_vendor = await _claim(conn, "retention.maximum_period", vendor="othervendor",
                                         mapped_version=v)
        crm_only = await _claim(conn, "recovery.automation_bypass", mapped_version=v)
        platform = await _claim(conn, "security.platform_hardening", mapped_version=v)
        other_vendor = await _claim(conn, "retention.maximum_period", "okta",
                                    vendor="othervendor", class_id="saas.identity",
                                    mapped_version=v)
        await capabilities.build(conn)

        # a target query: product-wide rows of an applicable dimension go to product_wide
        res = await capabilities.query(conn, CapFilter(vendor="ExampleVendor", target="okta"))
        assert _ids(res.facts) == {str(okta), str(okta_offering)}
        assert _ids(res.product_wide) == {str(wide), str(wide_offering)}
        assert _ids(res.product_level) == set() and res.hint is None
        fact = next(f for f in res.facts if f.claim_id == str(okta))
        assert (fact.value, fact.unit, fact.evidence_quote, fact.url, fact.title,
                fact.scope) == (365, "days", "synthetic quote", "http://docs/x", "Doc",
                                "product")
        assert fact.dimension_family == "retention"
        assert fact.doc_version is not None and datetime.fromisoformat(fact.doc_version) == T0
        # every vendor's, when no vendor is filtered; never a dimension the class lacks
        res = await capabilities.query(conn, CapFilter(target="okta"))
        assert _ids(res.facts) == {str(okta), str(okta_offering), str(other_vendor)}
        assert _ids(res.product_wide) == {str(wide), str(wide_offering), str(wide_other_vendor)}
        assert str(crm_only) not in _ids(res.product_wide) | _ids(res.facts)
        assert str(platform) not in _ids(res.product_wide) | _ids(res.facts)

        # no target filter: product-wide rows (an offering's too) are still kept out of facts
        res = await capabilities.query(conn, CapFilter(vendor="examplevendor"))
        assert _ids(res.product_wide) == {str(wide), str(wide_offering), str(crm_only)}
        assert _ids(res.facts) == {str(okta), str(okta_offering), str(platform)}
        assert res.product_level == []

        # an offering filter: rows with no offering (of that vendor) go to product_level
        res = await capabilities.query(conn, CapFilter(offering=OFFERING, target="okta"))
        assert _ids(res.facts) == {str(okta_offering)}
        assert _ids(res.product_level) == {str(okta)}
        assert _ids(res.product_wide) == {str(wide), str(wide_offering)}
        res = await capabilities.query(conn, CapFilter(offering="ECS"))  # an alias
        assert _ids(res.facts) == {str(okta_offering)}
        assert _ids(res.product_level) == {str(okta), str(platform)}
        assert _ids(res.product_wide) == {str(wide), str(wide_offering), str(crm_only)}
        # an edition filter: the offering's rows with no edition are less specific too
        premium = await _claim(conn, "retention.maximum_period", "okta",
                               class_id="saas.identity", offering=OFFERING, edition="premium",
                               mapped_version=v)
        await capabilities.build(conn)
        res = await capabilities.query(conn, CapFilter(offering=OFFERING, edition="premium",
                                                       target="okta"))
        assert _ids(res.facts) == {str(premium)}
        assert _ids(res.product_level) == {str(okta), str(okta_offering)}
        assert {f.claim_id: f.scope for f in res.product_level} == {
            str(okta): "product", str(okta_offering): "offering"}
        res = await capabilities.query(conn, CapFilter(offering=OFFERING, target="okta"))
        assert _ids(res.facts) == {str(okta_offering), str(premium)}
    finally:
        await conn.close()


async def test_target_descendants_are_included(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        aws = await _claim(conn, "coverage.backup", "aws", class_id="cloud", mapped_version=v)
        rds = await _claim(conn, "coverage.backup", "aws.rds", class_id="cloud.database",
                           object_type="database", mapped_version=v)
        oracle = await _claim(conn, "coverage.backup", "aws.rds.oracle",
                              class_id="cloud.database", object_type="database",
                              mapped_version=v)
        await _claim(conn, "coverage.backup", "okta", class_id="saas.identity",
                     mapped_version=v)
        await capabilities.build(conn)
        res = await capabilities.query(conn, CapFilter(target="aws"))
        assert _ids(res.facts) == {str(aws), str(rds), str(oracle)}
        res = await capabilities.query(conn, CapFilter(target="RDS"))  # an alias
        assert _ids(res.facts) == {str(rds), str(oracle)}
        res = await capabilities.query(conn, CapFilter(target="aws", object_type="database",
                                                       dimension="backed up objects"))
        assert _ids(res.facts) == {str(rds), str(oracle)}
    finally:
        await conn.close()


async def test_value_range_and_qualifier_filters(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        ids = {}
        for value in (30, 365, 3650, None):
            ids[value] = await _claim(conn, "retention.maximum_period", "salesforce",
                                      class_id="saas.crm", value=value, unit="days",
                                      vendor=f"vendor{value}", mapped_version=v)
        preview = await _claim(conn, "coverage.backup", "salesforce", class_id="saas.crm",
                               support="partial", availability="preview", mapped_version=v)
        await capabilities.build(conn)
        res = await capabilities.query(conn, CapFilter(value_min=100, value_max=1000))
        assert _ids(res.facts) == {str(ids[365])}
        res = await capabilities.query(conn, CapFilter(value_min=365))
        assert _ids(res.facts) == {str(ids[365]), str(ids[3650])}
        res = await capabilities.query(conn, CapFilter(family="retention"))
        assert _ids(res.facts) == {str(i) for i in ids.values()}
        res = await capabilities.query(conn, CapFilter(support="partial",
                                                       availability="preview"))
        assert _ids(res.facts) == {str(preview)}
    finally:
        await conn.close()


async def test_limit_caps_each_group_and_reports_totals(vkb_dsn, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        for _ in range(3):  # three facts for okta, one conflicting on support
            await _claim(conn, "coverage.backup", "okta", class_id="saas.identity",
                         object_type="user", mapped_version=v)
        await _claim(conn, "coverage.backup", "okta", class_id="saas.identity",
                     object_type="user", support="not_supported", mapped_version=v)
        for _ in range(2):
            await _claim(conn, "retention.maximum_period", mapped_version=v)  # product-wide
        await capabilities.build(conn)
        monkeypatch.setattr(capabilities, "CONFLICT_MEMBERS", 2)
        res = await capabilities.query(conn, CapFilter(target="okta"), limit=1)
        assert (len(res.facts), len(res.product_wide)) == (1, 1)
        assert res.totals == {"facts": 4, "product_level": 0, "product_wide": 2,
                              "unresolved_target": 0}
        [conflict] = res.conflicts
        assert (len(conflict.facts), conflict.size) == (2, 4)  # members capped, size kept
        # a limited group is filled round-robin across vendors
        await _claim(conn, "coverage.backup", "okta", vendor="othervendor",
                     class_id="saas.identity", object_type="user", mapped_version=v)
        await capabilities.build(conn)
        two = await capabilities.query(conn, CapFilter(target="okta"), limit=2)
        assert {f.vendor for f in two.facts} == {"examplevendor", "othervendor"}
        unbounded = await capabilities.query(conn, CapFilter(target="okta", vendor="examplevendor"))
        assert len(unbounded.facts) == 4 and unbounded.totals["facts"] == 4
        assert len(unbounded.conflicts[0].facts) == 2  # members are always capped
    finally:
        await conn.close()


async def test_a_target_text_that_did_not_map_is_not_product_wide(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        # "backup of Lotus Notes is not supported": the workload is named but not in the taxonomy
        lotus = await _claim(conn, "coverage.backup", target_text="Lotus Notes",
                             support="not_supported", mapped_version=v)
        await conn.execute("UPDATE raw_claims SET target_method = 'unmapped' WHERE claim_id = %s",
                           (lotus,))
        wide = await _claim(conn, "retention.maximum_period", mapped_version=v)  # no target text
        await capabilities.build(conn)
        caps = await _caps(conn)
        assert caps[lotus]["scope"] == "target_unmapped" and caps[wide]["scope"] == "product_wide"
        res = await capabilities.query(conn, CapFilter(vendor="examplevendor"))
        assert _ids(res.unresolved_target) == {str(lotus)}
        assert _ids(res.product_wide) == {str(wide)} and res.totals["unresolved_target"] == 1
        assert res.unresolved_target[0].target_raw == "Lotus Notes"
        res = await capabilities.query(conn, CapFilter(target="okta"))  # can't be a match
        assert str(lotus) not in _ids(res.facts + res.product_wide + res.unresolved_target)
    finally:
        await conn.close()


async def test_a_target_text_naming_the_product_is_not_an_unmapped_target(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        # the mapper leaves target_method NULL when the target text is the product itself
        cid = await _claim(conn, "retention.maximum_period", target_text="P", mapped_version=v)
        await conn.execute("UPDATE raw_claims SET target_method = NULL WHERE claim_id = %s",
                           (cid,))
        await capabilities.build(conn)
        assert (await _caps(conn))[cid]["scope"] == "product_wide"
    finally:
        await conn.close()


async def test_a_conflict_is_the_same_statement_both_supported_and_not(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        key = {"dimension": "coverage.backup", "target": "okta", "class_id": "saas.identity",
               "mapped_version": v}
        yes = await _claim(conn, **key, capability="group backup")
        no = await _claim(conn, **key, capability="Group  backup!", support="not_supported")
        partial = await _claim(conn, **key, capability="user backup", support="partial")
        full = await _claim(conn, **key, capability="user backup")
        other = await _claim(conn, **key, capability="role backup", support="not_supported")
        cond_yes = await _claim(conn, **key, capability="app backup", conditions="Enterprise")
        cond_no = await _claim(conn, **key, capability="app backup", support="not_supported",
                               conditions="Starter edition")
        scope_yes = await _claim(conn, **key, capability="object backup", target_scope="Users")
        scope_no = await _claim(conn, **key, capability="object backup", target_scope="Roles",
                                support="not_supported")
        await capabilities.build(conn)
        caps = await _caps(conn)
        # same phrase (normalized), same conditions, supported and not supported: a conflict
        assert caps[yes]["conflict_group"] == caps[no]["conflict_group"] is not None
        # partial and supported agree; different phrases, conditions or objects do not conflict
        for cid in (partial, full, other, cond_yes, cond_no, scope_yes, scope_no):
            assert caps[cid]["conflict_group"] is None, cid
    finally:
        await conn.close()


async def test_conflict_group_and_is_current(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        key = {"dimension": "coverage.backup", "target": "okta", "class_id": "saas.identity",
               "object_type": "group", "mapped_version": v}
        old = await _claim(conn, **key, support="not_supported", fetched_at=T0)
        new = await _claim(conn, **key, support="supported", fetched_at=T0 + timedelta(days=9))
        # same key, agreeing: no conflict, but only the newest is current
        agree = {**key, "object_type": "user"}
        agree_old = await _claim(conn, **agree, fetched_at=T0)
        agree_new = await _claim(conn, **agree, fetched_at=T0 + timedelta(days=1))
        # values that differ are a conflict too
        val = {"dimension": "retention.maximum_period", "mapped_version": v}
        v1 = await _claim(conn, **val, value=30, unit="days", fetched_at=T0)
        v2 = await _claim(conn, **val, value=90, unit="days", fetched_at=T0)
        alone = await _claim(conn, "coverage.restore", "okta", class_id="saas.identity",
                             mapped_version=v)
        await capabilities.build(conn)
        caps = await _caps(conn)
        group = caps[old]["conflict_group"]
        assert group is not None and caps[new]["conflict_group"] == group
        assert (caps[old]["is_current"], caps[new]["is_current"]) == (False, True)
        assert caps[agree_old]["conflict_group"] is None is caps[agree_new]["conflict_group"]
        assert (caps[agree_old]["is_current"], caps[agree_new]["is_current"]) == (False, True)
        assert caps[v1]["conflict_group"] == caps[v2]["conflict_group"] is not None
        assert caps[v1]["conflict_group"] != group
        assert caps[v1]["is_current"] and caps[v2]["is_current"]  # same fetch time: both
        assert caps[alone]["conflict_group"] is None and caps[alone]["is_current"]

        # the conflict is reported with every member, even one the filters left out
        res = await capabilities.query(conn, CapFilter(target="okta", support="supported",
                                                       object_type="group",
                                                       dimension="coverage.backup"))
        assert _ids(res.facts) == {str(new)}
        assert [c.conflict_group for c in res.conflicts] == [str(group)]
        assert _ids(res.conflicts[0].facts) == {str(old), str(new)}
        assert {f.claim_id: f.is_current for f in res.conflicts[0].facts} == {
            str(old): False, str(new): True}
    finally:
        await conn.close()


async def test_empty_result_hint(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        await _claim(conn, "coverage.backup", "salesforce", class_id="saas.crm",
                     mapped_version=v)
        await capabilities.build(conn)
        res = await capabilities.query(conn, CapFilter(vendor="examplevendor", target="okta",
                                                       dimension="schedule.backup_frequency"))
        assert (res.facts, res.product_level, res.product_wide, res.conflicts) == (
            [], [], [], [])
        assert res.hint is not None
        assert "examplevendor" in res.hint and "Okta" in res.hint
        assert "Backup frequency" in res.hint and "chunk" in res.hint
        # an unknown target is named as given
        res = await capabilities.query(conn, CapFilter(target="no such workload"))
        assert res.hint is not None and "no such workload" in res.hint
        res = await capabilities.query(conn, CapFilter(target="salesforce"))
        assert res.hint is None and len(res.facts) == 1
    finally:
        await conn.close()


async def test_unresolved_taxonomy_filter_returns_nothing(vkb_dsn):
    """A taxonomy filter that names nothing never falls back to less specific rows."""
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        await _claim(conn, "retention.maximum_period", "okta", class_id="saas.identity",
                     mapped_version=v)
        await _claim(conn, "retention.maximum_period", "okta", class_id="saas.identity",
                     offering=OFFERING, mapped_version=v)
        await _claim(conn, "retention.maximum_period", mapped_version=v)
        await capabilities.build(conn)
        # sanity: the resolvable filters do return rows
        res = await capabilities.query(conn, CapFilter(offering=OFFERING, target="okta"))
        assert res.facts and res.product_level and res.hint is None
        cases = [
            (CapFilter(offering=OFFERING, edition="no such edition", target="okta"),
             "edition 'no such edition'"),
            (CapFilter(vendor="examplevendor", edition="no such edition"),
             "edition 'no such edition'"),
            (CapFilter(offering="no such offering"), "offering 'no such offering'"),
            (CapFilter(vendor="examplevendor", target="nowhere"), "target 'nowhere'"),
            (CapFilter(vendor="examplevendor", dimension="nothing.at_all"),
             "dimension 'nothing.at_all'"),
        ]
        for f, named in cases:
            res = await capabilities.query(conn, f)
            assert (res.facts, res.product_level, res.product_wide, res.conflicts) == (
                [], [], [], []), f
            assert res.hint is not None and named in res.hint and "chunk" in res.hint, f
        # an edition that does resolve but has no rows of its own still gets the less
        # specific rows: only a filter naming nothing returns nothing
        res = await capabilities.query(conn, CapFilter(offering=OFFERING, edition="premium",
                                                       target="okta"))
        assert res.facts == [] and len(res.product_level) == 2 and res.hint is None
    finally:
        await conn.close()


async def test_rebuild_is_idempotent(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        key = {"dimension": "coverage.backup", "target": "okta", "class_id": "saas.identity",
               "mapped_version": v}
        await _claim(conn, **key, support="not_supported", fetched_at=T0)
        await _claim(conn, **key, support="supported", fetched_at=T0 + timedelta(days=1))
        await _claim(conn, "retention.maximum_period", offering=OFFERING, edition="premium",
                     mapped_version=v)
        first_count = await capabilities.build(conn)
        first = await _caps(conn)
        res1 = await capabilities.query(conn, CapFilter(target="okta"))
        assert await capabilities.build(conn) == first_count == 3
        assert await _caps(conn) == first
        res2 = await capabilities.query(conn, CapFilter(target="okta"))
        assert dataclasses.asdict(res1) == dataclasses.asdict(res2)
        # a remap under a new taxonomy version: the old rows are not served or rebuilt
        await conn.execute("UPDATE taxonomy_meta SET value = '2-newer000'"
                           " WHERE key = 'taxonomy_version'")
        assert (await capabilities.query(conn, CapFilter(target="okta"))).facts == []
        assert await capabilities.build(conn) == 0 and await _caps(conn) == {}
    finally:
        await conn.close()
