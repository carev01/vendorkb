"""The claim support check (claims.support) against a real Postgres, with fake clients.

Every vendor, phrase and quote here is invented for the tests."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import yaml

from tests.integration.test_capabilities_v2 import _claim
from vendorkb.claims import support
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import capabilities, load

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
TOPIC = {"id": "loss", "name": "Data loss", "triggers": ["data loss"],
         "aspects": [
             {"id": "keeping", "name": "Keeping copies", "query": "How long copies are kept",
              "cues": ["Retention"],
              "dimensions": ["retention.maximum_period", "schedule.backup_frequency"]},
             {"id": "getting_back", "name": "Getting data back", "query": "Restoring objects",
              "dimensions": ["coverage.restore"]}]}
RETENTION, FREQUENCY, RESTORE = ("retention.maximum_period", "schedule.backup_frequency",
                                 "coverage.restore")
CHECKER = "anthropic/claude-haiku-5.5@high/support-v2"
REVIEWER = "anthropic/claude-opus-5.5@low/review-v1"


def _settings(**overrides) -> VendorkbSettings:
    return VendorkbSettings(_env_file=None, support_backend="openai",
                            support_model="anthropic/claude-haiku-5.5",
                            support_review_model="anthropic/claude-opus-5.5", **overrides)


class Fake:
    """OpenAI-shaped: the verdict for a claim is looked up by its capability text; an exception
    is raised, a string is returned as the reply text."""

    def __init__(self, verdicts: dict[str, object]):
        self.verdicts = verdicts
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def asked(self, system: str | None = None) -> list[str]:
        """The capabilities asked about (with ``system``: by that step only), sorted."""
        return sorted(_capability(c) for c in self.calls
                      if system is None or c["messages"][0]["content"] == system)

    async def create(self, **params):
        self.calls.append(params)
        verdict = self.verdicts[_capability(params)]
        if isinstance(verdict, Exception):
            raise verdict
        text = verdict if isinstance(verdict, str) else json.dumps({"supported": verdict})
        message = SimpleNamespace(content=text, refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=message)],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=20, cost=0.0002))


def _capability(params: dict) -> str:
    return re.search(r"^Capability: (.*)$", params["messages"][1]["content"], re.M).group(1)


async def _fresh(conn, tmp_path: Path) -> str:
    await conn.execute("DROP TABLE IF EXISTS claim_support, capabilities, raw_claims,"
                       " rejected_claims, extractions, extract_batches, chunks, documents CASCADE")
    await init_schema(conn, embed_dim=8)
    doc = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    doc["topics"] = [TOPIC]
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    await load.load_taxonomy(conn, path)
    version = await load.current_version(conn)
    assert version is not None
    return version


async def _fact(conn, version: str, dimension: str, capability: str, *, vendor="Alpha",
                text="synthetic text", **kw) -> str:
    kw.setdefault("mapped_version", version)
    claim_id = await _claim(conn, dimension, vendor=vendor, capability=capability, **kw)
    await conn.execute("UPDATE chunks SET text = %s WHERE chunk_id ="
                       " (SELECT chunk_id FROM raw_claims WHERE claim_id = %s)", (text, claim_id))
    return str(claim_id)


async def _verdict(conn, claim_id: str, supported: bool, review: bool | None = None) -> None:
    await conn.execute(
        "INSERT INTO claim_support (claim_id, supported, checker, review_supported, reviewer)"
        " VALUES (%s, %s, 'test', %s, %s)",
        (claim_id, supported, review, None if review is None else "test"))


async def _rows(conn) -> dict[str, tuple]:
    cur = await conn.execute("SELECT claim_id::text, supported, checker, review_supported,"
                             " reviewer, cost_usd FROM claim_support")
    return {row[0]: row[1:] for row in await cur.fetchall()}


async def _capability_ids(conn) -> set[str]:
    cur = await conn.execute("SELECT claim_id::text FROM capabilities")
    return {row[0] for row in await cur.fetchall()}


async def test_only_unchecked_positive_claims_on_topic_dimensions_are_due(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn, tmp_path)
        due = {await _fact(conn, version, RETENTION, "kept for a year"),
               await _fact(conn, version, RESTORE, "restores users", support="partial")}
        await _fact(conn, version, RETENTION, "not kept", support="not_supported")
        await _fact(conn, version, "coverage.backup", "backs up users")  # no topic lists it
        await _fact(conn, version, FREQUENCY, "nightly", mapped_version="2-older")
        checked = await _fact(conn, version, FREQUENCY, "hourly")
        await _verdict(conn, checked, True)
        assert {r.claim_id for r in await support.due_for_check(conn, version)} == due
        assert len(await support.due_for_check(conn, version, limit=1)) == 1
    finally:
        await conn.close()


async def test_a_run_stores_each_verdict_once_and_asks_with_the_checker_settings(vkb_dsn,
                                                                                   tmp_path):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn, tmp_path)
        good = await _fact(conn, version, RETENTION, "kept for a year")
        bad = await _fact(conn, version, RETENTION, "generic advice")
        fake = Fake({"kept for a year": True, "generic advice": False})
        report = await support.run(conn, client=fake, settings=_settings(), check_max_usd=1.0,
                                   review_max_usd=0.0)
        assert (report.check.claims, report.check.calls) == (2, 2)
        assert (report.check.supported, report.check.unsupported) == (1, 1)
        rows = await _rows(conn)
        assert rows[good][:2] == (True, CHECKER) and rows[bad][:2] == (False, CHECKER)
        assert rows[good][4] > 0  # the reported cost
        request = fake.calls[0]
        assert request["model"] == "anthropic/claude-haiku-5.5"
        assert request["extra_body"]["reasoning"] == {"effort": "high"}
        assert request["messages"][0]["content"] == support.CHECK_SYSTEM
        assert "Vendor: Alpha\nProduct: P\nPage title: Doc\n" in request["messages"][1]["content"]
        # no review budget: `bad` stays flagged (and in capabilities) until a later run
        assert report.review.skipped_by_budget == 1
        assert (report.flagged, report.rejected) == (1, 0)

        again = await support.run(conn, client=fake, settings=_settings(), check_max_usd=1.0,
                                  review_max_usd=1.0)
        assert again.check.claims == 0 and again.review.claims == 1
    finally:
        await conn.close()


async def test_a_failed_or_unusable_call_writes_no_verdict(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn, tmp_path)
        broken = await _fact(conn, version, RETENTION, "provider error")
        garbled = await _fact(conn, version, RETENTION, "garbled reply")
        fake = Fake({"provider error": RuntimeError("upstream 502"),
                     "garbled reply": '{"supported": "maybe"}'})
        report = await support.run(conn, client=fake, settings=_settings(), check_max_usd=1.0,
                                   review_max_usd=0.0)
        assert report.check.invalid == 2 and len(report.check.errors) == 2
        assert await _rows(conn) == {}
        assert {r.claim_id for r in await support.due_for_check(conn, version)} == {broken,
                                                                                   garbled}
    finally:
        await conn.close()


async def test_the_budget_stops_a_step_before_it_spends(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn, tmp_path)
        for n in range(3):
            await _fact(conn, version, RETENTION, f"kept {n}")
        fake = Fake({})
        report = await support.run(conn, client=fake, settings=_settings(),
                                   check_max_usd=0.000001, review_max_usd=0.0)
        assert report.check.stopped_by_budget and report.check.skipped_by_budget == 3
        assert fake.calls == [] and await _rows(conn) == {}
    finally:
        await conn.close()


async def test_flagged_claims_stay_until_reviewed_and_only_rejected_ones_are_left_out(
        vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn, tmp_path)
        alpha_ok = await _fact(conn, version, RETENTION, "alpha ok")
        alpha_flag = await _fact(conn, version, RETENTION, "alpha flagged")
        beta_flag = await _fact(conn, version, RETENTION, "beta flagged", vendor="Beta")
        gamma_flag = await _fact(conn, version, RESTORE, "gamma flagged", vendor="Gamma")
        await _verdict(conn, alpha_ok, True)
        for flagged in (alpha_flag, beta_flag, gamma_flag):
            await _verdict(conn, flagged, False)
        assert await support.flagged_ids(conn) == sorted([alpha_flag, beta_flag, gamma_flag])
        # a flag alone never costs a fact
        built = await capabilities.build(conn)
        assert built.support_excluded == 0
        assert {alpha_ok, alpha_flag, beta_flag, gamma_flag} <= await _capability_ids(conn)

        fake = Fake({"alpha flagged": False, "beta flagged": False, "gamma flagged": True})
        report = await support.run(conn, client=fake, settings=_settings(), check_max_usd=1.0,
                                   review_max_usd=1.0)
        assert fake.asked(support.CHECK_SYSTEM) == []
        assert fake.asked(support.REVIEW_SYSTEM) == ["alpha flagged", "beta flagged",
                                                     "gamma flagged"]
        request = fake.calls[-1]
        assert request["model"] == "anthropic/claude-opus-5.5"
        assert request["extra_body"]["reasoning"] == {"effort": "low"}
        assert request["messages"][0]["content"] == support.REVIEW_SYSTEM
        assert (report.review.calls, report.review.supported, report.review.unsupported) == (
            3, 1, 2)
        rows = await _rows(conn)
        assert rows[beta_flag][2:4] == (False, REVIEWER)
        assert rows[gamma_flag][2:4] == (True, REVIEWER)
        assert (report.flagged, report.rejected) == (0, 2)

        # the rejected claims are left out: Beta's only fact goes, so Beta loses the dimension
        built = await capabilities.build(conn)
        assert built.support_excluded == 2
        ids = await _capability_ids(conn)
        assert {alpha_ok, gamma_flag} <= ids and not {alpha_flag, beta_flag} & ids
        cur = await conn.execute("SELECT count(*) FROM capabilities WHERE vendor = 'Beta'")
        assert (await cur.fetchone())[0] == 0

        # a verdict is per claim: without the topics the rejected claims stay out
        doc = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
        path = tmp_path / "no-topics.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        await load.load_taxonomy(conn, path)
        assert await load.current_version(conn) == version  # topics are not in the version
        assert await support.topic_dimensions(conn) == []
        assert (await capabilities.build(conn)).support_excluded == 2
        assert not {alpha_flag, beta_flag} & await _capability_ids(conn)
        again = await support.run(conn, client=fake, settings=_settings(), check_max_usd=1.0,
                                  review_max_usd=1.0)
        assert (again.check.claims, again.review.claims) == (0, 0) and len(fake.calls) == 3
    finally:
        await conn.close()


async def test_a_held_lock_does_nothing(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    other = await connect(vkb_dsn)
    try:
        version = await _fresh(conn, tmp_path)
        await _fact(conn, version, RETENTION, "kept for a year")
        await other.execute("SELECT pg_advisory_lock(%s)", (support.LOCK_KEY,))
        fake = Fake({})
        report = await support.run(conn, client=fake, settings=_settings(), check_max_usd=1.0,
                                   review_max_usd=1.0)
        assert report.locked and fake.calls == []
    finally:
        await other.execute("SELECT pg_advisory_unlock(%s)", (support.LOCK_KEY,))
        await other.close()
        await conn.close()


async def test_without_topics_nothing_is_in_scope(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await conn.execute("DROP TABLE IF EXISTS claim_support, capabilities, raw_claims,"
                           " rejected_claims, extractions, extract_batches, chunks, documents"
                           " CASCADE")
        await init_schema(conn, embed_dim=8)
        await load.load_taxonomy(conn, FIXTURE)
        version = await load.current_version(conn)
        await _fact(conn, version, RETENTION, "kept for a year")
        assert await support.due_for_check(conn, version) == []
        assert await support.topic_dimensions(conn) == []
        assert (await capabilities.build(conn)).support_excluded == 0
    finally:
        await conn.close()


class Status(Exception):
    """An API error with an HTTP status, as both SDKs' errors carry it."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


async def test_a_claim_deleted_while_asked_is_skipped_and_the_run_finishes(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    other = await connect(vkb_dsn)
    try:
        version = await _fresh(conn, tmp_path)
        doomed = await _fact(conn, version, RETENTION, "deleted meanwhile")
        kept = await _fact(conn, version, RETENTION, "kept for a year")

        class Deleting(Fake):
            async def create(self, **params):
                if _capability(params) == "deleted meanwhile":
                    await other.execute("DELETE FROM raw_claims WHERE claim_id = %s", (doomed,))
                return await super().create(**params)

        fake = Deleting({"deleted meanwhile": True, "kept for a year": True})
        report = await support.run(conn, client=fake, settings=_settings(), check_max_usd=1.0,
                                   review_max_usd=1.0)
        assert (report.check.calls, report.check.vanished, report.check.supported) == (2, 1, 1)
        assert set(await _rows(conn)) == {kept}
        cur = await other.execute("SELECT pg_try_advisory_lock(%s)", (support.LOCK_KEY,))
        assert (await cur.fetchone())[0]  # released after every call finished
        await other.execute("SELECT pg_advisory_unlock(%s)", (support.LOCK_KEY,))
    finally:
        await other.close()
        await conn.close()


async def test_a_status_no_retry_can_fix_stops_the_step(vkb_dsn, tmp_path):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn, tmp_path)
        for n in range(4):
            await _fact(conn, version, RETENTION, f"kept {n}")
        fake = Fake({f"kept {n}": Status(404) for n in range(4)})
        try:
            await support.run(conn, client=fake, settings=_settings(support_concurrency=1),
                              check_max_usd=1.0, review_max_usd=1.0)
        except Status as e:
            assert e.status_code == 404
        else:
            raise AssertionError("a 404 must stop the run")
        assert len(fake.calls) == 1 and await _rows(conn) == {}
        # a transient status is this claim's failure only
        fake = Fake({f"kept {n}": Status(503) for n in range(4)})
        report = await support.run(conn, client=fake, settings=_settings(), check_max_usd=1.0,
                                   review_max_usd=1.0)
        assert report.check.invalid == 4 and len(fake.calls) == 4
    finally:
        await conn.close()
