# Taxonomy v2 and Claims v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract claims with structured qualifiers (claims-v3), load a class-based taxonomy v2, map
claims onto it, and serve them through `aspects()` and `capabilities()`, measured by a findability
check.

**Architecture:**
- Extraction keeps its pipeline and evidence gate. Only the schema, the prompt and the stored
  columns change.
- Taxonomy v2 replaces v1 in place: new YAML format, new tables, a guarded one-time migration.
- The mapper resolves, in order:
  - target, then child target, variant or object type;
  - dimension (by alias, then LLM adjudication over nearest-neighbour candidates);
  - offering and edition.
- `capabilities` v2 is rebuilt from mapped claims and queried with scope labels.
- Paid runs and anything touching the internal comparison tables are **controller tasks**, done by
  the main session and never by subagents.

**Tech Stack:** Python 3.12, psycopg 3 (async), Postgres + pgvector, PyYAML, numpy, the Anthropic
SDK and the OpenAI-compatible client already in `vendorkb`, pytest with testcontainers, uv.

**Spec:** `docs/superpowers/specs/2026-11-06-taxonomy-v2-claims-v3-design.md` (revision 2,
approved 2026-11-06).

## Global Constraints

- The repo is public. **Nothing from `~/Documents/Shared/projects/vendorkb-temp/` is ever read by
  a subagent, committed, or quoted.** Scripts take such paths as arguments, and tests use
  synthetic fixtures.
- Secrets live only in the untracked `.env`. Never print, copy or commit it. Run gitleaks before
  every push.
- Paid runs (Anthropic, OpenRouter, Voyage) are controller tasks. Each one proves the variable is
  engaged on about 2 items first and has a hard cost cap.
- `PROMPT_VERSION = "claims-v3"`.
- Claim enums:
  - `kind`: `capability | licensing | service`.
  - `support`: `supported | partial | not_supported | unclear`.
  - `availability`: `generally_available | preview | early_access | limited_availability | planned | deprecated | unstated`.
  - `default_state`: `on | off | null`.
  - `bound`: `max | min | exact | null`.
- Dimension families, in this order: `coverage, protection, schedule, retention, storage,
  recovery, security, cyber, operations, governance, data_use, vendor`.
- Required coverage dimensions, in this order: `coverage.backup, coverage.restore,
  coverage.export, coverage.compare`.
- No `requires` field anywhere, and pricing is never a claim.
- Nearest neighbour alone never maps a dimension or a target.
- Lint covers tests: `uv run ruff check src tests`, plus `uv run mypy src` and
  `uv run python -m pytest`. CI runs all three.
- Production-changing actions (helm, pushes to `main`, merges) need the user's go-ahead. Writing
  claims and taxonomy rows into the production DB through the CLI is allowed, as in the pilot.

## Review Focus

1. **A reply that breaks the schema.** The OpenAI-compatible backend may not enforce it, so a
   reply can give `value: "99"` or `kind: "pricing"`. That claim is rejected with reason
   `schema:<field>` and the chunk's other claims are still stored (Task 1).
2. **A taxonomy YAML with problems.** For example, two dimensions sharing an alias plus an
   `object_aliases` key for an object type the class lacks. The load must refuse, list *every*
   problem, and leave the DB untouched (Task 3).
3. **`init_schema` on the production DB, which still holds v1 tables.** The first run drops the v1
   `capabilities`, `features`, `aliases` and `targets` tables. Later runs are no-ops. `raw_claims`
   rows of every prompt version survive (Task 3).
4. **A target query.** Product-wide facts must appear only in the `product_wide` group, never in
   `facts`. Facts with no offering appear only in `product_level` when an offering is filtered
   (Task 12).
5. **A `target_scope` that names an unrelated target,** for example target "AWS" with scope "Azure
   SQL Database". It must not map to `azure...` as a child of AWS. It falls through to variant,
   then object type, then stays unmapped (Task 11).

---

### Task 1: claims-v3 schema, prompt and storage

**Files:**
- Modify: `src/vendorkb/claims/extract.py` (constants, `CLAIM_SCHEMA`, `SYSTEM_PROMPT`, `_record`)
- Modify: `src/vendorkb/schema.sql` (`raw_claims` and `rejected_claims` columns)
- Test: `tests/integration/test_claims_extract.py`

**Interfaces:**
- Produces:
  - Constants: `PROMPT_VERSION = "claims-v3"`, `KIND_VALUES`, `AVAILABILITY_VALUES`,
    `DEFAULT_STATE_VALUES = ("on", "off")`, `BOUND_VALUES = ("max", "min", "exact")`.
  - `validate_claim(claim: dict) -> str | None`, which returns `"schema:<field>"` for the first
    invalid field, else None.
  - New `raw_claims` columns: `kind text NOT NULL DEFAULT 'capability'` (CHECK in
    `KIND_VALUES`), `target_scope text`, `offering text`, `edition text`, `availability text`
    (CHECK in values or NULL), `default_state text` (CHECK), `value double precision`,
    `unit text`, `bound text` (CHECK).
  - The same columns on `rejected_claims`, all nullable and with no CHECKs.

- [ ] **Step 1: Write the failing tests**
  - `test_schema_has_the_v3_fields`: the item `required` list is exactly `["kind", "capability",
    "target", "target_scope", "support", "offering", "edition", "availability", "default_state",
    "value", "unit", "bound", "conditions", "evidence_quote"]`. `additionalProperties` is False.
    `default_state` and `bound` accept null.
  - `test_v3_claim_fields_are_stored`: a fake reply with `kind="licensing"`,
    `target_scope="users"`, `edition="Premium"`, `availability="preview"`,
    `default_state="off"`, `value=99`, `unit="years"`, `bound="max"`. Every value lands in
    `raw_claims`, and `prompt_version == "claims-v3"`.
  - `test_invalid_qualifier_rejects_only_that_claim`: one reply carries one claim with
    `value="99"` and one valid claim. The bad one goes to `rejected_claims` with
    `reason == "schema:value"`, and the valid one is accepted.
  - `test_prompt_states_the_v3_rules`: `SYSTEM_PROMPT` contains `"licensing"`, `"service"`,
    `"one claim per object type"` and `"never a price"`, and does not contain `"requires"`.
  - Update the existing prompt-version assertion to `"claims-v3"`.

- [ ] **Step 2: Run them and confirm they fail**

  Run: `uv run python -m pytest tests/integration/test_claims_extract.py -q`
  Expected: the new tests FAIL.

- [ ] **Step 3: Implement**
  - Schema: extend `CLAIM_SCHEMA`. Use `{"anyOf": [{"type": "string", "enum": [...]}, {"type":
    "null"}]}` for nullable enums, and `{"anyOf": [{"type": "number"}, {"type": "null"}]}` for
    `value`.
  - Storage: `_record` calls `validate_claim` before the gate. Invalid claims go to
    `rejected_claims` with the schema reason. The gate's rejection path stays as it is.
  - Prompt: rewrite `SYSTEM_PROMPT` to the spec §3.1 field table and §3.2 rules. Keep v2's
    support definitions and exclusions, minus "licensing", and add these rules:
    - Licensing and support terms are claims, with `kind` set.
    - "One claim per object type": a sentence listing object types gives one claim each, all
      sharing the quote.
    - `offering`, `edition`, `availability` and `default_state` only when the text says so;
      `unstated` and null otherwise.
    - `value`, `unit` and `bound` only for an explicit number in the quote.
    - A dependency stated outright goes in `conditions`.
    - A price is never a claim.
  - Columns: add them with `ADD COLUMN IF NOT EXISTS`. Add each CHECK only when absent, in the
    same `DO $$ ... $$` guard style used for `extractions_status_check`.

- [ ] **Step 4: Run the full suite**

  Run: `uv run ruff check src tests && uv run mypy src && uv run python -m pytest -q`
  Expected: all pass.

- [ ] **Step 5: Commit** with message `feat(claims): claims-v3 schema, prompt and storage`.

### Task 2 (controller): Opus smoke for claims-v3

- [ ] Re-run the scratchpad smoke (`opus-smoke/smoke.py`, adapted to v3) on:
  - the same 20 spot-check chunks;
  - 10 production chunks chosen by SQL for qualifier words (`Premium`, `preview`, `by default`,
    `up to`, `licens`, `per user`).

  Cap: $2. Assert `PROMPT_VERSION == "claims-v3"` and the response model.
- [ ] Pass when all four hold:
  - the 14 v1 errors stay fixed;
  - every qualifier on the 10 chosen chunks is right on manual inspection;
  - licensing sentences come back with `kind="licensing"`;
  - no price is extracted.

  Otherwise revise the prompt (this is Task 1's code) and repeat. Ledger the result.

### Task 3: taxonomy v2 format, validation, tables and loader

**Files:**
- Rewrite: `src/vendorkb/taxonomy/load.py`
- Modify: `src/vendorkb/schema.sql` (v2 tables, guarded v1 drop)
- Create: `tests/fixtures/taxonomy_v2_small.yaml` (synthetic, our own words)
- Test: `tests/integration/test_taxonomy_load_v2.py`. Delete the v1 load tests in
  `tests/integration/test_taxonomy_pipeline.py`. Tasks 10–12 replace the rest of that file.

**Interfaces:**
- Produces:
  - `FAMILIES` and `REQUIRED_DIMENSIONS` (the Global Constraints values).
  - `class TaxonomyError(ValueError)`, with `.problems: list[str]`.
  - `parse_doc(data: bytes, where) -> dict` (unique keys, as today).
  - `validate(doc) -> Taxonomy`, a frozen dataclass with `dimensions`, `classes`, `targets` and
    `offerings` lists of typed entries, plus `lineage(class_id) -> list[str]` (the class first,
    then its ancestors).
  - `async load_taxonomy(conn, path) -> dict`, returning the version and per-table counts.
  - `async current_version(conn) -> str | None`. The version string is `"2-<sha8>"`.
- YAML format (top-level keys `version: 2`, `dimensions`, `classes`, `targets`, `offerings`):
  - **dimension:** `id` (`<family>.<snake>`), `name`, `definition`, `aliases` (list), `applies_to`
    (`all` or a list of class ids), optional `value_unit`, optional `platform: true`.
  - **class:** `id` (dotted, and every dotted prefix must be a class), `name`, `definition`,
    `object_types` (a list of `{id, name, aliases}`).
  - **target:** `id` (dotted, and every dotted prefix must be a target), `name`, `class`,
    `aliases`, optional `object_aliases` (object-type id → aliases), optional `variants` (a list
    of `{id, name, aliases}`).
  - **offering:** `vendor`, `id` (`<snake>.<snake>`), `name`, `aliases`, `editions` (a list of
    `{id, name, aliases}`).
- Tables:
  - `dimensions(dimension_id pk, family, name, definition, applies_to text[] NULL=all,
    value_unit, platform bool)`;
  - `classes(class_id pk, parent_id, name, definition)`;
  - `object_types(class_id, object_type_id, name, pk(class_id, object_type_id))`;
  - `targets(target_id pk, parent_id, class_id, name)`;
  - `variants(target_id, variant_id, name, pk)`;
  - `offerings(offering_id pk, vendor, name)`;
  - `editions(offering_id, edition_id, name, pk)`;
  - `aliases(alias, entity_type, scope, entity_id, pk(alias, entity_type, scope))`.

  `entity_type` and `scope` pairs:

  | entity_type | scope |
  |---|---|
  | `dimension` | `''` |
  | `target` | `''` |
  | `object_type` | class_id |
  | `target_object_type` | target_id |
  | `variant` | target_id |
  | `offering` | vendor |
  | `edition` | offering_id |

  Aliases are stored normalized with `cluster.normalize`, and each entity's own name is stored as
  an alias too.

- [ ] **Step 1: Write the failing tests**
  - `test_valid_fixture_loads`: the counts match the fixture, the version starts with `"2-"`,
    and `lineage("saas.identity") == ["saas.identity", "saas"]`.
  - `test_every_problem_is_reported`: a doc with
    - a duplicate dimension alias,
    - an unknown family,
    - a target whose class is missing,
    - an `object_aliases` key not in the class lineage,
    - and a missing `coverage.export`

    raises `TaxonomyError` with ≥5 problems naming each, and the DB tables are unchanged.
  - `test_child_class_cannot_redefine_parent_object_type`: raises `TaxonomyError`.
  - `test_v1_tables_are_dropped_once`: create the v1 `features`, `targets(kind)`, `capabilities
    (feature_id)` and `aliases(entity_type IN feature,target)` tables with rows. Insert one
    `raw_claims` row. Run `init_schema` twice. Afterwards the v2 tables exist, the v1 tables
    are gone, and the `raw_claims` row survives.
  - `test_reload_replaces_everything`: loading a second fixture with one dimension removed
    leaves no rows for it in `dimensions` or `aliases`.

- [ ] **Step 2: Run them and confirm they fail**

  Run: `uv run python -m pytest tests/integration/test_taxonomy_load_v2.py -q`
  Expected: FAIL.

- [ ] **Step 3: Implement**
  - Validation collects every problem before raising.
  - The v1 drop is a `DO $$` block that drops the v1 tables only when `capabilities.feature_id`
    exists (`information_schema.columns`). Drop `capabilities` first, then `aliases`,
    `features` and `targets`, and then create the v2 tables.
  - The load replaces all v2 rows in one transaction and sets `taxonomy_meta.taxonomy_version`.
  - Update `vendorkb taxonomy load` to the new return value.

- [ ] **Step 4: Run the full suite**, as in Task 1 Step 4. Expected: pass.

- [ ] **Step 5: Commit** with message `feat(taxonomy): v2 format, validation and loader; drop v1
  tables`.

### Task 4: `aspects()`

**Files:**
- Create: `src/vendorkb/taxonomy/aspects.py`
- Modify: `src/vendorkb/cli.py` (`vendorkb taxonomy aspects (--target T | --class C)`, JSON
  lines; unknown → stderr message, exit 2)
- Test: `tests/integration/test_taxonomy_aspects.py`

**Interfaces:**
- Consumes: the Task 3 tables. The lineage is computed from `classes.parent_id`, the DB
  equivalent of `Taxonomy.lineage`.
- Produces:
  - `@dataclass(frozen=True) class Aspect`, with fields `kind: Literal["coverage", "dimension",
    "platform"]`, `dimension_id: str`, `family: str`, `name: str`, `definition: str`,
    `object_type_id: str | None` and `value_unit: str | None`.
  - `async aspects(conn, *, target: str | None = None, cls: str | None = None) -> list[Aspect]`.
    Exactly one of the two arguments must be given. The target resolves by id or alias. Unknown
    input raises `LookupError`.
  - Order:
    - coverage aspects first (lineage object types in YAML order, times `REQUIRED_DIMENSIONS`
      order);
    - then the class's non-coverage dimensions, in `FAMILIES` order, then by id;
    - then platform dimensions.
  - A dimension applies to class C when `applies_to` is NULL, or any entry is C or an ancestor of
    C.

- [ ] **Step 1: Write the failing tests**
  - `test_okta_gets_entra_aspects`: the fixture has `entra_id` and `okta`, both
    `saas.identity`. `aspects(target="okta")` and `aspects(target="entra_id")` return equal
    lists.
  - `test_alias_resolves_target`: `aspects(target="Okta Workforce Identity")` works.
  - `test_class_only_dimension`: a dimension with `applies_to: [saas.crm]` is absent for `okta`
    and present for `salesforce`.
  - `test_parent_entry_applies_to_children`: `applies_to: [saas]` reaches `okta`.
  - `test_unknown_target_raises`.
  - A CLI test: an unknown target exits 2.
- [ ] **Step 2: Run them and confirm they fail**, then **Step 3: implement**, then **Step 4: run
  the full suite** (all pass).
- [ ] **Step 5: Commit** with message `feat(taxonomy): aspects() and the taxonomy aspects
  command`.

### Task 5: findability script, part 1 (taxonomy coverage)

**Files:**
- Create: `eval/findability.py`, with subcommand `coverage --map PATH --taxonomy PATH
  [--json OUT]`
- Test: `tests/unit/test_findability.py` (synthetic map and taxonomy only)

**Interfaces:**
- Consumes: Task 3 `parse_doc` and `validate`.
- Map format: a YAML list of `{table: str, row: str, vendor: str | null, ids: [str],
  out_of_scope: str | null}`. `vendor` is the table's vendor; Task 13 uses it. An
  id is one of:
  - a dimension id;
  - `target:<target_id>`;
  - `object:<class_id>:<object_type_id>` (the object type must be in that class's lineage).
- Produces: `coverage_report(map_rows, taxonomy) -> dict`, with keys `rows`, `out_of_scope`,
  `in_scope`, `mapped`, `unmapped: [{table, row}]` and `unknown_ids: [{table, row, id}]`. The CLI
  prints the counts and the two lists, and exits 1 if `unmapped` or `unknown_ids` is non-empty.
  It never prints anything but table names, row labels and ids.

- [ ] **Step 1: Write the failing tests**
  - Counts on a 5-row synthetic map (one out of scope, one unmapped, one unknown id).
  - Exit code 1 when anything is unmapped.
  - An `object:` id whose type comes from a parent class is accepted.
- [ ] **Step 2: Run them and confirm they fail**, then **Step 3: implement**, then **Step 4: run
  the full suite** (all pass).
- [ ] **Step 5: Commit** with message `feat(eval): findability coverage report`.

### Task 6 (controller): taxonomy v2 seed and coverage map

- [ ] Write `taxonomy/taxonomy.yaml` (v2) on this branch, in our own words:
  - all families and dimensions;
  - the spec §4.2 classes plus their top-level parents (`saas`, `cloud`, `virtualization`,
    `kubernetes`, `app`, `file`, `endpoint`, `storage`, `backup`) and object types;
  - targets for every workload a corpus vendor protects, with `object_aliases` and `variants`;
  - offerings for the vendors whose documents name them.

  Sources: the spec, domain knowledge, and the v1 taxonomy on branch `taxonomy-v1`.
- [ ] Write `~/Documents/Shared/projects/vendorkb-temp/coverage-map.yaml` with one entry per
  comparison-table row, and never commit it.
- [ ] Iterate until `uv run python eval/findability.py coverage --map <that> --taxonomy
  taxonomy/taxonomy.yaml` exits 0 and `vendorkb taxonomy load` accepts the file.
- [ ] Commit `taxonomy/taxonomy.yaml` with message `feat(taxonomy): v2 seed`.

### Task 7 (controller): claims-v3 precision check

- [ ] Sample 300 random production chunks of ≥200 characters, excluding the pilot ids, and write
  them to `eval/m3v3_precision_ids.txt`.
- [ ] Run `vendorkb extract --ids-file ... --mode sync` with `EXTRACT_MODEL=claude-opus-5-5`,
  `EXTRACT_EFFORT=low` and `EXTRACT_BUDGET_USD` set to the ledger total plus 6. Smoke on 2 chunks
  first.
- [ ] Draw 100 random claims and grade them with Fable subagents (model `fable`), using a brief
  that gives the spec §3.1 field definitions. A claim is correct only if all fields are right.
  Record field-level accuracy.
- [ ] Gate: ≥95%. If it fails, revise the prompt (Task 1 code, a new prompt version), re-run on
  300 fresh chunks, and re-grade, at most 3 times (about $15 in total). After that, stop and
  report to the user.

### Task 8: teacher-set sampler

**Files:**
- Create: `eval/teacher_sample.py`
- Test: `tests/integration/test_teacher_sample.py`

**Interfaces:**
- CLI: `teacher_sample.py --must-include PATH --total N --heldout-frac F --seed S --exclude
  IDSFILE... --out-train PATH --out-heldout PATH`.
- `--must-include` is a YAML list of `{vendor, product_like: str | null, title_like: str |
  null}` (SQL ILIKE patterns). It lives outside the repo when it is derived from the internal
  tables.
- Produces: `select_chunks(conn, *, must: list[dict], total: int, heldout_frac: float, seed: int,
  exclude: set[str]) -> tuple[list[str], list[str], dict]`, returning the train ids, the held-out
  ids and a stats dict.
  - Must-include chunks come first, all of them, up to `total // 2`, with any overflow sampled
    evenly across their documents.
  - The rest is spread across all vendors in proportion to `sqrt(vendor chunk count)`, with at
    least `min(50, available)` per vendor. Within a vendor, chunks are taken round-robin across
    documents.
  - The held-out set is a seeded random `heldout_frac` of the selection, disjoint from train.
  - Excluded ids never appear, and the output is deterministic for a given seed.

- [ ] **Step 1: Write the failing tests** on a synthetic DB with three vendors and uneven
  documents:
  - totals and disjointness;
  - every vendor gets at least the minimum;
  - must-include chunks are present;
  - excluded ids are absent;
  - the same seed gives the same output.
- [ ] **Step 2: Run them and confirm they fail**, then **Step 3: implement**, then **Step 4: run
  the full suite** (all pass).
- [ ] **Step 5: Commit** with message `feat(eval): stratified teacher-set sampler`.

### Task 9 (controller): Opus teacher capture

- [ ] Write the must-include file outside the repo, from the comparison tables' workloads. Run the
  sampler with `--total 10000 --heldout-frac 0.05`, excluding the precision ids.
- [ ] Run `vendorkb extract --mode batch --backend anthropic` with Opus and effort low.
  `EXTRACT_BUDGET_USD` is the ledger total plus 100. Smoke one 2-chunk batch first. Run it in the
  background; Tasks 10–13 proceed meanwhile.
- [ ] When it is done, record the counts, cost, gate pass rate and truncations in the ledger, and
  re-run any truncated chunks with `--retry-truncated` within the cap.

### Task 10: alias proposals for v2 (`taxonomy propose`)

**Files:**
- Rewrite: `src/vendorkb/taxonomy/propose.py`
- Modify: `src/vendorkb/cli.py` (`--kind dimension|target|object_type|offering`)
- Test: `tests/integration/test_taxonomy_pipeline.py` (replace the v1 propose tests)

**Interfaces:**
- Consumes: Task 3 `validate`, `cluster.cluster_strings` and the existing embedder and client.
- Inputs: v3 claims only (`prompt_version = 'claims-v3'`).
  - `dimension` clusters `capability_raw`.
  - `target` clusters `target_raw`.
  - `object_type` clusters `target_scope` per resolved target class (the class comes from the
    target alias; unresolved ones are skipped).
  - `offering` clusters `offering` and `edition` per vendor.
- For each cluster, the LLM gets the members and the top 5 existing entries by embedding
  similarity (`id`, `name`, `definition`). It answers with strict JSON:
  `{"action": "alias" | "new" | "none", "id": str | null, "new": {...} | null}`.
  - `new` needs the fields of its section: `family` for a dimension, `class` for a target, and
    so on.
- Produces: `async propose(conn, *, client, embedder, settings, kind, distance_threshold,
  out_path, max_usd, limit) -> ProposeReport`.
  - It merges aliases into existing entries of `out_path`, and appends new entries with
    `proposed: true`.
  - The output must pass `validate`. Entries that would break validation are skipped and
    counted in `invalid`.
  - The cost cap and the long-tail file behave as today.

- [ ] **Step 1: Write the failing tests** with a fake client and embedder:
  - an `alias` answer adds the members to that entry;
  - a `new` dimension with an unknown family is counted invalid and not written;
  - the output file loads through `validate`;
  - the budget stop works.
- [ ] **Step 2: Run them and confirm they fail**, then **Step 3: implement**, then **Step 4: run
  the full suite** (all pass).
- [ ] **Step 5: Commit** with message `feat(taxonomy): v2 alias proposals`.

### Task 11: mapper v2 with LLM adjudication

**Files:**
- Rewrite: `src/vendorkb/taxonomy/mapper.py`
- Modify: `src/vendorkb/schema.sql` (`raw_claims`: `dimension_id`, `dimension_method`,
  `class_id`, `variant_id`, `object_type_id`, `offering_id`, `edition_id`; keep `target_id`,
  `target_method`, `mapped_version`; methods are `alias | llm | unmapped`)
- Modify: `src/vendorkb/config.py` (`map_model: str = "claude-haiku-5-5"`, `map_backend` with the
  same Literal as `extract_backend`, `map_candidates: int = 5`)
- Test: `tests/integration/test_taxonomy_mapper_v2.py`

**Interfaces:**
- Consumes: the Task 3 tables and aliases, the embedder, and the extract client builder.
- Produces: `async map_claims(conn, *, embedder, client, settings, max_usd: float) -> MapReport`.
  It works on v3 claims only. Per distinct normalized string:
  1. **Target:** an alias exact match (scope `''`). Otherwise the top `map_candidates` targets by
     cosine over name+aliases go to an LLM adjudication that may answer `none`.
  2. **`target_scope`, given a resolved target T**, tried in this order:
     - a target alias whose entry has T as an ancestor;
     - a variant alias, scope T;
     - a `target_object_type` alias, scope T;
     - an `object_type` alias, scope each class in T's lineage.

     If T is null, try `target_scope` as a target alias.
  3. **Dimension:** an alias exact match. Otherwise the top candidates are drawn only from
     dimensions that apply to T's class (all of them when T is null) and adjudicated. A
     `coverage.*` result needs an object type to count as coverage. Otherwise the dimension is
     kept and the object type stays null.
  4. **Offering:** an alias within the claim's vendor. **Edition:** within that offering, or
     unique across the vendor's offerings when the offering is null.
- **The adjudication call:** a strict JSON reply `{"choice": "<candidate id>" | "none"}`, with the
  claim's capability, target, target_scope and vendor as context. One call per distinct (kind,
  string, candidate set), cached for the run. It stops at `max_usd`, and leaves the rest
  unmapped and counted.
- **Embeddings:** key vectors are embedded once per run, as float32.

- [ ] **Step 1: Write the failing tests** with a fake embedder and client:
  - alias hits make no LLM call;
  - an LLM answer of `none` leaves the claim unmapped;
  - the unrelated-scope case (Review Focus 5) stays unmapped;
  - "FSx for Lustre" maps to a variant, "RDS for Oracle" to a child target, and "users" to an
    object type;
  - dimension candidates exclude dimensions that do not apply;
  - the budget stop leaves claims unmapped;
  - v1 and v2 claims are untouched.
- [ ] **Step 2: Run them and confirm they fail**, then **Step 3: implement**, then **Step 4: run
  the full suite** (all pass).
- [ ] **Step 5: Commit** with message `feat(taxonomy): mapper v2 with LLM adjudication`.

### Task 12: `capabilities` v2 build and query

**Files:**
- Rewrite: `src/vendorkb/taxonomy/capabilities.py`
- Modify: `src/vendorkb/schema.sql` (the v2 `capabilities` table, with spec §6.2 columns plus
  `scope text CHECK (scope IN ('offering', 'product', 'product_wide'))`, `conflict_group uuid`
  and `is_current bool`)
- Modify: `src/vendorkb/cli.py` (`capabilities build`, and `capabilities query` with filter
  flags, printing JSON)
- Test: `tests/integration/test_capabilities_v2.py`

**Interfaces:**
- Produces:
  - `async build(conn) -> int`. It builds one row per v3 claim mapped under the current version
    whose dimension is not null.
    - `scope` is `offering` when `offering_id` is set, else `product_wide` when `target_id` is
      null and the dimension is not platform, else `product`.
    - Rows sharing the conflict key but differing in support or value get one `conflict_group`.
    - `is_current` marks the row from the most recently fetched document (`documents.fetched_at`).
  - `async query(conn, f: CapFilter) -> CapResult`.
    - `CapFilter` is a dataclass with optional `vendor`, `offering`, `edition`, `dimension`,
      `family`, `target` (descendants included), `object_type`, `support`, `availability`,
      `value_min` and `value_max`.
    - `CapResult` has `facts`, `product_level`, `product_wide`, `conflicts` and `hint: str |
      None`. Each fact carries the qualifiers, the evidence quote, the URL and the title.
    - When `target` is filtered, `product_wide` rows of the same vendor and dimension go to
      `product_wide`.
    - When `offering` is filtered, rows with no offering go to `product_level`.
    - When everything is empty, the hint names the vendor, target and dimension to search for in
      chunks.

- [ ] **Step 1: Write the failing tests:**
  - the scope assignment;
  - Review Focus 4 (product-wide facts never in `facts`; product-level only with an offering
    filter);
  - target descendants are included;
  - a value range filter;
  - the conflict group and `is_current`;
  - the empty result's hint;
  - a rebuild is idempotent.
- [ ] **Step 2: Run them and confirm they fail**, then **Step 3: implement**, then **Step 4: run
  the full suite** (all pass).
- [ ] **Step 5: Commit** with message `feat(taxonomy): capabilities v2 build and query`.

### Task 13: findability script, parts 2 and 3

**Files:**
- Modify: `eval/findability.py`. Add these subcommands:
  - `claims --map PATH --dsn ...`;
  - `chunks-candidates --map PATH --out JSONL`;
  - `chunks-score --candidates JSONL --labels JSONL`.
- Test: `tests/integration/test_findability_db.py`

**Interfaces:**
- **`claims`:** for each in-scope row (it needs the row's vendor, given in the map as an optional
  `vendor` field), calls `capabilities.query` with the row's ids. It reports rows with ≥1 fact.
  - A `target:` id filters target.
  - An `object:` id filters object type with the `coverage.*` dimensions.
  - A dimension id filters dimension.
- **`chunks-candidates`:** for each row with no claim, builds the question `"<vendor> <target
  name> <dimension or object name>"`, runs `search.hybrid_candidates` with the rerank, and writes
  the top 10 as `{row_key, chunk_id, text}`.
- **`chunks-score`:** reads the labels `{row_key, chunk_id, relevant: bool}` and reports the rows
  with ≥1 relevant chunk.
- Output never includes table notes. Row labels and ids only.

- [ ] **Step 1: Write the failing tests** on a synthetic DB:
  - claim findability counts;
  - candidates are written only for rows without claims;
  - the score arithmetic.
- [ ] **Step 2: Run them and confirm they fail**, then **Step 3: implement**, then **Step 4: run
  the full suite** (all pass).
- [ ] **Step 5: Commit** with message `feat(eval): findability claims and chunks reports`.

### Task 14 (controller): run, measure, review

- [ ] Load v2 and run the alias proposals (Task 10) on the teacher claims, capped at $3. Review
  the proposals, fold them into `taxonomy.yaml`, and reload.
- [ ] **Mapper sample:** have Opus subagents label 300 v3 claims (the correct dimension, target,
  object type and so on, from the taxonomy). Send the labels to the user for a check, then
  measure the mapper's per-field precision with `map_model`.
- [ ] Run the full mapping (capped at $5) and `capabilities build`. Run findability: claims, then
  chunk candidates, Opus-subagent relevance labels, and the score. Write
  `docs/superpowers/taxonomy-v2-claims-v3-<date>.md` with numbers only, no table content.
- [ ] Open the taxonomy v2 PR for the user's review and close PR #1 as superseded.
- [ ] Update `CLAUDE.md`: the v3 claim fields, taxonomy v2 commands, `aspects`, and the
  `capabilities` query.
- [ ] Final whole-branch review, then ask the user before merging or pushing to `main`.
