# Topic Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep a page's facts through page updates (Part A), then answer broad ransomware questions with the user's eight-aspect kit in every answer shape (Part B).

**Architecture:** Part A replaces "delete the document and its whole cascade" with "supersede the chunks that carry facts": superseded chunks leave search but keep their claims, which are retired once the new version is extracted and mapped. Part B adds curated `topics` to the taxonomy, detects a topic in a question, retrieves evidence per aspect (search plus the chunks behind capability facts) within a rerank budget, prompts the answer per aspect, renders "which vendors" questions as a code-built grid, and exposes the topic to MCP agents.

**Tech Stack:** Python 3.12, psycopg 3 (async), Postgres + pgvector, Voyage rerank, the OpenAI-compatible answer client, FastMCP; pytest with testcontainers; uv.

**Spec:** `docs/superpowers/specs/2026-11-08-topic-expansion-design.md`

## Global Constraints

- Precision first: every answer claim cited; never state that a product lacks an aspect because nothing was retrieved ("the retrieved sources do not cover this").
- Questions that do not trigger a topic behave exactly as today.
- Rerank budget per expanded question: `TOPIC_RERANK_DOCS = 240` documents; catch-up search: at most 20 candidates per aspect.
- Two PRs: PR A = Tasks 1, 2, 2b (deploy needs a pg_dump first), PR B = Tasks 3-9. Each: `uv run ruff check src tests`, `uv run mypy src`, `uv run --extra dev pytest` green; `gitleaks detect --source . --log-opts "origin/main..HEAD" --no-banner && git push`.
- Schema changes go in `src/vendorkb/schema.sql` (idempotent `IF NOT EXISTS` DDL); production applies them through the sync job's `init_schema`.
- New topic eval questions go in `eval/topic_questions.yaml`, never in `eval/questions.yaml` (other evals iterate that file).
- Never read `/home/openclaw/Documents/Shared/projects/vendorkb-temp`.

## Review Focus

1. A page re-synced with fewer chunks than before: the new ordinals must not collide with superseded ones (Task 1 test `test_fewer_chunks_after_an_update`).
2. A metadata-only re-tag (`store.update_metadata`) must not supersede anything (Task 1 test `test_a_retag_supersedes_nothing`).
3. A broad question that mentions encryption incidentally ("recover after ransomware encrypted my files") must still expand: cues are specific phrases, never a bare "encrypt" (Task 4 test `test_incidental_words_do_not_count_as_naming_an_aspect`).
4. A reranker failure during topic retrieval: fall back to RRF order and still answer (Task 5 test `test_topic_hits_without_a_reranker`).
5. An unknown topic in `query_capabilities(topic=...)`: a ToolError that lists the known topics (Task 8 test `test_unknown_topic_lists_the_known_ones`).

---

## Part A: facts survive page updates

### Task 1: Supersede instead of delete on re-sync; retire once the new version is mapped

**Files:**
- Modify: `src/vendorkb/schema.sql` (chunks table and upgrades)
- Modify: `src/vendorkb/store.py:35-50` (`write_document`), add `retire_superseded`
- Test: `tests/integration/test_supersede.py` (new)

**Interfaces:**
- Produces: column `chunks.superseded_at timestamptz` (NULL = current); `store.write_document(conn, meta, content_hash, chunks, vectors) -> None` (same signature, new behaviour); `store.retire_superseded(conn, taxonomy_version: str | None) -> int` (chunks deleted).

- [ ] **Step 1: Write the failing tests** in `tests/integration/test_supersede.py`, using the document/chunk/claim helpers of `tests/integration/test_capabilities_v2.py` (`_fresh`, `_claim`) or local equivalents that call `store.write_document`:

```python
async def test_an_update_keeps_the_extracted_version_until_the_new_one_is_mapped(vkb_dsn):
    # v1 written, its chunk extracted with one claim mapped at the current version
    # v2 written: v1's chunks are superseded (superseded_at set), the claim still exists,
    #   v2's chunks are current
    # v3 written before v2 was extracted: v2's chunks are deleted, v1's stay superseded,
    #   the claim still exists
    # retire_superseded(conn, version) == 0 while v3's chunks are unextracted
    # v3's chunks get a 'done' extraction and their claims are stamped mapped_version=version:
    #   retire_superseded(conn, version) == number of v1 chunks; the v1 claim is gone
async def test_fewer_chunks_after_an_update(vkb_dsn):
    # v1 with 3 chunks (one carries a claim), v2 with 1 chunk: no unique-violation; ordinals
    # 0..2 superseded and 0 current coexist
async def test_a_retag_supersedes_nothing(vkb_dsn):
    # update_metadata changes vendor/title: no chunk gets superseded_at
async def test_removal_still_deletes_everything(vkb_dsn):
    # delete_document on a document with superseded and current chunks leaves no chunk,
    # extraction or claim of it
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev pytest tests/integration/test_supersede.py -q`
Expected: FAIL (column `superseded_at` does not exist / claims deleted on update).

- [ ] **Step 3: Schema.** In `schema.sql`: drop the inline `UNIQUE (doc_id, ordinal)` from `CREATE TABLE chunks`; after the table add `ALTER TABLE chunks ADD COLUMN IF NOT EXISTS superseded_at timestamptz;`, `ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_doc_id_ordinal_key;`, `CREATE UNIQUE INDEX IF NOT EXISTS chunks_doc_ordinal_current_idx ON chunks (doc_id, ordinal) WHERE superseded_at IS NULL;`, `CREATE INDEX IF NOT EXISTS chunks_superseded_idx ON chunks (doc_id) WHERE superseded_at IS NOT NULL;`, with a comment citing the spec (Part A).

- [ ] **Step 4: Implement `write_document`** as: upsert the `documents` row (`INSERT ... ON CONFLICT (doc_id) DO UPDATE` of every column, `fetched_at = now()`); if any current chunk of the document has a `raw_claims` row, delete the document's superseded chunks and set `superseded_at = now()` on its current chunks, else delete its current chunks; then insert the new chunks. Docstring: the rule "the facts of the last extracted version stay until the next one is mapped".

- [ ] **Step 5: Implement `retire_superseded(conn, taxonomy_version)`**: delete superseded chunks of every document none of whose current chunks lacks a final extraction (`status IN ('done','failed')`, `prompt_version LIKE claims.extract.PROMPT_FAMILY`) or has a claim of that family whose `mapped_version IS DISTINCT FROM taxonomy_version` (with `taxonomy_version` None, the extraction condition only). The cascade removes their extractions, claims and capability rows.

- [ ] **Step 6: Run the tests and the whole suite**

Run: `uv run --extra dev pytest tests/integration/test_supersede.py -q && uv run --extra dev pytest -q`
Expected: PASS (fix any existing test that relied on `DELETE FROM documents` replacing chunks).

- [ ] **Step 7: Commit** `store: supersede the chunks that carry facts on re-sync; retire them once the new version is mapped`

### Task 2: Superseded chunks leave search and extraction; facts carry the marker

**Files:**
- Modify: `src/vendorkb/search.py` (`_filters`, `_vec_sql`, `vendor_candidates`, `read_section`)
- Modify: `src/vendorkb/claims/extract.py` (`unextracted_ids`), `src/vendorkb/claims/refresh.py` (`new_chunks`)
- Modify: `src/vendorkb/schema.sql` (capabilities), `src/vendorkb/taxonomy/capabilities.py` (`build`, `Fact`, `_FACT_COLUMNS`), `src/vendorkb/mcp_server.py` (`format_fact`)
- Test: `tests/integration/test_vendorkb_search.py`, `tests/integration/test_claims_extract.py`, `tests/integration/test_capabilities_v2.py`, `tests/unit/test_mcp_tools.py`

**Interfaces:**
- Consumes: `chunks.superseded_at`, `store.retire_superseded` (Task 1).
- Produces: `capabilities.evidence_superseded boolean NOT NULL DEFAULT false`; `Fact.evidence_superseded: bool` (last field); `capabilities.build` calls `store.retire_superseded(conn, version)` before rebuilding.

- [ ] **Step 1: Write the failing tests:**
  - `test_superseded_chunks_are_never_retrieved` (search): a superseded chunk with the best FTS and vector match appears in none of `hybrid_candidates`, `facet_candidates`, `vendor_candidates`, `search_docs`; `read_section(anchor, "1")` of a current chunk never returns superseded neighbours, and of a superseded anchor returns only superseded rows.
  - `test_unextracted_ids_skip_superseded_chunks` (extract) and the same for `refresh.new_chunks`.
  - `test_build_marks_superseded_facts_and_retires_replaced_ones` (capabilities): a claim on a superseded chunk yields a row with `evidence_superseded is True` and `query()` returns it with `Fact.evidence_superseded`; once the new version is extracted and mapped, `build` drops it.
  - `test_superseded_fact_is_marked` (MCP unit): `format_fact` output contains `from an earlier version of the page`.
- [ ] **Step 2: Run them to verify they fail** (`uv run --extra dev pytest -q -k "superseded"`).
- [ ] **Step 3: Implement:** every chunk query in search adds `c.superseded_at IS NULL` (put it in `_filters` and `_vec_sql`; `read_section` matches the anchor's own superseded state); the two extraction selectors add it; DDL `ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS evidence_superseded boolean NOT NULL DEFAULT false;`; `build` selects `ch.superseded_at IS NOT NULL` into it and starts with `retire_superseded`; `Fact`/`_FACT_COLUMNS` carry it; `format_fact` appends ` (from an earlier version of the page)` when set.
- [ ] **Step 4: Run the whole suite** (`uv run --extra dev pytest -q`, ruff, mypy). Expected: PASS.
- [ ] **Step 5: Commit** `search, extract, capabilities: superseded chunks leave search; their facts stay, marked`.

### Task 2b: Unchanged text keeps its facts

(Added 2026-11-08 by user decision; see spec Part A, Decisions. This replaces Task 1's page-level rule (ledger Ruling 4) with a per-chunk rule. PR A = Tasks 1, 2, 2b; pg_dump production before deploying.)

**Files:**
- Modify: `src/vendorkb/chunker.py` (add `reuse_key`)
- Modify: `src/vendorkb/store.py` (`write_document`), `src/vendorkb/sync.py` (`SyncReport.chunks_reused`, summed from `write_document`)
- Modify: `src/vendorkb/schema.sql` comments only: the `chunks` comment, where the sentence "So (doc_id, ordinal) is unique among the current chunks only" now follows an unrelated sentence, and the orphan line near the `capabilities` comment ("No foreign keys into the")
- Test: `tests/unit/test_chunker.py` (or the existing chunker test file), `tests/integration/test_supersede.py`

**Interfaces:**
- Produces: `chunker.reuse_key(text: str) -> str`; `store.write_document(...) -> int` (the number of current chunks reused); `SyncReport.chunks_reused: int = 0`.
- Consumes: Tasks 1-2 (`superseded_at`, `retire_superseded`, the search filters).

- [ ] **Step 1: Write the failing tests.**
  - `reuse_key` unit tests:
    - Equal keys when the texts differ only in whitespace or line breaks, a markdown link or image target (`[List Media](list_media.html)` vs `[List Media](./list_media.html)`; `![AIX](https://x/a.gif)` vs `![AIX](/media/1/a.gif)`), backslash escapes (`2026\-03\-26` vs `2026-03-26`), or a date line (`Last Updated: 2026\-03\-26` vs `Last Updated: 2026\-07\-31`; `Updated this week` vs `Updated yesterday`).
    - Different keys when a changelog-style line changes (`Updated support for Proxmox VE` vs `Updated support for Hyper-V`), when a table cell changes (`| Supported |` vs `| Supported for Linux x86_64 only |`), or when the link text changes.
  - `test_identical_chunks_keep_their_facts` (integration):
    - v1 has 3 chunks, each extracted with a claim mapped at the current version.
    - v2 changes only chunk 1.
    - Chunks 0 and 2 keep their `chunk_id`, extraction and claim, and stay current (`superseded_at IS NULL`).
    - Old chunk 1 is superseded, and a new chunk 1 is current and unextracted.
    - `write_document` returns 2.
    - Once the new chunk is extracted and mapped, `retire_superseded` deletes only old chunk 1.
  - `test_cosmetic_changes_reuse_the_chunk_with_the_new_text`: a link target, a date line and line breaks change. The chunk is reused (same `chunk_id`, claim kept), and its `text`, `token_count` and embedding are the new version's.
  - `test_a_quote_no_longer_in_the_text_blocks_reuse`: the keys are equal, but the claim's `evidence_quote` spans a changed link target, so it is not in the new text (whitespace-collapsed, case-insensitive). The chunk is not reused: the old one is superseded and the new one is inserted.
  - `test_reordered_chunks_are_reused`: v1 [A, B], v2 [B, A]. Both are reused with swapped ordinals, with no unique violation on `chunks_doc_ordinal_current_idx`.
  - `test_a_dropped_chunk_is_kept_only_with_a_mapped_claim`:
    - A dropped chunk whose claim is mapped is superseded.
    - A dropped chunk whose claims are all unmapped (`mapped_version IS NULL`), or that has no claim, is deleted.
    - Superseded chunks are never deleted by `write_document`: v1 mapped, then v2 mapped but not retired, then v3, so both v1's and v2's dropped chunks stay superseded until `retire_superseded`.
  - Update the Task 1 tests that encode the page-level rule (Ruling 4) to the per-chunk rule, and list each changed assertion in the report.
- [ ] **Step 2: Run them to verify they fail** (`uv run --extra dev pytest -q -k "reuse or supersede"`).
- [ ] **Step 3: `reuse_key`.** Drop date lines, then reduce markdown links and images to their text, then drop backslashes, then collapse whitespace.
  - Date line: a whole line `^\s*(last\s+)?updated\s*(:|on)?\s*<when>\s*$`, case-insensitive, where `<when>` is an ISO-like date (escaped hyphens allowed), a `Month D, YYYY` date, `today`, `yesterday`, `this week`, `last week`, `N (minutes|hours|days|weeks|months|years) ago` or `a (day|week|month|year) ago`. Never a bare `^updated`.
  - Links: `!?\[([^\]]*)\]\([^)]*\)` → `\1`.
- [ ] **Step 4: `write_document`** (keep the document upsert):
  - Load the page's current chunks (`chunk_id`, `ordinal`, `text`) and the evidence quotes of their claims.
  - Match each new chunk, in ordinal order, to the first unused current chunk with an equal `reuse_key` whose every quote appears in the new text. Normalise both sides for that check: collapse whitespace and lowercase.
  - Reused chunks: update `section_path`, `ordinal`, `text`, `token_count` and `embedding` to the new version's.
  - Every other current chunk: supersede it (`superseded_at = now()`) if it has a claim with `mapped_version IS NOT NULL`, else delete it. Never delete superseded chunks here.
  - Insert the unmatched new chunks.
  - Avoid transient ordinal collisions under the partial unique index. For example, first supersede or delete the unmatched chunks, then move the reused chunks to temporary negative ordinals (`-1 - new_ordinal`), then set the final ordinals, then insert.
  - Return the number reused, and update the docstring to the per-chunk rule.
  - Sync adds the return value to `SyncReport.chunks_reused`.
- [ ] **Step 5: Run the whole suite** (`uv run --extra dev pytest -q`, ruff, mypy). Expected: PASS.
- [ ] **Step 6: Commit** `store: unchanged text keeps its chunk and facts on re-sync; dropped chunks kept only with mapped claims` — then PR A (Tasks 1, 2, 2b), review, merge; pg_dump production before deploying.

---

## Part B: the ransomware protection topic

### Task 3: Topics in the taxonomy (format, validation, tables, the ransomware topic)

**Files:**
- Modify: `src/vendorkb/taxonomy/load.py` (dataclasses, `validate`, `_TABLES`, `load_taxonomy`, module docstring)
- Modify: `src/vendorkb/schema.sql` (two tables)
- Modify: `taxonomy/taxonomy.yaml` (add the `topics:` section below)
- Test: `tests/integration/test_taxonomy_load_v2.py`

**Interfaces:**
- Produces: `load.Aspect(id: str, name: str, query: str, cues: tuple[str, ...], dimensions: tuple[str, ...])`, `load.Topic(id: str, name: str, triggers: tuple[str, ...], aspects: tuple[Aspect, ...])` (frozen dataclasses); `Taxonomy.topics: list[Topic]` (defaulted last field); tables `topics(topic_id text PRIMARY KEY, name text NOT NULL, triggers text[] NOT NULL)` and `topic_aspects(topic_id text NOT NULL REFERENCES topics ON DELETE CASCADE, aspect_id text NOT NULL, ordinal int NOT NULL, name text NOT NULL, query text NOT NULL, cues text[] NOT NULL, dimensions text[] NOT NULL, PRIMARY KEY (topic_id, aspect_id))`.

- [ ] **Step 1: Write the failing tests:** `test_topics_load_into_their_tables` (the fixture plus one topic over fixture dimensions loads; rows in both tables, aspects in YAML order, triggers and cues normalized lowercase) and validation cases in the existing `test_validation_rejects` parametrize: unknown dimension in an aspect, duplicate aspect id, empty `triggers`, empty `aspects`, unknown key in a topic, a topic id that is not `^[a-z][a-z0-9_]*$`.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement** the optional top-level key `topics` (each topic: `id`, `name`, `triggers`, `aspects`; each aspect: `id`, `name`, `query`, `dimensions`, optional `cues`), validated with the existing `_Checker`; tables loaded children first (`topic_aspects`, `topics` at the front of `_TABLES`).
- [ ] **Step 4: Add the topic to `taxonomy/taxonomy.yaml`** (after `offerings:`). Dimensions are the spec's table; phrasings are reviewed data:

```yaml
topics:
- id: ransomware_protection
  name: Ransomware protection
  triggers: [ransomware, cyber attack, cyberattack, cyber resilience, cyber recovery, cyber threat]
  aspects:
  - id: built_in_security
    name: Built-in security (RBAC, MFA, SSO)
    query: role-based access control, multi-factor authentication and single sign-on for the backup console, audit logging, SIEM alerts
    cues: [rbac, role-based access, mfa, multi-factor, two-factor, 2fa, single sign-on, sso, saml]
    dimensions: [security.rbac, security.mfa, security.sso, security.audit_logging, cyber.siem_integration]
  - id: immutability
    name: Immutability / WORM
    query: immutable backups, WORM storage, object lock, hardened repository, backups that cannot be changed or deleted before retention ends
    cues: [immutab, worm, object lock, hardened repository, write once]
    dimensions: [security.immutability, security.immutability_level, security.immutability_mode, security.immutability_scope, security.immutability_period]
  - id: anomaly_detection
    name: Anomaly detection
    query: anomaly detection of suspicious activity such as abnormal changes, deletions or renames and spikes in backup size or change rate
    cues: [anomal, unusual activity, suspicious activity, abnormal change]
    dimensions: [cyber.detection_behavior]
  - id: malware_detection
    name: Malware detection
    query: malware scanning of backups with antivirus signatures, entropy or machine-learning analysis, YARA rules, indicators of compromise and hash lists
    cues: [malware scan, antivirus, anti-virus, yara, indicator of compromise, ioc, entropy, signature scan]
    dimensions: [cyber.detection_signatures, cyber.detection_entropy, cyber.detection_rules, cyber.indicators_of_compromise, cyber.third_party_scanners]
  - id: clean_recovery
    name: Clean recovery
    query: clean recovery after an attack, scanning backups during restore, quarantining infected restore points, finding the last clean restore point, clean room recovery
    cues: [scan on restore, scan during restore, quarantin, clean restore point, last clean, clean room, cleanroom]
    dimensions: [cyber.scan_on_restore, cyber.quarantine, cyber.clean_point_identification, cyber.clean_room]
  - id: air_gap
    name: Air gap / isolated copies
    query: air-gapped or logically isolated backup copies and isolated vaults
    cues: [air gap, air-gap, airgap, isolated copy, isolated vault, cyber vault]
    dimensions: [security.air_gap]
  - id: deletion_safeguards
    name: Deletion safeguards
    query: protection against deleting backups, soft delete of backups, a second approver (multi-person authorization) for destructive operations
    cues: [soft delete, soft-delete, multi-person, four-eyes, four eyes, second approver, quorum]
    dimensions: [security.soft_delete_of_backups, security.multi_person_authorization]
  - id: encryption
    name: Backup encryption
    query: encryption of backup data at rest and in transit, client-side encryption, customer-managed keys
    cues: [backup encryption, encryption at rest, encryption in transit, encrypted backups, customer-managed key, kms]
    dimensions: [security.encryption_at_rest, security.encryption_in_transit, security.client_side_encryption, security.customer_managed_keys]
```

- [ ] **Step 5: Run the tests**, then validate the real file: `uv run python -c "from vendorkb.taxonomy.load import read_doc, validate; print(len(validate(read_doc('taxonomy/taxonomy.yaml')).topics))"` → `1`.
- [ ] **Step 6: Commit** `taxonomy: topics, and the ransomware protection topic (eight aspects)`

### Task 4: Topic detection

**Files:**
- Create: `src/vendorkb/topics.py`
- Test: `tests/unit/test_topics.py`, `tests/integration/test_topics_db.py` (new)

**Interfaces:**
- Consumes: `load.Topic`, `load.Aspect`, tables from Task 3.
- Produces: `async def load_topics(conn) -> list[Topic]`; `def detect_topic(question: str, topics: list[Topic]) -> Topic | None`.

- [ ] **Step 1: Write the failing tests** (unit, with a `Topic` built from the YAML block of Task 3):

```python
def test_a_broad_question_triggers_the_topic():
    assert detect_topic("How does Veeam protect backups against ransomware?", T).id == "ransomware_protection"
    assert detect_topic("How do Druva and Cohesity compare for cyber-resilience?", T) is not None  # hyphen = space
def test_naming_an_aspect_keeps_todays_behaviour():
    assert detect_topic("How does immutability protect against ransomware?", T) is None
    assert detect_topic("Does Veeam support YARA rules for ransomware?", T) is None
    assert detect_topic("What is the retention period?", T) is None  # no trigger
def test_incidental_words_do_not_count_as_naming_an_aspect():
    assert detect_topic("How do I recover after ransomware encrypted my files?", T) is not None
```

  and `test_load_topics_reads_the_loaded_taxonomy` (integration: load the fixture with a topic, `load_topics` returns it with aspects in order).
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement:** lowercase the question, hyphens to spaces, whitespace collapsed; a trigger matches as a whole phrase (`(?<!\w)…(?!\w)`), a cue matches at a word start (`(?<!\w)cue`); the topic fires when a trigger matches and no cue of any of its aspects does.
- [ ] **Step 4: Run them to verify they pass.**
- [ ] **Step 5: Commit** `topics: load the taxonomy's topics and detect one in a question`

### Task 5: Topic retrieval for single, comparison and vendor-neutral questions

**Files:**
- Modify: `src/vendorkb/topics.py`
- Test: `tests/integration/test_topics_db.py`

**Interfaces:**
- Consumes: `search.hybrid_candidates`, `search.search_docs`, `modes.compare_facets`, `modes._distinct`, `Reranker.rerank`, capabilities rows.
- Produces: `TOPIC_RERANK_DOCS = 240`; `@dataclass class TopicHits: hits: list[Hit]; aspect_of: dict[str, str]  # chunk_id -> aspect id; rerank_ok: bool; reranked_docs: int`; `async def fact_chunks(conn, aspect: Aspect, facet: Facet, version: str, limit: int) -> list[Hit]`; `async def topic_hits(conn, question: str, topic: Topic, *, facets: list[Facet], embedder, reranker, budget: int = TOPIC_RERANK_DOCS, per_aspect: int = 2, own: int = 3) -> TopicHits`.

- [ ] **Step 1: Write the failing tests** (corpus: two vendors, chunks whose texts each match one aspect query; FakeEmbedder/FakeReranker as in `test_vendorkb_search.py`; capabilities rows from `test_capabilities_v2._claim` on aspect dimensions):
  - `test_every_aspect_gets_its_own_passages`: one facet; the question's own top passages are all one aspect, yet each aspect with matching chunks contributes up to `per_aspect` hits, and `aspect_of` labels them.
  - `test_fact_chunks_join_the_candidates`: a chunk behind a supported capability on an aspect dimension, whose text matches no query, is among that aspect's hits; a superseded chunk never is.
  - `test_the_rerank_budget_holds`: with two facets and `budget=40`, `reranked_docs <= 40`.
  - `test_topic_hits_without_a_reranker`: reranker fails (`FakeReranker(fail=True)`) → hits in RRF order, `rerank_ok is False`.
  - `test_no_facet_searches_the_whole_corpus`: `facets=[]` returns hits from both vendors and no fact chunks.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement `fact_chunks`**: current (not superseded) chunks behind `capabilities` rows with `dimension_id = ANY(aspect.dimensions)`, `support IN ('supported','partial')`, `taxonomy_version = version`, matching the facet (`vendor` or `product`), ordered `support = 'supported' DESC, is_current DESC, doc_version DESC`, distinct chunks, `limit`.
- [ ] **Step 4: Implement `topic_hits`**: sides = `facets` or `[None]`; allowance per (side, aspect) = `max(4, budget // (len(sides) * len(topic.aspects)))`; candidates = `fact_chunks(..., limit=allowance // 2)` (sides only) then `hybrid_candidates` with the aspect query (prefixed by the side's name) scoped to the side, up to the allowance, deduplicated; rerank each candidate list with that query (concurrently, `top_k=len`), keep `_distinct(..., per_aspect)`; embed all aspect queries in one call. The question's own passages: `compare_facets` (first `own` per side) when there are two or more facets, else `search_docs(question, k=own)` scoped to the side. Order: own passages, then aspect passages in aspect order; duplicates by text removed.
- [ ] **Step 5: Run them to verify they pass.**
- [ ] **Step 6: Commit** `topics: retrieve evidence per aspect within a rerank budget`

### Task 6: Topic answers (single, comparison, vendor-neutral)

**Files:**
- Modify: `src/vendorkb/answer.py`
- Test: `tests/unit/test_vendorkb_answer.py`

**Interfaces:**
- Consumes: `topics.load_topics`, `topics.detect_topic`, `topics.topic_hits` (Tasks 4-5).
- Produces: `def topic_rules(topic: Topic, *, products_named: bool) -> str`; `answer()` envelope gains `topic: str | None` and, when set, `aspects: list[str]`.

- [ ] **Step 1: Write the failing tests** (patch `ans.load_topics`, `ans.topic_hits`):
  - `test_a_topic_question_is_answered_per_aspect`: single scoped question → `topic_hits` called with the question's facets, the system message contains `topic_rules(T, products_named=True)`, `max_tokens == ENUMERATION_MAX_TOKENS`, `out["topic"] == "ransomware_protection"`, citations resolve against the topic hits.
  - `test_a_comparison_topic_question_uses_both_sides`: `topic_hits` receives both facets.
  - `test_a_vendor_neutral_topic_question`: "How can I protect my backups against ransomware?" → `topic_hits` called with `facets=[]` and the system message contains `topic_rules(T, products_named=False)`.
  - `test_an_aspect_question_is_not_expanded`: "Does Veeam support YARA rules for ransomware?" → `topic_hits` not called, `out["topic"] is None`.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement** `topic_rules` with exactly this copy:

```python
head = (f"\n- The question asks about {topic.name.lower()} in general. Organise the answer by "
        f"these aspects, in this order: {'; '.join(a.name for a in topic.aspects)}.\n")
# products_named=True
head + ("- For each product in the question, state what its sources show for each aspect, with "
        "citations, and name the product each statement is about (a vendor may have several); "
        "for an aspect its sources do not cover, write \"the retrieved sources do not cover "
        "this\", never that the product lacks it.")
# products_named=False
head + ("- The question names no product: for each aspect, describe what the sources show, with "
        "citations, naming the vendor and product each statement comes from; do not present "
        "the vendors named as a complete list.")
```

  and in `answer()`: for single and comparison shapes, `topic = detect_topic(question, await load_topics(conn))`; when set, retrieve with `topic_hits` (facets: the comparison facets; a scoped single question's vendors/products as facets; none when unscoped), system `SYSTEM_RULES + topic_rules(topic, products_named=bool(facets))`, `max_tokens=ENUMERATION_MAX_TOKENS`; everything after retrieval as today.
- [ ] **Step 4: Run the tests and the suite.**
- [ ] **Step 5: Commit** `answer: broad topic questions are answered per aspect`

### Task 7: The grid for "which vendors" topic questions

**Files:**
- Modify: `src/vendorkb/topics.py`, `src/vendorkb/answer.py`
- Modify: `src/vendorkb/search.py` (`hybrid_candidates` gains `only_unextracted: bool = False`: only current chunks with no final extraction of the v3 prompt family)
- Test: `tests/integration/test_topics_db.py`, `tests/integration/test_vendorkb_search.py`, `tests/unit/test_vendorkb_answer.py`

**Interfaces:**
- Consumes: `ENUMERATION_MIN_SCORE` (modes), capabilities rows.
- Produces: `@dataclass class Cell: mark: Literal["supported", "partial", "recent"]; hit: Hit; superseded: bool`; `@dataclass class Grid: vendors: list[str]; cells: dict[tuple[str, str], Cell]; facts_as_of: datetime | None; no_fact: list[str]`; `async def topic_grid(conn, topic, *, vendors: list[str], embedder, reranker, catch_up: bool = True) -> Grid`; `def render_grid(grid: Grid, topic: Topic) -> tuple[str, list[Hit]]`; `answer.citation(n: int, hit: Hit) -> dict` (extracted from `resolve_markers`, same keys).

- [ ] **Step 1: Write the failing tests:**
  - `test_grid_cells_take_the_best_fact`: per (vendor, aspect), supported beats partial, current beats not current, newest page first; a fact on a superseded chunk gives `superseded=True`; `facts_as_of` = the newest `doc_version` among the topic's rows; a vendor with no fact lands in `no_fact`.
  - `test_only_unextracted_candidates` (search): with `only_unextracted=True`, a chunk with a 'done' extraction is never a candidate and one without is.
  - `test_catch_up_fills_only_empty_cells`: an unextracted chunk scoring ≥ `ENUMERATION_MIN_SCORE` for an aspect fills that vendor's empty cell with `mark="recent"`; it never replaces a fact; one scoring below does nothing.
  - `test_render_grid`: header row `Vendor | <aspect names>`; marks `✓ [n]`, `◐ [n]`, `○ [n]`, a superseded mark ends with `†`; blank cells empty; these footer lines verbatim (date formatted `YYYY-MM-DD`): `✓ supported · ◐ partial · ○ found in a page not yet analysed (not verified) · † from an earlier version of the page.`, `Facts current to {date}. A blank cell means no fact was found, not that the product lacks the capability.`, `No fact found for: {vendors}.`
  - `test_which_vendors_topic_question_gets_the_grid` (answer unit, `topic_grid` patched): enumeration shape + topic → the answer is the model summary, a blank line, then the rendered grid; citations built with `citation()`; `out["topic"]` set; `enumerate_vendors` not called.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement** the facts query (one SQL, `DISTINCT ON (vendor, aspect)` with the aspect computed from the topic's dimension lists, joined to raw_claims/chunks/documents for the hit), the catch-up (per aspect, `hybrid_candidates(aspect.query, only_unextracted=True, limit=20)` reranked with `aspect.query`), `render_grid` (rows ordered by number of marks, then name), and the summary call: system `"Summarise this coverage table of backup vendors in at most four sentences, using only the counts and names given; no citations, no claims beyond the table."`, user = one line per aspect `"<aspect name>: <n> vendors: <names>"`, `max_tokens=2000`; a failed summary call leaves the grid alone.
- [ ] **Step 4: Run the tests and the suite.**
- [ ] **Step 5: Commit** `answer: "which vendors" topic questions get a cited vendor-by-aspect grid`

### Task 8: MCP: query by topic, topics in list_dimensions

**Files:**
- Modify: `src/vendorkb/taxonomy/capabilities.py` (`CapFilter`), `src/vendorkb/mcp_backend.py`, `src/vendorkb/mcp_server.py`
- Test: `tests/unit/test_mcp_tools.py`, `tests/integration/test_mcp_backend.py`

**Interfaces:**
- Consumes: `topics.load_topics`.
- Produces: `CapFilter.dimension_ids: tuple[str, ...] | None = None` (exact ids, used instead of resolving `dimension`); `Backend.topics(query: str | None) -> list[Topic]`; `query_capabilities(..., topic: str | None = None)`; `list_dimensions` output starts with matching topics.

- [ ] **Step 1: Write the failing tests:**
  - `test_query_by_topic_groups_facts_by_aspect` (unit): `topic="ransomware protection"` → one backend call per aspect with that aspect's `dimension_ids` and the other filters, `limit` split as `max(1, limit // len(aspects))`; text has one heading per aspect (`## <aspect name>`); structured `{"topic": id, "aspects": [{"id", "name", ...}]}`.
  - `test_topic_and_dimension_are_exclusive` (unit): both given → ToolError.
  - `test_unknown_topic_lists_the_known_ones` (unit).
  - `test_list_dimensions_shows_matching_topics` (unit): query "ransomware" → the text starts with `Topic ransomware_protection — Ransomware protection:` listing its aspects.
  - `test_capfilter_dimension_ids` (integration): facts on two dimensions returned together.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement**: topic resolution by id, name or trigger (case-insensitive, exact first); per-aspect queries; the topic block in `list_dimensions`; tool descriptions mention `topic`.
- [ ] **Step 4: Run the tests and the suite.**
- [ ] **Step 5: Commit** `mcp: query_capabilities by topic; list_dimensions shows topics`

### Task 9: Topic eval set and scorer; before/after measurement

**Files:**
- Create: `eval/topic_questions.yaml`, `eval/topic_eval.py`, `eval/TOPIC_GRADING_BRIEF.md`
- Test: `tests/unit/test_topic_eval.py`

**Interfaces:**
- Consumes: answers JSONL from `eval/answer_eval.py run --questions eval/topic_questions.yaml` (its row format).
- Produces: `topic_eval.py build ANSWERS.jsonl OUT.json` (one blind grading item per answer: question, aspect names, answer text, numbered source texts; runs shuffled together, labels in a separate key file) and `topic_eval.py score KEY.json GRADES.json...` (per run: aspect coverage = aspects covered with support over 8 per product, unsupported-claim rate = unsupported claims over claims).

- [ ] **Step 1: Write `eval/topic_questions.yaml`** (fields `id`, `type`, `question`, `vendors`, `topic: ransomware_protection`, `holdout: true`):
  1. `tp-01` single_vendor: "How does Veeam protect backups against ransomware?"
  2. `tp-02` single_vendor: "How does Commvault protect backups against ransomware?"
  3. `tp-03` single_vendor: "What ransomware protection does Rubrik offer?"
  4. `tp-04` single_vendor: "How does Cohesity help me recover from a ransomware attack?"
  5. `tp-05` cross_vendor: "How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware?"
  6. `tp-06` cross_vendor: "Compare Commvault and Rubrik on ransomware protection."
  7. `tp-07` cross_vendor: "How do Druva and Cohesity compare for cyber resilience?"
  8. `tp-08` cross_vendor: "How can I protect my backups against ransomware?"
  9. `tp-09` cross_vendor: "Which vendors protect backups against ransomware?"
  10. `tp-10` cross_vendor: "Which backup vendors offer ransomware protection?"
- [ ] **Step 2: Write the failing scorer tests** (`test_topic_eval.py`): coverage and rate from a small grades fixture; "not covered" counts against coverage, a grader's `corpus_lacks` flag removes the aspect from the denominator; build output contains no run labels.
- [ ] **Step 3: Implement `topic_eval.py`** and the brief (per answer, product and aspect: `covered_supported` | `covered_unsupported` | `not_covered` | `corpus_lacks`, plus a list of unsupported claims; graders read only the item and the brief; for grids, grade a sample of 30 marks for support).
- [ ] **Step 4: Run the tests.** Commit `eval: ransomware topic questions and scorer`, open PR B, review, merge.
- [ ] **Step 5: Measure** (operational): before merging PR B, run the "before" answers with main's code: `uv run python eval/answer_eval.py run --questions <branch>/eval/topic_questions.yaml --out /tmp/.../topic-before.jsonl`; after the pg_dump, deploy and `vendorkb taxonomy load` + `vendorkb capabilities build`, run "after" the same way; build the blind set from both, grade with an Opus and a Fable subagent, adjudicate disagreements, score. Pass (spec): coverage rises clearly; unsupported-claim rate rises by at most 2 points. Rerun the agent eval gate (`eval/agent_eval.py`) and compare to `eval/agent_baseline.json`. Report in `docs/superpowers/full-corpus-2026-11-08.md`; BACKLOG entry.
