# M3/M4 Claims and Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract evidence-gated capability claims from a stratified pilot with Claude Haiku 5.5,
grow taxonomy v1 from them, and build and measure a `capabilities` table with
`query_capabilities`.

**Architecture:**
- A new package `src/vendorkb/claims/` handles extraction: client, prompt, gate, runner. A new
  package `src/vendorkb/taxonomy/` handles cluster, propose, load, map and build.
- Both store into new Postgres tables created by `init_schema`.
- Paid calls go through the Anthropic SDK (Haiku 5.5, structured output, the Batches API).
  Embeddings use the existing voyage-4 `Embedder`.
- CLI subcommands: `extract`, `taxonomy propose|load`, `capabilities build|query`.

**Tech Stack:** Python 3.12, psycopg 3, pgvector, anthropic (new), rapidfuzz (new), scikit-learn
(new), pytest + testcontainers.

**Spec:** `docs/superpowers/specs/2026-11-06-m3-m4-claims-taxonomy-design.md`. It is binding;
DESIGN.md §4–§6 are its background.

## Global Constraints

- **Every claim needs a verbatim `evidence_quote`.** The gate is: normalized substring, else
  `rapidfuzz.fuzz.partial_ratio` ≥ 90, else rejected into `rejected_claims` with a reason.
- **`vendor`, `product` and `version` come from the chunk's document**, never from the model.
- **Model `claude-haiku-5-5`** through the official `anthropic` SDK with `output_config.format`
  json_schema and `output_config.effort` "low". No OpenAI shim.
- **Budget:** `extract_budget_usd` = 25.
  - Cost is estimated at $0.10 per MTok in and $0.50 per MTok out, halved for batch.
  - Refuse to submit when spend plus the estimate exceeds the budget.
  - Record per-chunk token usage in `extractions`.
- **Idempotency:** `extractions` is unique on (chunk_id, extractor_model, prompt_version). A
  re-run skips chunks that are done.
- **Tests never call real APIs** (fake Anthropic client, fake embedder) and never read `.env`.
- **CI gate:** `uv run ruff check src tests`, `uv run mypy src`, `uv run pytest`.
- **The taxonomy YAML is never merged to main without the user.** Code can merge.

## Review Focus

1. **A model reply with an evidence quote that differs only in whitespace, case or curly quotes.**
   It is accepted. Test in Task 1.
2. **A batch result that errors or expires for some chunks.** Those chunks get
   `extractions.status` = errored, are retried on re-run, and are never counted as done with 0
   claims. Test in Task 1.
3. **A chunk whose document has a NULL product.** The claim stores a NULL product, with no crash.
   Test in Task 1.
4. **A target query on a parent id (`aws`) includes child targets (`aws.rds`).** Test in Task 3.
5. **Re-running `capabilities build` after a taxonomy change replaces the rows**, never
   duplicates them. Test in Task 3.

---

### Task 1: Extraction pipeline (schema, gate, client, runner, CLI)

**Files:**
- Create: `src/vendorkb/claims/__init__.py`, `claims/gate.py`, `claims/extract.py`
- Modify: `src/vendorkb/db.py` (schema), `src/vendorkb/config.py`, `src/vendorkb/cli.py`,
  `pyproject.toml` (deps: anthropic, rapidfuzz)
- Test: `tests/unit/test_claims_gate.py`, `tests/integration/test_claims_extract.py`

**Interfaces:**
- `gate.normalize(s: str) -> str`.
- `gate.check(quote: str, chunk_text: str) -> tuple[bool, str]`. The second value is a reason:
  "exact", "fuzzy:<score>", "empty", or "no-match:<score>".
- `extract.CLAIM_SCHEMA: dict`. JSON schema `{"claims": [{capability: str, target: str|null,
  support: enum[supported, partial, not_supported, unclear], conditions: str|null,
  evidence_quote: str}]}`, with `additionalProperties: false` and every field required.
- `extract.PROMPT_VERSION = "claims-v1"`, plus `extract.SYSTEM_PROMPT` as in spec §4.
- `extract.build_request(chunk: ChunkRow, settings) -> dict`. These are Messages API params:
  model, max_tokens 2000, system, messages, and output_config {format, effort: "low"}.
- `ChunkRow` is a dataclass: chunk_id, doc_id, vendor, product, version, title, section_path,
  text.
- `async extract.run(conn, chunk_ids: list[str], *, client, settings, mode: "sync"|"batch",
  poll_seconds=30) -> ExtractReport`.
  - It skips done chunks and checks the budget before submitting.
  - In sync mode it uses `client.messages.create`; in batch mode it uses
    `client.messages.batches.create/retrieve/results`.
  - It writes `extractions`, `raw_claims` and `rejected_claims` in one transaction per chunk's
    result.
- `ExtractReport` fields: chunks, done, skipped, errored, claims_accepted, claims_rejected,
  input_tokens, output_tokens, est_cost_usd, stopped_by_budget.
- CLI: `vendorkb extract --ids-file PATH [--mode sync|batch] [--limit N]`. It prints the report
  as JSON. The Anthropic client is built from `ANTHROPIC_API_KEY`.
- Settings: `extract_model = "claude-haiku-5-5"`, `extract_budget_usd = 25.0`.
- Schema: `extractions`, `raw_claims`, `rejected_claims` per spec §3, plus `features`, `targets`,
  `aliases` and `capabilities` per DESIGN.md §4. Create all of them here, so Task 3 only fills
  them.

- [ ] **Step 1: Write failing tests:**
  - gate: exact; case and whitespace; curly quotes and dashes; fuzzy at 92 accepted; 80
    rejected; empty rejected.
  - extract with a fake client returning canned JSON: accepted and rejected rows written; vendor
    and product from the document; a NULL product kept; idempotent re-run skips the chunk; the
    budget refusal; batch mode with one succeeded, one errored and one expired result, where the
    errored and expired chunks are retried on the next run; a refusal or invalid JSON is recorded
    as errored.
- [ ] **Step 2: Run them and expect a FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gate.**
- [ ] **Step 5: Commit** `feat(claims): Haiku 5.5 claim extraction with evidence gate`.

### Task 2: Pilot sample, smoke, full extraction, spot check (controller-run)

- [ ] **Step 1:** Write `eval/m3_sample.py` to build the id list per spec §4, with seed 7, and
  write `eval/m3_pilot_ids.txt`, committed. Part (c) uses `search.vendor_candidates` top 5 per
  vendor per enumeration question (free; one embedding per question).
- [ ] **Step 2:** Smoke test: `vendorkb extract --limit 2 --mode sync` against production. Check
  for nonzero tokens and rows written.
- [ ] **Step 3:** Run the full batch, monitored. Report the gate pass rate and cost.
- [ ] **Step 4:** Spot check 100 random accepted claims with 2 Opus subagents (50 each). Record
  precision.

### Task 3: Taxonomy and capabilities pipeline

**Files:**
- Create: `src/vendorkb/taxonomy/__init__.py`, `taxonomy/cluster.py`, `taxonomy/propose.py`,
  `taxonomy/load.py`, `taxonomy/mapper.py`, `taxonomy/capabilities.py`
- Modify: `src/vendorkb/cli.py`, `pyproject.toml` (scikit-learn)
- Test: `tests/unit/test_taxonomy_cluster.py`, `tests/integration/test_taxonomy_pipeline.py`

**Interfaces:**
- `cluster.cluster_strings(strings: list[str], vectors: list[list[float]], *,
  distance_threshold: float) -> list[list[int]]`. Agglomerative, cosine, average linkage.
- `async propose.propose(conn, *, client, embedder, settings, kind: "feature"|"target",
  distance_threshold, out_path) -> ProposeReport`.
  - It reads distinct normalized strings from `raw_claims` with counts and vendor counts.
  - It embeds them, clusters them, and calls Haiku per cluster, with structured output and
    batching allowed.
  - It writes or merges `taxonomy.yaml` under `features:` / `targets:`.
  - Each entry has: id, name, category (seed list), definition, aliases, support_count and
    vendor_count.
  - Targets get dotted ids and a parent.
- `load.load_taxonomy(conn, path) -> dict`. It upserts features, targets and aliases. The YAML
  version becomes `taxonomy_version`.
- `async mapper.map_claims(conn, *, embedder, threshold: float) -> MapReport`. It maps each
  raw_claim's capability to a feature and its target to a target: exact alias, then nearest
  neighbour at ≥ threshold, else unmapped. It stores mapping_conf.
- `capabilities.build(conn) -> int`. It rebuilds `capabilities` for the current taxonomy version
  (delete then insert in one transaction).
- `capabilities.query(conn, feature: str, target: str | None = None, vendor: str | None = None,
  support: str | None = None) -> list[CapRow]`. A target matches itself and its descendants
  (`target_id = t OR target_id LIKE t || '.%'`). Each CapRow carries vendor, product, support,
  conditions, evidence_quote, chunk_id and url.
- CLI:
  - `vendorkb taxonomy propose --kind feature|target --threshold X --out taxonomy/taxonomy.yaml`
  - `vendorkb taxonomy load PATH`
  - `vendorkb capabilities build`
  - `vendorkb capabilities query FEATURE [--target T] [--vendor V] [--support S]`

- [ ] **Step 1: Write failing tests:** clustering groups near-duplicates; load is idempotent;
  mapper exact, nearest-neighbour and unmapped; build replaces rows on a re-run; query includes
  descendant targets; propose with a fake client writes valid YAML.
- [ ] **Step 2: Run them and expect a FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gate.**
- [ ] **Step 5: Commit** `feat(taxonomy): cluster, propose, load, map, capabilities`.

### Task 4: Taxonomy v1, capability table and measurement (controller-run)

- [ ] **Step 1:** Run `taxonomy propose` for features and targets against production, tuning the
  threshold by reading samples. Commit `taxonomy/taxonomy.yaml` on a branch `taxonomy-v1` and
  open a draft PR.
- [ ] **Step 2:** Load, map (threshold calibrated on 50 hand-checked mappings), and build.
- [ ] **Step 3:** Write `eval/m4_measure.py`: question → feature/target resolution, then vendor
  recall and precision of `query_capabilities`, the union with retrieval, and the split by sample
  part. Run it and write `docs/superpowers/m3-m4-pilot-2026-11-06.md`.
