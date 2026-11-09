# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A backup-vendor knowledge system. It answers questions about backup vendors' technical documentation
**precisely and with citations: cite or abstain**. `DESIGN.md` is the architecture and milestone plan
(M0–M5), so read it before non-trivial work.

vendorkb replaced the Graphiti graph in the sibling `graph-rag` repo, which is parked indefinitely. On
2026-11-04 the two systems were compared head-to-head on 100 deciding questions and vendorkb won 28–0. It
had 50 perfect answers against the graph's 22, and was about 8x faster
(`docs/superpowers/head-to-head-report-2026-11-04.md`). The code was moved here, with its history, from
graph-rag's `src/vendorkb/`. The head-to-head harness stays in graph-rag because it depends on the graph
stack.

**Built (M1/M2 retrieval core):**
- section chunking (`chunker.py`);
- the shared document store (`store.py`) and the sync tables in `schema.sql`;
- `vendorkb sync` (`sync.py`): bootstrap then incremental pulls of the DocExtractor delta feed, in
  pages (one embed call + one transaction each), single-flight via an advisory lock. Budget guards:
  in bootstrap mode `--max-tokens` caps the cumulative total across attempts
  (`sync_state.bootstrap_tokens`); a page that aborts after embedding 3 times runs once in
  quarantine mode (record by record; failures go to `sync_failures` with their
  `content_hash`, so a replay of the same content is skipped unembedded; the rest commit; if
  two or more records all fail, the page is refused until it changes or the breaker keys are
  cleared); the bootstrap's end prunes documents it never served (`sync_seen`), refusing more
  than max(500, 2% of documents) unless `--allow-prune N` covers it. The Helm sync CronJob
  runs with `--max-tokens 5000000`;
- hybrid retrieval (`search.py`): Postgres full-text search (OR terms) plus pgvector HNSW, merged by
  reciprocal rank fusion (k=60), then a Voyage rerank of the top 40;
- question scope detection (`scope.py`, aliases in `scope_aliases.json`; ordinary-word names
  such as "Own" or "Slide" match case-sensitively);
- question shape (`shape.py`): single / comparison / enumeration, deterministic. Two named
  facets never enumerate; a one-vendor "which tools…" question enumerates only for a workload
  vendor (`WORKLOAD_VENDORS`);
- cross-vendor retrieval (`modes.py`): `enumerate_vendors` (all vendors at once via the
  batched `search.vendor_candidates`, 2/3 chunks per vendor, one rerank, floor
  `ENUMERATION_MIN_SCORE` 0.46 calibrated 2026-11-06) and `compare_facets` (per-facet quotas
  that survive the rerank);
- a cite-or-abstain answer step (`answer.py`). Enumeration answers are one line per vendor; a
  line survives only if it cites a chunk of its own vendor, and the closing line "No
  supporting passage found for: …" is built in code, never by the model;
- cross-vendor eval (`eval/cross_vendor.py`, `eval/answer_eval.py`, labels in `eval/labels/`;
  `docs/superpowers/cross-vendor-retrieval-2026-11-06.md`);
- the CLI: `search | answer | stats | sync | reembed | promote-embedding`.

**M3/M4 pilot (2026-11-06; `docs/superpowers/m3-m4-pilot-2026-11-06.md`):**
- claim extraction (`claims/`, `vendorkb extract`):
  - Claude Haiku 5.5 through the Anthropic SDK or any OpenAI-compatible endpoint, with a
    spend ledger, a run lock and a budget cap;
  - a verbatim evidence gate: word-bounded exact match, or fuzzy match that keeps every
    number and negation; `vendorkb claims regate` re-applies it to stored claims.
  - `vendorkb extract --unextracted [--vendor V]`: every chunk with no done or failed v3
    extraction by any model (new and re-synced chunks; the Opus teacher chunks are not redone).
  - `vendorkb mcp` (`mcp_app.py` HTTP + bearer gate, `mcp_server.py` tools, `mcp_backend.py`
    data access): the agent interface of DESIGN §6 over streamable HTTP at `/mcp`
    (`MCP_BEARER_TOKEN` required; Helm `mcp.enabled`, docs/deploy/k3s.md §7b).
  - `vendorkb refresh` (`claims/refresh.py`, hourly CronJob `vendorkb-claims-refresh`): acts
    only when the sync brought chunks no extraction touched, once the feed is quiet; then wakes
    the scaled-to-zero extraction endpoint (`deploy/azure`), extracts, maps (MAP_BASE_URL /
    MAP_API_KEY, else the EXTRACT_ ones) and rebuilds capabilities.
  - a re-synced page (`store.write_document`) is written per chunk. A new chunk whose
    `chunker.reuse_key` equals a current chunk's (whitespace, link targets, backslashes and
    "Last Updated" lines ignored), with every claim quote still in the text, reuses that chunk:
    same `chunk_id`, extraction and claims, new text and embedding (`SyncReport.chunks_reused`;
    a chunk equal in text, section path and ordinal is not written at all).
    A chunk the new version drops is kept with `chunks.superseded_at` set when a claim on it is
    mapped, so its facts survive until the new version is extracted and mapped, else deleted
    (`SyncReport.chunks_superseded` counts the kept ones); `capabilities build` then retires the
    superseded ones (`store.retire_superseded`; the build's JSON `retired` counts them).
    Superseded chunks stay out of search, extraction and the refresh trigger (every chunk query
    that ranks or selects needs `superseded_at IS NULL`); their citations still resolve
    (`read_section` of a superseded chunk returns the chunks superseded in the same write).
- Production holds 74,885 v1 (Haiku) pilot claims.

**Taxonomy v2 and claims v3 (2026-11-06; spec `specs/2026-11-06-taxonomy-v2-claims-v3-design.md`,
results `docs/superpowers/taxonomy-v2-claims-v3-2026-11-06.md`):**
- **Purpose:** the system *finds* comparison information for an MCP agent or people; it never
  builds tables or rates vendors.
- **claims v3** (`PROMPT_VERSION` `claims-v3.1`; every prompt change bumps the minor version):
  - 14 fields, including kind (capability, licensing, service), target_scope, offering,
    edition, availability, default_state, and value/unit/bound. No `requires`; prices are never
    claims.
  - The system prompt is cached. Prices are per model (`extract.PRICES_PER_MTOK`), and an
    unpriced model is refused.
  - Opus 5.5 precision is 96/100 (gate 95).
  - Select the family with `prompt_version LIKE 'claims-v3%'`.
- **taxonomy v2** (`taxonomy/taxonomy.yaml`, version 2):
  - dimensions (`<family>.<id>`, 12 families, `applies_to` classes, `platform`, `value_unit`);
  - classes with object-type templates (dotted ids nest);
  - targets (class instances with `object_aliases` and `variants`);
  - offerings with editions;
  - topics (`topics:` key; tables `topics`, `topic_aspects`): a broad subject as a kit of aspects
    (each: `dimensions`, search `query`, `cues`); `topics.detect_topic` fires on a trigger when no
    aspect cue is named (product names are masked first). Used by `answer` (per-aspect
    retrieval, unless a workload vendor qualifies it; a broad "which vendors" question gets a
    cited grid) and MCP (`query_capabilities(topic=)` by aspect; `list_dimensions` shows the
    topic). The taxonomy version hash leaves the `topics:` section out: a topic edit needs a
    `taxonomy load` only, no re-map or build.

  New entries from `taxonomy propose` carry `proposed: true` until reviewed. v1 of the
  reviewed taxonomy, with its decision log, is in `taxonomy/review-2026-11-06.tsv`.
  `taxonomy load|propose|map|aspects`; `aspects(target|class)` lists what applies to a workload.
- **mapper:** exact alias first, otherwise a Haiku adjudication over the nearest candidates
  (`map_model`, `map_candidates`) with one fixed reply schema. Nearest neighbour alone never
  maps. On the reviewed taxonomy, dimension precision is 0.88 (alias 0.96, LLM 0.84)
  and target precision 0.98. `taxonomy map` maps only claims not yet stamped with the
  current taxonomy version (`mapped_version`), so earlier decisions are kept; `--all` re-maps
  every claim. Claims waiting on a skipped or failed adjudication stay unstamped and are
  retried on the next run. Decisions persist in `map_decisions` (a new candidate makes a new
  question): with nothing new offered the earlier answer is carried, and a re-asked target
  answered none keeps an earlier answer still offered. Aliases are candidate-search keys, so add
  curated phrasings as `lookup_aliases` (exact lookup only): those ask nothing again. A new
  entity re-asks the phrases near it, and an entity chosen before for a question stays among its
  candidates. Run `taxonomy map --max-usd 0` after a load to count the re-asks
  (`skipped_by_budget`) before a paid re-map. A decision is keyed on candidate ids, not their
  definitions: after sharpening a definition, pass `--reask ID` (on the dry run and the paid
  run) to ask again, and replace, the decisions that answered it. A new dimension draws
  phrases that had no home into it, so blind-check its LLM picks before trusting it; a
  dimension the model cannot file reliably can be `lookup_only: true` (never a candidate, not
  even pinned: only its curated phrasings map).
- **capabilities v2:** `capabilities build|query`, with scope labels (offering, product,
  product_wide), conflict groups and `is_current` (by `documents.fetched_at`, except that a row
  on a superseded chunk ranks below every current-version row of its key, whatever the date).
  `capabilities.evidence_superseded` marks facts whose claim is on a superseded chunk (stored
  at build, and `query` also reads the chunk's own state, so a sync marks its facts before the
  next build; `is_current` is as built); the MCP fact text says "from an earlier version of the
  page", as does a `read_section` of a superseded chunk. Product-wide facts never appear as
  target facts.
- **Not on production yet:** the branch's `init_schema` drops the v1
  features/targets/aliases/capabilities tables, so never run it against production before
  merge and deploy. Extraction scripts that write to production must skip `init_schema`.
- **Evals:** `eval/findability.py coverage|claims|chunks-candidates|chunks-score`, run against a
  local, never-committed map of the internal comparison tables (418/423 rows findable on
  the reviewed taxonomy).
  `eval/teacher_sample.py` draws the stratified teacher set.
  `eval/agent_eval.py prepare|call|score` is the agent-level regression gate: one fresh Claude
  Code subagent (Sonnet) per brief answers a labelled cross-vendor question through the
  deployed MCP server, then `score RUN --baseline eval/agent_baseline.json` exits 1 on a
  regression. Run it after a taxonomy, mapper or MCP tool change.
  `eval/map_replay.py run|score|recall` replays the dimension adjudication on the labelled
  sample (`eval/labels/dimension-sample-2026-11-08.json`) under variants, storing nothing, for
  about $0.03 a variant: measure a mapper change there before paying for a re-map.

**Not built yet:**
- capabilities in `answer()`;

## Commands

Dependencies are managed with `uv`:

- **Install:** `uv sync --extra dev`
- **Lint (CI gate):** `uv run ruff check src tests`. The rule set is pinned in `pyproject.toml`.
- **Types:** `uv run mypy src`
- **Tests:** `uv run pytest`. The integration tests start pgvector through testcontainers, so they need
  Docker. `@live` tests are excluded by default.
- **A single test:** `uv run pytest tests/unit/test_vendorkb_chunker.py -q`
- **Local database:** `docker compose up -d`. It is pgvector pg16 on `127.0.0.1:55442`, and needs
  `VENDORKB_DB_PASSWORD`.
- **CLI:** `uv run vendorkb search "question" [--vendor X] [--k 20]`. `--rerank` is a paid Voyage call.

CI (`.github/workflows/ci.yml`) runs exactly: ruff, mypy, pytest.

## Invariants

- **Cite or abstain.** Every factual statement traces to a stored chunk. The LLM emits chunk markers
  only, never URLs. Citations resolve deterministically to document URLs.
- **One embedding space.** voyage-4 at 1,024 dimensions since 2026-11-04
  (`docs/superpowers/voyage-4-embedding-ab-2026-11-04.md`). Queries and chunks must use the same model
  and dimension; `embed_dim` is checked on every reply. Change models only with `vendorkb reembed`
  followed by `promote-embedding`. The 200M free Voyage tokens cover the full corpus about once, so
  never re-embed unchanged chunks.
- **Settings come from `VendorkbSettings` (`config.py`)**, loaded from `.env`. Tests never read `.env`
  (`tests/conftest.py`).
- **Writes are idempotent.** Documents are keyed on the article id and gated on `content_hash`
  (`store.py`); callers wrap a page of writes in one transaction.

## Rules

- **Secrets live only in the untracked `.env`.** Never print, copy or commit its contents. Run gitleaks
  before every push.
- **Prove the variable is engaged before any paid run.** Smoke-test on about 2 items, check that real
  calls happened, and get the user's go-ahead for paid runs.
- **Ask before pushing to a shared branch, merging, or changing production** (k3s).
