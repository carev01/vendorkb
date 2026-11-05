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
  (`sync_state.bootstrap_tokens`); a page that aborts after embedding 3 times is refused until
  `abort_position`/`abort_count` are deleted; the bootstrap's end prunes documents it never served
  (`sync_seen`), refusing more than max(500, 2% of documents) unless `--allow-prune N` covers
  it. The Helm sync CronJob runs with `--max-tokens 5000000`;
- hybrid retrieval (`search.py`): Postgres full-text search (OR terms) plus pgvector HNSW, merged by
  reciprocal rank fusion (k=60), then a Voyage rerank of the top 40;
- question scope detection (`scope.py`, aliases in `scope_aliases.json`);
- a cite-or-abstain answer step (`answer.py`);
- the CLI: `search | answer | stats | sync | reembed | promote-embedding`.

**Not built yet:**
- the MCP server;
- claim extraction (M3);
- the taxonomy and capability table (M4);
- the production deployment;

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
