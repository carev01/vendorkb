# Production deploy and delta-feed sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** vendorkb keeps itself current from DocExtractor's delta feed (`vendorkb sync`) and ships as an image plus a Helm chart for a `vendorkb` namespace in k3s. Large tables are split into header-repeating row groups so no chunk exceeds the token budget.

**Architecture:** A new `sync.py` drives one advisory-locked run. It reads the NDJSON feed through a new `DocExtractor.delta()` iterator, in pages of 200 records in stream order. For each page it hash-gates, chunks, embeds the whole page in one call, and writes the page in one transaction. Only then does it move the bootstrap watermark. The cursor moves only on the feed's terminal line. A shared `store.py` owns document writes and replaces the corpus-file `ingest`, which is removed. The deployment is a pgvector StatefulSet, a suspended sync CronJob and an opt-in bootstrap Job, all built from one image.

**Tech Stack:** Python 3.12, httpx (streaming), psycopg 3 async, pgvector, pytest with testcontainers and `httpx.MockTransport`, Docker, GitHub Actions to GHCR, Helm 3, k3s.

**Spec:** `docs/superpowers/specs/2026-11-04-production-and-delta-sync-design.md` (approved 2026-11-05). The feed contract is graph-rag's `CLIENT-USAGE-GUIDE.md` §6 (`../graph-rag/CLIENT-USAGE-GUIDE.md`). The table split was approved in chat on 2026-11-05: split tables by row groups and repeat the header. No chonkie.

## Global Constraints

- **Embedding budget.** 200M free voyage-4 tokens cover the corpus about once. A sync must never re-embed an article whose `content_hash` is unchanged.
- **The cursor is opaque.** Store `next_since` byte-for-byte and never decode it. Advance `cursor` only after the terminal `{"control":"cursor"}` line.
- **The bootstrap watermark.** Keep the first attempt's `bootstrap_start.next_since` and never overwrite it on resume. Resume with `bootstrap_after=<last applied id>`. Pages are applied strictly in stream order, and the watermark moves only after the page's transaction commits.
- **Tests never read `.env`** (`tests/conftest.py`). Unit tests are hermetic. Integration tests use the `vkb_dsn` pgvector testcontainer fixture. `@live` is opt-in.
- **CI gate:** `uv run ruff check src tests`, `uv run mypy src`, `uv run pytest`. All three stay green after every task.
- **No secret in the repo, image, chart or values.** Credentials arrive only from the out-of-band Secret `vendorkb-secret`. Its keys are `POSTGRES_PASSWORD`, `DOCEXT_READ_KEY`, `EMBED_API_KEY`, `RERANK_API_KEY` and `JUDGE_API_KEY`.
- **Image tags** are `ghcr.io/carev01/vendorkb:sha-<7 hex>` and `:main`. The chart refuses any other tag format.
- **Nothing in this plan deploys.** Pushing an image, `helm install`, or creating a namespace or Secret are operational steps. Each needs the user's go-ahead (see "After the plan").

## Review Focus

1. **The feed drops mid-page during bootstrap.** The partial page is not written, the watermark stays at the previous page's last id, the run exits 1, and the next run resumes with `bootstrap_after`. This is pinned in Task 4, `test_bootstrap_drop_resumes_after_last_committed_page`.
2. **The Voyage call fails for a whole page** (outage, 429s past the retries). Nothing on the page is written, the watermark does not move, and the run exits 1. It must not record 200 "failed" articles and move past them. This is pinned in Task 4, `test_embed_failure_aborts_without_moving_watermark`. (Ruling, below.)
3. **An unchanged article's metadata changes** (renamed product or title). The metadata is updated in place, with no embed call. This is pinned in Task 4, `test_unchanged_hash_updates_metadata_without_embedding`.
4. **A record whose `source_id` is not in the cached source table.** The table is refreshed from the API once per run. If the id is still unknown, `source` is stored as NULL and the record is still written. This is pinned in Task 4, `test_unknown_source_refreshes_names_once`.
5. **A table row longer than the whole budget**, such as a huge JSON cell. It is hard-split so that no chunk exceeds `max_tokens`. This is pinned in Task 1, `test_single_table_row_over_budget_is_hard_split`.

## Rulings made while planning (check these)

- **Failures split into transient (abort) and deterministic (record and skip).** The spec said a failed embed would be recorded in `sync_failures` and skipped; the first plan aborted on every embed or database error. Ruling (revised in the Task 4 review): a page that fails deterministically would stall the sync forever and re-spend its tokens on every run, so failures are classed. Transient: embed 5xx/408/409/425/429 after the Embedder's retries, httpx transport errors, `psycopg.OperationalError`, and anything unexpected. These abort the page with nothing written and the watermark unmoved, because an outage or an expired key must not skip up to 200 articles. Deterministic, isolated per record and then committed: missing markdown or 0 chunks; an embed 4xx other than 408/409/425/429, after which each record is embedded alone and the ones rejected again go to `sync_failures` as `embed: <status> <message>` (a transient error during isolation still aborts); and `psycopg.DataError`, after which the page is redone with a savepoint per item, reusing the vectors already computed, with the offenders recorded as `db: <message>`. NUL characters are stripped from markdown before chunking so the common DataError does not occur. If this is wrong, a provider that returns a plain 400 for an outage would skip a page's articles until they next change. They stay visible in `sync_failures` and `vendorkb stats`.
- **`ingest` is removed in this plan, not after rollout.** The local database is already loaded, the restore moves it to production, and `sync` does everything after that. Removing `ingest` now lets `graph_content_hash` be dropped in the same task. If this is wrong, the frozen-list path is gone, but it is still in git history and in graph-rag.
- **The 137 locally stored articles with oversized table chunks are re-chunked operationally, not with new code.** After the restore into production, their `documents` rows are deleted, and the bootstrap re-adds them with the new chunker (about 0.5M tokens). If this is wrong, it costs nothing extra: it is a one-off SQL statement.
- **The spec's `@live` feed test is dropped.** The smoke bootstrap at `--max-tokens 50000` in production exercises the same path against the real feed, with a hard check that rows and tokens moved. If this is wrong, a feed-format surprise shows up at the smoke step rather than in CI.
- **The hash comes from the feed.** The gate compares the record's `content_hash`, which DocExtractor defines as the SHA-256 of the served markdown, the same value vendorkb stores. When the record has none, it falls back to `sha256(markdown)`.
- **The image pull method is left to the user and decided at deploy time.** `imagePullSecrets` is a values list that is empty by default, so the chart works for either answer.

---

### Task 1: Split oversized tables into header-repeating row groups

**Files:**
- Modify: `src/vendorkb/chunker.py` (`_split_block`, plus a new `_split_table` helper)
- Test: `tests/unit/test_vendorkb_chunker.py`. Replace `test_table_is_never_split` with the tests below.

**Interfaces:**
- Produces: `chunk_markdown(title, markdown, *, min_tokens=300, max_tokens=1200) -> list[Chunk]`. The signature is unchanged. The new guarantee is that every chunk has `approx_tokens(text) <= max_tokens`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_vendorkb_chunker.py`. Delete `test_table_is_never_split`.

```python
def _table(n_rows: int, cell: str = "x" * 90, header=("| a | b |", "|---|---|")) -> str:
    return "\n".join([*header, *[f"| row{i} | {cell} |" for i in range(n_rows)]])


def test_large_table_splits_into_row_groups_with_header_repeated():
    md = "## Data\n\n" + _table(400) + "\n"
    chunks = chunk_markdown("T", md)
    assert len(chunks) > 1
    for c in chunks:
        assert approx_tokens(c.text) <= 1200
        body = c.text.split("\n\n", 1)[1]
        assert body.startswith("| a | b |\n|---|---|\n| row")
    rows = [line for c in chunks for line in c.text.splitlines() if line.startswith("| row")]
    assert rows == [f"| row{i} | {'x' * 90} |" for i in range(400)]


def test_small_table_stays_whole_with_its_section():
    md = "## Data\n\n" + para(100) + "\n\n" + _table(5) + "\n"
    chunks = chunk_markdown("T", md)
    assert len(chunks) == 1
    assert chunks[0].text.count("| a | b |") == 1


def test_table_without_separator_line_splits_without_repeating_a_header():
    md = "## Data\n\n" + _table(400, header=()) + "\n"
    chunks = chunk_markdown("T", md)
    assert len(chunks) > 1
    assert all(approx_tokens(c.text) <= 1200 for c in chunks)
    assert sum(c.text.count("| row") for c in chunks) == 400


def test_single_table_row_over_budget_is_hard_split():
    md = "## Data\n\n" + _table(3, cell="y" * 20000) + "\n"
    chunks = chunk_markdown("T", md)
    assert all(approx_tokens(c.text) <= 1200 for c in chunks)
    assert sum(c.text.count("y") for c in chunks) == 3 * 20000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_vendorkb_chunker.py -q`
Expected: the large-table, no-separator and over-budget tests fail. The small-table test passes.

- [ ] **Step 3: Implement `_split_table(text: str, budget_chars: int) -> list[_Block]`** in `chunker.py`, and call it from `_split_block` for `block.is_table` when `len(block.text) > budget_chars`. Update the `_split_block` docstring.
  - The header is the first two lines when the second line matches `^\s*\|?\s*:?-{3,}`, which is a separator row. Otherwise there is no header.
  - Data rows are packed greedily into groups. Each group is `header + rows`, and its length including newlines stays within `budget_chars`.
  - A row that cannot fit even alone with the header is hard-split by characters into pieces of `budget_chars - len(header) - 1`. Each piece is emitted as its own block, with the header prefixed.
  - The pieces are table blocks (`is_table=True`), so `_split_section` packs them like any other block.

- [ ] **Step 4: Run the chunker tests and the full suite**

Run: `uv run pytest tests/unit/test_vendorkb_chunker.py -q && uv run ruff check src tests && uv run mypy src`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vendorkb/chunker.py tests/unit/test_vendorkb_chunker.py
git commit -m "feat(chunker): split oversized tables into header-repeating row groups"
```

---

### Task 2: Schema for sync, a shared document store, and removing the corpus-file ingest

**Files:**
- Modify: `src/vendorkb/schema.sql`
- Create: `src/vendorkb/store.py`
- Delete: `src/vendorkb/ingest.py` and `tests/integration/test_vendorkb_ingest.py`. In `src/vendorkb/clients.py`, delete `CorpusEntry` and `load_corpus`, plus the imports they leave unused.
- Modify: `src/vendorkb/cli.py`. Remove the `ingest` subcommand and its handler. Update the module docstring.
- Modify: `CLAUDE.md`. Update the "Built" and "Not built yet" lists. `ingest` is gone; `sync` arrives in Task 4.
- Test: `tests/integration/test_vendorkb_store.py` (new) and `tests/integration/test_vendorkb_db.py`

**Interfaces:**
- Produces, in `store.py`:
  - `@dataclass(frozen=True) class DocMeta: doc_id: uuid.UUID; vendor: str; product: str | None; source: str | None; source_id: uuid.UUID | None; title: str | None; url: str | None`
  - `async def stored_hashes(conn, doc_ids: list[uuid.UUID]) -> dict[uuid.UUID, str]`
  - `async def write_document(conn, meta: DocMeta, content_hash: str, chunks: list[Chunk], vectors: list[list[float]]) -> None`. It deletes and reinserts the document and its chunks. It does **not** open its own transaction, so the caller wraps a page.
  - `async def update_metadata(conn, meta: DocMeta) -> bool`. It updates vendor, product, source, source_id, title and url when any of them differ, and returns whether a row changed.
  - `async def delete_document(conn, doc_id: uuid.UUID) -> bool`
- Schema additions, which must be idempotent against the existing local and production database:
  - `ALTER TABLE documents DROP COLUMN IF EXISTS graph_content_hash;`
  - `ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_id uuid;`
  - `CREATE TABLE IF NOT EXISTS sync_state (key text PRIMARY KEY, value text NOT NULL, updated_at timestamptz NOT NULL DEFAULT now());`
  - `CREATE TABLE IF NOT EXISTS sources (source_id uuid PRIMARY KEY, name text NOT NULL, refreshed_at timestamptz NOT NULL DEFAULT now());`
  - `CREATE TABLE IF NOT EXISTS sync_failures (doc_id uuid PRIMARY KEY, reason text NOT NULL, failed_at timestamptz NOT NULL DEFAULT now());`

- [ ] **Step 1: Write the failing tests.**
  - `test_vendorkb_db.py`, `test_init_schema_upgrades_a_pre_sync_database`. Create the old `documents` table with `graph_content_hash`, run `init_schema` twice, and assert that `graph_content_hash` is gone, `source_id` exists, and `sync_state`, `sources` and `sync_failures` exist.
  - `test_vendorkb_store.py`:
    - `test_write_document_replaces_chunks`. Write two chunks, then write the same document again with one chunk. Assert one chunk row, and that `stored_hashes` returns the new hash.
    - `test_update_metadata_reports_change_only_when_different`. Assert `True`, then `False` with the same values.
    - `test_delete_document_cascades_chunks`. Assert 0 chunk rows, and `False` on a second delete.
  - Vectors are `[0.0]*767 + [1.0]` with `embed_dim=768`, the same convention as the other integration tests.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/integration/test_vendorkb_store.py tests/integration/test_vendorkb_db.py -q`
Expected: FAIL. `vendorkb.store` does not exist, and the schema assertions fail.

- [ ] **Step 3: Implement** the schema additions, `store.py` (move `_replace`'s SQL out of `ingest.py`), and the deletions listed under Files.

- [ ] **Step 4: Run the full gate**

Run: `uv run ruff check src tests && uv run mypy src && uv run pytest -q`
Expected: all pass. `grep -rn "graph_content_hash\|load_corpus\|CorpusEntry" src tests` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add -A src tests CLAUDE.md
git commit -m "refactor: shared document store and sync tables; drop the corpus-file ingest"
```

---

### Task 3: Delta-feed and source-name client

**Files:**
- Create: `src/vendorkb/feed.py` (record types and line parsing)
- Modify: `src/vendorkb/clients.py`. Add `DocExtractor.delta` and `DocExtractor.source_names`. The streaming read timeout is 300 s; other calls keep 60 s.
- Test: `tests/unit/test_vendorkb_feed.py`

**Interfaces:**
- Produces, in `feed.py`:
  - `@dataclass(frozen=True) class BootstrapStart: next_since: str`
  - `@dataclass(frozen=True) class CursorLine: next_since: str; count: int`
  - `@dataclass(frozen=True) class Content: doc_id: uuid.UUID; change_type: str; source_id: uuid.UUID | None; vendor: str; product: str | None; title: str | None; url: str | None; content_hash: str | None; markdown: str | None`. `url` comes from the record's `source_url` and `markdown` from `content_markdown`. A `change_type` of `added` or `updated` maps here.
  - `@dataclass(frozen=True) class Tombstone: doc_id: uuid.UUID`, for `change_type == "removed"`.
  - `FeedItem = BootstrapStart | CursorLine | Content | Tombstone`
  - `def parse_line(line: str) -> FeedItem | None`. It returns None for unknown `control` values and unknown `change_type`s, including `run_start`. Unknown fields are ignored. A line that is not valid JSON, or a record without `id`, raises `ValueError`.
- Produces, in `clients.py`:
  - `async def DocExtractor.delta(self, *, since: str | None = None, bootstrap_after: str | None = None) -> AsyncIterator[FeedItem]`. It streams `GET /api/articles/delta` with the `X-API-Key` header and yields parsed non-None items. A `ValueError` line propagates, so the caller treats the stream as dropped. An `httpx` transport error mid-stream also propagates.
  - `async def DocExtractor.source_names(self) -> dict[str, str]`, mapping source_id to name. It calls `GET /api/products?limit=200`, then `GET /api/sources?product_id=<id>&limit=200` per product, with a concurrency of 8. Why per product: `/api/sources` caps at 200 and ignores paging (graph-rag `src/graph_sync/catalog.py:83`). A product that returns 200 or more sources logs a warning.

- [ ] **Step 1: Write the failing tests** with `httpx.MockTransport`. Build settings the way `tests/unit/test_vendorkb_clients.py::_settings` does, with `docext_base_url="http://dx"` and `docext_read_key="k"`.
  - `test_parse_line_shapes`. Cover the bootstrap_start, cursor, added, updated, removed and `run_start` (gives None) lines, an unknown extra field (ignored), and a non-JSON line (raises `ValueError`).
  - `test_delta_streams_items_and_passes_params`. The transport asserts `request.url.params` contains `since=abc` and the `x-api-key` header. Its body is three NDJSON lines with a blank line in between. Assert the yielded types in order.
  - `test_delta_bootstrap_after_param`. Assert that `bootstrap_after` is sent and `since` is absent.
  - `test_source_names_unions_per_product`. Two products, whose sources come back from per-product requests; assert the merged dict.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_vendorkb_feed.py -q`
Expected: FAIL. `vendorkb.feed` does not exist.

- [ ] **Step 3: Implement** `feed.py` and the two client methods. Use `self._http.stream("GET", ...)` with `aiter_lines()` and skip blank lines.

- [ ] **Step 4: Run the gate**

Run: `uv run ruff check src tests && uv run mypy src && uv run pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vendorkb/feed.py src/vendorkb/clients.py tests/unit/test_vendorkb_feed.py
git commit -m "feat: DocExtractor delta-feed stream and per-product source names"
```

---

### Task 4: The sync engine (`vendorkb sync`)

**Files:**
- Create: `src/vendorkb/sync.py`
- Modify: `src/vendorkb/config.py`. Add `sync_page_size: int = 200`.
- Modify: `src/vendorkb/cli.py`. Add the `sync [--max-tokens N]` subcommand. Extend `stats` with `"sync": {"mode", "cursor_set": bool, "bootstrap_after", "failures": <count>}`.
- Modify: `CLAUDE.md`. Add `sync` to "Built" and drop "delta-feed ingest" from "Not built yet".
- Test: `tests/integration/test_vendorkb_sync.py` (new) and `tests/integration/test_vendorkb_cli.py` (stats keys)

**Interfaces:**
- Consumes: `store.*` (Task 2); `DocExtractor.delta`, `DocExtractor.source_names` and `feed.*` (Task 3); `Embedder.embed(texts, input_type="document")` and `Embedder.tokens_used`; `chunk_markdown`.
- Produces, in `sync.py`:
  - `LOCK_KEY = 0x766B62` (the advisory lock id)
  - `@dataclass class SyncReport: mode: str; records: int = 0; written: int = 0; unchanged: int = 0; metadata_updated: int = 0; removed: int = 0; failed: int = 0; tokens: int = 0; seconds: float = 0.0; complete: bool = False; stopped_by_budget: bool = False; locked: bool = False`. `mode` is `"bootstrap"` or `"incremental"`.
  - `async def sync(conn, *, docext: DocExtractor, embedder: Embedder, page_size: int = 200, max_tokens: int | None = None) -> SyncReport`
- The CLI prints `json.dumps(asdict(report))` on one line. Exit codes are 0 when `complete`, `stopped_by_budget` or `locked`; otherwise 1.

**Behaviour.** The test list pins it; the spec §3 has the rationale.
- `pg_try_advisory_lock(LOCK_KEY)` on `conn`. If it is not acquired, return `locked=True` right away. Release the lock in `finally`.
- `sync_state` keys are `cursor`, `bootstrap_next_since`, `bootstrap_after` and `mode`. With no `cursor`, run a bootstrap and call `delta(bootstrap_after=<stored or None>)`. Store `BootstrapStart.next_since` only if `bootstrap_next_since` is unset. Otherwise run an incremental pull with `delta(since=cursor)`.
- Load the source names from the `sources` table. On the first `Content` record whose `source_id` is missing, call `source_names()` once, upsert the result into `sources`, and continue.
- **Pages.** Buffer up to `page_size` `Content`/`Tombstone` items, then:
  1. read `stored_hashes` for the page's ids. A record's hash is its `content_hash`, or `sha256(markdown)` when that is absent;
  2. sort each item into one of:
     - tombstone, which deletes;
     - an unchanged hash, which calls `update_metadata`;
     - missing markdown or 0 chunks, which goes to `sync_failures` (upserted with the reason);
     - otherwise, to embed;
  3. make one `embedder.embed` call over all to-embed chunks, in page order;
  4. one transaction that applies every write. A written or unchanged doc is deleted from `sync_failures`.
  5. After the commit, a bootstrap sets `bootstrap_after` to the page's last item id.
  6. Then, if `max_tokens` is set and `embedder.tokens_used` (taken as a delta from the start of the run) is at or above it, stop with `stopped_by_budget=True`. The cursor is not touched.
- On `CursorLine`, flush the remaining page. Then a bootstrap sets `cursor` to the stored `bootstrap_next_since` and `mode` to `incremental`, and deletes `bootstrap_after`. An incremental pull sets `cursor` to `CursorLine.next_since`. Set `complete=True`.
- A stream that ends without a `CursorLine` (a drop), or an exception from the feed, embed or database, leaves the current page unwritten. It is logged, and the report comes back with `complete=False`. The CLI exits 1.

- [ ] **Step 1: Write the failing integration tests** in `tests/integration/test_vendorkb_sync.py`.
  - The DocExtractor is faked with `httpx.MockTransport`. A queue of NDJSON bodies is served one per `/api/articles/delta` request, and the transport records each request's params. `/api/products` and `/api/sources` are faked too.
  - The embedder is faked with an `httpx.MockTransport` that counts the texts embedded and returns 768-dim vectors with `usage.total_tokens = len(texts)`. Settings use `embed_dim=768`, `embed_input_type=False` and `embed_output_dimension=None`.
  - Use `page_size=2` throughout so that pages are observable.

```python
async def test_full_bootstrap_writes_all_and_switches_to_incremental(vkb_dsn): ...
    # stream: bootstrap_start(S0), 3 added, cursor(S9). Assert 3 documents, the cursor
    # equals S0 (not S9), mode=incremental, bootstrap_after deleted, report.complete.
async def test_bootstrap_drop_resumes_after_last_committed_page(vkb_dsn): ...
    # attempt 1: bootstrap_start(S0), a1, a2, a3, then EOF (no cursor line). Assert
    # a1, a2 written, a3 NOT written, bootstrap_after == a2, complete False.
    # attempt 2: bootstrap_start(S1), a3, cursor(S9). Assert request 2 has
    # bootstrap_after=a2 and no since; cursor == S0 (the first watermark kept).
async def test_incremental_update_unchanged_replay_and_tombstone(vkb_dsn): ...
    # seeded by a bootstrap. Incremental: updated a1 (new hash), added a2 (same hash),
    # removed a3, cursor(S10). Assert texts embedded only for a1, a3 deleted with its
    # chunks cascaded, cursor == S10, request had since=<old cursor>.
async def test_dropped_incremental_leaves_cursor_unchanged(vkb_dsn): ...
async def test_unchanged_hash_updates_metadata_without_embedding(vkb_dsn): ...
    # the same hash with a new title: the title is updated, 0 texts embedded,
    # report.metadata_updated == 1
async def test_embed_failure_aborts_without_moving_watermark(vkb_dsn): ...
    # the embedder transport returns 400 (not retried). Assert 0 documents written,
    # bootstrap_after unset, report.complete False, sync_failures empty
async def test_zero_chunk_record_goes_to_sync_failures_and_page_commits(vkb_dsn): ...
    # markdown "   ": the failure row exists, the other record on the page is written,
    # and the watermark moved
async def test_unknown_source_refreshes_names_once(vkb_dsn): ...
    # two records with unknown source ids: /api/products called exactly once,
    # documents.source is filled where known and NULL where still unknown
async def test_max_tokens_stops_after_the_current_page(vkb_dsn): ...
    # 4 added, page_size 2, max_tokens=1. One page is written, stopped_by_budget,
    # bootstrap_after == the 2nd id, cursor unset
async def test_second_concurrent_run_reports_locked(vkb_dsn): ...
    # hold pg_advisory_lock(LOCK_KEY) on another connection; sync returns locked=True
    # and makes no feed request
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/integration/test_vendorkb_sync.py -q`
Expected: FAIL. `vendorkb.sync` does not exist.

- [ ] **Step 3: Implement** `sync.py`, the config field, the CLI `sync` subcommand (calls `init_schema(conn, embed_dim=settings.embed_dim)` first) and the `stats` addition. Extend `test_stats_reports_counts_and_percentiles` to assert the `sync` key.

- [ ] **Step 4: Run the full gate**

Run: `uv run ruff check src tests && uv run mypy src && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vendorkb/sync.py src/vendorkb/config.py src/vendorkb/cli.py CLAUDE.md tests/integration/test_vendorkb_sync.py tests/integration/test_vendorkb_cli.py
git commit -m "feat: vendorkb sync -- bootstrap + incremental from the DocExtractor delta feed"
```

---

### Task 5: The container image and the GHCR workflow

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `.github/workflows/image.yml`

**Interfaces:**
- Produces the image `ghcr.io/carev01/vendorkb:sha-<7>` and `:main`, with entrypoint command `vendorkb`. It runs as non-root uid 10001 and contains no secret.

- [ ] **Step 1: Write the `Dockerfile`.** Follow the two-stage pattern of graph-rag's `Dockerfile`:
  - base `python:3.12-slim`, `uv` copied from `ghcr.io/astral-sh/uv:0.12.17`, `uv sync --frozen --no-dev`;
  - `LABEL org.opencontainers.image.source="https://github.com/carev01/vendorkb"`;
  - `USER 10001`;
  - `CMD ["vendorkb", "sync"]`.
  - `.dockerignore` excludes `.env*` (except `.env.example`), `.git`, `.venv`, `eval/runs`, `tests`, `docs`, `**/__pycache__`.

- [ ] **Step 2: Build it locally and check the package data**

Run: `docker build -t vendorkb:local . && docker run --rm vendorkb:local python -c "import importlib.resources as r; print(r.files('vendorkb').joinpath('schema.sql').is_file(), r.files('vendorkb').joinpath('scope_aliases.json').is_file())" && docker run --rm vendorkb:local vendorkb --help && docker run --rm --entrypoint sh vendorkb:local -c 'ls -a / /app | grep -c "^\.env$" || true'`
Expected: `True True`, the help output lists `sync`, and the `.env` count is `0`.

- [ ] **Step 3: Write `.github/workflows/image.yml`**, copying graph-rag's `.github/workflows/image.yml`:
  - the image name is `ghcr.io/carev01/vendorkb`;
  - `paths` are `src/**`, `pyproject.toml`, `uv.lock`, `Dockerfile`, `.dockerignore` and `.github/workflows/image.yml`;
  - keep `provenance: false` and `sbom: false`.

- [ ] **Step 4: Commit** (the workflow first runs when this is pushed, which is an operational step)

```bash
git add Dockerfile .dockerignore .github/workflows/image.yml
git commit -m "build: vendorkb container image and GHCR workflow"
```

---

### Task 6: The Helm chart and the deploy runbook

**Files:**
- Create: `deploy/helm/vendorkb/Chart.yaml`, `values.yaml`, and under `templates/`: `_helpers.tpl`, `configmap.yaml`, `postgres.yaml`, `sync-cronjob.yaml`, `bootstrap-job.yaml`
- Create: `docs/deploy/k3s.md`
- Modify: `.github/workflows/ci.yml`. Add a `helm lint` job and a `helm template` job, both with `--set image.tag=sha-0000000`.

**Interfaces and values.** Follow the patterns in graph-rag's `deploy/helm/graph-rag/` (helpers, pod and container security, the tag regex, the config checksum annotation).
- `image.repository: ghcr.io/carev01/vendorkb`, `image.tag: ""` (required, `^sha-[0-9a-f]{7}$`), `imagePullSecrets: []`
- `secretName: vendorkb-secret`
- `config:` goes into the ConfigMap:
  - `DOCEXT_BASE_URL: http://docextractor-backend.docextractor.svc:8000`, `DOCEXT_VERIFY_TLS: "false"`
  - `EMBED_BASE_URL: https://api.voyageai.com/v1`, `EMBED_MODEL: voyage-4`, `EMBED_DIM: "1024"`
  - `RERANK_BASE_URL: https://api.voyageai.com/v1`, `RERANK_MODEL: rerank-3`
  - `JUDGE_BASE_URL: https://openrouter.ai/api/v1`
- `postgres:`
  - `image: pgvector/pgvector:pg16`, `storageClass: vsphere-csi-silver-sc`, `size: 20Gi`
  - `database: vendorkb`, `user: vendorkb`
  - `fsGroup: 999` (the Debian postgres uid, not alpine's 70)
  - resources: requests `{cpu: 250m, memory: 1Gi}`, limits `{cpu: "2", memory: 4Gi}`. HNSW builds need memory.
  - a headless Service `vendorkb-postgres` with no ingress.
- **The DSN.** Each app container defines `POSTGRES_PASSWORD` from `secretKeyRef` first, then `VENDORKB_DSN: postgresql://vendorkb:$(POSTGRES_PASSWORD)@vendorkb-postgres:5432/vendorkb`, using Kubernetes dependent env expansion. The runbook requires a hex password: `openssl rand -hex 24`.
- `sync:` `schedule: "*/30 * * * *"`, `suspend: true`, `concurrencyPolicy: Forbid`, `activeDeadlineSeconds: 21600`, `backoffLimit: 0`, command `["vendorkb", "sync"]`, resources requests `{cpu: 100m, memory: 512Mi}` and limits `{cpu: "1", memory: 1Gi}`.
- `bootstrap:` `enabled: false`, `maxTokens: 85000000`, `activeDeadlineSeconds: 43200`, `backoffLimit: 6`. Each retry resumes from `bootstrap_after`, so the backoff is how a dropped feed recovers. The command is `["vendorkb", "sync", "--max-tokens", "<maxTokens>"]`, and the Job is named `vendorkb-bootstrap`.
- Every pod has `imagePullSecrets` from values, `envFrom` (the secret plus the ConfigMap), non-root uid 10001, and drops all capabilities.

- [ ] **Step 1: Write the chart.**

- [ ] **Step 2: Lint and render it**

Run: `helm lint deploy/helm/vendorkb --set image.tag=sha-0000000 && helm template v deploy/helm/vendorkb --set image.tag=sha-0000000 | grep -E "kind:|schedule|suspend|image:" && helm template v deploy/helm/vendorkb --set image.tag=main 2>&1 | grep -c "must match"`
Expected:
- the lint passes;
- the render shows a ConfigMap, Service, StatefulSet and CronJob, with `suspend: true` and no Job;
- the bad tag is refused (count 1);
- `--set bootstrap.enabled=true` adds the Job, with `--max-tokens` and `"85000000"` in its args;
- `grep -ri "key\|password" deploy/helm/vendorkb/values.yaml` finds only names, never values.

- [ ] **Step 3: Write `docs/deploy/k3s.md`**, the runbook for the operational steps below. Show exact commands, with placeholders and never real values.
  - `kubectl create namespace vendorkb` and `kubectl label namespace vendorkb stack.stage=production`.
  - Creating the Secret with `kubectl create secret generic vendorkb-secret --from-literal=...`. Every value comes from a shell variable the operator sets, never pasted inline.
  - The optional pull secret (a `docker-registry` secret plus `imagePullSecrets`), or making the GHCR package public.
  - `helm upgrade --install vendorkb deploy/helm/vendorkb -n vendorkb --set image.tag=sha-…`.
  - Restoring the local database: run `pg_dump -Fc` from `vendorkb-db`, then `kubectl port-forward svc/vendorkb-postgres 55443:5432` and `pg_restore --no-owner --role=vendorkb -d …`. The restore has to come after `CREATE EXTENSION vector` (the pgvector image allows it) and before any sync. Then run `vendorkb stats` through the port-forward.
  - The one-off re-chunk of the oversized table articles, with the measured count expected: `DELETE FROM documents WHERE doc_id IN (SELECT doc_id FROM chunks WHERE token_count > 1200);`
  - The bootstrap:
    1. smoke it with a manual Job at `--max-tokens 50000` (`kubectl create job --from=...` is not possible from a disabled template, so use `helm upgrade --set bootstrap.enabled=true,bootstrap.maxTokens=50000`);
    2. check rows and tokens;
    3. delete the Job and re-run with the full cap;
    4. watch with `kubectl logs -f job/vendorkb-bootstrap`.
  - Unsuspending the CronJob with `helm upgrade --set sync.suspend=false`.
  - Verifying with `vendorkb stats` and `eval/retrieval_compare.py` through the port-forward.
  - Access for development: the port-forward with `VENDORKB_DSN` pointed at it.

- [ ] **Step 4: Run the full gate, and confirm the CI file parses**

Run: `uv run ruff check src tests && uv run mypy src && uv run pytest -q && uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add deploy/helm/vendorkb docs/deploy/k3s.md .github/workflows/ci.yml
git commit -m "deploy: vendorkb Helm chart (pgvector, sync CronJob, bootstrap Job) and k3s runbook"
```

---

## After the plan (operational, each needs the user's go-ahead)

1. Push to `main`. Run gitleaks first. CI and the image workflow then build `sha-<7>`.
2. The user decides how the image is pulled (pull secret or public package) and creates `vendorkb-secret`, or authorises me to create it from `.env` without printing it.
3. Create the namespace and install the chart, with Postgres only and the CronJob suspended.
4. Run `pg_dump` locally and restore into production. Check that `vendorkb stats` matches local (32,282 chunks before the re-chunk delete).
5. Re-chunk delete, then the bootstrap smoke at `--max-tokens 50000`, then the full bootstrap (about 75M tokens). This is the paid step, and it needs an explicit go-ahead.
6. Unsuspend the CronJob, run the retrieval A/B against production, and remove the local Docker database once the user confirms.
