# vendorkb in production, kept current from DocExtractor's delta feed

> Status: draft for review, 2026-11-04. Covers steps 3 and 4 of the order the user approved:
> (1) move to its own repo — done; (2) voyage-4 re-embed + A/B — running; (3) delta-feed
> ingest + production deploy; (4) full-corpus ingest in production.

## 1. Intent

vendorkb runs today as a Docker container on the workstation, over a frozen list of the
20,899 articles the graph happened to extract (4 vendors). It should run in the k3s
cluster, hold the **whole DocExtractor corpus (121,374 articles, 400 sources, 43 vendors,
measured 2026-11-04)**, and stay current on its own, so later milestones (MCP server,
claim extraction, taxonomy) build on a complete, live index.

## 2. Goals, non-goals, anti-goals

**Goals**
- A `vendorkb` namespace in k3s: Postgres + pgvector (StatefulSet, vSphere storage, so
  Kasten backs it up) and a sync CronJob, deployed from a Helm chart in this repo with an
  image built by CI.
- `vendorkb sync`: bootstrap + incremental ingest from `GET /api/articles/delta`, following
  the DocExtractor contract (`CLIENT-USAGE-GUIDE.md` §6, in graph-rag).
- The full corpus loaded once, embedded with voyage-4, inside the 200M free-token budget.

**Non-goals**
- The MCP server and any public exposure (later; no ingress now).
- Claim extraction, taxonomy, capability table (M3–M5).
- Webhook-triggered sync (polling covers it; the webhook can be added later as a nudge).
- Document version history (DESIGN.md §1 non-goal: latest docs only).

**Anti-goals**
- Re-embedding what is already embedded: the 15M voyage-4 tokens spent on the current
  20,899 articles must carry over, and a re-run of a sync must never re-embed unchanged
  articles. The budget has room for the corpus once, not twice.
- A cursor that advances past data not yet applied (silent gaps).
- Secrets in the repo, the image or the chart.

## 3. The sync (`vendorkb sync`)

One command, single-flight (a Postgres advisory lock; a second run exits 0 immediately).
State lives in a new `sync_state` table (key/value): `cursor` (opaque, stored as-is),
`bootstrap_next_since`, `bootstrap_after`, `mode` (`bootstrap` | `incremental`).

**Bootstrap** (no `cursor` yet):
1. `GET /api/articles/delta` (resuming with `bootstrap_after=<highest applied id>` when a
   previous attempt dropped). Store the **first** attempt's `bootstrap_start.next_since`
   immediately and never overwrite it on resume.
2. Consume the stream **in order**, in pages of N records. For each `added` record: gate on
   `content_hash` against `documents.content_hash` (skip unchanged — this is what carries
   the existing 20,899 articles over), otherwise chunk, embed, upsert document + chunks.
   Embedding is batched across the page's articles (Voyage takes up to 1,000 texts /
   320K tokens a request); writes are serial, one transaction per page.
3. After each committed page, set `bootstrap_after` to the page's last article id. Order
   matters: the watermark means "every id up to here is applied", which only holds because
   pages are applied in stream order (graph-rag's sync rule, kept).
4. On the terminal `{"control":"cursor"}` line: `cursor = bootstrap_next_since` (the
   stored one, per the contract), `mode = incremental`. A stream without the terminal line
   is a dropped stream: exit non-zero, resume next run.

**Incremental** (`cursor` set): `GET /api/articles/delta?since=<cursor>`; `added`/`updated`
as above; `removed` deletes the document (chunks cascade). Tombstones may arrive in
thousands with `run_id: null` — fine. The cursor advances **only** after the terminal line;
a re-run replays idempotently.

**Metadata** comes from the feed record: `vendor`, `product`, `title`, `source_url`, and
`source_id`. `documents.source` currently holds the source *name* (used by scope detection
for distinctive source names); the feed gives only `source_id`, so sync resolves names once
per run from `GET /api/sources` (cached in a `sources` table). `graph_content_hash` is
dropped (head-to-head only). The corpus-file `ingest` command is removed once `sync` has
loaded production.

**Guards**
- `--max-tokens N`: stop cleanly (after the current page) once embedding tokens reach N.
  In bootstrap mode N caps the **cumulative** total across attempts (`sync_state`
  `bootstrap_tokens`, written in each page transaction, deleted at the bootstrap's end), so
  Job retries cannot spend it twice; in incremental mode it is per run. The bootstrap Job
  runs with a cap just above the projection; the sync CronJob with 5M (`sync.maxTokens`),
  so it can never continue an unfinished bootstrap uncapped.
- **Circuit breaker**: a page that aborts after it was embedded is recorded (autocommit)
  as `abort_position` (its first id) and `abort_count`. Only the commit of that same page,
  or the run's terminal line (the cursor moved or the bootstrap finished), clears both: an
  incremental run replays the unchanged pages before it, and their commits must not reset
  the count. At 3 the
  page is refused before any embed call (exit 1) until an operator deletes the two keys.
  `vendorkb stats` shows `abort_count`.
- **Bootstrap pruning**: each page records its ids in `sync_seen`; the terminal line
  deletes documents the bootstrap never served (deleted upstream after the local ingest),
  reported as `pruned`. A fresh bootstrap clears `sync_seen`; a dropped one never prunes.
- A malformed record whose id parses is a per-record failure (`malformed: ...`).
- Failures are split by class. **Deterministic** per-record failures are recorded in a
  `sync_failures` table (article id, reason, time) and skipped; the page still commits and
  the watermark moves. These are: missing markdown, markdown that yields 0 chunks, an embed
  4xx other than 401/403/404/408/409/425/429 (the page embed is retried record by record to
  isolate the offender; reason `embed: <status> <message>`; if isolation rejects every one
  of 2+ to-embed records the page aborts as transient instead), and a `psycopg.DataError` (the page is
  re-applied with a savepoint per record, reusing the vectors; reason `db: <message>`). NUL
  characters are stripped from markdown before chunking so the common DataError never
  happens. **Transient** failures (embed 401/403/404 — a bad key or endpoint — and
  5xx/408/409/425/429 after the client's retries,
  transport errors, `OperationalError`, anything unexpected) abort the page: nothing is
  written, the watermark stays, and the run exits non-zero to resume next time. Repeated
  failures are visible in `vendorkb stats`.
- Run report (stdout, one JSON line): mode, records seen, written, skipped-unchanged,
  removed, failed, pruned, tokens (this run), seconds.

## 4. Production deployment

- **Namespace** `vendorkb`, labelled `stack.stage=production` like the old graph-rag one.
- **Postgres**: `pgvector/pgvector:pg16` StatefulSet, `vsphere-csi-silver-sc`, **20 Gi**.
  Projection: ~190k chunks × (1,024-dim vectors + HNSW + text + tsvector) ≈ 3–4 GB; 20 Gi
  leaves room for growth, M3's claims table and vacuum. Headless service, no ingress.
- **Sync CronJob**: every 30 minutes, `concurrencyPolicy: Forbid`, same image. DocExtractor
  is in the same cluster, so the feed stream stays in-cluster.
- **One-off bootstrap Job** (same image, `vendorkb sync --max-tokens …`), run by hand once.
- **Image**: `Dockerfile` (uv, Python 3.12, non-root) and a GitHub Actions workflow pushing
  `ghcr.io/carev01/vendorkb:sha-<short>` and `:main` on changes to `src/`, `pyproject.toml`,
  `uv.lock` or the Dockerfile.
- **Secret** `vendorkb-secret` (POSTGRES_PASSWORD, DOCEXT_READ_KEY, EMBED_API_KEY,
  RERANK_API_KEY, JUDGE_API_KEY), created outside the chart. Non-secret settings
  (EMBED_MODEL=voyage-4, EMBED_DIM=1024, …) in a ConfigMap from `values.yaml`.
- **Access for development and evals**: `kubectl port-forward` to Postgres; the CLI and
  `eval/retrieval_compare.py` run unchanged against it.

## 5. Rollout

1. Finish step 2: if voyage-4 wins the A/B, `promote-embedding` locally.
2. Build and push the image; deploy the chart (Postgres only, CronJob suspended).
3. `pg_dump` the local database and restore it into production. This moves the 20,899
   articles with their voyage-4 vectors — no re-embedding.
4. Run the bootstrap Job. Expected: ~100k new articles, ~75M voyage-4 tokens (Voyage counts
   ~0.84× the chars/4 estimate, measured), embedding ≈ 10–20 min at 8M TPM; the whole Job
   is bounded by DB writes and HNSW maintenance, est. 1–2 h. Smoke it first with
   `--max-tokens 50000` and check rows and tokens moved.
5. Unsuspend the CronJob. Verify: `vendorkb stats`, the retrieval A/B script on the eval
   set (should be unchanged for the 4 original vendors), spot searches on new vendors.
6. Stop and remove the local Docker database only after the user confirms production.

## 6. Testing

- Unit: feed parsing (control lines, missing terminal line, tombstones, unknown fields
  ignored), page ordering, cursor/watermark rules, `--max-tokens` stop.
- Integration (testcontainers Postgres + an `httpx.MockTransport` DocExtractor): a full
  bootstrap; a dropped bootstrap resumed with `bootstrap_after` keeping the first
  `next_since`; an incremental pull with an update, an unchanged replay (no embed call) and
  a tombstone; a dropped incremental stream leaving the cursor unchanged; single-flight lock.
- `@live`: one bootstrap page against DocExtractor scoped by `source_id` (opt-in).

## 7. Decisions (mine unless marked)

- Polling every 30 minutes, no webhook (cheap: an empty pull is one control line).
- Hard delete on tombstones (no version history in v1).
- Embedding batches cross article boundaries (fewer requests; same vectors).
- Pages of 200 records per transaction (bounded memory; ~1 min of work per page).
- **Theirs:** how the image is pulled (the repo and its GHCR package are private — a pull
  secret with a read-only token, or a public image) and how `vendorkb-secret` is created
  (by the user, or by me from `.env` without printing it).
