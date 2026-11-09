# vendorkb on srv-k3s: deploy runbook

Chart: `deploy/helm/vendorkb`. Run each step in order and verify it before the next.

**Never** paste a real credential into a chat, ticket, commit or this document. Every
value below comes from a shell variable you set yourself (read it from your password
manager, never inline in a command you will keep in history), or is a placeholder
(`sha-abc1234`, `<pod>`).

## 1. Prerequisites

- `kubectl` context pointed at `srv-k3s.home.lan`; `helm` v3.16 or newer.
- The image built by the `image` workflow: `ghcr.io/carev01/vendorkb:sha-abc1234`
  (7-char short SHA). The chart rejects any other tag format (a moving tag such as
  `main`, uppercase hex, a full SHA).
- Local tools for the restore: `pg_dump`/`pg_restore` 16 and `uv`.

## 2. Namespace and Secret

```
kubectl create namespace vendorkb
kubectl label namespace vendorkb stack.stage=production
```

Set the shell variables first (`PG_PASSWORD` must be hex so it is safe inside the DSN):

```
PG_PASSWORD=$(openssl rand -hex 24)
# EMBED_KEY, RERANK_KEY, JUDGE_KEY, DOCEXT_KEY: set from your password manager / .env
kubectl -n vendorkb create secret generic vendorkb-secret \
  --from-literal=POSTGRES_PASSWORD="$PG_PASSWORD" \
  --from-literal=EMBED_API_KEY="$EMBED_KEY" \
  --from-literal=RERANK_API_KEY="$RERANK_KEY" \
  --from-literal=JUDGE_API_KEY="$JUDGE_KEY" \
  --from-literal=DOCEXT_READ_KEY="$DOCEXT_KEY"
```

Keep `PG_PASSWORD` in your shell for the restore step (or read it back later with
`kubectl -n vendorkb get secret vendorkb-secret -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d`).

### Image pull

Either make the GHCR package public (GitHub, repo, Packages, `vendorkb`, Package
settings, Change visibility), then verify with `docker logout ghcr.io && docker pull ghcr.io/carev01/vendorkb:sha-abc1234`;
or keep it private and create a pull secret:

```
kubectl -n vendorkb create secret docker-registry ghcr-pull \
  --docker-server=ghcr.io --docker-username="$GH_USER" --docker-password="$GH_READ_PACKAGES_TOKEN"
```

and add `--set 'imagePullSecrets[0].name=ghcr-pull'` to every `helm upgrade` below.

## 3. Install (Postgres only, CronJob suspended)

```
helm upgrade --install vendorkb deploy/helm/vendorkb -n vendorkb --set image.tag=sha-abc1234
kubectl -n vendorkb rollout status statefulset/vendorkb-postgres
```

The sync CronJob is created with `suspend: true`; nothing talks to DocExtractor yet.
Later upgrades use `--reset-then-reuse-values` (Helm 3.14 or newer), which keeps earlier
`--set`s on top of the current chart's defaults. Plain `--reuse-values` replays the values
stored with an older chart, so keys the chart has since gained render empty (a missing
`sync.maxTokens` would drop the CronJob's cap; the chart now refuses to render without it).

## 4. Restore the local database

The restore comes after `CREATE EXTENSION vector` and before any sync.

```
# local: dump from the vendorkb-db container's database
pg_dump -Fc -h 127.0.0.1 -p 55442 -U postgres -d vendorkb -f vendorkb.dump

# in another terminal: reach production Postgres
kubectl -n vendorkb port-forward svc/vendorkb-postgres 55443:5432

PGPASSWORD="$PG_PASSWORD" psql -h 127.0.0.1 -p 55443 -U vendorkb -d vendorkb \
  -c 'CREATE EXTENSION IF NOT EXISTS vector'
PGPASSWORD="$PG_PASSWORD" pg_restore --no-owner --role=vendorkb \
  -h 127.0.0.1 -p 55443 -U vendorkb -d vendorkb vendorkb.dump
```

The pgvector image lets the `vendorkb` user create the extension. Check the result
through the port-forward:

```
VENDORKB_DSN="postgresql://vendorkb:${PG_PASSWORD}@127.0.0.1:55443/vendorkb" uv run vendorkb stats
```

Counts must match the local database (32,282 chunks before the re-chunk below).

Access over the port-forward needs the password. The chart mounts its own `pg_hba.conf`
(`postgres.hba` in `values.yaml`): TCP connections, including those arriving through
`kubectl port-forward` (they look like 127.0.0.1 inside the pod, which the image's default
trusts without a password), must authenticate with `scram-sha-256`. Only the unix socket stays
trusted, and reaching it needs `kubectl exec`. Always use the DSN with `PG_PASSWORD` or
`PGPASSWORD`, as above. Changing `postgres.hba` rolls the Postgres pod (checksum annotation).

**Schema note.** The dump still has the old `graph_content_hash` column. The first
`vendorkb sync` runs `init_schema`, which drops it. Do not follow the restore with any
older vendorkb version (it would not know the schema).

## 5. One-off re-chunk of oversized table articles

Articles with a chunk over 1200 tokens predate the splitter fix. Deleting their
documents makes the sync re-fetch and re-chunk them. Through the port-forward, count first:

```
export PGPASSWORD="$PG_PASSWORD"
psql -h 127.0.0.1 -p 55443 -U vendorkb -d vendorkb -c \
  "SELECT count(*) FROM documents WHERE doc_id IN (SELECT doc_id FROM chunks WHERE token_count > 1200);"
```

Expect about 137 documents locally. If the number is far off, stop and investigate.
Then delete:

```
psql -h 127.0.0.1 -p 55443 -U vendorkb -d vendorkb -c \
  "DELETE FROM documents WHERE doc_id IN (SELECT doc_id FROM chunks WHERE token_count > 1200);"
```

## 6. Bootstrap

The bootstrap Job is a disabled chart template, so enable it through Helm.

**The bootstrap cap is cumulative.** `--max-tokens` in bootstrap mode compares against
`sync_state.bootstrap_tokens`, the running total of every attempt, retry and Job of this
bootstrap (the smoke run included), so a Job retry can never spend the cap twice. The key is
deleted when the bootstrap finishes. In incremental mode the cap is per run.

1. Smoke test at a small cap:
   ```
   helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reset-then-reuse-values \
     --set bootstrap.enabled=true,bootstrap.maxTokens=50000
   kubectl -n vendorkb logs -f job/vendorkb-bootstrap
   ```
2. Gate on the run report, the last JSON line of the log:
   ```
   kubectl -n vendorkb logs job/vendorkb-bootstrap | grep '^{' | tail -1
   ```
   Go on only if **`tokens` > 0** (the provider reports usage; at 0 the cap would never
   engage) **and `unchanged` > 0** (the hash gate skipped restored articles instead of
   re-embedding them). `stopped_by_budget` is true. `tokens` may exceed 50,000 by up to one
   page (200 articles): the cap is checked after each page commits. Check `sync.failures`
   in `vendorkb stats` is empty or understood.
3. Delete the Job (`kubectl -n vendorkb delete job vendorkb-bootstrap`), then re-run with
   the full cap (default 85,000,000, cumulative, so it includes the smoke run's tokens; this
   is the paid step):
   ```
   helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reset-then-reuse-values \
     --set bootstrap.enabled=true,bootstrap.maxTokens=85000000
   ```
   If Helm reports the Job already exists, delete it first (it is immutable).
4. Watch with `kubectl -n vendorkb logs -f job/vendorkb-bootstrap` (one line per committed
   page). A dropped feed exits 1 and the Job retries (`backoffLimit: 6`), each retry
   resuming from `bootstrap_after`. A budget stop or a held lock exits 0, so a **Complete
   Job does not mean a finished bootstrap**: check `vendorkb stats` (step 7).

At the end of the bootstrap, documents the feed no longer serves (deleted upstream after
the local ingest) are deleted; the report's `pruned` counts them.

**Prune refusal.** The bootstrap's end refuses to prune more than max(500, 2% of
documents): an empty or partial stream (a key scoped differently, an RBAC change, an
upstream bug) would otherwise delete the restored corpus. A refusal logs one ERROR line,
`bootstrap finish refused: it would prune N documents ... over the ceiling of C`, rolls the
finish back (cursor not set, `sync.mode` still `bootstrap`), reports `prune_refused: N` and
exits 1, so the Job retries. Each retry resumes after the last applied id (or from the start
of the feed if no page committed, a full download), the hash gate embeds nothing unchanged,
and it is refused again unless the feed has recovered (then it completes normally). Check N against what you expect deleted upstream since the local ingest
(e.g. DocExtractor's article count against `vendorkb stats`). If N is wrong, stop and
investigate the feed. If it is right, delete the Job and re-run with the override:
```
kubectl -n vendorkb delete job vendorkb-bootstrap
helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reset-then-reuse-values \
  --set bootstrap.enabled=true,bootstrap.allowPrune=N
```
Afterwards set `bootstrap.allowPrune=null` again.

**Failing page (quarantine).** `vendorkb stats` shows `sync.failures` and
`sync.abort_count`. A page that aborts after it was embedded is counted in `sync_state`
(`abort_position`, a hash of the page's items, and `abort_count`). After 3 aborts in a row the
next run embeds and writes that page record by record (tombstones and metadata updates
included): records that still fail are quarantined into `sync_failures` (reason `quarantined after 3 page aborts: ...`), the rest
commit and the sync carries on. The run logs one ERROR line naming the quarantined ids; the
report's `quarantined` counts them. No operator action is needed for the feed to keep
moving. Inspect the quarantined articles and fix the cause. A replay of the same content is
skipped without embedding (the report's `skipped_quarantined`).

If two or more records on the page all fail in quarantine, that is systemic (the write path,
the database, the embedding provider): the page aborts with nothing written, `sync_state`
gets `abort_quarantine_failed`, and later runs refuse that page before embedding it and exit
1, logging `sync refused: page starting at <id> failed for every record in quarantine mode`.
Fix the cause from the errors the runs logged, then clear the breaker:
```
psql -h 127.0.0.1 -p 55443 -U vendorkb -d vendorkb -c \
  "DELETE FROM sync_state WHERE key IN ('abort_position','abort_count','abort_quarantine_failed');"
```
An upstream change to the page (an update, or a new article joining it) also gives it a
fresh try without operator action.

**Recovering a quarantined article.** It is retried automatically when its content changes
upstream. To retry it sooner (the cause was on our side and is fixed), deleting its
`sync_failures` row alone is not enough: the cursor has moved past it, so the feed never
serves it again. Delete its row **and** re-bootstrap:
```
psql -h 127.0.0.1 -p 55443 -U vendorkb -d vendorkb -c \
  "DELETE FROM sync_failures WHERE doc_id = '<id>';
   DELETE FROM sync_state WHERE key IN ('cursor','mode');"
```
then run the bootstrap Job (step 6, with the CronJob suspended meanwhile). The hash gate
skips every stored unchanged document, so the re-bootstrap embeds only what is missing or
changed (the article(s) whose rows you deleted; quarantined articles whose rows you kept are
skipped again): its token cost is a few articles, but it downloads the whole feed. The cap
is cumulative as for any bootstrap, and the prune ceiling applies at its end (documents the
feed no longer serves are pruned; a large count is refused).

When the bootstrap has completed, disable the Job: `helm upgrade ... --reset-then-reuse-values --set bootstrap.enabled=false`.

## 7. Enable incremental sync

First confirm the bootstrap finished: through the port-forward, `uv run vendorkb stats` must
show `sync.mode` = `"incremental"` and `sync.cursor_set` = `true`. If it still says
`bootstrap`, do **not** unsuspend: go back to step 6.

```
helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reset-then-reuse-values --set sync.suspend=false
kubectl -n vendorkb get cronjob vendorkb-sync
```

The CronJob runs every 30 minutes with `concurrencyPolicy: Forbid` and
`--max-tokens 5000000` (`sync.maxTokens`, per run), which also bounds a large wave of
`updated` articles such as a VLM image-enrichment run.

## 7a. Reactive claims refresh

`vendorkb refresh` (`claims/refresh.py`) keeps claims and capabilities current without a GPU
running between DocExtractor refreshes. The CronJob `vendorkb-claims-refresh` runs it weekly (Sunday 03:45).
Until the sync has written chunks that no extraction has touched, a run is two queries and
contacts nothing. When such chunks exist and no new document has arrived for
`claimsRefresh.settleMinutes` (or the backlog fills a run), it wakes the distilled extractor
(`deploy/azure`, scaled to zero), extracts every chunk with no finished extraction, maps the
claims not yet mapped under the current taxonomy version (`claimsRefresh.map.maxUsd` per run),
support-checks the new claims of topic dimensions (`claimsRefresh.support`: the checker, then
the reviewer for every flagged claim, each with its own `maxUsd` per run; backend anthropic
needs `ANTHROPIC_API_KEY` in the secret), and rebuilds capabilities. Chunks that only errored never trigger a
run; they ride along with the next one.

Add the three keys to the secret (ANTHROPIC_API_KEY for the support check), then enable it with
the endpoint's URL:

```
kubectl -n vendorkb patch secret vendorkb-secret --type merge -p \
  "{\"stringData\":{\"EXTRACT_API_KEY\":\"$SERVE_KEY\",\"MAP_API_KEY\":\"$OPENROUTER_KEY\",\"ANTHROPIC_API_KEY\":\"$ANTHROPIC_KEY\"}}"
helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reset-then-reuse-values \
  --set claimsRefresh.enabled=true \
  --set claimsRefresh.extract.baseUrl=https://<app>.<region>.azurecontainerapps.io/v1
kubectl -n vendorkb create job --from=cronjob/vendorkb-claims-refresh refresh-check  # one run now
```

A run prints a JSON report (`new_chunks`, `due`, `reason`, and the extract, map and support
reports of a due run). By hand, through the port-forward: `vendorkb extract --unextracted
[--vendor V]` extracts what is missing, `vendorkb taxonomy map` maps only claims not yet mapped,
and `vendorkb claims support --check-max-usd N --review-max-usd M [--limit K]` checks the claims
of topic dimensions not yet checked and reviews the flagged ones (Anthropic API by default; with
`SUPPORT_BACKEND=openai` set `SUPPORT_MODEL` / `SUPPORT_REVIEW_MODEL` to the OpenRouter ids);
`vendorkb capabilities build` then leaves out the rejected claims.

## 7b. MCP server

`vendorkb mcp` (`src/vendorkb/mcp_app.py`, `mcp_server.py`) serves the agent tools
(`list_vendors`, `list_dimensions`, `list_targets`, `query_capabilities`, `search_docs`,
`read_section`, `answer_question`) over streamable HTTP at `/mcp`, stateless. Every request must
carry `Authorization: Bearer $MCP_BEARER_TOKEN`; the server refuses to start without the
token. Host headers outside `mcp.allowedHosts` get 421 (DNS-rebinding guard).

```
MCP_TOKEN=$(openssl rand -hex 32)
kubectl -n vendorkb patch secret vendorkb-secret --type merge -p \
  "{\"stringData\":{\"MCP_BEARER_TOKEN\":\"$MCP_TOKEN\"}}"
helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reset-then-reuse-values \
  --set mcp.enabled=true --set mcp.ingress.enabled=true \
  --set mcp.ingress.host=vendorkb-mcp.k3s.home.lan
```

Smoke test (an MCP `initialize`; the reply names the server `vendorkb`):

```
curl -s http://vendorkb-mcp.k3s.home.lan/mcp -H "Authorization: Bearer $MCP_TOKEN" \
  -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

The LAN ingress is plain HTTP like the cluster's other `*.k3s.home.lan` services. Exposing it
to Microsoft Copilot needs a public HTTPS route and OAuth in front (not built yet).

## 8. Verify

Through the port-forward, with `VENDORKB_DSN` set as in step 4:

```
uv run vendorkb stats
uv run python eval/retrieval_compare.py   # see eval/README.md for arguments
```

## Upgrading to taxonomy v2 (one-way)

The first `init_schema` after deploying the taxonomy v2 image drops the v1 taxonomy tables
(`features`, the v1 `targets`, `aliases`, `capabilities`) and creates the v2 ones. **This cannot
be undone by an image rollback.** The previous image's `schema.sql` fails on a v2 database (it
indexes `capabilities.feature_id`), so every sync run would fail. To go back, restore the
database from a backup taken before the upgrade.

Before upgrading:
1. Take a database backup (`pg_dump` through the port-forward or `kubectl exec`).
2. Check that nothing outside the v1 tables depends on them:
   `SELECT classid::regclass, objid, refobjid::regclass FROM pg_depend WHERE refobjid IN
   ('features'::regclass, 'capabilities'::regclass);`
3. Check `\d raw_claims`. The v3 claim columns were added to production by hand before the
   upgrade. `kind` must be `NOT NULL DEFAULT 'capability'`.

After upgrading, load the reviewed taxonomy, map, and build capabilities (`vendorkb taxonomy
load|map`, `vendorkb capabilities build`).

## Development access

```
kubectl -n vendorkb port-forward svc/vendorkb-postgres 55443:5432
export VENDORKB_DSN="postgresql://vendorkb:${PG_PASSWORD}@127.0.0.1:55443/vendorkb"
```
