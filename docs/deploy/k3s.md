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
Later upgrades use `--reuse-values` (or `--reset-then-reuse-values` on Helm 4 when the
chart's `values.yaml` gained keys) so earlier `--set`s persist.

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

1. Smoke test at a small cap:
   ```
   helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reuse-values \
     --set bootstrap.enabled=true,bootstrap.maxTokens=50000
   kubectl -n vendorkb logs -f job/vendorkb-bootstrap
   ```
2. Check rows and tokens: `vendorkb stats` through the port-forward (chunk and document
   counts grew, the token total is near 50,000, and `sync.failures` is empty or understood).
3. Delete the Job (`kubectl -n vendorkb delete job vendorkb-bootstrap`), then re-run with
   the full cap (default 85,000,000; this is the paid step):
   ```
   helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reuse-values \
     --set bootstrap.enabled=true,bootstrap.maxTokens=85000000
   ```
   If Helm reports the Job already exists, delete it first (it is immutable).
4. Watch with `kubectl -n vendorkb logs -f job/vendorkb-bootstrap`. A dropped feed
   exits 1 and the Job retries (`backoffLimit: 6`), each retry resuming from
   `bootstrap_after`. A budget stop or a held lock exits 0.

`vendorkb stats` shows `sync.failures`. A run that exits 1 every time with the same page
ids logged means a stuck page: look at those documents before retrying further.

When the bootstrap has completed, disable the Job: `helm upgrade ... --reuse-values --set bootstrap.enabled=false`.

## 7. Enable incremental sync

```
helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reuse-values --set sync.suspend=false
kubectl -n vendorkb get cronjob vendorkb-sync
```

The CronJob runs every 30 minutes with `concurrencyPolicy: Forbid`.

## 8. Verify

Through the port-forward, with `VENDORKB_DSN` set as in step 4:

```
uv run vendorkb stats
uv run python eval/retrieval_compare.py   # see eval/README.md for arguments
```

## Development access

```
kubectl -n vendorkb port-forward svc/vendorkb-postgres 55443:5432
export VENDORKB_DSN="postgresql://vendorkb:${PG_PASSWORD}@127.0.0.1:55443/vendorkb"
```
