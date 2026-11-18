# Observability: a Grafana dashboard for vendorkb (design)

2026-11-18. Status: for the user's approval.

## Intent

The user wants to glance at one screen and know whether vendorkb is healthy and what it is doing:
- which tasks are running, how far along they are, and roughly how long they have left;
- stats on the corpus and the pipeline;
- how vendorkb is being used through MCP;
- the database's size and performance.

Today they find out by asking Claude to run kubectl and SQL.

**Success:** the user opens the dashboard and can answer these without a terminal:
- Is the MCP server up?
- Is a sync, refresh, propose or re-map running, and when will it finish?
- Did the last scheduled jobs succeed?
- How much was asked and spent this week?
- Is the database growing or slowing down?

## Goals

The dashboard is one Grafana dashboard with four sections, top to bottom. The user approved the
layout mockup on 2026-11-18.

1. **Right now:**
   - the MCP server is up, with its version (image tag);
   - tasks running now;
   - time since the last successful sync;
   - failed jobs in the last 24 h;
   - questions in the last 24 h, with p95 latency;
   - spend over 7 days, split into answers and pipeline.
2. **Pipeline tasks.** The tasks are:
   - the sync;
   - the claims refresh and each of its stages (extract, map, support check, capabilities build),
     plus its batch rounds;
   - `taxonomy propose`;
   - `taxonomy map` re-maps.

   This covers runs started by hand from a workstation against the production database, not only
   the cluster's CronJobs.

   Each running or recent task shows its state, progress (done of total, where a total exists),
   rate, estimated time left, and spend. Further panels:
   - backlog over time: chunks to extract, claims to map, claims to support-check;
   - open provider batches, with their age and items in flight;
   - corpus counts: documents, chunks, claims, capability facts;
   - extraction errors;
   - the current taxonomy version, and claims not yet mapped under it.
3. **Usage** (MCP):
   - requests per tool;
   - latency p50 and p95 per tool;
   - errors;
   - abstentions from `answer_question`;
   - the answer model's tokens and cost;
   - requests by key (standard or full access);
   - the most-asked vendors.
4. **Database:**
   - size, and its growth over time;
   - the largest tables;
   - connections, and lock waits;
   - cache hit rate;
   - transactions and rows read or written per second;
   - the slowest queries;
   - free space on the Postgres volume.

## Non-goals

- **No alerting or notifications.** The user checks the dashboard. Nothing is built that pages or
  messages anyone.
- **No eval runs.** The answer, agent and bypass evals stay off the dashboard.
- **No ratings or quality scores.** The dashboard does not show answer-quality scores.
- **No log aggregation.** Loki and its like are out of scope.

## Anti-goals

This would be a failure even if it technically works:

- **Slowing the database or the MCP server.** Metric queries must stay cheap:
  - no full-table counts on every scrape;
  - the backlog counts on `raw_claims` (676k rows) and `chunks` (254k) are computed at most once a
    minute and cached;
  - nothing heavy runs on the request path.
- **Leaking content.** Metrics never carry question text, answer text, chunk text, keys or
  secrets as labels or values. Every label has a bounded set of values: tool, key kind, outcome,
  task kind, stage, and vendor, of which there are about 40.
- **A metrics route on the public ingress.** The public ingress exposes only `/mcp`. `/metrics`
  must not be reachable through it, nor through the bearer-gated app.
- **A time estimate that pretends to know.** Where a task has no known total (`propose`) or its
  end depends on the provider (an open Message Batch), the dashboard says so: "unknown", or
  "waiting N h". It never shows an invented time.

## Constraints

- **Cluster:** k3s, single node.
- **Monitoring namespace:** `monitoring` runs plain Prometheus v3.3.1 from a Deployment, with static
  `scrape_configs` in config map `prometheus-config` and `--web.enable-lifecycle`. It runs
  Grafana 13.2.0, which provisions dashboards from config map `grafana-dashboards` through
  provider config map `grafana-dashboard-provider` (folder "LLM"). Data source: Prometheus at
  `http://prometheus:9090`.
- **No exporters for vendorkb:** no kube-state-metrics, no Prometheus Operator CRDs, and no
  kubelet or cAdvisor scraping.
- **The monitoring stack's manifests are in no repository.** They were applied by hand.
- **vendorkb:** Helm release `vendorkb` in namespace `vendorkb`.
  - Postgres is the StatefulSet `vendorkb-postgres` (`pgvector/pgvector:pg16`), about 7 GB.
    `pg_stat_statements` is not loaded, `track_io_timing` is off, and its only extensions are
    `vector` and `plpgsql`.
  - The MCP server is the Deployment `vendorkb-mcp`: one uvicorn process serving a Starlette app
    (`mcp_app.py`) with one MCP server per key.
  - Jobs are the CronJobs `vendorkb-sync`, `vendorkb-claims-refresh` (weekly) and
    `vendorkb-claims-collect` (hourly).
- **What the database already records:**
  - `extractions`: status, tokens, cost and `created_at` per chunk;
  - `extract_batches` and `llm_batches` (with `llm_batch_items`): status, submit and end times,
    estimated and actual cost;
  - `propose_runs`: start, end and spend per run;
  - `claim_support`: `checked_at`, `reviewed_at` and cost;
  - `raw_claims.mapped_version` (no mapped-at time);
  - `sync_state`: cursor and `updated_at`.

  No table records "a task is running, at stage X, with N of M done".
- **The answer model:** Azure OpenAI `gpt-6.2-sol` (`JUDGE_PROVIDER=azure`). Azure returns token
  usage but no cost, so cost must come from configured prices.
- **Project rules (CLAUDE.md):**
  - secrets stay in `.env` and are never printed;
  - run gitleaks before every push;
  - ask before merging or changing production;
  - a column added to a filled table needs its own `ANALYZE` in `schema.sql`;
  - CI runs ruff, mypy and pytest.

## Decisions

The user handed all the "how" decisions to Claude: "I trust you to handle these". Each one below
is Claude's call.

1. **A task-run record in the database: new table `task_runs`.**
   - The CLI commands write a row when they start: sync, refresh and each of its stages, extract,
     `taxonomy map`, `claims support`, `capabilities build`, `taxonomy propose`, and batch
     collect rounds.
   - While running, they update a heartbeat and their progress (`done`, `total` if known, spend)
     about every 30 seconds or every few hundred items, whichever comes first.
   - At the end they set the state (succeeded or failed), the end time and a small `details`
     jsonb of counts.
   - Being in the database, it catches manual runs from a workstation, which Kubernetes metrics
     cannot see.
   - A running row whose heartbeat is older than about 5 minutes shows as stalled, which is what a
     crashed run looks like.
   - Writes are their own short transactions, never inside the work's own transaction.
   - Writing the row must not be able to fail the task: errors are logged and swallowed.
2. **Pipeline and usage metrics from vendorkb itself:** a Prometheus `/metrics` endpoint, using
   `prometheus_client`, on a separate port of the MCP server process (9100), outside the
   bearer-gated app and outside the ingress.
   - **Usage metrics, recorded in the tool handlers:**
     - requests by tool, key kind and outcome (ok, error, abstained);
     - a latency histogram by tool;
     - answer-model prompt and completion tokens;
     - answer cost, as tokens times prices from settings. New settings: `ANSWER_PRICE_IN_PER_MTOK`
       and `ANSWER_PRICE_OUT_PER_MTOK`, priced for gpt-6.2-sol;
     - questions by detected vendor scope.
   - **Pipeline gauges, from a collector that reads the database at most once a minute (cached):**
     - running, stalled and recent `task_runs` with their progress;
     - backlog counts;
     - open batches, with their age and items in flight;
     - corpus counts;
     - extraction errors;
     - claims not on the current taxonomy version;
     - pipeline spend summed from the ledgers.
   - **Build info:** `vendorkb_build_info{version}`, from the image tag, which the chart passes as
     an environment variable.
   - **Database role:** the collector reads as `VENDORKB_DSN` (full access), never the restricted
     role, so counts include hidden vendors. The dashboard is for the operator only.
3. **Database performance: `postgres_exporter`**, the prometheus-community exporter, as a sidecar
   in the `vendorkb-postgres` pod, added to the vendorkb chart behind `metrics.postgresExporter`.
   - It connects over localhost as a dedicated role with `pg_monitor`.
   - It runs the `stat_statements` collector and per-table sizes.
   - The chart loads `pg_stat_statements` (`shared_preload_libraries`) and turns
     `track_io_timing` on, and `schema.sql` creates the extension.
   - This needs one Postgres restart, at deploy, with the user's go-ahead.
4. **Job and pod status: kube-state-metrics** in `monitoring`, scraped by Prometheus. It provides
   CronJob last-success times, Job failures and pod restarts.
5. **Volume free space:** the kubelet's `kubelet_volume_stats_*`, scraped by Prometheus through the
   API server's node proxy. That needs a ServiceAccount with `nodes/metrics` and
   `nodes/proxy` read for the Prometheus Deployment. Alternatively, a cheaper route the builder
   finds equivalent.
6. **Time left:** computed in PromQL, as the remaining count divided by the recent rate of the
   done or backlog series, over a 10–15 minute window. Shown only while that rate is above zero;
   otherwise "unknown". Open batches show their age, not a time left.
7. **Scrape interval:** 60 s for the vendorkb jobs. The cluster's default is 15 s.
8. **Where things live.** The vendorkb repo holds only vendorkb's parts. It never holds whole
   shared config maps: the shared ones also carry the LLM dashboards and the other scrape jobs.
   - **In `deploy/monitoring/`, applied with kubectl** (the same pattern as `deploy/kasten/`):
     - the dashboard JSON;
     - kube-state-metrics;
     - the RBAC;
     - vendorkb's scrape jobs as a fragment.
   - **The scrape jobs:** merged into the live `prometheus-config` by a small script that replaces
     only the jobs named `vendorkb-*` (matched by `job_name`) and leaves every other job as it is.
     Prometheus is then reloaded through `/-/reload`.
   - **The dashboard:** gets its own config map, `grafana-dashboards-vendorkb`. It is mounted into
     Grafana at `/var/lib/grafana/dashboards-vendorkb`, with a second provider entry (folder
     "vendorkb") added to `grafana-dashboard-provider`. The existing `grafana-dashboards` config
     map is never touched. The Grafana Deployment gets one extra volume and mount, as a patch.
   - **The Helm chart:** the exporter sidecar and the metrics port.

   *Claude's call, from the builder check.*
9. **Prometheus retention is enough.** Its data is on a persistent volume (`prometheus-data`,
   10 Gi, 4% used) with 30-day retention, checked 2026-11-18. So answer usage can live in
   Prometheus counters alone, with no database ledger. A counter resets when the MCP pod
   restarts; `increase()` handles that.
10. **Answer prices.** `ANSWER_PRICE_IN_PER_MTOK` and `ANSWER_PRICE_OUT_PER_MTOK` default to 0.
    While they are 0, the dashboard shows tokens and leaves answer cost blank. The user supplies
    the Azure prices for gpt-6.2-sol when they want cost shown. *Claude's call.*
11. **Eval traffic stays out of usage and spend.** *The user's decision, 2026-11-18.*
    - **The marker:** the eval harnesses (`eval/agent_eval.py`, `eval/bypass_eval.py` and any
      other caller that runs evals against the deployed server) send the header
      `X-Vendorkb-Client: eval`. In the agent eval, it goes in the MCP config the subagents are
      given.
    - **The metric label:** every usage metric gets a `client` label: `eval` when the header says
      so, else `user`. There are no other values, and any other header value counts as `user`.
    - **The panels:** the Usage section and the answer-spend figures show `client="user"` only.
      Eval traffic is recorded but not displayed.
    - **Not a security boundary:** the header only labels the traffic, and anyone holding a key
      could set it.

## Left to the builder

- The exact metric names and the `task_runs` columns, within the rules above.
- How each command reports progress: what counts as "done" and "total" per stage.
- The dashboard's panel types, units, colors and thresholds, kept close to the approved mockup.
- Which kube-state-metrics version and install method: plain manifests or a rendered chart.
- Tests: unit tests for the `task_runs` writer, the collector's queries and the usage
  instrumentation, plus a render check of the Helm chart.

## Rollout

Each step that changes production needs the user's go-ahead.

1. Code and chart in one PR.
2. Deploy the chart: the Postgres restart for `pg_stat_statements`, and the metrics port.
3. Apply the monitoring manifests and the Prometheus scrape jobs, then reload Prometheus.
4. Load the dashboard.
5. Check every panel against the live system: a manual run of something small should appear as
   running, with progress.
