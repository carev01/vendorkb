# Observability Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One Grafana dashboard that shows four things:
- whether vendorkb is up;
- what its pipeline is doing, with progress and time left;
- how it is used through MCP (eval traffic excluded);
- how its database is doing.

**Architecture:**
- **Task progress:** a new `task_runs` table, written by every long-running command, records
  progress and heartbeats.
- **vendorkb's own metrics:** the MCP server process serves a Prometheus `/metrics` endpoint on
  port 9100. It carries usage counters from the tool handlers, plus pipeline gauges from a
  snapshot it refreshes from the database every 60 s.
- **Database and Kubernetes metrics:** a `postgres_exporter` sidecar, kube-state-metrics, and the
  kubelet's volume stats.
- **Monitoring stack:** vendorkb's scrape jobs and its dashboard are merged into the hand-applied
  stack without touching the other systems' entries.

**Tech Stack:** Python 3 + psycopg (async) + `prometheus_client`, Postgres 16
(`pg_stat_statements`), Helm, Prometheus v3.3.1, Grafana 13.2.0, kube-state-metrics v2.15.0, and
postgres_exporter v0.17.1.

**Spec:** `docs/superpowers/specs/2026-11-18-observability-dashboard-design.md` (read it first).

## Global Constraints

- **No content in metrics:** no question, answer or chunk text, no key, and no secret in any label
  or value. The labels are bounded:
  - `tool`;
  - `key`: the MCP key label;
  - `client`: `user` or `eval`;
  - `outcome`: `ok`, `error` or `abstained`;
  - `kind` and `stage`;
  - `vendor`: about 40 values;
  - `table`;
  - `run`, which only appears on series for runs that are running.
- **`/metrics` stays off the public route:** it is served on its own port (`METRICS_PORT`, default
  9100) and is never routed by the ingress or the bearer-gated app.
- **No database queries on the scrape path:** the pipeline snapshot is refreshed in the background
  at most every `METRICS_REFRESH_S` (60) seconds.
- **Eval traffic:** marked by the header `X-Vendorkb-Client: eval`, matched case-insensitively.
  Any other value, or no header, means `client="user"`. Usage and answer-spend panels show
  `client="user"` only.
- **Answer cost:** prompt and completion tokens times `ANSWER_PRICE_IN_PER_MTOK` and
  `ANSWER_PRICE_OUT_PER_MTOK` (default 0). While both are 0, cost isn't recorded and the panel
  shows tokens only.
- **Time left:** shown only while the rate is above 0, otherwise "unknown". Open batches show their
  age, never a time left.
- **Stalled runs:** a `running` task whose heartbeat is older than 300 s is shown as stalled. The
  heartbeat interval is 30 s.
- **Scrape interval:** 60 s for every `vendorkb-*` job.
- **The tasks' own work comes first:** writing to `task_runs` must never fail or slow the task.
  Write errors are logged and swallowed, and every write is its own short autocommit transaction
  on its own connection.
- **Shared monitoring maps are merged, never replaced:** the shared config maps
  (`prometheus-config`, `grafana-dashboard-provider`) are merged by name, and the
  `grafana-dashboards` config map is never touched.
- **Project rules:**
  - every new column on a filled table gets its own `ANALYZE` in `schema.sql` (`task_runs` is new
    and empty, so it needs none);
  - CI runs ruff, mypy and pytest;
  - run gitleaks before every push;
  - ask the user before merging or touching production.

## Review Focus

1. **A crashed run:** a run killed mid-task (OOM, a Job deadline, Ctrl-C) must show as stalled
   after 300 s, never "running" forever. Test in Task 4.
2. **Database outage during a snapshot refresh:** `/metrics` keeps serving the last snapshot, with
   `vendorkb_metrics_refresh_ok 0`, and MCP requests are unaffected. Test in Task 4.
3. **`task_runs` writes fail:** the database is unreachable, or the connection drops mid-run. The
   command still runs, and returns its normal result and exit code. Test in Task 1.
4. **Two runs of one kind at once:** for example a manual `taxonomy map` during the cron refresh's
   map stage. Both appear as running with their own progress, and neither overwrites the other.
   Test in Task 4.
5. **Errors and refusals:** a tool that raises `ToolError` is counted as `outcome="error"` and its
   latency is still observed. An `answer_question` that returns the refusal text is counted as
   `outcome="abstained"`, not `ok`. Test in Task 3.

---

### Task 1: The `task_runs` table and tracker

**Files:**
- Modify: `src/vendorkb/schema.sql`: append the table.
- Create: `src/vendorkb/task_runs.py`.
- Test: `tests/integration/test_vendorkb_task_runs.py`; follow the existing integration tests'
  testcontainer fixture.

**Interfaces:**
- Produces:
  - `track(dsn: str, kind: str, *, stage: str | None = None) -> AsyncContextManager[TaskRun]`;
  - `TaskRun.run_id: str`;
  - `await TaskRun.progress(done: int, total: int | None = None, spent_usd: float | None = None, *, force: bool = False) -> None`;
  - `await TaskRun.finish(state: Literal["succeeded", "failed"], details: dict | None = None) -> None`;
  - the module-level `await progress(done, total=None, spent_usd=None)`, which forwards to the
    current run or does nothing;
  - `recording() -> ContextManager[Recorder]`, for tests: `Recorder.calls: list[tuple[int, int | None, float | None]]`;
  - the constants `KINDS = ("sync", "refresh", "extract", "map", "support", "capabilities", "propose")`, `HEARTBEAT_S = 30.0` and `STALLED_AFTER_S = 300`.

The table:

```sql
CREATE TABLE IF NOT EXISTS task_runs (
    run_id uuid PRIMARY KEY,
    parent_id uuid REFERENCES task_runs ON DELETE CASCADE,
    kind text NOT NULL,
    stage text,
    state text NOT NULL CHECK (state IN ('running', 'succeeded', 'failed')),
    origin text NOT NULL CHECK (origin IN ('cluster', 'manual')),
    done bigint NOT NULL DEFAULT 0,
    total bigint,
    spent_usd double precision NOT NULL DEFAULT 0,
    details jsonb NOT NULL DEFAULT '{}',
    started_at timestamptz NOT NULL DEFAULT now(),
    heartbeat_at timestamptz NOT NULL DEFAULT now(),
    ended_at timestamptz
);
CREATE INDEX IF NOT EXISTS task_runs_running ON task_runs (kind) WHERE state = 'running';
CREATE INDEX IF NOT EXISTS task_runs_ended ON task_runs (kind, ended_at DESC);
```

- [ ] **Step 1: Write the failing tests** in `tests/integration/test_vendorkb_task_runs.py`:
  - `test_track_records_success`: inside `track(dsn, "map")`, call
    `progress(5, 10, 0.25, force=True)`. Afterwards the row has `state='succeeded'`, `done=5`,
    `total=10`, `spent_usd=0.25`, `ended_at` set and `origin='manual'`.
  - `test_track_records_failure_and_reraises`: the body raises `RuntimeError`. `pytest.raises`
    sees it, and the row is `failed`.
  - `test_nested_track_sets_parent`: `track(dsn, "map")` inside `track(dsn, "refresh")` gives the
    inner row's `parent_id` = the outer row's `run_id`.
  - `test_explicit_finish_wins`: `await run.finish("failed", {"errored": 3})` inside a body that
    returns normally leaves `state='failed'` and `details={"errored": 3}`.
  - `test_progress_is_throttled`: monkeypatch the module clock. Two `progress()` calls within
    `HEARTBEAT_S` give one write (`done` = the first value). A call after `HEARTBEAT_S`, or with
    `force=True`, writes.
  - `test_unreachable_database_never_fails_the_task`: `track("postgresql://nobody@127.0.0.1:1/x", "sync")`
    runs its body, returns its value, and `progress()` inside it doesn't raise. (Review Focus 3.)
  - `test_origin_cluster`: with `KUBERNETES_SERVICE_HOST` set in the environment, the row has
    `origin='cluster'`.
  - `test_old_finished_rows_pruned`: a finished row with `ended_at` 91 days ago is deleted when a
    new `track()` starts. A `running` row of the same age is kept.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/integration/test_vendorkb_task_runs.py -q`

  Expected: errors on importing `vendorkb.task_runs`.

- [ ] **Step 3: Implement `src/vendorkb/task_runs.py` and the `schema.sql` table**
  - **Connection:** `track` opens its own autocommit `psycopg.AsyncConnection`. If the connection
    fails, it logs a warning and yields a no-op `TaskRun`.
  - **Start:** it inserts the row, then prunes finished rows older than 90 days.
  - **Current run:** a `ContextVar` holds the current `TaskRun`, which gives children their parent
    and lets the module-level `progress()` find it.
  - **Heartbeat:** a background `asyncio` task updates `heartbeat_at` every `HEARTBEAT_S` while the
    body runs.
  - **End:** state is `failed` on an exception (re-raised), else `succeeded`, unless `finish()` was
    already called.
  - **Writes:** each write is wrapped in try/except that logs and swallows the error.
  - **Origin:** `cluster` when `KUBERNETES_SERVICE_HOST` is set.
  - **`recording()`:** sets a fake current run that appends to `calls`. It doesn't need a database.

- [ ] **Step 4: Run the tests to verify they pass**

  Run: `uv run pytest tests/integration/test_vendorkb_task_runs.py -q`

  Expected: all pass.

- [ ] **Step 5: Commit**

  `git commit -m "feat(task_runs): a progress record every long-running command writes"`

---

### Task 2: Every command and refresh stage runs inside `track()`

**Files:**
- Modify: `src/vendorkb/cli.py`: `_sync`, `_extract`, `_refresh`, `_claims` (support),
  `_taxonomy` (map, propose) and `_capabilities` (build).
- Modify: `src/vendorkb/claims/refresh.py`: the stages in `refresh()`.
- Test: `tests/integration/test_vendorkb_task_runs.py` (add), and the existing refresh and CLI
  tests.

**Interfaces:**
- Consumes: `track` and `TaskRun.finish` from Task 1.
- Produces: rows with these `kind` values:
  - `sync`, `extract`, `support`, `capabilities` and `propose` from their CLI commands;
  - `map` from `taxonomy map`;
  - `refresh` from `refresh`, with children `extract`, `map`, `support` and `capabilities`.

  The batch kinds go in `stage`: `map-target`, `map-dimension`, `support-check`,
  `support-review`, wherever the code knows which round it is in.

- [ ] **Step 1: Write the failing tests**
  - `test_refresh_stages_are_children`: drive `refresh.refresh()` with the fake clients the
    existing refresh tests use, on a database with new chunks. There's one `refresh` row, and
    `extract`, `map`, `support` and `capabilities` rows whose `parent_id` is that row's id.
  - `test_refresh_not_due_records_reason`: on a database with nothing to do, the `refresh` row is
    `succeeded` and its `details.reason` equals the report's `reason`.
  - `test_cli_exit_code_sets_state`: run `cli.main(["extract", ...])` in a configuration whose
    report has `errored > 0` (exit code 1, as today). The `extract` row is `failed`, and `details`
    holds the report's counts.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/integration/test_vendorkb_task_runs.py -q -k "refresh or cli"`

  Expected: FAIL (no rows).

- [ ] **Step 3: Wrap each command body**
  - In each command: `async with track(settings.vendorkb_dsn, kind) as run:`. When the command
    computes its exit code, call `await run.finish("failed" if code else "succeeded", details)`.
    `details` is the printed report dict with list fields cut to 20 items.
  - In `refresh.py`: wrap each stage. Pass `settings.vendorkb_dsn`, which `refresh()` already has
    through `settings`.

- [ ] **Step 4: Run all tests**

  Run: `uv run pytest -q`

  Expected: all pass, including the existing CLI and refresh tests unchanged.

- [ ] **Step 5: Commit**

  `git commit -m "feat(task_runs): commands and refresh stages record their runs"`

---

### Task 3: Progress from inside the long loops

**Files:**
- Modify:
  - `src/vendorkb/claims/extract.py` (`run`);
  - `src/vendorkb/taxonomy/mapper.py` (`map_claims`);
  - `src/vendorkb/claims/support.py` (the loop over due claims);
  - `src/vendorkb/taxonomy/propose.py` (the loop over clusters);
  - `src/vendorkb/sync.py` (the page loop).
- Test: the existing unit and integration tests of each module, with new assertions.

**Interfaces:**
- Consumes: `task_runs.progress()` and `task_runs.recording()` from Task 1.

The progress each loop reports (`done`, `total`, `spent_usd`):

| Loop | `done` | `total` | `spent_usd` |
|---|---|---|---|
| extract | chunks finished (`done` + `errored` + `failed`) | `len(chunk_ids)` | cost so far |
| map | claims mapped so far | claims to map, counted at the start | spend so far |
| support | claims checked and reviewed | claims due, at the start | spend so far |
| propose | clusters asked | clusters to ask, if known before the first call, else `None` | spend so far |
| sync | records processed | `None` | `None` |

In batch mode, report after each collected batch, not per item.

- [ ] **Step 1: Write the failing tests**

  Add one assertion block per module's existing happy-path test, run under
  `task_runs.recording()`:
  - `rec.calls` is not empty;
  - the last call's `done` equals the report's processed count;
  - where there is a total, the last call's `done == total`.

  The tests are named `test_extract_reports_progress`, `test_map_reports_progress`,
  `test_support_reports_progress`, `test_propose_reports_progress` and
  `test_sync_reports_progress`.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest -q -k reports_progress`

  Expected: 5 failures (`rec.calls == []`).

- [ ] **Step 3: Add `await task_runs.progress(...)`** at the point in each loop where one unit
  completes. The throttling lives in `TaskRun`, so call it on every unit.

- [ ] **Step 4: Run all tests**

  Run: `uv run pytest -q`

  Expected: all pass.

- [ ] **Step 5: Commit**

  `git commit -m "feat(task_runs): extract, map, support, propose and sync report progress"`

---

### Task 4: The pipeline snapshot (database → gauges)

**Files:**
- Create: `src/vendorkb/metrics_db.py`.
- Modify: `pyproject.toml`: add `prometheus_client>=0.21,<1` to the dependencies, then `uv lock`.
- Test: `tests/integration/test_vendorkb_metrics_db.py`.

**Interfaces:**
- Consumes: `task_runs.STALLED_AFTER_S`, `support.waiting(conn, version)`,
  `load.current_version(conn)`, and the unextracted-chunk condition of
  `extract.unextracted_ids`. Reuse its SQL as a count; don't fetch the ids.
- Produces:
  - `async def snapshot(conn) -> Snapshot`;
  - `class PipelineCollector(prometheus_client.registry.Collector)`, with `.update(s: Snapshot | None, ok: bool)` and `.collect()`;
  - `async def refresh_loop(dsn: str, collector: PipelineCollector, interval_s: float) -> None`, which runs until cancelled.

Metric families from `collect()` (all prefixed `vendorkb_`):

| Metric | Type | Labels | Source |
|---|---|---|---|
| `task_running` | gauge (1) | `kind`, `stage`, `run`, `origin` | rows `running` with heartbeat ≤ 300 s old |
| `task_stalled` | gauge (1) | `kind`, `stage`, `run`, `origin` | rows `running` with heartbeat > 300 s old |
| `task_done`, `task_total`, `task_spent_usd` | gauge | `kind`, `stage`, `run` | running and stalled rows (`task_total` only when not null) |
| `task_started_timestamp_seconds` | gauge | `kind`, `stage`, `run` | running and stalled rows |
| `task_last_success_timestamp_seconds`, `task_last_failure_timestamp_seconds` | gauge | `kind` | `max(ended_at)` per kind and state, top-level rows (`parent_id IS NULL`) and children alike |
| `backlog` | gauge | `kind` = `extract`, `map` or `support` | unextracted chunks; claims whose `mapped_version` is distinct from the current version; `support.waiting` |
| `batches_open` | gauge | `kind` | `llm_batches` and `extract_batches` with `status='submitted'` (`kind="extract"` for the latter) |
| `batch_oldest_age_seconds` | gauge | `kind` | `now() - min(submitted_at)` over open batches |
| `batch_items` | gauge | `kind`, `status` | `llm_batch_items` grouped |
| `corpus` | gauge | `what` = `documents`, `chunks`, `claims` or `capabilities` | counts; `chunks` excludes `superseded_at IS NOT NULL` |
| `extraction_errors` | gauge | — | `extractions` with status `errored` or `failed` |
| `taxonomy_info` | gauge (1) | `version` | `current_version` |
| `pipeline_spend_usd_total` | counter | `source` = `extract`, `extract_batch`, `llm_batch`, `propose` or `support` | running sums of the ledgers' cost columns (actual cost, else the estimate, for batches) |
| `table_size_bytes` | gauge | `table` | `pg_total_relation_size` for the 15 largest tables in `public` |
| `sync_cursor_timestamp_seconds` | gauge | — | `sync_state.updated_at` where `key='cursor'` |
| `metrics_refresh_ok` | gauge | — | 1 after a good refresh, 0 after a failed one |
| `metrics_refresh_timestamp_seconds` | gauge | — | time of the last good refresh |

- [ ] **Step 1: Write the failing tests**

  Seed the testcontainer database, then call `snapshot()` and run `collect()` through a fresh
  `CollectorRegistry` read with `get_sample_value`:
  - `test_running_and_stalled`: one row `running` with a fresh heartbeat (`kind=map`), and one
    with a heartbeat 10 minutes old (`kind=support`). This gives `task_running{kind="map"}==1` and
    `task_stalled{kind="support"}==1`, with no `task_running` for support. (Review Focus 1.)
  - `test_two_runs_same_kind`: two `running` `map` rows give two `task_done` series with distinct
    `run` labels and their own values. (Review Focus 4.)
  - `test_backlog_counts`: 3 unextracted chunks; 2 claims on an old `mapped_version` and 1 on the
    current one. This gives `backlog{kind="extract"}==3` and `backlog{kind="map"}==2`.
  - `test_batches`: one open `llm_batches` row of kind `map-target` submitted 2 h ago gives
    `batches_open{kind="map-target"}==1` and `batch_oldest_age_seconds{kind="map-target"}` within
    7,200 ± 60.
  - `test_spend_sums`: extraction costs 0.5 + 0.25 give
    `pipeline_spend_usd_total{source="extract"}==0.75`.
  - `test_refresh_failure_keeps_last_snapshot`: `update(s, True)`, then `update(None, False)`.
    `collect()` still yields the `s` values, with `metrics_refresh_ok==0`. (Review Focus 2.)
  - `test_no_question_text_in_labels`: every label value from `collect()` is in the bounded sets
    listed in Global Constraints, or is a run id, a table name or a taxonomy version.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/integration/test_vendorkb_metrics_db.py -q`

  Expected: errors on importing `vendorkb.metrics_db`.

- [ ] **Step 3: Implement `metrics_db.py`**
  - `refresh_loop` opens one connection per refresh, so it doesn't hold an idle connection, and
    wraps the refresh in try/except. A failure calls `update(None, False)` and logs once per
    failure streak.
  - `collect()` builds families from the last snapshot only.

- [ ] **Step 4: Run the tests to verify they pass**

  Run: `uv run pytest tests/integration/test_vendorkb_metrics_db.py -q`

  Expected: all pass.

- [ ] **Step 5: Commit**

  `git commit -m "feat(metrics): pipeline snapshot from the database as Prometheus gauges"`

---

### Task 5: Usage metrics, the eval marker and the `/metrics` server

**Files:**
- Create: `src/vendorkb/metrics.py`.
- Modify:
  - `src/vendorkb/config.py`: the new settings;
  - `src/vendorkb/mcp_server.py`: every tool records usage, and `answer_question` records
    tokens, cost and vendors;
  - `src/vendorkb/mcp_app.py`: `BearerTokenGate` sets the client from the header; the lifespan
    starts the metrics server and `refresh_loop`;
  - `eval/agent_eval.py` (`mcp_call`): send `X-Vendorkb-Client: eval`. `bypass_eval` reuses
    `mcp_call`.
- Test: `tests/unit/test_vendorkb_metrics.py`. Use `build_app(settings, backends=...)` with fake
  backends, the way the existing MCP app tests do.

**Interfaces:**
- Consumes: `PipelineCollector` and `refresh_loop` from Task 4.
- Produces:
  - `metrics.REGISTRY: CollectorRegistry`;
  - `metrics.client_kind: ContextVar[str]`, default `"user"`;
  - `metrics.observe(tool: str, key: str) -> AsyncContextManager[Observation]`, where
    `Observation.outcome` defaults to `"ok"`, an exception sets `"error"`, and the handler can set
    `"abstained"`;
  - `metrics.record_answer(out: dict, settings: VendorkbSettings) -> None`;
  - `metrics.start(settings) -> None`, which starts the HTTP server on `settings.metrics_port`,
    or does nothing when it is 0.

The new settings in `VendorkbSettings`:
- `metrics_port: int = 9100`;
- `metrics_refresh_s: float = 60.0`;
- `vendorkb_version: str = ""`;
- `answer_price_in_per_mtok: float = 0.0`;
- `answer_price_out_per_mtok: float = 0.0`.

The usage metrics:

| Metric | Type | Labels |
|---|---|---|
| `vendorkb_mcp_requests_total` | counter | `tool`, `key`, `client`, `outcome` |
| `vendorkb_mcp_request_seconds` | histogram, buckets 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160 | `tool`, `client` |
| `vendorkb_answer_tokens_total` | counter | `kind` = `prompt` or `completion`, `client` |
| `vendorkb_answer_cost_usd_total` | counter, only incremented when a price is > 0 | `client` |
| `vendorkb_questions_by_vendor_total` | counter, one increment per vendor in `out["scope"]["vendors"]` | `vendor`, `client` |
| `vendorkb_build_info` | gauge (1) | `version` |

- [ ] **Step 1: Write the failing tests**
  - `test_tool_call_counted`: one `search_docs` call through the app with the full key gives
    `requests_total{tool="search_docs",key=<that key's label>,client="user",outcome="ok"}==1` and a
    latency count of 1.
  - `test_eval_header_labels_client`: `X-Vendorkb-Client: eval` gives `client="eval"`.
    `X-VENDORKB-CLIENT: EVAL` does too. `X-Vendorkb-Client: other` gives `client="user"`.
    (Review Focus 5 and Global Constraints.) This test is also the proof that the gate's context
    variable reaches the tool handler under stateless streamable HTTP. If it doesn't, read the
    header from the MCP request context in the handler instead. The test stays the same.
  - `test_tool_error_counted`: a fake backend that raises gives `outcome="error"`, and the latency
    count still increments. (Review Focus 5.)
  - `test_abstention_counted`: a fake `answer()` returning `answer=answer.REFUSAL` gives
    `outcome="abstained"`.
  - `test_answer_tokens_and_cost`: usage `{"prompt_tokens": 1000, "completion_tokens": 200}` with
    prices of 2.0 and 8.0 per million tokens. Tokens are 1,000 and 200, and cost is
    `0.002 + 0.0016 = 0.0036`. With both prices at 0, no cost sample exists.
  - `test_vendor_counted`: scope vendors `["Veeam", "Rubrik"]` give one increment each.
  - `test_metrics_not_on_app_routes`: `GET /metrics` on the Starlette app returns 404.
  - `test_build_info`: with `vendorkb_version="sha-abc1234"`, the gauge has
    `version="sha-abc1234"`.
  - `test_agent_eval_sends_marker`: monkeypatch `httpx.post` in `eval.agent_eval`.
    `mcp_call("search_docs", {...})` sends the header `X-Vendorkb-Client: eval`.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_vendorkb_metrics.py -q`

  Expected: errors on importing `vendorkb.metrics`.

- [ ] **Step 3: Implement**
  - **Metric objects:** all of them are registered on `REGISTRY`, not the global default registry,
    so tests can create fresh state. Expose a `reset_for_tests()`, or build them in a factory.
  - **Tool handlers:** wrap each body in `async with metrics.observe(<tool>, label) as obs:`.
  - **The server:** `metrics.start` uses `prometheus_client.start_http_server(port, registry=REGISTRY)`,
    and registers `PipelineCollector` on `REGISTRY`.
  - **The lifespan:** starts `refresh_loop(settings.vendorkb_dsn, ...)` as a task when
    `metrics_port` is set and `backends is None`, and cancels it on shutdown.

- [ ] **Step 4: Run all tests and the CI gates**

  Run: `uv run ruff check src tests && uv run mypy src && uv run pytest -q`

  Expected: all pass.

- [ ] **Step 5: Commit**

  `git commit -m "feat(metrics): MCP usage metrics, the eval marker and a /metrics port"`

---

### Task 6: The Helm chart: the metrics port, the Postgres exporter and `pg_stat_statements`

**Files:**
- Modify:
  - `deploy/helm/vendorkb/values.yaml`;
  - `deploy/helm/vendorkb/templates/mcp.yaml`;
  - `deploy/helm/vendorkb/templates/postgres.yaml`;
  - `src/vendorkb/schema.sql`.
- Test: the existing `helm` CI job (`helm lint`), plus the render checks below.

**Interfaces:**
- Produces, for Task 7:
  - the Service `vendorkb-mcp` gains port `metrics` (9100);
  - a new Service `vendorkb-postgres-metrics` with port `metrics` (9187);
  - the role `vendorkb_monitor` (LOGIN, member of `pg_monitor`).

**Values:**

```yaml
metrics:
  enabled: true
  port: 9100
  answerPriceInPerMtok: "0"
  answerPriceOutPerMtok: "0"
  postgresExporter:
    enabled: true
    image: quay.io/prometheuscommunity/postgres-exporter:v0.17.1
```

**Changes:**
- **`mcp.yaml`:**
  - container port `metrics`;
  - env `METRICS_PORT`, `VENDORKB_VERSION={{ .Values.image.tag }}`,
    `ANSWER_PRICE_IN_PER_MTOK` and `ANSWER_PRICE_OUT_PER_MTOK`;
  - a Service port `metrics`.

  The ingress keeps routing only `/mcp` to the existing port. Check the ingress template names
  that port explicitly.
- **`postgres.yaml`:**
  - add `-c shared_preload_libraries=pg_stat_statements` and `-c track_io_timing=on`;
  - mount an `emptyDir` named `pg-socket` at `/var/run/postgresql` in both containers;
  - the sidecar `postgres-exporter` runs with
    `DATA_SOURCE_NAME=host=/var/run/postgresql user=vendorkb_monitor dbname=<postgres.database> sslmode=disable`
    and the args `--collector.stat_statements`. It needs no password: `local all all trust`
    already trusts the socket;
  - the Service `vendorkb-postgres-metrics` selects the Postgres pod on 9187.
- **`schema.sql`:**
  - create the role `vendorkb_monitor` if it doesn't exist (LOGIN, in a `DO` block) and
    `GRANT pg_monitor TO vendorkb_monitor`;
  - add `CREATE EXTENSION IF NOT EXISTS pg_stat_statements`.

- [ ] **Step 1: Render and check** with `helm template vendorkb deploy/helm/vendorkb --set image.tag=sha-abcdef1`:
  - the Postgres pod has 2 containers;
  - the args include `shared_preload_libraries=pg_stat_statements`;
  - the MCP Service has a port named `metrics`;
  - `VENDORKB_VERSION=sha-abcdef1` is set;
  - the ingress backend port is unchanged.

  With `--set metrics.postgresExporter.enabled=false`, the Postgres pod has 1 container.

- [ ] **Step 2: Run the integration tests** (`init_schema` runs the new `schema.sql`)

  Run: `uv run pytest -q tests/integration`

  Expected: all pass.

- [ ] **Step 3: Lint**

  Run: `helm lint deploy/helm/vendorkb --set image.tag=sha-abcdef1`

  Expected: `1 chart(s) linted, 0 chart(s) failed`.

- [ ] **Step 4: Commit**

  `git commit -m "feat(chart): metrics port, postgres_exporter sidecar, pg_stat_statements"`

---

### Task 7: The monitoring stack: scrape jobs, kube-state-metrics, kubelet stats and the dashboard

**Files:**
- Create in `deploy/monitoring/`:
  - `kube-state-metrics.yaml`;
  - `prometheus-rbac.yaml`;
  - `scrape-vendorkb.yaml`;
  - `apply.py`;
  - `vendorkb-dashboard.json`;
  - `README.md`.
- Test: `tests/unit/test_monitoring_apply.py`, which loads `deploy/monitoring/apply.py` by path.

**Interfaces:**
- Consumes: the Services from Task 6, and the metric names from Tasks 4 and 5.
- Produces:
  - `merge_scrape(config_yaml: str, fragment_yaml: str) -> str`, which replaces every job whose
    `job_name` starts with `vendorkb-`, adds the fragment's jobs, and keeps the others in order;
  - `merge_provider(provider_yaml: str) -> str`, which adds or replaces the provider named
    `vendorkb` (folder `vendorkb`, path `/var/lib/grafana/dashboards-vendorkb`) and keeps the
    others;
  - the CLI `python deploy/monitoring/apply.py [--dry-run]`.

**The files:**
- **`kube-state-metrics.yaml`:**
  - a ServiceAccount, plus a ClusterRole with list and watch on cronjobs, jobs, pods,
    deployments, statefulsets and persistentvolumeclaims, and its binding;
  - a Deployment running `registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.15.0` with
    `--resources=cronjobs,jobs,pods,deployments,statefulsets,persistentvolumeclaims --namespaces=vendorkb`;
  - a Service on 8080.

  Everything is in the `monitoring` namespace.
- **`prometheus-rbac.yaml`:** a ServiceAccount `prometheus`, plus a ClusterRole with get, list and
  watch on `nodes`, `nodes/metrics` and `nodes/proxy`, and its binding. The Prometheus Deployment
  is patched to use it (`apply.py` does the patch).
- **`scrape-vendorkb.yaml`:** a list of jobs, each with `scrape_interval: 60s`:
  - `vendorkb-app`: `vendorkb-mcp.vendorkb.svc:9100`;
  - `vendorkb-postgres`: `vendorkb-postgres-metrics.vendorkb.svc:9187`;
  - `vendorkb-kube-state-metrics`: `kube-state-metrics.monitoring.svc:8080`;
  - `vendorkb-kubelet`: `kubernetes_sd_configs` with `role: node`, scheme https through
    `kubernetes.default.svc:443` and `metrics_path` `/api/v1/nodes/${1}/proxy/metrics`, using the
    in-cluster service-account token and CA. It has `metric_relabel_configs` keeping only
    `kubelet_volume_stats_.*` with `namespace="vendorkb"`.
- **`apply.py`:** runs `kubectl`, and with `--dry-run` prints diffs and changes nothing. The
  steps:
  1. apply the two manifests;
  2. patch the Prometheus Deployment's `serviceAccountName` to `prometheus`;
  3. read `prometheus-config`, `merge_scrape` it, apply it, and POST `/-/reload` with
     `kubectl exec deploy/prometheus -- wget -qO- --post-data= http://localhost:9090/-/reload`.
     The patch in step 2 restarts the pod anyway, but the reload must still work alone;
  4. create or replace the config map `grafana-dashboards-vendorkb` from `vendorkb-dashboard.json`;
  5. read `grafana-dashboard-provider`, `merge_provider` it, and apply it;
  6. patch the Grafana Deployment to add volume `dashboards-vendorkb` from that config map,
     mounted at `/var/lib/grafana/dashboards-vendorkb`. The patch is idempotent: skip it if the
     mount exists.
- **`vendorkb-dashboard.json`:** uid `vendorkb`, title "vendorkb", default refresh 1m, 24 h range,
  and the default data source (panel `datasource` omitted). The four rows follow the approved
  mockup:
  - **Right now:**
    - MCP up: `up{job="vendorkb-app"}`, with `vendorkb_build_info`;
    - running now: `count(vendorkb_task_running)`, plus a stalled count in red;
    - last sync: `time() - vendorkb_task_last_success_timestamp_seconds{kind="sync"}`;
    - failed in 24 h:
      - failed Jobs started in the last 24 h, from `kube_job_status_failed` and
        `kube_job_status_start_time`, in namespace `vendorkb`;
      - plus kinds whose `task_last_failure_timestamp_seconds` falls in the last 24 h;
    - questions in 24 h: `sum(increase(vendorkb_mcp_requests_total{client="user",tool=~"answer_question|gather_evidence"}[24h]))`,
      with the p95 from the histogram;
    - spend in 7 d: `sum(increase(vendorkb_pipeline_spend_usd_total[7d])) + sum(increase(vendorkb_answer_cost_usd_total{client="user"}[7d]))`.
  - **Pipeline:**
    - a table of running and stalled tasks: kind, stage, origin, done, total, %, rate, time left
      and spend, where:
      - the rate is `deriv(vendorkb_task_done[15m])*60` per minute;
      - time left is `(vendorkb_task_total - vendorkb_task_done) / deriv(vendorkb_task_done[15m])`,
        shown only when the rate is > 0, else the text "unknown";
    - the last success and failure per kind;
    - `vendorkb_backlog` over time, with time-left stats `vendorkb_backlog / -deriv(vendorkb_backlog[15m])`
      when the derivative is < 0;
    - open batches with "waiting" age, and items by status;
    - corpus stats, extraction errors and the taxonomy version.
  - **Usage**, all `client="user"`:
    - rate by tool;
    - p50 and p95 by tool;
    - errors and abstentions by tool;
    - answer tokens, and cost when present;
    - requests by key;
    - the top 10 vendors over the range.
  - **Database:**
    - `pg_database_size_bytes{datname="vendorkb"}` and its growth;
    - `vendorkb_table_size_bytes` top 10;
    - `pg_stat_activity_count` and waiting locks;
    - the cache hit rate from `pg_stat_database_blks_hit` and `pg_stat_database_blks_read`;
    - transactions and tuples per second;
    - the top 10 queries by mean time from `pg_stat_statements_*`;
    - the volume's free percentage from `kubelet_volume_stats_available_bytes / kubelet_volume_stats_capacity_bytes`
      for PVC `data-vendorkb-postgres-0`.

- [ ] **Step 1: Write the failing tests**
  - `test_merge_scrape_keeps_other_jobs`: a config with jobs `llamacpp-x`, `vendorkb-app` (old
    target) and `node-y`, and a fragment with `vendorkb-app` (new target) and `vendorkb-postgres`.
    The result holds `llamacpp-x` and `node-y` unchanged and in order, `vendorkb-app` with the new
    target, and `vendorkb-postgres`. `global:` is unchanged.
  - `test_merge_scrape_idempotent`: merging twice equals merging once.
  - `test_merge_provider_adds_once`: the existing provider "LLM Dashboards" is kept; one
    `vendorkb` provider is added; merging twice gives one.
  - `test_dashboard_json_valid`: the JSON loads, its uid is `vendorkb`, and every `expr` names
    only metrics from this plan's tables or the exporters (`pg_*`, `kube_*`, `kubelet_*`, `up`).

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_monitoring_apply.py -q`

  Expected: FAIL (no module at that path).

- [ ] **Step 3: Implement the files**, and `README.md`, which gives the apply and dry-run commands
  and what each step changes.

- [ ] **Step 4: Run the tests to verify they pass**

  Run: `uv run pytest tests/unit/test_monitoring_apply.py -q`

  Expected: all pass. Also `kubectl apply --dry-run=server -f deploy/monitoring/kube-state-metrics.yaml -f deploy/monitoring/prometheus-rbac.yaml`
  is accepted. This is read-only against the cluster.

- [ ] **Step 5: Commit**

  `git commit -m "feat(monitoring): scrape jobs, kube-state-metrics, kubelet stats and the vendorkb dashboard"`

---

### Task 8: Docs, PR, and the production rollout (with the user's go-ahead at each gate)

**Files:**
- Modify:
  - `CLAUDE.md`: a short "Observability" entry under Built;
  - `docs/deploy/k3s.md`: an "Observability" section covering the metrics port, the exporter,
    the `deploy/monitoring/apply.py` usage, and setting the answer prices.

- [ ] **Step 1: Write the docs. Run the full CI gate and gitleaks, push the branch and open the
  PR.**

  Run: `uv run ruff check src tests && uv run mypy src && uv run pytest -q && gitleaks protect --staged --no-banner`

  Expected: all pass, and "no leaks found".

- [ ] **Step 2: GATE: the user approves the merge.** Then merge, and wait for the image build
  `sha-<merge>`.
- [ ] **Step 3: GATE: the user approves the deploy**, including one Postgres restart for
  `pg_stat_statements`.
  - Run: `helm -n vendorkb upgrade vendorkb deploy/helm/vendorkb --reuse-values --set image.tag=sha-<merge> --wait`
  - Then check:
    - `kubectl -n vendorkb exec vendorkb-postgres-0 -c postgres -- psql -U vendorkb -d vendorkb -Atc "select count(*) from pg_stat_statements"`
      returns a number;
    - `kubectl -n vendorkb exec deploy/vendorkb-mcp -- wget -qO- localhost:9100/metrics | grep -c '^vendorkb_'`
      returns more than 20.
- [ ] **Step 4: GATE: the user approves the monitoring changes.**
  - Run: `python deploy/monitoring/apply.py --dry-run`, show the diff, then
    `python deploy/monitoring/apply.py`.
  - Check, in the Prometheus targets (`kubectl -n monitoring exec deploy/prometheus -- wget -qO- localhost:9090/api/v1/targets`):
    - all four `vendorkb-*` jobs are `up`;
    - the earlier jobs are still present.
- [ ] **Step 5: Verify every panel against the live system.**
  - Run a small manual task against production: `uv run vendorkb capabilities build`, or a
    `taxonomy map --max-usd 0` dry run, which spends nothing. Check it appears under "Running
    now", then as the last success.
  - Make one `answer_question` call with and one without the eval header. Only the second counts.
  - Every panel shows data, or "unknown" or "no data" where the spec expects it: time left for
    batches and for propose.
- [ ] **Step 6:** Report the panels to the user, with a screenshot if possible, and anything
  still empty.
