# vendorkb monitoring

The Prometheus and Grafana in namespace `monitoring` were applied by hand and live in no repo.
This directory adds what vendorkb needs to them, and nothing else: the other scrape jobs, the
other dashboard provider and the config map `grafana-dashboards` are left as they are.

| File | What |
|---|---|
| `kube-state-metrics.yaml` | kube-state-metrics v2.15.0 for namespace `vendorkb` only (CronJobs, Jobs, Pods, Deployments, StatefulSets, PVCs), Service `kube-state-metrics.monitoring.svc:8080` |
| `prometheus-rbac.yaml` | ServiceAccount `prometheus` with get/list/watch on nodes (discovery) and get on `nodes/proxy` (kubelet metrics through the API server) |
| `scrape-vendorkb.yaml` | the `vendorkb-*` scrape jobs (60 s): app metrics, postgres_exporter, kube-state-metrics, kubelet volume stats of namespace `vendorkb` |
| `vendorkb-dashboard.json` | the Grafana dashboard (uid `vendorkb`): Right now, Pipeline, Usage, Database |
| `apply.py` | applies all of it |

The app and postgres_exporter targets need the chart's metrics Services
(`vendorkb-mcp.vendorkb.svc:9100`, `vendorkb-postgres-metrics.vendorkb.svc:9187`, Helm
`metrics.enabled` and `metrics.postgresExporter.enabled`).

## Apply

```bash
uv run python deploy/monitoring/apply.py --dry-run   # print the diffs, change nothing
uv run python deploy/monitoring/apply.py             # apply
```

`--dry-run` reads the cluster only: `kubectl diff` (a server-side dry run) for the manifests and
the dashboard config map, text diffs for the two merged config maps, and the patches it would
send. Run it with the cluster's kubeconfig; it needs `ruamel.yaml` (the project's environment).

What each step changes:

1. **Manifests:** `kubectl apply` of `kube-state-metrics.yaml` and `prometheus-rbac.yaml`
   (ServiceAccounts, ClusterRoles and bindings `kube-state-metrics` and `prometheus`, the
   kube-state-metrics Deployment and Service).
2. **Prometheus strategy and service account:** patches Deployment `prometheus` to
   `strategy: Recreate` (one replica on a ReadWriteOnce volume: a rolling update would start
   the new pod while the old one holds the TSDB lock, and it would crash-loop) and
   `serviceAccountName: prometheus`, which restarts its pod. Each part is skipped when already
   set.
3. **Scrape jobs:** reads config map `prometheus-config`, replaces every job named `vendorkb-*`
   with the ones in `scrape-vendorkb.yaml` (added at the end), keeps every other job, `global:`
   and the comments, and applies it. Then it waits for the rollout and for the pod to see the
   new file, and POSTs `/-/reload` (`kubectl exec deploy/prometheus -- wget -qO- --post-data=
   http://localhost:9090/-/reload`; Prometheus runs with `--web.enable-lifecycle`). The reload
   runs even when nothing changed.
4. **Dashboard:** creates or replaces config map `grafana-dashboards-vendorkb` (key
   `vendorkb.json`) from `vendorkb-dashboard.json`.
5. **Provider:** adds (or replaces) the provider `vendorkb` (folder `vendorkb`, path
   `/var/lib/grafana/dashboards-vendorkb`) in config map `grafana-dashboard-provider`, keeping
   "LLM Dashboards".
6. **Grafana strategy and mount:** patches Deployment `grafana` to `strategy: Recreate` (same
   reason: its sqlite database is on a ReadWriteOnce volume) and adds volume
   `dashboards-vendorkb` (from that config map) mounted read-only at
   `/var/lib/grafana/dashboards-vendorkb`, which restarts the pod. Each part is skipped when
   already there; if no pod restart was patched and step 5 changed the provider, Grafana is
   restarted, as it reads providers at start only.

Re-running is safe: every step is idempotent. A dashboard edit needs step 4 only (Grafana
re-reads the file within 30 s); a scrape job edit needs step 3.

Re-applying the original hand-written Prometheus or Grafana manifests drops the vendorkb
volume, service account and Recreate strategy: run `apply.py` again afterwards.

Comments in `scrape-vendorkb.yaml` belong inside a job: a comment between two jobs is merged
as part of the job above it.

## The dashboard

Default data source (no panel names one), refresh 1 min, last 24 h. Usage panels count
`client="user"` only, so eval runs (`X-Vendorkb-Client: eval`) stay out. Time left is shown only
while the rate is above 0 ("unknown" otherwise); open batches show how long they have waited,
as a batch reports no progress. Spend is `vendorkb_pipeline_spend_usd{window}` (leaf task runs
started in the last 24 h / 7 d) plus the answer model's cost.

To change it, edit the JSON (or export it from Grafana after editing there) and run `apply.py`.
`tests/unit/test_monitoring_apply.py` checks that every `expr` names only metrics vendorkb or
the exporters serve.
