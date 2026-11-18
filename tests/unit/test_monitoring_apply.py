"""deploy/monitoring/apply.py: the scrape and provider merges, and the dashboard's expressions."""
import importlib.util
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / "deploy" / "monitoring"

_spec = importlib.util.spec_from_file_location("monitoring_apply", MON / "apply.py")
assert _spec and _spec.loader
apply = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apply)

CONFIG = """\
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # the llama.cpp router
  - job_name: "llamacpp-x"
    static_configs:
      - targets: ["srv:8080"]
  # an old vendorkb job (a comment after a replaced job goes with it)
  - job_name: "vendorkb-app"
    static_configs:
      - targets: ["old:9100"]
  - job_name: "node-y"
    static_configs:
      - targets: ["srv:9100"]
"""

FRAGMENT = """\
- job_name: vendorkb-app
  scrape_interval: 60s
  static_configs:
    - targets: ["new:9100"]
- job_name: vendorkb-postgres
  scrape_interval: 60s
  static_configs:
    - targets: ["pg:9187"]
"""

PROVIDER = """\
apiVersion: 1
providers:
  - name: "LLM Dashboards"
    folder: "LLM"
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false"""


def _jobs(text):
    return {j["job_name"]: j for j in yaml.safe_load(text)["scrape_configs"]}


def test_merge_scrape_keeps_other_jobs():
    out = apply.merge_scrape(CONFIG, FRAGMENT)
    doc = yaml.safe_load(out)
    before = yaml.safe_load(CONFIG)
    assert doc["global"] == before["global"]
    names = [j["job_name"] for j in doc["scrape_configs"]]
    assert names == ["llamacpp-x", "node-y", "vendorkb-app", "vendorkb-postgres"]
    jobs, old = _jobs(out), _jobs(CONFIG)
    assert jobs["llamacpp-x"] == old["llamacpp-x"]
    assert jobs["node-y"] == old["node-y"]
    assert jobs["vendorkb-app"]["static_configs"][0]["targets"] == ["new:9100"]
    assert jobs["vendorkb-postgres"]["static_configs"][0]["targets"] == ["pg:9187"]
    assert "# the llama.cpp router" in out  # comments survive the round trip
    assert "# an old vendorkb job" in out


def test_merge_scrape_idempotent():
    for fragment in (FRAGMENT, (MON / "scrape-vendorkb.yaml").read_text()):
        once = apply.merge_scrape(CONFIG, fragment)
        assert apply.merge_scrape(once, fragment) == once


def test_merge_scrape_with_the_repo_fragment():
    fragment = (MON / "scrape-vendorkb.yaml").read_text()
    jobs = _jobs(apply.merge_scrape(CONFIG, fragment))
    assert {"vendorkb-app", "vendorkb-postgres", "vendorkb-kube-state-metrics",
            "vendorkb-kubelet", "llamacpp-x", "node-y"} == set(jobs)
    assert all(j.get("scrape_interval") == "60s" for n, j in jobs.items()
               if n.startswith("vendorkb-"))
    kubelet = jobs["vendorkb-kubelet"]
    assert kubelet["kubernetes_sd_configs"] == [{"role": "node"}]
    assert kubelet["scheme"] == "https"


def test_merge_provider_adds_once():
    once = apply.merge_provider(PROVIDER)
    doc = yaml.safe_load(once)
    names = [p["name"] for p in doc["providers"]]
    assert names == ["LLM Dashboards", "vendorkb"]
    assert doc["providers"][0] == yaml.safe_load(PROVIDER)["providers"][0]
    ours = doc["providers"][1]
    assert ours["folder"] == "vendorkb"
    assert ours["options"]["path"] == "/var/lib/grafana/dashboards-vendorkb"
    twice = apply.merge_provider(once)
    assert [p["name"] for p in yaml.safe_load(twice)["providers"]] == names
    assert twice == once


# metrics the plan's tables define, plus the exporters' families
ALLOWED = {
    "vendorkb_task_running", "vendorkb_task_stalled", "vendorkb_task_done",
    "vendorkb_task_total", "vendorkb_task_spent_usd", "vendorkb_task_started_timestamp_seconds",
    "vendorkb_task_last_success_timestamp_seconds",
    "vendorkb_task_last_failure_timestamp_seconds", "vendorkb_backlog", "vendorkb_batches_open",
    "vendorkb_batch_oldest_age_seconds", "vendorkb_batch_items", "vendorkb_corpus",
    "vendorkb_extraction_errors", "vendorkb_taxonomy_info", "vendorkb_pipeline_spend_usd",
    "vendorkb_table_size_bytes", "vendorkb_sync_cursor_timestamp_seconds",
    "vendorkb_metrics_refresh_ok", "vendorkb_metrics_refresh_timestamp_seconds",
    "vendorkb_mcp_requests_total", "vendorkb_mcp_request_seconds_bucket",
    "vendorkb_mcp_request_seconds_count", "vendorkb_mcp_request_seconds_sum",
    "vendorkb_answer_tokens_total", "vendorkb_answer_cost_usd_total",
    "vendorkb_questions_by_vendor_total", "vendorkb_build_info", "up",
}
EXPORTER_PREFIXES = ("pg_", "kube_", "kubelet_")
# PromQL words that look like metric names: functions, keywords, label names in by/on/without
NOT_METRICS = {
    "sum", "count", "max", "min", "avg", "topk", "bottomk", "rate", "irate", "increase",
    "deriv", "time", "vector", "histogram_quantile", "by", "without", "on", "ignoring",
    "group_left", "group_right", "and", "or", "unless", "clamp_min", "clamp_max", "label_replace",
    "max_over_time", "min_over_time", "avg_over_time", "scalar", "abs", "offset", "bool",
    "sort_desc", "delta", "last_over_time", "absent", "round", "sgn",
}


def _metric_names(expr):
    expr = re.sub(r'"[^"]*"', '""', expr)          # string literals
    expr = re.sub(r"\{[^}]*\}", "", expr)          # matchers
    expr = re.sub(r"\[[^\]]*\]", "", expr)         # ranges
    expr = re.sub(r"\b(by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)", "", expr)
    words = re.findall(r"\b[a-zA-Z_:][a-zA-Z0-9_:]*\b", expr)
    return {w for w in words if w not in NOT_METRICS and not re.fullmatch(r"\d+[smhdwy]?", w)
            and w not in ("e", "Inf")}


def _exprs(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "expr" and isinstance(v, str):
                yield v
            else:
                yield from _exprs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _exprs(v)


def test_dashboard_json_valid():
    dash = json.loads((MON / "vendorkb-dashboard.json").read_text())
    assert dash["uid"] == "vendorkb"
    assert dash["title"] == "vendorkb"
    assert dash["refresh"] == "1m"
    assert dash["time"]["from"] == "now-24h"
    exprs = list(_exprs(dash))
    assert len(exprs) > 20
    for expr in exprs:
        for name in _metric_names(expr):
            assert name in ALLOWED or name.startswith(EXPORTER_PREFIXES), (name, expr)
    assert not any("pipeline_spend_usd_total" in e for e in exprs)
    assert any('vendorkb_pipeline_spend_usd{window="7d"}' in e for e in exprs)
    # no panel pins a data source: the default one serves them all
    panels = dash["panels"] + [p for row in dash["panels"] for p in row.get("panels", [])]
    assert not any("datasource" in p for p in panels)
    assert [p["title"] for p in dash["panels"] if p["type"] == "row"] == [
        "Right now", "Pipeline", "Usage", "Database"]


def test_dashboard_usage_is_user_only():
    dash = json.loads((MON / "vendorkb-dashboard.json").read_text())
    for expr in _exprs(dash):
        for name in ("vendorkb_mcp_request", "vendorkb_answer_", "vendorkb_questions_by_vendor"):
            if name in expr:
                assert 'client="user"' in expr, expr


def _deploy(strategy, sa=None, mounts=None, volumes=None):
    spec = {"containers": [{"name": "grafana", "volumeMounts": mounts or []}],
            "volumes": volumes or []}
    if sa:
        spec["serviceAccountName"] = sa
    return {"metadata": {"name": "x"},
            "spec": {"strategy": strategy, "template": {"spec": spec}}}


ROLLING = {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": "25%", "maxUnavailable": "25%"}}
RECREATE_OP = {"op": "add", "path": "/spec/strategy", "value": {"type": "Recreate"}}


def test_prometheus_patch_recreates_and_sets_sa():
    ops = apply.prometheus_patch(_deploy(ROLLING))
    assert RECREATE_OP in ops  # replaces the whole strategy: rollingUpdate goes
    assert {"op": "add", "path": "/spec/template/spec/serviceAccountName",
            "value": "prometheus"} in ops
    assert apply.prometheus_patch(_deploy({"type": "Recreate"}, sa="prometheus")) == []
    assert apply.prometheus_patch(_deploy(ROLLING, sa="prometheus")) == [RECREATE_OP]


def test_grafana_patch_recreates_and_mounts_once():
    ops = apply.grafana_patch(_deploy(ROLLING, mounts=[{"name": "d", "mountPath": "/x"}],
                                      volumes=[{"name": "d"}]))
    assert ops[0] == RECREATE_OP
    paths = [o["path"] for o in ops]
    assert "/spec/template/spec/volumes/-" in paths
    assert "/spec/template/spec/containers/0/volumeMounts/-" in paths
    done = _deploy({"type": "Recreate"},
                   mounts=[{"name": "dashboards-vendorkb",
                            "mountPath": "/var/lib/grafana/dashboards-vendorkb"}],
                   volumes=[{"name": "dashboards-vendorkb"}])
    assert apply.grafana_patch(done) == []


def test_prometheus_rbac_is_minimal():
    docs = list(yaml.safe_load_all((MON / "prometheus-rbac.yaml").read_text()))
    role = next(d for d in docs if d["kind"] == "ClusterRole")
    rules = sorted((tuple(r["resources"]), tuple(sorted(r["verbs"]))) for r in role["rules"])
    assert rules == [(("nodes",), ("get", "list", "watch")), (("nodes/proxy",), ("get",))]


def _panel(dash, title):
    panels = dash["panels"] + [p for row in dash["panels"] for p in row.get("panels", [])]
    (panel,) = [p for p in panels if p.get("title") == title]
    return panel


def test_questions_count_answer_question_only():
    dash = json.loads((MON / "vendorkb-dashboard.json").read_text())
    exprs = list(_exprs(_panel(dash, "Questions in 24 h")))
    assert len(exprs) == 2
    for expr in exprs:  # gather_evidence parts are not questions
        assert 'tool="answer_question"' in expr and "gather_evidence" not in expr


def test_last_success_table_shows_the_stage():
    dash = json.loads((MON / "vendorkb-dashboard.json").read_text())
    panel = _panel(dash, "Last success and failure per kind and stage")
    organize = next(t for t in panel["transformations"] if t["id"] == "organize")
    assert organize["options"]["indexByName"]["stage"] == 1
