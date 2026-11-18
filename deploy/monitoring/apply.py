"""Apply the vendorkb monitoring pieces to the hand-applied stack in namespace ``monitoring``.

    uv run python deploy/monitoring/apply.py --dry-run   # print what would change, change nothing
    uv run python deploy/monitoring/apply.py             # apply

The steps (README.md says what each one changes):
  1. apply kube-state-metrics.yaml and prometheus-rbac.yaml;
  2. switch the Prometheus Deployment to strategy Recreate and the service account
     ``prometheus``;
  3. merge scrape-vendorkb.yaml into the config map ``prometheus-config`` and reload Prometheus;
  4. create or replace the config map ``grafana-dashboards-vendorkb`` from vendorkb-dashboard.json;
  5. add the ``vendorkb`` provider to the config map ``grafana-dashboard-provider``;
  6. switch Grafana to strategy Recreate and mount ``grafana-dashboards-vendorkb`` at
     /var/lib/grafana/dashboards-vendorkb.

The config map ``grafana-dashboards`` (the other dashboards) is never touched.
"""
import argparse
import difflib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

HERE = Path(__file__).resolve().parent
NS = "monitoring"
PREFIX = "vendorkb-"
PROVIDER_NAME = "vendorkb"
DASHBOARD_DIR = "/var/lib/grafana/dashboards-vendorkb"
DASHBOARD_CM = "grafana-dashboards-vendorkb"
VOLUME = "dashboards-vendorkb"
PROM_CONFIG_PATH = "/etc/prometheus/prometheus.yml"
LABELS = {"app.kubernetes.io/part-of": "llm-monitoring"}


def _yaml() -> YAML:
    y = YAML()  # round trip: comments, quotes and key order survive
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 4096
    return y


def _dump(doc) -> str:
    buf = io.StringIO()
    _yaml().dump(doc, buf)
    return buf.getvalue()


def merge_scrape(config_yaml: str, fragment_yaml: str) -> str:
    """``config_yaml`` with every job named ``vendorkb-*`` replaced by the fragment's jobs (added
    at the end); the other jobs, ``global:`` and the comments are kept in order."""
    doc = _yaml().load(config_yaml)
    jobs = _yaml().load(fragment_yaml) or []
    bad = [j.get("job_name") for j in jobs if not str(j.get("job_name", "")).startswith(PREFIX)]
    if bad:
        raise ValueError(f"fragment jobs must be named {PREFIX}*: {bad}")
    scrape = doc.get("scrape_configs")
    if scrape is None:
        doc["scrape_configs"] = scrape = []
    for i in reversed(range(len(scrape))):
        if str(scrape[i].get("job_name", "")).startswith(PREFIX):
            del scrape[i]
    scrape.extend(jobs)
    return _dump(doc)


def merge_provider(provider_yaml: str) -> str:
    """The Grafana dashboard provider file with the ``vendorkb`` provider added or replaced; the
    other providers are kept."""
    doc = _yaml().load(provider_yaml)
    providers = doc.get("providers")
    if providers is None:
        doc["providers"] = providers = []
    ours = {"name": PROVIDER_NAME, "folder": "vendorkb", "type": "file",
            "disableDeletion": False, "editable": True, "updateIntervalSeconds": 30,
            "options": {"path": DASHBOARD_DIR, "foldersFromFilesStructure": False}}
    for i, p in enumerate(providers):
        if p.get("name") == PROVIDER_NAME:
            providers[i] = ours
            break
    else:
        providers.append(ours)
    return _dump(doc)


# --- kubectl ---------------------------------------------------------------------------------

def kubectl(*args: str, stdin: str | None = None, check: bool = True) -> str:
    proc = subprocess.run(["kubectl", "-n", NS, *args], input=stdin, text=True,
                          capture_output=True)
    if check and proc.returncode != 0:
        sys.exit(f"kubectl {' '.join(args)} failed:\n{proc.stderr}")
    return proc.stdout


def kubectl_diff(manifest: str) -> None:
    """``kubectl diff`` (a server-side dry run: read only) of a manifest on stdin."""
    proc = subprocess.run(["kubectl", "-n", NS, "diff", "-f", "-"], input=manifest, text=True,
                          capture_output=True)
    if proc.returncode > 1:
        sys.exit(f"kubectl diff failed:\n{proc.stderr}")
    print(proc.stdout or "  (no change)")


def text_diff(old: str, new: str, name: str) -> bool:
    lines = list(difflib.unified_diff(old.splitlines(keepends=True),
                                      new.splitlines(keepends=True), f"{name} (live)",
                                      f"{name} (merged)"))
    print("".join(lines) if lines else "  (no change)")
    return bool(lines)


def configmap(name: str, data: dict[str, str], labels: dict[str, str]) -> str:
    return json.dumps({"apiVersion": "v1", "kind": "ConfigMap",
                       "metadata": {"name": name, "namespace": NS, "labels": labels},
                       "data": data})


def get_json(*args: str) -> dict:
    return json.loads(kubectl("get", *args, "-o", "json"))


def container_index(deploy: dict, name: str) -> int:
    for i, c in enumerate(deploy["spec"]["template"]["spec"]["containers"]):
        if c["name"] == name:
            return i
    sys.exit(f"no container {name!r} in deployment {deploy['metadata']['name']}")


# --- the steps -------------------------------------------------------------------------------

def step_manifests(dry: bool) -> None:
    for f in ("kube-state-metrics.yaml", "prometheus-rbac.yaml"):
        print(f"== 1. {f}")
        text = (HERE / f).read_text()
        if dry:
            kubectl_diff(text)
        else:
            print(kubectl("apply", "-f", "-", stdin=text), end="")


def recreate_ops(deploy: dict) -> list[dict[str, Any]]:
    """Switch a Deployment to ``strategy: Recreate`` (dropping ``rollingUpdate``) unless it is
    already. One replica on a ReadWriteOnce volume: a rolling update would start the new pod
    while the old one still holds the data directory (the TSDB lock, Grafana's sqlite)."""
    if (deploy["spec"].get("strategy") or {}).get("type") == "Recreate":
        return []
    return [{"op": "add", "path": "/spec/strategy", "value": {"type": "Recreate"}}]


def prometheus_patch(deploy: dict) -> list[dict[str, Any]]:
    """The JSON patch for Deployment ``prometheus``: Recreate, then the service account."""
    ops = recreate_ops(deploy)
    if deploy["spec"]["template"]["spec"].get("serviceAccountName") != "prometheus":
        ops.append({"op": "add", "path": "/spec/template/spec/serviceAccountName",
                    "value": "prometheus"})
    return ops


def grafana_patch(deploy: dict) -> list[dict[str, Any]]:
    """The JSON patch for Deployment ``grafana``: Recreate, then the dashboards volume and its
    mount, each only when missing."""
    ops = recreate_ops(deploy)
    i = container_index(deploy, "grafana")
    spec = deploy["spec"]["template"]["spec"]
    volumes = spec.get("volumes") or []
    if not any(v.get("name") == VOLUME for v in volumes):
        volume = {"name": VOLUME, "configMap": {"name": DASHBOARD_CM}}
        ops.append({"op": "add", "path": "/spec/template/spec/volumes/-", "value": volume}
                   if volumes else
                   {"op": "add", "path": "/spec/template/spec/volumes", "value": [volume]})
    mounts = spec["containers"][i].get("volumeMounts") or []
    if not any(m.get("mountPath") == DASHBOARD_DIR for m in mounts):
        mount: dict[str, Any] = {"name": VOLUME, "mountPath": DASHBOARD_DIR, "readOnly": True}
        base = f"/spec/template/spec/containers/{i}/volumeMounts"
        ops.append({"op": "add", "path": base + "/-", "value": mount} if mounts
                   else {"op": "add", "path": base, "value": [mount]})
    return ops


def patch_deployment(name: str, ops: list[dict[str, Any]], dry: bool) -> None:
    print(f"  patch deployment/{name}: {json.dumps(ops)}")
    if not dry:
        print(kubectl("patch", "deployment", name, "--type", "json", "-p", json.dumps(ops)),
              end="")


def step_prometheus_sa(dry: bool) -> bool:
    print("== 2. Prometheus: strategy Recreate, service account prometheus")
    ops = prometheus_patch(get_json("deployment", "prometheus"))
    if not ops:
        print("  already Recreate with service account prometheus")
        return False
    patch_deployment("prometheus", ops, dry)
    return True


def reload_prometheus(expected: str, timeout_s: float = 180) -> None:
    """Wait for the rollout and for the mounted config to match, then POST /-/reload."""
    kubectl("rollout", "status", "deployment/prometheus", f"--timeout={int(timeout_s)}s")
    deadline = time.monotonic() + timeout_s
    while True:
        live = kubectl("exec", "deploy/prometheus", "--", "cat", PROM_CONFIG_PATH, check=False)
        if live == expected:
            break
        if time.monotonic() > deadline:
            sys.exit(f"the mounted {PROM_CONFIG_PATH} did not update within {timeout_s:.0f} s")
        time.sleep(5)
    kubectl("exec", "deploy/prometheus", "--", "wget", "-qO-", "--post-data=",
            "http://localhost:9090/-/reload")
    print("  reloaded")


def step_scrape(dry: bool) -> None:
    print("== 3. prometheus-config scrape jobs")
    cm = get_json("configmap", "prometheus-config")
    old = cm["data"]["prometheus.yml"]
    new = merge_scrape(old, (HERE / "scrape-vendorkb.yaml").read_text())
    changed = text_diff(old, new, "prometheus.yml")
    if dry:
        if changed:
            print("  then: POST /-/reload once the pod sees the new file")
        return
    if changed:
        print(kubectl("apply", "-f", "-", stdin=configmap(
            "prometheus-config", {"prometheus.yml": new},
            cm["metadata"].get("labels") or LABELS)), end="")
    reload_prometheus(new)


def step_dashboard(dry: bool) -> None:
    print(f"== 4. config map {DASHBOARD_CM}")
    dash = (HERE / "vendorkb-dashboard.json").read_text()
    json.loads(dash)  # refuse a broken file
    manifest = configmap(DASHBOARD_CM, {"vendorkb.json": dash},
                         {"app.kubernetes.io/name": "grafana", **LABELS})
    if dry:
        kubectl_diff(manifest)
    else:
        print(kubectl("apply", "-f", "-", stdin=manifest), end="")


def step_provider(dry: bool) -> bool:
    print("== 5. grafana-dashboard-provider")
    cm = get_json("configmap", "grafana-dashboard-provider")
    old = cm["data"]["dashboard-provider.yml"]
    new = merge_provider(old)
    changed = text_diff(old, new, "dashboard-provider.yml")
    if changed and not dry:
        print(kubectl("apply", "-f", "-", stdin=configmap(
            "grafana-dashboard-provider", {"dashboard-provider.yml": new},
            cm["metadata"].get("labels") or LABELS)), end="")
    return changed


def step_grafana_mount(dry: bool, provider_changed: bool) -> None:
    print(f"== 6. Grafana: strategy Recreate, mount {DASHBOARD_DIR}")
    ops = grafana_patch(get_json("deployment", "grafana"))
    restarts = any(o["path"].startswith("/spec/template/") for o in ops)
    if ops:
        patch_deployment("grafana", ops, dry)
    else:
        print("  already Recreate with the mount")
    if provider_changed and not restarts:
        # Grafana reads the provider file at start only
        print("  provider changed: rollout restart deployment/grafana")
        if not dry:
            kubectl("rollout", "restart", "deployment/grafana")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print the diffs, change nothing")
    dry = ap.parse_args(argv).dry_run
    step_manifests(dry)
    step_prometheus_sa(dry)
    step_scrape(dry)
    step_dashboard(dry)
    step_grafana_mount(dry, step_provider(dry))
    print("dry run: nothing changed" if dry else "done")


if __name__ == "__main__":
    main()
