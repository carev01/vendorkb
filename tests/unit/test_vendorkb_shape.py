from pathlib import Path

import yaml

from vendorkb.scope import load_scope_index
from vendorkb.shape import Shape, detect_shape

VENDORS = [
    "Acronis", "Afi.ai", "Arcserve", "Arctera", "Atempo", "AvePoint", "AWS", "Axcient", "Bacula",
    "Barracuda", "CloudAlly", "CloudCasa", "Cohesity", "Commvault", "Datto", "Dell", "Druva", "Eon",
    "Flosum", "Gearset", "Google", "GRAX", "Hornetsecurity (Altaro)", "HYCU", "IBM", "Keepit",
    "Microsoft", "N-Able", "Nakivo", "NinjaOne (Dropsuite)", "Portworx", "Red Hat", "Rubrik",
    "Salesforce", "Slide", "Spanning", "Storware", "Synology", "Trilio", "Unitrends", "Veeam",
    "Velero", "Zerto"]
PRODUCTS = ["AWS Backup", "Azure Backup", "Microsoft 365 Backup", "Entra ID Backup", "FortKnox",
            "Veeam Backup & Replication", "Veeam Backup for Microsoft 365", "Veeam Data Cloud"]
INDEX = load_scope_index(VENDORS, PRODUCTS)
QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "questions.yaml"

ENUM = {"ex-rt-10", "ex-rt-12", *(f"cv-{i:02d}" for i in range(1, 10))}
COMPARE = {"ex-rt-07", "ex-rt-08", "ex-rt-09", *(f"cv-{i:02d}" for i in range(10, 16))}


def _load():
    return yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))


def test_eval_shapes():
    qs = _load()
    by_id = {q["id"]: q for q in qs}
    wrong = []
    for qid in ENUM | COMPARE:
        want = "enumeration" if qid in ENUM else "comparison"
        got = detect_shape(by_id[qid]["question"], INDEX).kind
        if got != want:
            wrong.append((qid, want, got))
    assert not wrong, wrong
    singles = [q for q in qs if q["type"] == "single_vendor"]
    bad = [q["id"] for q in singles if detect_shape(q["question"], INDEX).kind != "single"]
    assert len(bad) <= 0.05 * len(singles), f"misclassified single_vendor ids: {bad}"


def test_other_vendors_besides_one_is_enumeration():
    s = detect_shape("Do other vendors besides Veeam support tape?", INDEX)
    assert s == Shape("enumeration", [("vendor", "Veeam")])


def test_product_and_its_vendor_is_one_comparison_side():
    s = detect_shape("Compare AWS Backup and Azure Backup", INDEX)
    assert s.kind == "comparison"
    assert s.facets == [("product", "AWS Backup"), ("product", "Azure Backup")]
    one = detect_shape("How does AWS Backup in AWS handle retention?", INDEX)
    assert one.kind == "single"
