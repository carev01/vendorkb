from pathlib import Path

import pytest
import yaml

from vendorkb.scope import detect_scope, load_scope_index
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
PRODUCTS += ["Own", "Veeam Service Provider Console", "Veeam Kasten for Kubernetes"]
PRODUCT_VENDORS = {
    "AWS Backup": "AWS", "Azure Backup": "Microsoft", "Microsoft 365 Backup": "Microsoft",
    "Entra ID Backup": "Microsoft", "FortKnox": "Cohesity",
    "Veeam Backup & Replication": "Veeam", "Veeam Backup for Microsoft 365": "Veeam",
    "Veeam Data Cloud": "Veeam", "Veeam Service Provider Console": "Veeam",
    "Veeam Kasten for Kubernetes": "Veeam", "Own": "Salesforce"}
INDEX = load_scope_index(VENDORS, PRODUCTS, product_vendors=PRODUCT_VENDORS)
QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "questions.yaml"


def _load():
    return yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))


def test_eval_shapes():
    qs = _load()
    labelled = [q for q in qs if q.get("shape")]
    assert sum(q["shape"] == "enumeration" for q in labelled) == 23
    assert sum(q["shape"] == "comparison" for q in labelled) == 9
    wrong = [(q["id"], q["shape"], detect_shape(q["question"], INDEX).kind) for q in labelled
             if detect_shape(q["question"], INDEX).kind != q["shape"]]
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


def test_from_wording_is_a_comparison():
    s = detect_shape("Compare Exchange backup from Veeam and from Cohesity", INDEX)
    assert s == Shape("comparison", [("vendor", "Veeam"), ("vendor", "Cohesity")])
    s = detect_shape("How does restore from Veeam compare to restore from Rubrik?", INDEX)
    assert s.kind == "comparison"


def test_vendor_after_for_is_still_a_side():
    s = detect_shape("Which backup is better for Veeam: Rubrik or Cohesity?", INDEX)
    assert s.kind == "comparison"
    assert {f[1] for f in s.facets} == {"Veeam", "Rubrik", "Cohesity"}


def test_two_vendors_without_comparison_wording_is_single():
    s = detect_shape("Does Veeam back up Microsoft SQL Server?", INDEX)
    assert s.kind == "single"
    assert [f[1] for f in s.facets] == ["Veeam", "Microsoft"]


def test_two_products_is_a_comparison_without_wording():
    s = detect_shape("Does AWS Backup and Azure Backup support SAP HANA?", INDEX)
    assert s.kind == "comparison"


def test_vendor_alias_owns_its_product():
    s = detect_shape("Compare Azure Backup with Microsoft's retention", INDEX)
    assert s.kind == "single"
    assert s.facets == [("product", "Azure Backup")]


def test_product_wording_enumerates_only_without_facets():
    assert detect_shape("Which products can protect Kubernetes workloads?", INDEX) == Shape(
        "enumeration", [])
    assert detect_shape("Which of these vendors support tape?", INDEX).kind == "enumeration"
    s = detect_shape("Which products does Veeam offer for Kubernetes?", INDEX)
    assert s == Shape("single", [("vendor", "Veeam")])
    assert detect_scope("Which products does Veeam offer for Kubernetes?", INDEX) == (
        ["Veeam"], [])


def test_same_vendor_products_and_different_are_single():
    by_id = {q["id"]: q for q in _load()}
    for qid in ("ma-a-07", "ma-b-02", "ma-b-03"):
        assert detect_shape(by_id[qid]["question"], INDEX).kind == "single", qid
    q = "Is the Enterprise edition a different build from the free one on Google GKE?"
    assert detect_shape(q, INDEX).kind == "single"


def test_difference_between_is_comparison_wording():
    s = detect_shape("What is the difference between Veeam and Rubrik retention?", INDEX)
    assert s.kind == "comparison"


def test_which_version_is_single():
    assert detect_shape("Which version of VBR supports immutable repositories?", INDEX).kind == "single"


def test_sweep_wording_with_one_facet():
    # product facet -> single
    q = "Which tools can I use to keep an eye on AWS Backup jobs?"
    assert detect_shape(q, INDEX) == Shape("single", [("product", "AWS Backup")])
    # vendor preceded by does/from/by/of/in/at -> single
    for q in ("Which tools does Veeam ship for tape?", "Which products from Veeam do this?",
              "Which consoles by Veeam support MFA?", "Which tools in Veeam do X?"):
        assert detect_shape(q, INDEX).kind == "single", q
    # vendor as a workload -> enumeration
    q = "Which products can back up our Google Workspace?"
    assert detect_shape(q, INDEX) == Shape("enumeration", [("vendor", "Google")])
    assert detect_shape("Which tools can back up Salesforce metadata?", INDEX).kind == "enumeration"
    # original vendors rule unchanged
    assert detect_shape("Which vendors does Veeam compete with?", INDEX).kind == "enumeration"


def test_ordinary_word_names_match_case_sensitively():
    by_id = {q["id"]: q for q in _load()}
    assert detect_shape(by_id["cv-09"]["question"], INDEX).kind == "enumeration"
    assert detect_shape(by_id["sv-b-10"]["question"], INDEX).kind == "single"
    assert detect_shape("Does Own back up Salesforce metadata?", INDEX).facets[0] == ("product", "Own")
    assert ("product", "Own") not in detect_shape("I run my own scripts", INDEX).facets
    assert detect_shape("does veeam support tape?", INDEX).facets == [("vendor", "Veeam")]


# --- final fix wave: I-1, I-2, I-3 -------------------------------------------------------

@pytest.mark.parametrize("q", [
    "Which platforms can Cohesity protect?",
    "Which products can Veeam back up?",
    "Which platforms can a Rubrik cluster protect?",
    "What tools do I need for Commvault IntelliSnap?",
    "Which tools can I use to monitor Rubrik backups?",
    "What software do I need to install on the Veeam proxy?",
    "Which software versions does the Nakivo transporter support?",
    "With Commvault, which tools let me monitor jobs?",
    "Which storage vendors does Veeam Backup & Replication integrate with for snapshots?",
])
def test_one_facet_sweep_or_vendors_wording_is_single(q):
    assert detect_shape(q, INDEX).kind == "single", q


@pytest.mark.parametrize("q", [
    "Do other vendors besides Veeam support tape?",
    "Which vendors except Rubrik support tape?",
    "Which vendors can back up Salesforce metadata?",
    "Which tools can back up Microsoft Exchange?",
    "Which products protect Cohesity clusters?",
    "Which tools work for our Nakivo deployment?",
])
def test_one_facet_workload_or_protected_vendor_is_enumeration(q):
    assert detect_shape(q, INDEX).kind == "enumeration", q


@pytest.mark.parametrize("q", [
    "What products does Commvault offer for Microsoft 365?",
    "Which of these vendors support immutable backups: Veeam, Rubrik, Cohesity?",
    "Which vendors among Veeam and Rubrik support tape?",
])
def test_two_or_more_facets_are_never_enumeration(q):
    s = detect_shape(q, INDEX)
    assert s.kind == "single", (q, s)
    assert len(s.facets) >= 2


@pytest.mark.parametrize("q", [
    "Can Veeam Backup & Replication back up both Hyper-V and Azure VMs?",
    "How do I configure retention for each job in Veeam Backup & Replication when backing up"
    " Azure VMs?",
    "Can each Commvault MediaAgent use Azure Blob storage?",
])
def test_each_and_both_are_not_comparison_wording(q):
    assert detect_shape(q, INDEX).kind == "single", q


def test_two_facets_with_comparison_wording_stay_comparison():
    assert detect_shape("Is Veeam better than Rubrik for tape?", INDEX).kind == "comparison"
    assert detect_shape("Which one handles tape: Veeam or Rubrik?", INDEX).kind == "comparison"
