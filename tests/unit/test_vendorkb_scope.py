import json
import sys

from vendorkb.scope import ScopeIndex, detect_scope, load_scope_index, product_surfaces

ALIASES = {
    "VBR": "Veeam Backup & Replication",
    "Amazon": "AWS",
    "Veeam Agent": ["Veeam Agent for Linux", "Veeam Agent for Mac"],
}
VENDORS = ["Veeam", "AWS", "Microsoft"]
PRODUCTS = ["Veeam Backup & Replication", "Veeam Backup for Microsoft 365", "AWS Backup"]
SOURCE_PRODUCTS = {
    "Veeam Agent for Linux": ["Veeam Backup & Replication"],
    "Veeam Agent for Mac": ["Veeam Backup & Replication"],
    "User Guide": ["AWS Backup"],
}
SOURCE_VENDORS = {
    "Veeam Agent for Linux": "Veeam",
    "Veeam Agent for Mac": "Veeam",
    "User Guide": "AWS",
}


def _index(tmp_path, source_products=None, source_vendors=None):
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(ALIASES))
    return load_scope_index(VENDORS, PRODUCTS, p, source_products, source_vendors)


def test_alias_vbr_maps_to_product(tmp_path):
    v, p = detect_scope("How does VBR handle retention?", _index(tmp_path))
    assert p == ["Veeam Backup & Replication"]
    assert v == []


def test_alias_to_vendor(tmp_path):
    v, p = detect_scope("What does Amazon offer?", _index(tmp_path))
    assert v == ["AWS"]
    assert p == []


def test_multi_target_alias_veeam_agent_returns_all_targets(tmp_path):
    sp = {"Veeam Agent for Linux": ["Veeam Backup & Replication"],
          "Veeam Agent for Mac": ["AWS Backup"]}
    v, p = detect_scope("How do I install Veeam Agent?", _index(tmp_path, sp))
    assert p == ["Veeam Backup & Replication", "AWS Backup"]
    assert v == []


def test_multi_target_alias_single_product_after_dedupe(tmp_path):
    v, p = detect_scope("How do I install Veeam Agent?", _index(tmp_path, SOURCE_PRODUCTS))
    assert p == ["Veeam Backup & Replication"]


def test_directly_named_source_maps_to_products(tmp_path):
    idx = _index(tmp_path, SOURCE_PRODUCTS, SOURCE_VENDORS)
    v, p = detect_scope("Back up with Veeam Agent for Linux", idx)
    assert p == ["Veeam Backup & Replication"]
    assert v == []


def test_generic_source_name_is_not_a_surface(tmp_path):
    idx = _index(tmp_path, SOURCE_PRODUCTS, SOURCE_VENDORS)
    v, p = detect_scope("Where is the User Guide for AWS Backup?", idx)
    assert p == ["AWS Backup"]
    assert v == []
    # nothing but the generic name: still unscoped
    assert detect_scope("Open the User Guide", idx) == ([], [])


def test_unknown_alias_target_ignored(tmp_path):
    idx = _index(tmp_path)  # no source_products: Agent targets are unknown
    assert detect_scope("Veeam Agent setup", idx)[1] == []


def test_longest_match_wins(tmp_path):
    v, p = detect_scope("Veeam Backup for Microsoft 365 restore", _index(tmp_path))
    assert p == ["Veeam Backup for Microsoft 365"]
    assert "Microsoft" not in v


def test_case_insensitive_whole_word(tmp_path):
    idx = _index(tmp_path)
    assert detect_scope("how does aws work", idx)[0] == ["AWS"]
    assert detect_scope("the laws of backup", idx) == ([], [])


def test_cross_vendor_wording_is_unscoped(tmp_path):
    idx = _index(tmp_path)
    assert detect_scope("Compare AWS Backup across vendors", idx) == ([], [])
    assert detect_scope("Which vendor supports AWS immutability?", idx) == ([], [])


def test_single_vendor_across_regions_stays_scoped(tmp_path):
    v, p = detect_scope("How do I copy AWS Backup recovery points across regions?",
                        _index(tmp_path))
    assert p == ["AWS Backup"]
    assert v == []


def test_dedupe_preserves_first_match_order(tmp_path):
    v, p = detect_scope("VBR and Veeam Backup & Replication", _index(tmp_path))
    assert p == ["Veeam Backup & Replication"]


def test_aliases_file_is_read_as_data():
    before = {m for m in sys.modules if m.startswith("answer_api")}
    idx = load_scope_index(["Veeam"], ["Veeam Backup & Replication"], None)
    assert idx.aliases["VBR"] == ["Veeam Backup & Replication"]
    assert isinstance(idx, ScopeIndex)
    assert {m for m in sys.modules if m.startswith("answer_api")} == before


def test_a_generic_product_name_needs_its_vendor(tmp_path):
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps({"Entra Backup and Recovery": "Entra ID Backup"}))
    idx = load_scope_index(["Microsoft", "Storware", "Cohesity"],
                           ["Entra ID Backup", "Backup and Recovery", "Security Center"], p,
                           product_vendors={"Entra ID Backup": "Microsoft",
                                            "Backup and Recovery": "Storware",
                                            "Security Center": "Cohesity"})
    # alone, "Backup and Recovery" names nobody's product in particular
    assert detect_scope("How long can backups be kept with Backup and Recovery?", idx) == ([], [])
    assert detect_scope("Does Storware Backup and Recovery keep 90 days?", idx) == (
        [], ["Backup and Recovery"])
    assert detect_scope("What does Cohesity Security Center report?", idx) == (
        [], ["Security Center"])
    # the cv-12 wording: Microsoft's Entra product, not Storware's
    assert detect_scope("Microsoft's built-in Entra Backup and Recovery", idx) == (
        ["Microsoft"], ["Entra ID Backup"])


def test_product_surfaces_are_the_product_names_scope_detection_reads(tmp_path):
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(ALIASES))
    idx = load_scope_index([*VENDORS, "Storware"], [*PRODUCTS, "Backup and Recovery"], p,
                           SOURCE_PRODUCTS, SOURCE_VENDORS,
                           product_vendors={"Backup and Recovery": "Storware"})
    surfaces = product_surfaces(idx)
    # product names, aliases of products, and distinctive sources (which stand for products)
    for name in (*PRODUCTS, "VBR", "Veeam Agent", "Veeam Agent for Linux"):
        assert name in surfaces, name
    # a generic product name only after its vendor's name, as find_facets reads it
    assert "Storware Backup and Recovery" in surfaces and "Backup and Recovery" not in surfaces
    # vendor names and their aliases are no product surfaces; nor is a shared source name
    for name in (*VENDORS, "Storware", "Amazon", "User Guide"):
        assert name not in surfaces, name
