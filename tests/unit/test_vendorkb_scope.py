import json
import sys

from vendorkb.scope import ScopeIndex, detect_scope, load_scope_index

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
}


def _index(tmp_path, source_products=None):
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(ALIASES))
    return load_scope_index(VENDORS, PRODUCTS, p, source_products)


def test_alias_vbr_maps_to_product(tmp_path):
    v, p = detect_scope("How does VBR handle retention?", _index(tmp_path))
    assert p == ["Veeam Backup & Replication"]
    assert v == []


def test_alias_to_vendor(tmp_path):
    v, p = detect_scope("What does Amazon offer?", _index(tmp_path))
    assert v == ["AWS"]
    assert p == []


def test_multi_target_alias_veeam_agent_returns_all_targets(tmp_path):
    idx = _index(tmp_path, SOURCE_PRODUCTS)
    v, p = detect_scope("How do I install Veeam Agent?", idx)
    assert p == ["Veeam Backup & Replication"]


def test_directly_named_source_maps_to_products(tmp_path):
    v, p = detect_scope("Back up with Veeam Agent for Linux", _index(tmp_path, SOURCE_PRODUCTS))
    assert p == ["Veeam Backup & Replication"]
    assert v == ["Veeam"] or v == []


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
    assert v == ["AWS"] or v == []


def test_dedupe_preserves_first_match_order(tmp_path):
    v, p = detect_scope("VBR and Veeam Backup & Replication", _index(tmp_path))
    assert p == ["Veeam Backup & Replication"]


def test_aliases_file_is_read_as_data():
    before = {m for m in sys.modules if m.startswith("answer_api")}
    idx = load_scope_index(["Veeam"], ["Veeam Backup & Replication"], None)
    assert idx.aliases["VBR"] == ["Veeam Backup & Replication"]
    assert isinstance(idx, ScopeIndex)
    assert {m for m in sys.modules if m.startswith("answer_api")} == before
