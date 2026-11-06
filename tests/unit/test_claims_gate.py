from rapidfuzz import fuzz

from vendorkb.claims import gate

CHUNK = ("Protecting Databases / Amazon RDS\n\n"
         "Backups of Amazon RDS for PostgreSQL instances are supported with immutable "
         "snapshots. The “retention lock” — once set — can’t be removed.")

# The reviewer's probe chunk (task-1-review.md, I4).
PROBE = ("Storage targets\n\n"
         "Immutable backups are supported for Amazon S3 and S3-compatible storage with "
         "object lock enabled. Protection covers VMware vSphere and Microsoft Hyper-V virtual "
         "machines, including application-consistent snapshots, instant recovery and "
         "file-level restore.")

TABLE = "| Feature | Supported |\n|---|---|\n| **Object lock** | Yes |\n| Tape | No |"


def test_exact_substring():
    assert gate.check("Amazon RDS for PostgreSQL instances", CHUNK) == (True, "exact")


def test_case_and_whitespace_are_normalized():
    assert gate.check("  amazon   RDS for\n postgresql  ", CHUNK) == (True, "exact")


def test_curly_quotes_and_dashes_are_normalized():
    quote = 'The "retention lock" - once set - can\'t be removed.'
    assert gate.check(quote, CHUNK) == (True, "exact")


def test_normalize_unifies_quotes_dashes_case_whitespace():
    assert gate.normalize("“A”–‘b’  \tC") == "\"a\"-'b' c"


def test_normalize_drops_zero_width_and_soft_hyphen_and_maps_ellipsis():
    assert gate.normalize("im­mu​table…") == "immutable..."


def test_evidence_span_is_verbatim_chunk_text():
    r = gate.match('the "RETENTION lock" - once set', CHUNK)
    assert r.ok and r.reason == "exact"
    assert r.span == "The “retention lock” — once set"


def test_punctuation_insensitive_exact_matches_table_rows_and_markdown():
    r = gate.match("Object lock: Yes", TABLE)
    assert (r.ok, r.reason) == (True, "exact-nopunct")
    assert r.span == "Object lock** | Yes"


def test_fuzzy_with_every_word_present_is_accepted():
    quote = "amazon rds postgresql instances are supported"
    assert fuzz.partial_ratio(quote, gate.normalize(CHUNK)) > 90
    r = gate.match(quote, CHUNK)
    assert r.ok and r.reason.startswith("fuzzy:")
    assert r.span == "Amazon RDS for PostgreSQL instances are supported"


def test_fuzzy_at_80_is_rejected():
    quote = "axazon rdx fxr pxstgresqk"
    assert fuzz.partial_ratio(quote, gate.normalize(CHUNK)) == 80.0
    assert gate.check(quote, CHUNK) == (False, "no-match:80.0")


def test_fuzzy_above_90_with_a_substituted_word_is_rejected():
    quote = "amazon rdx for postgresqk"
    assert fuzz.partial_ratio(quote, gate.normalize(CHUNK)) == 92.0
    ok, reason = gate.check(quote, CHUNK)
    assert not ok and reason.startswith("no-match:92.0")


def test_negation_flip_is_rejected():
    quote = ("Immutable backups are not supported for Amazon S3 and S3-compatible storage "
             "with object lock enabled.")
    assert fuzz.partial_ratio(gate.normalize(quote), gate.normalize(PROBE)) >= 90
    ok, reason = gate.check(quote, PROBE)
    assert not ok and "not" in reason


def test_target_swap_azure_blob_is_rejected():
    quote = ("Immutable backups are supported for Azure Blob and S3-compatible storage with "
             "object lock enabled.")
    assert fuzz.partial_ratio(gate.normalize(quote), gate.normalize(PROBE)) >= 90
    assert gate.check(quote, PROBE)[0] is False


def test_target_swap_nutanix_is_rejected():
    quote = ("Protection covers VMware vSphere and Nutanix AHV virtual machines, including "
             "application-consistent snapshots, instant recovery and file-level restore.")
    assert fuzz.partial_ratio(gate.normalize(quote), gate.normalize(PROBE)) >= 90
    assert gate.check(quote, PROBE)[0] is False


def test_empty_quote_is_rejected():
    assert gate.check("", CHUNK) == (False, "empty")
    assert gate.check(" \n\t ", CHUNK) == (False, "empty")


def test_short_quotes_are_rejected():
    assert gate.check("Yes", TABLE) == (False, "too-short")
    assert gate.check("Supported", TABLE) == (False, "too-short")
    assert gate.check("Object lock", TABLE) == (False, "too-short")  # 2 words, 11 chars
    # three words, or fifteen characters, is enough
    assert gate.check("Feature Supported Object", TABLE)[0] is True
    assert gate.check("PostgreSQL instances", CHUNK) == (True, "exact")  # 20 chars


def test_quote_longer_than_chunk_is_never_fuzzy_accepted():
    ok, reason = gate.check(CHUNK + " and also it supports everything else", CHUNK)
    assert not ok and reason.startswith("no-match:")


LIMITS = ("Retention\n\nRetention for immutable backups can be configured for up to 7 years on "
          "Amazon S3 storage. File-level restore requires an agent on the guest virtual "
          "machine. Instant recovery has no support for encrypted virtual machine disks in "
          "this release.")


def _fuzzy_candidate(quote: str) -> None:
    assert fuzz.partial_ratio(gate.normalize(quote), gate.normalize(LIMITS)) >= 90


def test_numeric_swap_is_rejected():
    quote = ("Retention for immutable backups can be configured for up to 5 years on Amazon S3 "
             "storage.")
    _fuzzy_candidate(quote)
    ok, reason = gate.check(quote, LIMITS)
    assert not ok and "5" in reason


def test_inserted_short_negation_is_rejected():
    quote = "File-level restore requires no agent on the guest virtual machine."
    _fuzzy_candidate(quote)
    ok, reason = gate.check(quote, LIMITS)
    assert not ok and "no" in reason


def test_dropped_negation_is_rejected():
    quote = ("Instant recovery has support for encrypted virtual machine disks in this "
             "release.")
    _fuzzy_candidate(quote)
    ok, reason = gate.check(quote, LIMITS)
    assert not ok and "negation" in reason


def test_fuzzy_with_matching_numbers_and_negation_is_accepted():
    quote = ("Instant recovery has no support for encrypted virtual machine disks in this "
             "release")  # no full stop; same words
    assert gate.match(quote, LIMITS).ok
    quote = "immutable backups can be configured up to 7 years on Amazon S3 storage"
    _fuzzy_candidate(quote)
    r = gate.match(quote, LIMITS)
    assert r.ok and r.reason.startswith("fuzzy:")


def test_exact_match_inside_a_word_is_not_exact():
    # "supported" inside "unsupported": no exact match; the fuzzy path sees the negation
    chunk = "Agents\n\nInstant recovery is unsupported on Linux servers."
    ok, reason = gate.check("supported on Linux servers", chunk)
    assert not ok and not reason.startswith("exact")
    r = gate.match("mutable backups are supported for Amazon S3", PROBE)
    assert not r.ok
    r = gate.match("recovery is unsupported on Linux servers", chunk)
    assert (r.ok, r.reason, r.span) == (True, "exact", "recovery is unsupported on Linux servers")


def test_nopunct_match_inside_a_word_is_not_exact():
    chunk = "| Instant recovery | unsupported on Linux |"
    ok, reason = gate.check("supported: on Linux", chunk)
    assert not ok and not reason.startswith("exact")
    assert gate.check("Instant recovery: unsupported on Linux", chunk) == (True, "exact-nopunct")


def test_exact_match_may_start_or_end_on_punctuation():
    r = gate.match('"retention lock" - once set', CHUNK)
    assert r.ok and r.reason == "exact"
    assert r.span == "\u201cretention lock\u201d \u2014 once set"
