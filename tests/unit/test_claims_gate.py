from rapidfuzz import fuzz

from vendorkb.claims import gate

CHUNK = ("Protecting Databases / Amazon RDS\n\n"
         "Backups of Amazon RDS for PostgreSQL instances are supported with immutable "
         "snapshots. The “retention lock” — once set — can’t be removed.")


def test_exact_substring():
    assert gate.check("Amazon RDS for PostgreSQL instances", CHUNK) == (True, "exact")


def test_case_and_whitespace_are_normalized():
    assert gate.check("  amazon   RDS for\n postgresql  ", CHUNK) == (True, "exact")


def test_curly_quotes_and_dashes_are_normalized():
    quote = 'The "retention lock" - once set - can\'t be removed.'
    assert gate.check(quote, CHUNK) == (True, "exact")


def test_normalize_unifies_quotes_dashes_case_whitespace():
    assert gate.normalize("“A”–‘b’  \tC") == "\"a\"-'b' c"


def test_fuzzy_at_92_is_accepted():
    quote = "amazon rdx for postgresqk"
    assert fuzz.partial_ratio(quote, gate.normalize(CHUNK)) == 92.0
    ok, reason = gate.check(quote, CHUNK)
    assert ok and reason == "fuzzy:92.0"


def test_fuzzy_at_80_is_rejected():
    quote = "axazon rdx fxr pxstgresqk"
    assert fuzz.partial_ratio(quote, gate.normalize(CHUNK)) == 80.0
    assert gate.check(quote, CHUNK) == (False, "no-match:80.0")


def test_empty_quote_is_rejected():
    assert gate.check("", CHUNK) == (False, "empty")
    assert gate.check(" \n\t ", CHUNK) == (False, "empty")


def test_quote_longer_than_chunk_is_never_fuzzy_accepted():
    ok, reason = gate.check(CHUNK + " and also it supports everything else", CHUNK)
    assert not ok and reason.startswith("no-match:")
