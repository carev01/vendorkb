from vendorkb.chunker import Chunk, approx_tokens, chunk_markdown


def para(n_tokens: int, word: str = "word") -> str:
    # ~n_tokens tokens at chars/4: each "word " is 5 chars
    return " ".join([word] * max(1, (n_tokens * 4) // 5))


def test_heading_trail_becomes_section_path():
    md = "## A\n\n### B\n\n" + para(400) + "\n"
    chunks = chunk_markdown("T", md)
    assert any(c.section_path == ("T", "A", "B") for c in chunks)
    assert all(isinstance(c, Chunk) for c in chunks)


def test_text_is_prefixed_with_title_and_path():
    md = "## A\n\n### B\n\n" + para(400) + "\n"
    chunks = chunk_markdown("T", md)
    c = next(c for c in chunks if c.section_path == ("T", "A", "B"))
    assert c.text.startswith("T\nA > B\n\n")
    assert c.token_count == approx_tokens(c.text)


def test_no_headings_gives_one_path_title_chunk():
    md = para(400) + "\n\n" + para(400) + "\n"
    chunks = chunk_markdown("T", md)
    assert len(chunks) >= 1
    assert all(c.section_path == ("T",) for c in chunks)


def test_single_h1_only_still_chunks():
    chunks = chunk_markdown("T", "# T\n\n" + para(400) + "\n")
    assert len(chunks) == 1
    assert chunks[0].section_path == ("T",)


def test_leading_h1_other_than_title_is_in_path():
    chunks = chunk_markdown("T", "# Other\n\n" + para(400) + "\n")
    assert chunks[0].section_path == ("T", "Other")


def test_tiny_sections_merge_with_their_parent():
    md = "## P\n\n" + "".join(f"### S{i}\n\n{para(20)}\n\n" for i in range(3))
    chunks = chunk_markdown("T", md)
    assert len(chunks) == 1
    assert "S2" in chunks[0].text


def test_oversized_section_splits_on_paragraphs():
    md = "## Big\n\n" + "\n\n".join(para(100, f"word{i:02d}") for i in range(30)) + "\n"
    chunks = chunk_markdown("T", md)
    assert len(chunks) >= 3
    assert all(c.token_count <= 1200 for c in chunks)
    assert all(c.section_path == ("T", "Big") for c in chunks)


def _table(n_rows: int, cell: str = "x" * 90, header=("| a | b |", "|---|---|")) -> str:
    return "\n".join([*header, *[f"| row{i} | {cell} |" for i in range(n_rows)]])


def test_large_table_splits_into_row_groups_with_header_repeated():
    md = "## Data\n\n" + _table(400) + "\n"
    chunks = chunk_markdown("T", md)
    assert len(chunks) > 1
    for c in chunks:
        assert approx_tokens(c.text) <= 1200
        body = c.text.split("\n\n", 1)[1]
        assert body.startswith("| a | b |\n|---|---|\n| row")
    rows = [line for c in chunks for line in c.text.splitlines() if line.startswith("| row")]
    assert rows == [f"| row{i} | {'x' * 90} |" for i in range(400)]


def test_small_table_stays_whole_with_its_section():
    md = "## Data\n\n" + para(100) + "\n\n" + _table(5) + "\n"
    chunks = chunk_markdown("T", md)
    assert len(chunks) == 1
    assert chunks[0].text.count("| a | b |") == 1


def test_table_without_separator_line_splits_without_repeating_a_header():
    md = "## Data\n\n" + _table(400, header=()) + "\n"
    chunks = chunk_markdown("T", md)
    assert len(chunks) > 1
    assert all(approx_tokens(c.text) <= 1200 for c in chunks)
    assert sum(c.text.count("| row") for c in chunks) == 400


def test_single_table_row_over_budget_is_hard_split():
    md = "## Data\n\n" + _table(3, cell="y" * 20000) + "\n"
    chunks = chunk_markdown("T", md)
    assert all(approx_tokens(c.text) <= 1200 for c in chunks)
    assert sum(c.text.count("y") for c in chunks) == 3 * 20000


def test_headings_inside_code_fences_are_ignored():
    md = "## Real\n\n```\n# not a heading\n## nor this\n```\n\n" + para(400) + "\n"
    chunks = chunk_markdown("T", md)
    assert all(c.section_path == ("T", "Real") for c in chunks)
    assert "# not a heading" in "".join(c.text for c in chunks)


def test_ordinals_are_dense_from_zero():
    md = "".join(f"## S{i}\n\n{para(700)}\n\n" for i in range(4))
    chunks = chunk_markdown("T", md)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len(chunks) >= 4


def test_setext_headings_build_section_path():
    md = (
        "Amazon S3 backups\n=================\n\n"
        "Overview\n--------\n\n" + para(400) + "\n\n"
        "Details\n=======\n\n" + para(400) + "\n"
    )
    chunks = chunk_markdown("amazon s3 backups", md)
    paths = [c.section_path for c in chunks]
    assert ("amazon s3 backups", "Overview") in paths
    assert ("amazon s3 backups", "Details") in paths
    assert all("Amazon S3 backups" not in p for p in paths)


def test_horizontal_rule_is_not_a_heading():
    md = para(400) + "\n\n---\n\n" + para(400) + "\n"
    chunks = chunk_markdown("T", md)
    assert all(c.section_path == ("T",) for c in chunks)
    assert "---" in "".join(c.text for c in chunks)


def test_title_only_chunk_prefix_does_not_repeat_title():
    chunks = chunk_markdown("T", para(400) + "\n")
    assert chunks[0].text.startswith("T\n\n")
    assert not chunks[0].text.startswith("T\nT\n")
