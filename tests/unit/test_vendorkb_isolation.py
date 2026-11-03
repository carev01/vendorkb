import ast
from pathlib import Path

FORBIDDEN = {"graphiti_core", "neo4j", "graph_extract", "answer_api", "graph_sync", "theme_builder"}


def test_vendorkb_imports_nothing_from_the_graph_stack():
    root = Path(__file__).resolve().parents[2] / "src" / "vendorkb"
    files = list(root.rglob("*.py"))
    assert files
    bad = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            bad += [(path.name, n) for n in names if n.split(".")[0] in FORBIDDEN]
    assert not bad
