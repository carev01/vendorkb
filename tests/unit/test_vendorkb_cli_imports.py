"""``vendorkb sync`` runs in a 1 Gi pod: the CLI module must not drag in the claims and
taxonomy stacks (sklearn, scipy, numpy, anthropic) until their commands run."""
import subprocess
import sys

HEAVY = ("sklearn", "scipy", "numpy", "anthropic", "vendorkb.taxonomy", "vendorkb.claims")


def test_importing_the_cli_skips_the_heavy_stacks():
    code = ("import sys, vendorkb.cli\n"
            f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == ""


def test_the_answer_path_skips_the_numeric_stacks():
    # `vendorkb answer` imports the topics code on first use (vendorkb.answer.load_topics): the
    # taxonomy loader and normalize must come without scikit-learn, scipy and numpy
    code = ("import sys, vendorkb.cli, vendorkb.topics, vendorkb.taxonomy.load\n"
            "print(','.join(m for m in ('sklearn', 'scipy', 'numpy') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == ""


def test_normalize_is_one_function_wherever_it_is_imported():
    from vendorkb import text
    from vendorkb.taxonomy import cluster
    assert cluster.normalize is text.normalize
    assert text.normalize("  Immutable\tBackups \n") == "immutable backups"
