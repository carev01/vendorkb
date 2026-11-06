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
