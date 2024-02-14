"""Execute all files in `kfs`, creating their stdout."""

from os import path
from subprocess import run

HEREDIR = path.dirname(path.abspath(__file__))
REPODIR = path.dirname(HEREDIR)
SCRIPT = path.join(REPODIR, "tex", "execute_code_blocks.py")


def test_execute_code_blocks():
    """Dummy test."""
    run(["python", SCRIPT], capture_output=True, check=True)
