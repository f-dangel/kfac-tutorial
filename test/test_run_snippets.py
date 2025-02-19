"""Execute all files in `kfs`, creating their stdout."""

from glob import glob
from os import path
from subprocess import CalledProcessError, run

from pytest import mark

HEREDIR = path.dirname(path.abspath(__file__))
REPODIR = path.dirname(HEREDIR)
CODEDIR = path.join(REPODIR, "kfs")

# find all python files in the code directory
py_files = glob(
    path.join(CODEDIR, "**", "*.py"), recursive=True
)


@mark.parametrize(
    "snippet",
    py_files,
    ids=[f.replace(REPODIR, "")[1:] for f in py_files],
)
def test_run_snippets(snippet: str):
    """Execute a snippet.

    Args:
        snippet: Path to the snippet.

    Raises:
        CalledProcessError: If the snippet fails to run.
    """
    cmd = ["python", snippet]
    if snippet.endswith(
        "linearized_rosenbrock.py"
    ) or snippet.endswith("synthetic_hessian.py"):
        cmd.append("--disable_tex")

    try:
        print(f"Running command: {' '.join(cmd)}")
        job = run(
            cmd, capture_output=True, text=True, check=True
        )
        print(f"STDOUT:\n{job.stdout}")
        print(f"STDERR:\n{job.stderr}")
    except CalledProcessError as e:
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise e
