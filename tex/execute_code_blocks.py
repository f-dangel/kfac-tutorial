"""Run all `.py` files in `kfs` as main and save their output to `output`.

Output files are stored in the same file tree as the `.py` files.
"""

from glob import glob
from os import makedirs, path, remove
from subprocess import run

HEREDIR = path.dirname(path.abspath(__file__))
REPODIR = path.dirname(HEREDIR)
CODEDIR = path.join(REPODIR, "kfs")
OUTDIR = path.join(HEREDIR, "output")

# remove all existing output files
makedirs(OUTDIR, exist_ok=True)
for out_file in glob(path.join(OUTDIR, "**", "*.txt"), recursive=True):
    print(f"Removing {out_file!r}")
    remove(out_file)

# find all python files in the code directory
py_files = glob(path.join(CODEDIR, "**", "*.py"), recursive=True)
# generate names of output files
out_files = [
    f"{path.splitext(py_file)[0]}.txt".replace(CODEDIR, OUTDIR) for py_file in py_files
]
# generate folders
for out_file in out_files:
    makedirs(path.dirname(out_file), exist_ok=True)

# execute them and write their output to a file
for py_file, out_file in zip(py_files, out_files):
    print(f"Executing {py_file!r}")
    job = run(["python", py_file], capture_output=True, text=True, check=True)
    with open(out_file, "w") as f:
        f.write(job.stdout)
