# <img alt="KFAC" src="./tex/figures/logo.pdf" height="90"> KFAC From Scratch – A Tutorial

This repository contains a self-contained introduction to Kronecker-factored approximate curvature (KFAC) in math and code ([arXiv paper](https://arxiv.org/abs/2507.05127)).

*We hope this tutorial is a helpful resource for both newcomers to the field who want to learn more about curvature matrices,
their approximations, and common pitfalls, as well as experienced researchers who are seeking a pedagogical introduction
and implementation they can use as a starting point to prototype their research idea.*

**We invite anyone to contribute to this fully open-source effort to further improve the tutorial over time.**

## Getting Started

The paper contains code snippets you can modify and execute as you read.
To do that, follow these steps:

```bash
# 1) Clone the repository
git clone git@github.com:f-dangel/kfac-tutorial.git

# 2) Navigate into the repository's directory
cd kfac-tutorial

# 3) Install the package and dependencies in editable mode
#    so you can modify and run the code without re-installing
pip install -e .
#    (if you use conda, you can instead run
#        `make conda-env`
#    and activate the environment with
#        `conda activate kfs`
#    )

# 4) Verify the installation by executing one of the files
python kfs/basics/flattening.py
#    (you can run and modify any file in the kfs directory)
```

## Citing

If you find this repository useful for your work, consider citing the arXiv paper

```bib

@article{dangel2025kroneckerfactored,
  title =        {Kronecker-factored Approximate Curvature (KFAC) from Scratch},
  author =       {Dangel, Felix and Mucsányi, Bálint and Weber, Tobias and
  Eschenhagen, Runa},
  journal =      {arXiv},
  url =          {https://github.com/f-dangel/kfac-tutorial},
  year =         2025,
}

```

# Developer Guide

This guide describes principles and workflows for contributors.

## Setup

We recommend programming in a fresh virtual environment. You can set up the
`conda` environment and activate it

```bash
make conda-env
conda activate kfs
```

If you don't use `conda`, set up your preferred environment and run

```bash
pip install -e ."[lint,test]"
```
to install the package in editable mode, along with all required development dependencies
(the quotes are for OS compatibility, see
[here](https://github.com/mu-editor/mu/issues/852#issuecomment-498759372)).

## Continuous integration

To standardize code style and enforce high quality, checks are carried out with
Github actions when you push. You can also run them locally, as they are managed
via `make`:

- Run tests with `make test`

- Run all linters with `make lint`, or separately with:

    - Run auto-formatting and import sorting with `make black` and `make isort`

    - Run linting with `make flake8`

    - Run docstring checks with `make pydocstyle-check` and `make darglint-check`

## Documentation

We use the [Google docstring
convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
