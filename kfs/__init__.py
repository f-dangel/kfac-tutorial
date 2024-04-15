"""KFAC from scratch (KFS) library."""

# register functionality to compute KFAC-expand on Linear layers
from kfs import kfac_expand_Linear  # noqa:
from kfs.kfac import KFAC

__all__ = ["KFAC"]
