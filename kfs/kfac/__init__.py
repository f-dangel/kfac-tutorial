"""A didactic implementation of KFAC."""

# register functionality to compute KFAC-expand on Linear layers
from kfs.kfac import expand_Linear  # noqa:
from kfs.kfac.scaffold import KFAC

__all__ = ["KFAC"]
