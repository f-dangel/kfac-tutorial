"""A didactic implementation of KFAC."""

# register functionality to compute KFAC-expand/reduce on
# Linear/ConvNd layers
from kfs.kfac import (  # noqa: F401
    expand_ConvNd,
    expand_Linear,
    reduce_ConvNd,
    reduce_Linear,
)
from kfs.kfac.scaffold import KFAC

__all__ = ["KFAC"]
