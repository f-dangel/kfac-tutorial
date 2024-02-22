from torch import (
    Tensor,
    arange,
    unravel_index,
    zeros,
    zeros_like,
)

from kfs.flattening import cvec, rvec
from kfs.hessian_product import hvp


def hess(f: Tensor, x: Tensor) -> Tensor:
    H = zeros(x.shape + x.shape)

    for d in arange(x.numel()):
        d_unraveled = unravel_index(d, x.shape)
        one_hot_d = zeros_like(x)
        one_hot_d[d_unraveled] = 1.0
        H[d_unraveled] = hvp(f, x, one_hot_d, retain_graph=True)

    return H


def cvec_hess(f: Tensor, x: Tensor):
    H = hess(f, x)
    # flatten row indices
    H = cvec(H, end_dim=x.ndim - 1)
    # flatten column indices
    return cvec(H, start_dim=1)


def rvec_hess(f: Tensor, x: Tensor):
    H = hess(f, x)
    # flatten row indices
    H = rvec(H, end_dim=x.ndim - 1)
    # flatten column indices
    return rvec(H, start_dim=1)
