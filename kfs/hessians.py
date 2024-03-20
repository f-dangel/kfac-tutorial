"""Implementation of Hessian tensors and matrices."""

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
    r"""Compute Hessian tensor of a tensor-to-scalar function.

    See $\text{\Cref{def:general_hessian}}$.

    Args:
        f: The function whose Hessian is computed.
        x: The variable w.r.t. which the Hessian is taken.

    Returns:
        The Hessian tensor of f w.r.t. x.
        Has shape (*x.shape, *x.shape).
    """
    H = zeros(x.shape + x.shape)

    for d in arange(x.numel()):
        d_unraveled = unravel_index(d, x.shape)
        one_hot_d = zeros_like(x)
        one_hot_d[d_unraveled] = 1.0
        H[d_unraveled] = hvp(
            f, x, one_hot_d, retain_graph=True
        )

    return H


def cvec_hess(f: Tensor, x: Tensor) -> Tensor:
    r"""Compute the cvec-Hessian of f w.r.t. x.

    See $\text{\Cref{def:cvec_hessian}}$.

    Args:
        f: The function whose cvec-Hessian is computed.
        x: Variable w.r.t. which the cvec-Hessian is taken.

    Returns:
        The cvec-Hessian of f w.r.t. x.
        Has shape (x.numel(), x.numel()).
    """
    H = hess(f, x)
    # flatten row indices
    H = cvec(H, end_dim=x.ndim - 1)
    # flatten column indices
    return cvec(H, start_dim=1)


def rvec_hess(f: Tensor, x: Tensor) -> Tensor:
    r"""Compute the rvec-Hessian of f w.r.t. x.

    See $\text{\Cref{def:rvec_hessian}}$.

    Args:
        f: The function whose rvec-Hessian is computed.
        x: Variable w.r.t. which the rvec-Hessian is taken.

    Returns:
        The rvec-Hessian of f w.r.t. x.
        Has shape (x.numel(), x.numel()).
    """
    H = hess(f, x)
    # flatten row indices
    H = rvec(H, end_dim=x.ndim - 1)
    # flatten column indices
    return rvec(H, start_dim=1)
