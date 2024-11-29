"""Jacobian implementations."""

from torch import (
    Tensor,
    arange,
    unravel_index,
    zeros,
    zeros_like,
)

from kfs.basics.flattening import cvec, rvec
from kfs.basics.jacobian_products import vjp


def jac(f: Tensor, x: Tensor) -> Tensor:
    r"""Compute the general Jacobian tensor of f w.r.t. x.

    See $\text{\Cref{def:general_jacobian}}$.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.

    Returns:
        Jacobian tensor of f w.r.t. x. Has shape
        (*f.shape, *x.shape).
    """
    J = zeros(f.shape + x.shape)

    for d in arange(f.numel()):
        d_unraveled = unravel_index(d, f.shape)
        one_hot_d = zeros_like(f)
        one_hot_d[d_unraveled] = 1.0
        J[d_unraveled] = vjp(
            f, x, one_hot_d, retain_graph=True
        )

    return J


def vec_jac(f: Tensor, x: Tensor, vec: str) -> Tensor:
    r"""Compute the Jacobian in row- or column-flattening convention.

    See $\text{\Cref{def:cvec_jacobian}}$ and $\text{\Cref{def:rvec_jacobian}}$.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.
        vec: Name of the flattening scheme.
            Must be either 'rvec' or 'cvec'.

    Returns:
        The rvec- or cvec-Jacobian matrix of f w.r.t. x.
        Has shape (f.numel(), x.numel()).
    """
    vec = {"cvec": cvec, "rvec": rvec}[vec]
    J = jac(f, x)
    # flatten row indices
    J = vec(J, end_dim=f.ndim - 1)
    # flatten column indices
    return vec(J, start_dim=1)
