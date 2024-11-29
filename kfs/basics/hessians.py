"""Implementation of Hessian tensors and matrices."""

from typing import Tuple, Union

from torch import (
    Tensor,
    arange,
    unravel_index,
    zeros,
    zeros_like,
)

from kfs.basics.flattening import cvec, rvec
from kfs.basics.hessian_product import hvp


def hess(
    f: Tensor, x: Union[Tensor, Tuple[Tensor, Tensor]]
) -> Tensor:
    r"""Compute Hessian tensor of a tensor-to-scalar function.

    See $\text{\Cref{def:general_hessian}}$.

    Args:
        f: The function whose Hessian is computed.
        x: The variable w.r.t. which the Hessian is taken.
            If x is a tuple, the mixed Hessian is computed
            (see $\text{\Cref{sec:basics_dl_hessian}}$).

    Returns:
        The Hessian tensor of f w.r.t. x with shape
        (*x.shape, *x.shape). If x was a tuple, returns the
        mixed Hessian of shape (*x[0].shape, *x[1].shape).
    """
    x1, x2 = (x, x) if isinstance(x, Tensor) else x
    H = zeros(x1.shape + x2.shape)

    for d in arange(x1.numel()):
        d_unraveled = unravel_index(d, x1.shape)
        one_hot_d = zeros_like(x1)
        one_hot_d[d_unraveled] = 1.0
        H[d_unraveled] = hvp(
            f, (x2, x1), one_hot_d, retain_graph=True
        )

    return H


def vec_hess(
    f: Tensor, x: Union[Tensor, Tuple[Tensor, Tensor]], vec: str
) -> Tensor:
    r"""Compute the rvec- or cvec-Hessian of f w.r.t. x.

    See $\text{\Cref{def:cvec_hessian}}$ and $\text{\Cref{def:rvec_hessian}}$.

    Args:
        f: The function whose rvec- or cvec-Hessian is computed.
        x: Variable w.r.t. which the rvec- or cvec-Hessian is taken.
            If x is a tuple, the mixed cvec-Hessian is
            computed.
        vec: Name of the flattening scheme.
            Must be either 'rvec' or 'cvec'.

    Returns:
        The rvec- or cvec-Hessian of f w.r.t. x. Has shape
        (x.numel(), x.numel()). If x was a tuple, the
        result has shape (x[0].numel(), x[1].numel()).
    """
    vec = {"cvec": cvec, "rvec": rvec}[vec]
    H = hess(f, x)
    # flatten row indices
    row_ndim = (
        x.ndim if isinstance(x, Tensor) else x[0].ndim
    )
    H = vec(H, end_dim=row_ndim - 1)
    # flatten column indices
    return vec(H, start_dim=1)
