"""Generalized Gauss-Newton tensor and matrices."""

from typing import Tuple, Union

from torch import (
    Tensor,
    arange,
    unravel_index,
    zeros,
    zeros_like,
)

from kfs.basics.flattening import cvec, rvec
from kfs.basics.ggn_product import ggnvp


def ggn(
    f: Tensor,
    x: Union[Tensor, Tuple[Tensor, Tensor]],
    h: Tensor,
) -> Tensor:
    r"""Compute the GGN of f linearized at h w.r.t. x .

    See $\text{\Cref{def:vector_ggn}}$.

    Args:
        f: The function whose GGN is multiplied with.
        x: The input of the function. If x is a tuple, the
            mixed GGN is computed.
        h: The output of the sub-function at which
            the dependency on x is linearized.

    Returns:
        The GGN tensor of f linearized at h w.r.t. x.
        Has shape x.shape + x.shape. If x was a tuple,
        returns the mixed GGN of shape
        (*x[0].shape, *x[1].shape).
    """
    x1, x2 = (x, x) if isinstance(x, Tensor) else x
    (dtype,) = {x1.dtype, x2.dtype}
    G = zeros(x1.shape + x2.shape, dtype=dtype)

    for d in arange(x1.numel()):
        d_unraveled = unravel_index(d, x1.shape)
        one_hot_d = zeros_like(x1)
        one_hot_d[d_unraveled] = 1.0
        G[d_unraveled] = ggnvp(
            f, (x2, x1), h, one_hot_d, retain_graph=True
        )  # ggnvp is defined in Snippet 15

    return G


def vec_ggn(
    f: Tensor,
    x: Union[Tensor, Tuple[Tensor, Tensor]],
    h: Tensor,
    vec: str,
) -> Tensor:
    r"""Compute the GGN matrix of f linearized at h w.r.t. x.

    See $\text{\Cref{def:rvec_ggn,def:cvec_ggn}}$.

    Args:
        f: The function whose GGN is multiplied with.
        x: The input of the function. If x is a tuple, the
            mixed GGN is computed.
        h: The output of the sub-function at which
            the dependency on x is linearized.
        vec: Name of the flattening scheme.
            Must be either 'rvec' or 'cvec'.

    Returns:
        The rvec- or cvec-GGN matrix of f linearized at h
        w.r.t. x. Has shape (x.numel(), x.numel()).
        If x was a tuple, returns the mixed GGN of shape
        (x[0].numel(), x[1].numel()).
    """
    vec = {"cvec": cvec, "rvec": rvec}[vec]
    G = ggn(f, x, h)
    # flatten row indices
    row_ndim = (
        x.ndim if isinstance(x, Tensor) else x[0].ndim
    )
    G = vec(G, end_dim=row_ndim - 1)
    # flatten column indices
    return vec(G, start_dim=1)
