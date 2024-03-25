"""Generalized Gauss-Newton tensor and matrices."""

from torch import (
    Tensor,
    arange,
    unravel_index,
    zeros,
    zeros_like,
)

from kfs.flattening import cvec, rvec
from kfs.ggn_product import ggnvp


def ggn(f: Tensor, x: Tensor, g: Tensor) -> Tensor:
    r"""Compute the GGN of f linearized at g w.r.t. x .

    See $\text{\Cref{def:vector_ggn}}$.

    Args:
        f: The function whose GGN is multiplied with.
        x: The input of the function.
        g: The output of the sub-function at which
            the dependency on x is linearized.

    Returns:
        The GGN tensor of f linearized at g w.r.t. x.
        Has shape x.shape + x.shape.
    """
    G = zeros(x.shape + x.shape)

    for d in arange(x.numel()):
        d_unraveled = unravel_index(d, x.shape)
        one_hot_d = zeros_like(x)
        one_hot_d[d_unraveled] = 1.0
        G[d_unraveled] = ggnvp(
            f, x, g, one_hot_d, retain_graph=True
        )

    return G


def vec_ggn(
    f: Tensor, x: Tensor, g: Tensor, vec: str
) -> Tensor:
    r"""Compute the GGN matrix of f linearized at g w.r.t. x.

    See $\text{\Cref{def:rvec_ggn,def:cvec_ggn}}$.

    Args:
        f: The function whose GGN is multiplied with.
        x: The input of the function.
        g: The output of the sub-function at which
            the dependency on x is linearized.
        vec: Name of the flattening scheme.
            Must be either 'rvec' or 'cvec'.

    Returns:
        The rvec- or cvec-GGN matrix of f linearized at g
        w.r.t. x. Has shape (x.numel(), x.numel()).
    """
    vec = {"cvec": cvec, "rvec": rvec}[vec]
    G = ggn(f, x, g)
    # flatten row indices
    H = vec(G, end_dim=x.ndim - 1)
    # flatten column indices
    return vec(H, start_dim=1)
