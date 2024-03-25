"""GGN-vector multiplication."""

from torch import Tensor

from kfs.hessian_product import hvp
from kfs.jacobian_products import jvp, vjp


def ggnvp(
    f: Tensor,
    x: Tensor,
    g: Tensor,
    v: Tensor,
    retain_graph: bool = False,
) -> Tensor:
    r"""Multiply v with the GGN of f linearized at g w.r.t. x.

    The GGN corresponds to the Hessian of a function that
    uses a linearization at linearize w.r.t. x.

    See $\text{\Cref{def:ggnvp}}$.

    Args:
        f: The function whose GGN is multiplied with.
        x: The input of the function.
        g: The output of the sub-function at which
            the dependency on x is linearized.
        v: The vector that is multiplied with the GGN.
            Has same shape as x.
        retain_graph: Whether to retain the computation graph
            of f for future differentiation. Default: False.

    Returns:
        The product of the GGN of f w.r.t. x with v.
        Has same shape as v.
    """
    Jv = jvp(g, x, v, retain_graph=retain_graph)
    HJ_v = hvp(f, g, Jv, retain_graph=retain_graph)
    JTHJ_v = vjp(g, x, HJ_v, retain_graph=retain_graph)
    return JTHJ_v
