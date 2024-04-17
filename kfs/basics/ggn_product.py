"""GGN-vector multiplication."""

from typing import Tuple, Union

from torch import Tensor

from kfs.basics.hessian_product import hvp
from kfs.basics.jacobian_products import jvp, vjp


def ggnvp(
    f: Tensor,
    x: Union[Tensor, Tuple[Tensor, Tensor]],
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
        x: The input of the function. If x is a tuple, the
            mixed GGN is multiplied onto v.
        g: The output of the sub-function at which
            the dependency on x is linearized.
        v: The vector that is multiplied with the GGN.
            Has same shape as x (or x[1] if it is a tuple).
        retain_graph: Whether to retain the computation graph
            of f for future differentiation. Default: False.

    Returns:
        The product of the GGN of f w.r.t. x with v.
        Has same shape as v and x. If x was a tuple, has the
        same shape as x[1].
    """
    x1, x2 = (x, x) if isinstance(x, Tensor) else x
    Jv = jvp(g, x2, v, retain_graph=retain_graph)
    HJ_v = hvp(f, g, Jv, retain_graph=retain_graph)
    JTHJ_v = vjp(g, x1, HJ_v, retain_graph=retain_graph)
    return JTHJ_v
