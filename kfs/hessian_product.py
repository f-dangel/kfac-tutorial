"""Hessian multiplication routine (Pearlmutter trick)."""

from torch import Tensor
from torch.autograd import grad


def hvp(
    f: Tensor,
    x: Tensor,
    v: Tensor,
    retain_graph: bool = False,
) -> Tensor:
    r"""Multiply v with the Hessian of f w.r.t. x.

    See $\text{\Cref{def:hvp}}$.

    Args:
        f: The function whose Hessian is multiplied with.
        x: The variable w.r.t. which the Hessian is taken.
        v: The vector to be multiplied with the Hessian.
            Must have the same shape as x.
        retain_graph: Whether to retain the graph which will
            allow future differentiations of f. Default: False.

    Returns:
        The Hessian-vector product (H_x f) @ v of same shape
        as v.
    """
    assert f.numel() == 1  # f must be a scalar-valued
    (grad_f,) = grad(f, x, create_graph=True)
    v_dot_grad_f = (v * grad_f).sum()
    (result,) = grad(
        v_dot_grad_f, x, retain_graph=retain_graph
    )
    return result
