"""Jacobian multiplication routines."""

from torch import Tensor, zeros_like
from torch.autograd import grad


def vjp(
    f: Tensor,
    x: Tensor,
    v: Tensor,
    retain_graph: bool = False,
    detach=True,
) -> Tensor:
    r"""Multiply the transpose Jacobian of f w.r.t. x onto v.

    See $\text{\Cref{def:vjp}}$.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.
        v: Tensor to be multiplied with the Jacobian.
            Has same shape as f(x).
        retain_graph: If True, keep the computation graph of
            f for future differentiation. Default: False.
        detach: If True, detach the result from the
            computation graph. Default: True.

    Returns:
        Vector-Jacobian product v @ (J_x f).T with shape of x.
    """
    (result,) = grad(
        f, x, grad_outputs=v, retain_graph=retain_graph
    )
    return result.detach() if detach else result


def jvp(
    f: Tensor,
    x: Tensor,
    v: Tensor,
    retain_graph: bool = False,
    detach=True,
) -> Tensor:
    r"""Multiply the Jacobian of f w.r.t. x onto v.

    See $\text{\Cref{def:jvp}}$.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.
        v: Tensor to be multiplied with the Jacobian.
            Has same shape as x.
        create_graph: If True, keep the computation graph for
            higher-order differentiation. Default: False.
        detach: If True, detach the result from the
            computation graph. Default: True.

    Returns:
        Jacobian-Vector product (J_x f) @ v with shape of f.
    """
    u = zeros_like(f, requires_grad=True)
    (ujp,) = grad(f, x, grad_outputs=u, create_graph=True)
    (result,) = grad(
        ujp, u, grad_outputs=v, retain_graph=retain_graph
    )
    return result.detach() if detach else result
