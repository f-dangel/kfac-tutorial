"""Jacobians and Jacobian multiplications routines."""

from torch import (
    Tensor,
    arange,
    unravel_index,
    zeros,
    zeros_like,
)
from torch.autograd import grad

from kfs.flattening import cvec, rvec


def vjp(
    f: Tensor,
    x: Tensor,
    v: Tensor,
    create_graph: bool = False,
    detach=True,
) -> Tensor:
    """Multiply the transpose Jacobian of f w.r.t. x onto v.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.
        v: Tensor to be multiplied with the Jacobian.
           Has same shape as f(x).
        create_graph: If True, keep the computation graph for
            higher-order differentiation. Default: False.
        detach: If True, detach the result from the computation graph.
            Default: True.

    Returns:
        Vector-Jacobian product v @ (J_x f).T with shape of x.
    """
    (result,) = grad(
        f, x, grad_outputs=v, create_graph=create_graph
    )
    return result.detach() if detach else result


def jvp(
    f: Tensor,
    x: Tensor,
    v: Tensor,
    create_graph: bool = False,
    detach=True,
) -> Tensor:
    """Multiply the Jacobian of f w.r.t. x onto v.

    Uses the double-backward trick from TODO

    Let g(u) = (J_x f).T @ u.
    Then (J_u g).T @ v evaluated at u = 0 equals (J_x f) @ v.
    """
    u = zeros_like(f, requires_grad=True)
    (ujp,) = grad(f, x, grad_outputs=u, create_graph=True)
    (result,) = grad(
        ujp, x, grad_outputs=v, create_graph=create_graph
    )
    return result.detach() if detach else result


def jac(f: Tensor, x: Tensor) -> Tensor:
    J = zeros(f.shape + x.shape)

    for d in arange(f.numel()):
        d_unraveled = unravel_index(d, f.shape)
        one_hot_d = zeros_like(f)
        one_hot_d[d_unraveled] = 1.0
        J[d_unraveled] = vjp(
            f, x, one_hot_d, create_graph=True
        )

    return J


def cvec_jac(f: Tensor, x: Tensor) -> Tensor:
    J = jac(f, x)
    # flatten indices of f
    J = cvec(J, end_dim=f.ndim - 1)
    # flatten indices of x
    return cvec(J, start_dim=1)


def rvec_jac(f: Tensor, x: Tensor) -> Tensor:
    J = jac(f, x)
    # flatten indices of f
    J = rvec(J, end_dim=f.ndim - 1)
    # flatten indices of x
    return rvec(J, start_dim=1)
