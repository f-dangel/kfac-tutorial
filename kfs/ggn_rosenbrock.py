"""Compute the Rosenbrock function's GGN."""

from math import sqrt

from torch import (
    Tensor,
    allclose,
    arange,
    cat,
    manual_seed,
    rand,
    randn,
    zeros,
    zeros_like,
)

from kfs.flattening import rvec
from kfs.ggn_product import ggn
from kfs.hessians import hess
from kfs.jacobians import jac


def g(x, alpha):
    g0, g1 = 1 - x[[0]], sqrt(alpha) * (
        x[[1]] - x[[0]] ** 2
    )
    return cat([g0, g1])


def l(g):
    return (g**2).sum()


def rosenbrock(x, alpha):
    assert x.ndim == 1 and x.numel() == 2
    g_value = g(x, alpha)
    return l(g_value), g_value


def rosenbrock_partially_linearized(x, x_prime, alpha):
    assert x_prime.ndim == 1 and x_prime.numel() == 2

    g_prime = g(x_prime, alpha)
    J_prime = jac(g_prime, x_prime)

    g_linearized = g_prime + J_prime @ (x - x_prime)

    return l(g_linearized), g_linearized


if __name__ == "__main__":
    manual_seed(0)
    x = rand(2, requires_grad=True)
    alpha = 1.0

    value, linearize = rosenbrock(x, alpha)

    # compute Hessian
    H = hess(value, x)

    H_manual = Tensor(
        [
            [
                2
                + 12 * alpha * x[0] ** 2
                - 4 * alpha * x[1],
                -4 * alpha * x[0],
            ],
            [-4 * alpha * x[0], 2 * alpha],
        ]
    )
    assert allclose(H, H_manual)

    # compute GGN
    G = ggn(value, x, linearize)

    G_manual = 2 * Tensor(
        [
            [
                1 + 4 * alpha * x[0] ** 2,
                -2 * sqrt(alpha) * x[0],
            ],
            [-2 * sqrt(alpha) * x[0], alpha],
        ]
    )
    assert allclose(G, G_manual)

    # compute GGN via Hessian of linearized function
    x_prime = x.clone().detach().requires_grad_(True)
    value, _ = rosenbrock_partially_linearized(
        x, x_prime, alpha
    )
    H_lin = hess(value, x)
    assert allclose(H_lin, G_manual)
