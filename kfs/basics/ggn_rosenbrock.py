"""Compute the Rosenbrock function's GGN."""

from functools import partial
from math import sqrt
from typing import Tuple, Union

from torch import Tensor, allclose, cat, manual_seed, rand

from kfs.basics.ggns import ggn
from kfs.basics.hessians import hess
from kfs.basics.linearization import linearize


def rosenbrock_first(x: Tensor, alpha: float):
    """The first sub-function of the Rosenbrock function.

    Args:
        x: A two-dimensional vector.
        alpha: The Rosenbrock function'ss parameter.

    Returns:
        A two-dimensional vector containing the evaluation.
    """
    assert x.ndim == 1 and x.numel() == 2
    g0, g1 = 1 - x[[0]], sqrt(alpha) * (
        x[[1]] - x[[0]] ** 2
    )
    return cat([g0, g1])


def rosenbrock_last(g: Tensor) -> Tensor:
    """The last sub-function of the Rosenbrock function.

    Args:
        g: A two-dimensional vector containing the evaluation
            of the first sub-function.

    Returns:
        The evaluation of the last sub-function (a scalar).
    """
    assert g.ndim == 1 and g.numel() == 2
    return (g**2).sum()


def rosenbrock(
    x: Tensor, alpha: float, return_first: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Evaluate the Rosenbrock function.

    Args:
        x: A two-dimensional vector.
        alpha: The Rosenbrock function's parameter.
        return_first: Whether to return the evaluation of the
            first sub-function as well. Defaults to False.

    Returns:
        The evaluation of the Rosenbrock function.
        If return_first is True, a tuple containing the
        evaluation of the last sub-function and the first
        sub-function is returned.
    """
    assert x.ndim == 1 and x.numel() == 2
    first = rosenbrock_first(x, alpha)
    last = rosenbrock_last(first)
    return last, first if return_first else last


def rosenbrock_partially_linearized(
    x: Tensor, x_prime: Tensor, alpha: float
) -> Tensor:
    """Evaluate the partially linearized Rosenbrock function.

    Args:
        x: A two-dimensional vector.
        x_prime: Point at which to linearize the first
            composite.
        alpha: The Rosenbrock function's parameter.

    Returns:
        The evaluation of the linearized Rosenbrock function.
    """
    assert x_prime.ndim == 1 and x_prime.numel() == 2
    rosenbrock_first_linearized = linearize(
        partial(rosenbrock_first, alpha=alpha), x_prime
    )
    g_linearized = rosenbrock_first_linearized(x)
    return rosenbrock_last(g_linearized)


if __name__ == "__main__":
    manual_seed(0)
    x = rand(2, requires_grad=True)
    alpha = 1.0

    value, lin = rosenbrock(x, alpha, return_first=True)

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
    G = ggn(value, x, lin)

    G_manual = 2 * Tensor(
        [
            [
                1 + 4 * alpha * x[0] ** 2,
                -2 * alpha * x[0],
            ],
            [-2 * alpha * x[0], alpha],
        ]
    )
    assert allclose(G, G_manual)

    # compute GGN via Hessian of linearized function
    x_prime = x.clone().detach().requires_grad_(True)
    value = rosenbrock_partially_linearized(
        x, x_prime, alpha
    )
    H_lin = hess(value, x)
    assert allclose(H_lin, G_manual)
