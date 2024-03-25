"""Function linearization."""

from typing import Callable

from einops import einsum
from torch import Tensor, allclose, cos, manual_seed, rand

from kfs.jacobians import jac


def linearize(
    f: Callable[[Tensor], Tensor], x0: Tensor
) -> Callable[[Tensor], Tensor]:
    r"""Linearize a tensor-to-tensor function.

    See $\text{\Cref{def:tensor_linearization}}$.

    Args:
        f: The function to linearize.
        x0: The anchor point of the linearized function.

    Returns:
        The linearized tensor-to-tensor function.
    """
    g0 = f(x0)
    J0 = jac(g0, x0)
    g0, J0 = g0.detach(), J0.detach()
    dims = " ".join([f"d{i}" for i in range(x0.ndim)])
    equation = f"... {dims}, {dims} -> ..."
    return lambda x: g0 + einsum(J0, x - x0, equation)


if __name__ == "__main__":
    manual_seed(0)

    shape = (2, 3, 4)
    x = rand(shape, requires_grad=True)
    x0 = x.clone().detach().requires_grad_(True)

    f = cos
    f_lin = linearize(f, x0)

    assert allclose(f(x), f_lin(x0))
    assert allclose(jac(f(x), x), jac(f_lin(x), x))
