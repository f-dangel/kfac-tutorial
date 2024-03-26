"""Hessian multiplication routine (Pearlmutter trick)."""

from typing import Tuple, Union

from torch import Tensor
from torch.autograd import grad


def hvp(
    f: Tensor,
    x: Union[Tensor, Tuple[Tensor, Tensor]],
    v: Tensor,
    retain_graph: bool = False,
    detach: bool = True,
) -> Tensor:
    r"""Multiply v with the Hessian of f w.r.t. x.

    See $\text{\Cref{def:hvp}}$.

    Args:
        f: The function whose Hessian is multiplied with.
        x: The variable w.r.t. which the Hessian is taken.
            If x is a tuple, multiplication with the mixed
            Hessian ($\text{\Cref{sec:basics_dl_hessian}}$)
            is computed.
        v: The vector to be multiplied with the Hessian.
            Must have the same shape as x (or x[1] if x is
            a tuple).
        retain_graph: Whether to retain the graph which will
            allow future differentiations of f. Default: False.
        detach: Whether to detach the result from the graph.
            Default: True.

    Returns:
        The Hessian-vector product (H_x f) @ v of same shape
        as v. If x was a tuple, the result the shape of x[0].
    """
    x1, x2 = (x, x) if isinstance(x, Tensor) else x
    assert f.numel() == 1  # f must be a scalar-valued
    (grad_f,) = grad(f, x2, create_graph=True)
    v_dot_grad_f = (v * grad_f).sum()
    (result,) = grad(
        v_dot_grad_f, x1, retain_graph=retain_graph
    )
    return result.detach() if detach else result
