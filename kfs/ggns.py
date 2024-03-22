"""Generalized Gauss-Newton tensor and matrices."""

from torch import (
    Tensor,
    allclose,
    arange,
    eye,
    kron,
    manual_seed,
    outer,
    rand,
    unravel_index,
    zeros,
    zeros_like,
)
from torch.nn import Linear, MSELoss

from kfs.ggn_product import ggnvp


def ggn(f: Tensor, x: Tensor, g: Tensor) -> Tensor:
    r"""Compute the GGN of f linearized at g w.r.t. x .

    See $\text{\Cref{sec:ggn}}$.

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


if __name__ == "__main__":
    manual_seed(0)
    x = rand(3)
    y = rand(2)
    model = Linear(3, 2, bias=False)
    loss_func = MSELoss(reduction="sum")

    f = model(x)
    w = model.weight
    loss = loss_func(f, y)

    # v = rand_like(w)
    # Gv = ggnvp(loss, w, f, v)
    # print(Gv)
    G = ggn(loss, w, f)
    print(G)
    G = G.reshape(w.numel(), w.numel())
    # print(G)

    G_manual = kron(2 * eye(2), outer(x, x))
    print(G_manual)
    print(allclose(G, G_manual))
