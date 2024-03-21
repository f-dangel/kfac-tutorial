"""GGN-vector multiplication."""

from torch import (
    Tensor,
    allclose,
    arange,
    eye,
    kron,
    manual_seed,
    outer,
    rand,
    rand_like,
    unravel_index,
    zeros,
    zeros_like,
)
from torch.nn import Linear, MSELoss

from kfs.hessian_product import hvp
from kfs.jacobian_products import jvp, vjp


def ggnvp(
    f: Tensor,
    x: Tensor,
    linearize: Tensor,
    v: Tensor,
    retain_graph: bool = False,
) -> Tensor:
    """Multiply v with the GGN of f w.r.t. x.

    The GGN corresponds to the Hessian of a function that
    uses a linearization at linearize w.r.t. x.
    """
    Jv = jvp(linearize, x, v, retain_graph=retain_graph)
    HJ_v = hvp(f, linearize, Jv, retain_graph=retain_graph)
    JHJT_v = vjp(
        linearize, x, HJ_v, retain_graph=retain_graph
    )
    return JHJT_v


def ggn(f: Tensor, x: Tensor, linearize: Tensor) -> Tensor:
    G = zeros(x.shape + x.shape)

    for d in arange(x.numel()):
        d_unraveled = unravel_index(d, x.shape)
        one_hot_d = zeros_like(x)
        one_hot_d[d_unraveled] = 1.0
        G[d_unraveled] = ggnvp(
            f, x, linearize, one_hot_d, retain_graph=True
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
