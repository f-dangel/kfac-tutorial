"""Compute the GGN of linear regression."""

from einops import einsum
from torch import allclose, eye, kron, manual_seed, rand
from torch.nn import Linear, MSELoss

from kfs.basics.ggns import vec_ggn

if __name__ == "__main__":
    manual_seed(0)  # make deterministic

    # set up model, data, and cost function
    D_in, D_out = 5, 3
    N = 10
    X = rand(N, D_in)
    y = rand(N, D_out)
    model = Linear(D_in, D_out, bias=False)
    loss_func = MSELoss(reduction="sum")

    # set up the computation graph
    f = model(X)
    W = model.weight
    loss = loss_func(f, y)

    # sum the input outer products
    xxT = einsum(X, X, "n d_in1, n d_in2 -> d_in1 d_in2")

    # cvec-GGN computation and comparison
    G_cvec = vec_ggn(loss, W, f, "cvec")
    G_cvec_manual = kron(xxT, 2 * eye(D_out))
    assert allclose(G_cvec, G_cvec_manual)

    # rvec-GGN computation and comparison
    G_rvec = vec_ggn(loss, W, f, "rvec")
    G_rvec_manual = kron(2 * eye(D_out), xxT)
    assert allclose(G_rvec, G_rvec_manual)
