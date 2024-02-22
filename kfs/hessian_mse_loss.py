"""Hessian of mean-squared error."""

from torch import (
    allclose,
    eye,
    manual_seed,
    rand,
    rand_like,
)
from torch.nn import MSELoss

from kfs.hessians import cvec_hess, hess, rvec_hess

if __name__ == "__main__":
    C = 3
    manual_seed(0)
    x = rand(C, requires_grad=True)
    y = rand_like(x)

    l = MSELoss(reduction="sum")(x, y)

    H = hess(l, x)
    assert allclose(H, 2 * eye(C))

    S = 4
    X = rand(C, S, requires_grad=True)
    Y = rand_like(X)

    l = MSELoss(reduction="sum")(X, Y)
    H = cvec_hess(l, X)
    assert allclose(H, 2 * eye(C * S))

    H = rvec_hess(l, X)
    assert allclose(H, 2 * eye(C * S))
