"""Hessian of mean-squared error."""

from torch import (
    allclose,
    eye,
    manual_seed,
    rand,
    rand_like,
)
from torch.nn import MSELoss

from kfs.basics.hessians import hess, vec_hess

if __name__ == "__main__":
    manual_seed(0)

    C = 3  # create random input data
    x = rand(C, requires_grad=True)
    y = rand_like(x)
    loss = MSELoss(reduction="sum")(x, y)

    H_manual = 2 * eye(C)
    H = hess(loss, x)
    H_cvec = vec_hess(loss, x, vec="cvec")
    H_rvec = vec_hess(loss, x, vec="rvec")

    assert allclose(H, H_manual)
    assert allclose(H_cvec, H_manual)
    assert allclose(H_rvec, H_manual)
