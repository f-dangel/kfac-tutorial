"""Hessian of softmax cross-entropy loss."""

from torch import (
    allclose,
    manual_seed,
    outer,
    rand,
    randint,
)
from torch.nn import CrossEntropyLoss

from kfs.hessians import cvec_hess, hess, rvec_hess

if __name__ == "__main__":
    manual_seed(0)

    C = 3  # generate random input data
    x = rand(C, requires_grad=True)
    y = randint(0, C, ())
    loss = CrossEntropyLoss()(x, y)

    p = x.softmax(0)  # manual computation
    H_manual = p.diag() - outer(p, p)

    H = hess(loss, x)  # autodiff computation
    H_cvec = cvec_hess(loss, x)
    H_rvec = rvec_hess(loss, x)

    assert allclose(H, H_manual)  # comparison
    assert allclose(H_cvec, H_manual)
    assert allclose(H_rvec, H_manual)
