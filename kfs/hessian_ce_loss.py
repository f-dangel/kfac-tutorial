"""Hessian of softmax cross-entropy loss."""

from torch import (
    allclose,
    manual_seed,
    outer,
    rand,
    randint,
)
from torch.nn import CrossEntropyLoss

from kfs.hessians import hess

if __name__ == "__main__":
    C = 3
    manual_seed(0)
    x = rand(C, requires_grad=True)
    y = randint(0, C, ())

    l = CrossEntropyLoss(reduction="sum")(x, y)

    H = hess(l, x)

    p = x.softmax(0)
    assert allclose(H, p.diag() - outer(p, p))
