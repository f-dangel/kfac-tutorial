"""Compute KFAC factors for Linear layers."""

from collections import OrderedDict

from torch import allclose, kron, manual_seed, rand
from torch.nn import Linear, MSELoss, Sequential

from kfs import KFAC
from kfs.ggns import vec_ggn

if __name__ == "__main__":
    manual_seed(0)

    batch_size = 2
    D_in, D_hidden, D_out = 4, 3, 2
    X, y = rand(batch_size, D_in), rand(batch_size, D_out)

    layers = OrderedDict(
        {
            "first": Linear(D_in, D_hidden, bias=False),
            "middle": Linear(
                D_hidden, D_hidden, bias=False
            ),
            "last": Linear(D_hidden, D_out, bias=False),
        }
    )
    model = Sequential(layers)
    loss_func = MSELoss(reduction="mean")

    # GGN computation for last layer weight
    output = model(X)
    loss = loss_func(output, y)
    rvec_ggn = vec_ggn(
        loss, model.last.weight, output, "rvec"
    )
    cvec_ggn = vec_ggn(
        loss, model.last.weight, output, "cvec"
    )

    # KFAC computation for last layer weight
    kfacs = KFAC.compute(
        model,
        loss_func,
        (X, y),
        "input_only",
        "expand",
        None,
    )
    A, B = kfacs["last"]

    print(kron(A, B)[0])
    print(cvec_ggn[0])
    assert allclose(kron(A, B), cvec_ggn)
    assert allclose(kron(B, A), rvec_ggn)
