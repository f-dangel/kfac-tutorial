"""Test KFAC on an MLP with batch size 1."""

from collections import OrderedDict
from itertools import product

from torch import kron, manual_seed, rand, randint
from torch.nn import (
    CrossEntropyLoss,
    Linear,
    MSELoss,
    Sequential,
    Tanh,
)

from kfs.basics.ggns import vec_ggn
from kfs.kfac import KFAC
from kfs.utils import report_nonclose

# allow testing different depths, loss & activation functions
depths = [3]
activations = [Tanh]
loss_funcs = [MSELoss, CrossEntropyLoss]
reductions = ["mean", "sum"]

# architecture and data set specifications
D_in, D_hidden, D_out = 5, 4, 2
batch_size = 1
manual_seed(0)
X = rand(batch_size, D_in).double()
# classification and regression labels
ys = {
    MSELoss: rand(batch_size, D_out).double(),
    CrossEntropyLoss: randint(
        low=0, high=D_out, size=(batch_size,)
    ),
}

# loop over all settings and test KFAC equivalences
for L, phi, c, reduction in product(
    depths, activations, loss_funcs, reductions
):
    loss_func = c(reduction=reduction)
    y = ys[c]

    # set up the MLP
    manual_seed(1)
    layers = OrderedDict()
    dims = [D_in] + (L - 1) * [D_hidden] + [D_out]
    for l in range(L):
        layers[f"linear{l}"] = Linear(
            dims[l], dims[l + 1], bias=False
        )
        layers[f"activation{l}"] = phi()
    model = Sequential(layers).double()

    # compute GGN with autodiff
    output = model(X)
    loss = loss_func(output, y)
    params = list(model.parameters())
    rvec_ggn = [
        vec_ggn(loss, p, output, "rvec") for p in params
    ]
    cvec_ggn = [
        vec_ggn(loss, p, output, "cvec") for p in params
    ]

    # compute KFAC-type-II and compare
    kfac = KFAC.compute(
        model, loss_func, (X, y), "type-2", "expand", None
    )
    assert len(kfac) == len(rvec_ggn) == len(cvec_ggn) == L
    for (A, B), G_cvec, G_rvec in zip(
        kfac.values(), cvec_ggn, rvec_ggn
    ):
        report_nonclose(G_cvec, kron(A, B))
        report_nonclose(G_rvec, kron(B, A))

    # compute KFAC-MC with large number of samples and compare
    kfac = KFAC.compute(
        model,
        loss_func,
        (X, y),
        "mc=25_000",
        "expand",
        None,
    )
    assert len(kfac) == len(rvec_ggn) == len(cvec_ggn) == L
    tols = {"atol": 5e-4, "rtol": 5e-2}
    for (A, B), G_cvec, G_rvec in zip(
        kfac.values(), cvec_ggn, rvec_ggn
    ):
        report_nonclose(G_cvec, kron(A, B), **tols)
        report_nonclose(G_rvec, kron(B, A), **tols)

    # TODO compute the EF

    # TODO compute KFAC-empirical and compare
