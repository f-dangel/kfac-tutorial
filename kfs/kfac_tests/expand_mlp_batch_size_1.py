"""Test KFAC on an MLP with batch size 1."""

from collections import OrderedDict
from itertools import product

from torch import kron, manual_seed, rand, randint
from torch.nn import (
    CrossEntropyLoss,
    Linear,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)

from kfs.basics.emp_fishers import vec_empfisher
from kfs.basics.ggns import vec_ggn
from kfs.kfac import KFAC
from kfs.utils import report_nonclose

# allow testing different depths, loss & activation functions
depths = [3]
activations = [Tanh, Sigmoid, ReLU]
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

combinations = list(
    product(depths, activations, loss_funcs, reductions)
)
# loop over all settings and test KFAC equivalences
for idx, (L, phi, c, reduction) in enumerate(
    combinations, start=1
):
    print(
        f"{idx}/{len(combinations)} Testing L={L},"
        + f" phi={phi.__name__}, c={c.__name__},"
        + f" reduction={reduction}"
    )
    loss_func = c(reduction=reduction).double()
    y = ys[c]

    # set up the MLP
    manual_seed(1)
    layers = OrderedDict()
    dims = [D_in] + (L - 1) * [D_hidden] + [D_out]
    for i in range(L):
        layers[f"linear{i}"] = Linear(
            dims[i], dims[i + 1], bias=False
        )
        layers[f"activation{i}"] = phi()
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
        model, loss_func, (X, y), "type-2", "expand"
    )
    assert len(kfac) == len(rvec_ggn) == len(cvec_ggn) == L
    for (A, B), G_cvec, G_rvec in zip(
        kfac.values(), cvec_ggn, rvec_ggn
    ):
        report_nonclose(G_cvec, kron(A, B))
        report_nonclose(G_rvec, kron(B, A))

    # compute KFAC-MC with large number of samples and compare
    kfac = KFAC.compute(
        model, loss_func, (X, y), "mc=25_000", "expand"
    )
    assert len(kfac) == len(rvec_ggn) == len(cvec_ggn) == L
    tols = {"atol": 5e-4, "rtol": 5e-2}
    for (A, B), G_cvec, G_rvec in zip(
        kfac.values(), cvec_ggn, rvec_ggn
    ):
        report_nonclose(G_cvec, kron(A, B), **tols)
        report_nonclose(G_rvec, kron(B, A), **tols)

    # compute the EF
    rvec_ef = [
        vec_empfisher(
            model, loss_func, X, y, (p, p), vec="rvec"
        )
        for p in params
    ]
    cvec_ef = [
        vec_empfisher(
            model, loss_func, X, y, (p, p), vec="cvec"
        )
        for p in params
    ]

    # compute KFAC-empirical and compare
    kfac = KFAC.compute(
        model, loss_func, (X, y), "empirical", "expand"
    )
    assert len(kfac) == len(rvec_ef) == len(cvec_ef) == L
    for (A, B), e_cvec, e_rvec in zip(
        kfac.values(), cvec_ef, rvec_ef
    ):
        report_nonclose(e_cvec, kron(A, B))
        report_nonclose(e_rvec, kron(B, A))
