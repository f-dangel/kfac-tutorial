"""Test KFAC for square loss with a deep MLP."""

from collections import OrderedDict
from itertools import product

from torch import kron, manual_seed, rand
from torch.nn import Linear, MSELoss, Sequential

from kfs.basics.ggns import vec_ggn
from kfs.kfac import KFAC
from kfs.utils import report_nonclose

# allow testing different depths and reductions
depths = [3]
reductions = ["mean", "sum"]
sequence_lengths = [1, 5]  # weight sharing

batch_size = 6
D_in, D_hidden, D_out = 4, 3, 2

combinations = list(
    product(depths, sequence_lengths, reductions)
)

for idx, (L, S, reduction) in enumerate(
    combinations, start=1
):
    print(
        f"{idx}/{len(combinations)} Testing L={L},"
        + "f S={S}, reduction={reduction}"
    )
    loss_func = MSELoss(reduction=reduction).double()

    # use double-precision for exact comparison
    manual_seed(0)
    if S == 1:
        X = rand(batch_size, D_in).double()
        y = rand(batch_size, D_out).double()
    else:
        X = rand(batch_size, S, D_in).double()
        y = rand(batch_size, S, D_out).double()

    # set up the MLP
    manual_seed(1)
    dims = [D_in] + (L - 1) * [D_hidden] + [D_out]
    layers = OrderedDict(
        {
            f"linear{i}": Linear(
                dims[i], dims[i + 1], bias=False
            )
            for i in range(L)
        }
    )
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

    # compute KFAC type-2 and compare
    kfac = KFAC.compute(
        model, loss_func, (X, y), "type-2", "expand", None
    )
    assert len(rvec_ggn) == len(cvec_ggn) == L
    for (A, B), G_cvec, G_rvec in zip(
        kfac.values(), cvec_ggn, rvec_ggn
    ):
        report_nonclose(G_rvec, kron(B, A))
        report_nonclose(G_cvec, kron(A, B))

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
    assert len(rvec_ggn) == len(cvec_ggn) == L
    tols = {"atol": 5e-4, "rtol": 5e-2}
    for (A, B), G_cvec, G_rvec in zip(
        kfac.values(), cvec_ggn, rvec_ggn
    ):
        report_nonclose(G_rvec, kron(B, A), **tols)
        report_nonclose(G_cvec, kron(A, B), **tols)
