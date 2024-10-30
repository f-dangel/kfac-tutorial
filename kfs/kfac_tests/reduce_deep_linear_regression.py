from itertools import product

from torch import kron, manual_seed, rand
from torch.nn import MSELoss

from kfs.basics.ggns import vec_ggn
from kfs.kfac import KFAC
from kfs.kfac_tests.deep_linear_net import (
    create_deep_linear_net,
)
from kfs.utils import report_nonclose

batch_size = 6
D_in, D_hidden, D_out = 4, 3, 2
depths = [1, 3]
sequence_lengths = [5]
loss_reductions = ["sum", "mean"]
shared_reduction = ["sum", "mean"]

configurations = product(
    depths,
    sequence_lengths,
    loss_reductions,
    shared_reduction,
)

for (
    L,
    S,
    loss_reduction,
    shared_reduction,
) in configurations:
    print(
        f"Testing L={L}, S={S},"
        + f" loss_reduction={loss_reduction},"
        + f" shared_reduction={shared_reduction}"
    )

    loss_func = MSELoss(reduction=loss_reduction).double()

    # use double-precision for exact comparison
    manual_seed(0)
    X = rand(batch_size, S, D_in).double()
    y = rand(batch_size, D_out).double()

    # set up the MLP
    manual_seed(1)
    dims = [D_in] + (L - 1) * [D_hidden] + [D_out]
    model = create_deep_linear_net(
        dims,
        reduce_shared=shared_reduction,
        reduction_pos=L,
    ).double()
    print(model)

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
        model, loss_func, (X, y), "type-2", "reduce"
    )
    assert len(rvec_ggn) == len(cvec_ggn) == L
    for (A, B), G_cvec, G_rvec in zip(
        kfac.values(), cvec_ggn, rvec_ggn
    ):
        report_nonclose(G_rvec, kron(B, A))
        report_nonclose(G_cvec, kron(A, B))

    # compute KFAC MC with large number of samples and compare
    kfac = KFAC.compute(
        model, loss_func, (X, y), "mc=25_000", "reduce"
    )
    assert len(rvec_ggn) == len(cvec_ggn) == L
    for (A, B), G_cvec, G_rvec in zip(
        kfac.values(), cvec_ggn, rvec_ggn
    ):
        tols = {
            "atol": 5e-3 * G_cvec.abs().max(),
            "rtol": 1e-2,
        }
        report_nonclose(G_rvec, kron(B, A), **tols)
        report_nonclose(G_cvec, kron(A, B), **tols)
