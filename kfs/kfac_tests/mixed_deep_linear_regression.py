from itertools import product

from pytest import raises
from torch import kron, manual_seed, rand
from torch.nn import MSELoss

from kfs.basics.ggns import vec_ggn
from kfs.kfac import KFAC
from kfs.kfac_tests.deep_linear_net import (
    create_deep_linear_net,
)
from kfs.utils import report_nonclose

batch_size = 6
D_in, D_hidden, D_out = 7, 3, 2
L = 4
reduce_sequence_after = 2
S = 5
loss_reductions = ["sum", "mean"]
shared_reduction = ["sum", "mean"]

configurations = list(
    product(loss_reductions, shared_reduction)
)

for idx, (loss_reduction, shared_reduction) in enumerate(
    configurations, start=1
):
    print(
        f"{idx}/{len(configurations)} Testing"
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
        reduction_pos=reduce_sequence_after,
    ).double()

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
    kfac_expand = KFAC.compute(
        model, loss_func, (X, y), "type-2", "expand"
    )
    kfac_reduce = KFAC.compute(
        model, loss_func, (X, y), "type-2", "reduce"
    )
    assert (
        len(rvec_ggn)
        == len(cvec_ggn)
        == L
        == len(kfac_expand)
        == len(kfac_reduce)
    )
    for idx, (
        (A_expand, B_expand),
        (A_reduce, B_reduce),
        G_cvec,
        G_rvec,
    ) in enumerate(
        zip(
            kfac_expand.values(),
            kfac_reduce.values(),
            cvec_ggn,
            rvec_ggn,
        )
    ):
        if idx >= reduce_sequence_after:
            # after reduction there is no shared axis,
            # hence KFAC-expand equals KFAC-reduce
            report_nonclose(A_expand, A_reduce)
            report_nonclose(B_expand, B_reduce)
        else:
            # before reduction KFAC-expand is different
            # from KFAC-reduce (grad-output-based factor),
            # and KFAC-expand does not equal the GGN
            report_nonclose(B_expand, B_reduce)
            with raises(ValueError):
                report_nonclose(
                    A_expand, A_reduce, verbose=False
                )
            with raises(ValueError):
                report_nonclose(
                    G_rvec,
                    kron(B_expand, A_expand),
                    verbose=False,
                )
            with raises(ValueError):
                report_nonclose(
                    G_cvec,
                    kron(A_expand, B_reduce),
                    verbose=False,
                )

        A, B = (
            A_reduce,
            B_reduce,
        )  # or _expand, both are the same
        report_nonclose(G_rvec, kron(B, A))
        report_nonclose(G_cvec, kron(A, B))
