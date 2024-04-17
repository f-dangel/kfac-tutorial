"""Compute KFAC factors for Linear layers."""

from collections import OrderedDict

from torch import kron, manual_seed, rand
from torch.nn import Linear, MSELoss, Sequential

from kfs.basics.ggns import vec_ggn
from kfs.kfac import KFAC
from kfs.utils import report_nonclose

if __name__ == "__main__":
    manual_seed(0)

    batch_size = 5
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
    for reduction in ["sum", "mean"]:
        loss_func = MSELoss(reduction=reduction)

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

        report_nonclose(kron(A, B), cvec_ggn)
        report_nonclose(kron(B, A), rvec_ggn)
