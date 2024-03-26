"""Visualize the Hessian of a synthetic empirical risk."""

from argparse import ArgumentParser
from os import path

from matplotlib import pyplot as plt
from torch import cat, manual_seed, rand
from torch.nn import Linear, MSELoss, ReLU, Sequential
from tueplots import bundles

from kfs.hessians import cvec_hess, rvec_hess
from kfs.plots import SAVEDIR

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in matplotlib.",
    )
    args = parser.parse_args()

    manual_seed(0)
    N = 100
    D_in, D_hidden, D_out = 5, 4, 3
    X = rand(N, D_in)
    y = rand(N, D_out)

    model = Sequential(
        Linear(D_in, D_hidden),
        ReLU(),
        Linear(D_hidden, D_hidden),
        ReLU(),
        Linear(D_hidden, D_out),
    )
    loss_func = MSELoss()

    loss = loss_func(model(X), y)

    hess_funcs = {"cvec": cvec_hess, "rvec": rvec_hess}
    for flattening, hess_func in hess_funcs.items():

        # compute Hessian tiles and combine them
        H = []
        for p_i in model.parameters():
            H_i = []
            for p_j in model.parameters():
                H_ij = hess_func(loss, (p_i, p_j))
                H_i.append(H_ij)
            H.append(H_i)

        # concatenate along the column axis
        H = [cat(H_i, dim=1) for H_i in H]
        # concatenate along the row axis
        H = cat(H)

        with plt.rc_context(
            bundles.icml2024(usetex=not args.disable_tex)
        ):
            fig, ax = plt.subplots()
            ax.imshow((H.detach().abs() + 1e-5).log10())
            ax.set_xlabel("$j$")
            ax.set_ylabel("$i$")
            plt.savefig(
                path.join(
                    SAVEDIR,
                    f"synthetic_{flattening}_hessian.pdf",
                ),
                transparent=True,
                bbox_inches="tight",
            )
            plt.close(fig)
