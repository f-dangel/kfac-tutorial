"""Visualize various full curvatures together with KFAC."""

from argparse import ArgumentParser
from functools import partial
from os import path
from typing import Dict, Tuple

from matplotlib import pyplot as plt
from torch import Tensor, block_diag, kron, manual_seed
from torch.nn import MSELoss
from tueplots import bundles

from kfs.kfac import KFAC
from kfs.plots import SAVEDIR
from kfs.plots.synthetic_fisher import (
    curv_mat,
    cvec_empfisher,
    cvec_ggn,
    cvec_mcfisher,
    get_model,
    get_rand_data,
    highlight_blocks,
    rvec_empfisher,
    rvec_ggn,
    rvec_mcfisher,
)


def kfac_mat(
    kfac: Dict[str, Tuple[Tensor, Tensor]], vec: str
) -> Tensor:
    r"""Extend KFAC factors to dense matrix.

    Args:
        kfac: KFAC approximation of curvature
            per layer.
        vec: Name of the flattening scheme. Must be
            either 'rvec' or 'cvec'.

    Returns:
        The dense representation of the KFAC
        approximation.
    """
    matrices = []

    for _, (A, B) in kfac.items():
        matrices.append(
            kron(A, B) if vec == "cvec" else kron(B, A)
        )

    # Construct dense block diagonal matrix
    return block_diag(*matrices)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in matplotlib.",
    )
    parser.add_argument(
        "--reduce_mc_samples",
        action="store_true",
        default=False,
        help=(
            "Reduce the number of MC samples for "
            "CI integration."
        ),
    )
    args = parser.parse_args()

    config = bundles.icml2024(usetex=not args.disable_tex)
    plt.rcParams.update(config)
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} "
        r"\usepackage{bm}"
    )

    manual_seed(0)
    N = 100
    D_in, D_hidden, D_out = 5, 4, 3
    X, y = get_rand_data(N, D_in, D_out)

    max_num_mc_samples = (
        10 if args.reduce_mc_samples else float("inf")
    )

    model = get_model(D_in, D_hidden, D_out)
    param_dims = [p.numel() for p in model.parameters()]
    num_params = sum(param_dims)
    loss_func = MSELoss()

    output = model(X)
    loss = loss_func(output, y)

    empirical_risk_setting = {
        "model": model,
        "loss_func": loss_func,
        "X": X,
        "y": y,
    }

    curvature_funcs = {
        "cvec_ggn": {
            "full": partial(
                cvec_ggn, loss=loss, output=output
            ),
            "kfac": ("type-2", "cvec"),
        },
        "rvec_ggn": {
            "full": partial(
                rvec_ggn, loss=loss, output=output
            ),
            "kfac": ("type-2", "rvec"),
        },
        "cvec_mcfisher_100": {
            "full": partial(
                cvec_mcfisher,
                **empirical_risk_setting,
                num_samples=100,
                max_samples=max_num_mc_samples,
            ),
            "kfac": ("mc=100", "cvec"),
        },
        "rvec_mcfisher_100": {
            "full": partial(
                rvec_mcfisher,
                **empirical_risk_setting,
                num_samples=100,
                max_samples=max_num_mc_samples,
            ),
            "kfac": ("mc=100", "rvec"),
        },
        "cvec_empfisher": {
            "full": partial(
                cvec_empfisher, **empirical_risk_setting
            ),
            "kfac": ("empirical", "cvec"),
        },
        "rvec_empfisher": {
            "full": partial(
                rvec_empfisher, **empirical_risk_setting
            ),
            "kfac": ("empirical", "rvec"),
        },
    }

    for name, example in curvature_funcs.items():
        func = example["full"]
        fisher_type, vec = example["kfac"]

        # compute curvature matrix tiles and combine them
        C = curv_mat(func, model, bda=False)
        C_rescaled = (C.detach().abs() + 1e-5).log10()

        # compute kfac approximation
        kfac = KFAC.compute(
            model, loss_func, (X, y), fisher_type, "expand"
        )
        C_kfac = kfac_mat(kfac, vec)
        C_kfac_rescaled = (
            C_kfac.detach().abs() + 1e-5
        ).log10()

        # compute min and max over both matrices
        min_val = min(
            C_rescaled.min().item(),
            C_kfac_rescaled.min().item(),
        )
        max_val = max(
            C_rescaled.max().item(),
            C_kfac_rescaled.max().item(),
        )

        imshow_kwargs = {
            "vmin": min_val,
            "vmax": max_val,
            "interpolation": "none",
            "extent": (
                0.5,
                num_params + 0.5,
                num_params + 0.5,
                0.5,
            ),
        }

        # plot full curvature matrix and the KFAC matrix
        for C, C_name in zip(
            [C_rescaled, C_kfac_rescaled], ["full", "kfac"]
        ):
            fig, ax = plt.subplots()
            ax.imshow(C, **imshow_kwargs)
            ax.set_xlabel("$j$")
            ax.set_ylabel("$i$")
            highlight_blocks(ax, param_dims)
            plt.savefig(
                path.join(
                    SAVEDIR,
                    f"synthetic_{name}_{C_name}.pdf",
                ),
                transparent=True,
                bbox_inches="tight",
            )
            plt.close(fig)
