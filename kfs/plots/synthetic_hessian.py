"""Visualize the Hessian of a synthetic empirical risk."""

from argparse import ArgumentParser
from os import path
from typing import List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch import cat, manual_seed, rand, zeros_like
from torch.nn import Linear, MSELoss, Sequential, Sigmoid
from tueplots import bundles

from kfs.basics.hessians import vec_hess
from kfs.plots import SAVEDIR


def highglight_blocks(
    ax: Axes,
    block_dims: List[int],
    color: str = "white",
    linewidth: float = 0.25,
):
    """Highlight the block structure of a square matrix.

    Args:
        ax: Matplotlib axes.
        block_dims: Dimensions of the blocks.
        color: Color of the lines. Defaults to "white".
        linewidth: Width of the lines. Defaults to 0.25.
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    assert xmin == ymax and xmax == ymin

    style = dict(color=color, linewidth=linewidth)
    current = xmin
    for dim in block_dims[:-1]:
        current += dim
        ax.axvline(current, **style)
        ax.axhline(current, **style)


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
        Sigmoid(),
        Linear(D_hidden, D_hidden),
        Sigmoid(),
        Linear(D_hidden, D_out),
    )
    param_dims = [p.numel() for p in model.parameters()]
    num_params = sum(param_dims)
    loss_func = MSELoss()

    output = model(X)
    loss = loss_func(output, y)

    curvature_funcs = {
        "cvec_hessian": lambda p_i, p_j: vec_hess(
            loss, (p_i, p_j), vec="cvec"
        ),
        "rvec_hessian": lambda p_i, p_j: vec_hess(
            loss, (p_i, p_j), vec="rvec"
        ),
    }

    for bda in [False, True]:
        for name, func in curvature_funcs.items():
            # compute curvature matrix tiles and combine them
            C = []
            for i, p_i in enumerate(model.parameters()):
                C_i = []
                for j, p_j in enumerate(model.parameters()):
                    C_ij = func(p_i, p_j)
                    if bda and i != j:
                        C_ij = zeros_like(C_ij)
                    C_i.append(C_ij)
                C.append(C_i)

            # concatenate along the column axis
            C = [cat(C_i, dim=1) for C_i in C]
            # concatenate along the row axis
            C = cat(C)

            with plt.rc_context(
                bundles.icml2024(
                    usetex=not args.disable_tex
                )
            ):
                fig, ax = plt.subplots()
                ax.imshow(
                    (C.detach().abs() + 1e-5).log10(),
                    interpolation="none",
                    extent=(
                        0.5,
                        num_params + 0.5,
                        num_params + 0.5,
                        0.5,
                    ),
                )
                ax.set_xlabel("$j$")
                ax.set_ylabel("$i$")
                highglight_blocks(ax, param_dims)
                suffix = "_bda" if bda else ""
                plt.savefig(
                    path.join(
                        SAVEDIR,
                        f"synthetic_{name}{suffix}.pdf",
                    ),
                    transparent=True,
                    bbox_inches="tight",
                )
                plt.close(fig)
