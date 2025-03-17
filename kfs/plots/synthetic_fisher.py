"""Visualize the Fisher of a synthetic empirical risk."""

from argparse import ArgumentParser
from functools import partial
from os import path
from typing import Callable, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch import (
    Tensor,
    cat,
    manual_seed,
    rand,
    tensor,
    zeros_like,
)
from torch.linalg import matrix_norm
from torch.nn import (
    Linear,
    Module,
    MSELoss,
    Sequential,
    Sigmoid,
)
from tueplots import bundles

from kfs.basics.emp_fishers import vec_empfisher
from kfs.basics.ggns import vec_ggn
from kfs.basics.mc_fishers_quick import vec_mcfisher_quick
from kfs.plots import SAVEDIR


def get_rand_data(
    N: int, D_in: int, D_out: int
) -> Tuple[Tensor, Tensor]:
    """Get random toy dataset.

    Args:
        N: Number of data points.
        D_in: Input dimensions.
        D_out: Output dimensions.

    Returns:
        A tuple of N inputs and outputs.
    """
    X = rand(N, D_in)
    y = rand(N, D_out)

    return X, y


def get_model(
    D_in: int, D_hidden: int, D_out: int
) -> Sequential:
    """Get toy model.

    Args:
        D_in: Input dimensions.
        D_hidden: Hidden dimensions.
        D_out: Output dimensions.

    Returns:
        A three-layer fully-connected network with sigmoid
        activations and no final activation.
    """
    return Sequential(
        Linear(D_in, D_hidden),
        Sigmoid(),
        Linear(D_hidden, D_hidden),
        Sigmoid(),
        Linear(D_hidden, D_out),
    )


def curv_mat(
    curv: Callable, model: Module, bda: bool
) -> Tensor:
    """Calculate dense curvature estimate of model.

    Args:
        curv: The curvature estimate to use. Accepts two
            parameter tensors.
        model: The model of interest.
        bda: Whether to provide a block-diagonal
            approximation of the dense curvature matrix.

    Returns:
        The dense curvature matrix.
    """
    H = []
    for i, p_i in enumerate(model.parameters()):
        H_i = []
        for j, p_j in enumerate(model.parameters()):
            H_ij = curv(p_i, p_j)

            if bda and i != j:
                H_ij = zeros_like(H_ij)

            H_i.append(H_ij)
        H.append(H_i)

    # concatenate along the column axis
    H = [cat(H_i, dim=1) for H_i in H]
    # concatenate along the row axis
    H = cat(H)

    return H


def plot_diff_spec_norms(
    diff_spec_norms: Tensor,
    mc_sample_list: List[int],
    name: str,
) -> None:
    """Plot spectral norm of MC and exact Fisher diff.

    Args:
        diff_spec_norms: Spectral norm of MC and exact
            Fisher differences for various number of
            MC samples.
        mc_sample_list: The number of MC samples used
            for each entry of diff_spec_norms.
        name: Whether the matrices are in cvec or rvec
            convention.
    """
    y_coords = diff_spec_norms.mean(dim=1)
    y_stds = diff_spec_norms.std(dim=1)
    fig, ax = plt.subplots()
    ax.plot(mc_sample_list, y_coords)
    ax.fill_between(
        mc_sample_list,
        y_coords - 1.96 * y_stds,
        y_coords + 1.96 * y_stds,
        alpha=0.3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MC Samples")
    ax.set_ylabel(
        r"$\Vert \tilde{F}^{\text{I}} "
        r"- F^{\text{I}} \Vert_2$"
    )

    plt.savefig(
        path.join(
            SAVEDIR,
            f"synthetic_{name}_diff_spec_norm.pdf",
        ),
        transparent=True,
        bbox_inches="tight",
    )
    plt.close(fig)


def highlight_blocks(
    ax: Axes,
    block_dims: List[int],
    color: str = "white",
    linewidth: float = 0.25,
) -> None:
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


def cvec_ggn(
    p_i: Tensor, p_j: Tensor, loss: Tensor, output: Tensor
) -> Tensor:
    """Calculate GGN block in cvec convention.

    Args:
        p_i: First parameter tensor.
        p_j: Second parameter tensor.
        loss: Loss value.
        output: Output of the neural network.

    Returns:
        The cvec GGN block for p_i and p_j.
    """
    return vec_ggn(loss, (p_i, p_j), output, vec="cvec")


def rvec_ggn(
    p_i: Tensor, p_j: Tensor, loss: Tensor, output: Tensor
) -> Tensor:
    """Calculate GGN block in rvec convention.

    Args:
        p_i: First parameter tensor.
        p_j: Second parameter tensor.
        loss: Loss value.
        output: Output of the neural network.

    Returns:
        The rvec GGN block for p_i and p_j.
    """
    return vec_ggn(loss, (p_i, p_j), output, vec="rvec")


def cvec_mcfisher(
    p_i: Tensor,
    p_j: Tensor,
    model: Sequential,
    loss_func: MSELoss,
    X: Tensor,
    y: Tensor,
    num_samples: int,
    max_samples: int,
    seed: int = 42,
) -> Tensor:
    """Calculate MC Fisher block in cvec convention.

    Args:
        p_i: First parameter tensor.
        p_j: Second parameter tensor.
        model: Neural network.
        loss_func: Loss function.
        X: Input tensor.
        y: Target tensor.
        num_samples: Number of MC samples to use.
        max_samples: Maximum number of MC samples allowed.
        seed: Optional seed for reproducibility.

    Returns:
        The cvec MC Fisher block for p_i and p_j.
    """
    return vec_mcfisher_quick(
        model,
        loss_func,
        X,
        y,
        (p_i, p_j),
        min(num_samples, max_samples),
        vec="cvec",
        seed=seed,
    )


def rvec_mcfisher(
    p_i: Tensor,
    p_j: Tensor,
    model: Sequential,
    loss_func: MSELoss,
    X: Tensor,
    y: Tensor,
    num_samples: int,
    max_samples: int,
    seed: int = 42,
) -> Tensor:
    """Calculate MC Fisher block in rvec convention.

    Args:
        p_i: First parameter tensor.
        p_j: Second parameter tensor.
        model: Neural network.
        loss_func: Loss function.
        X: Input tensor.
        y: Target tensor.
        num_samples: Number of MC samples to use.
        max_samples: Maximum number of MC samples allowed.
        seed: Optional seed for reproducibility.

    Returns:
        The rvec MC Fisher block for p_i and p_j.
    """
    return vec_mcfisher_quick(
        model,
        loss_func,
        X,
        y,
        (p_i, p_j),
        min(num_samples, max_samples),
        vec="rvec",
        seed=seed,
    )


def cvec_empfisher(
    p_i: Tensor,
    p_j: Tensor,
    model: Sequential,
    loss_func: MSELoss,
    X: Tensor,
    y: Tensor,
) -> Tensor:
    """Calculate empirical Fisher block in cvec convention.

    Args:
        p_i: First parameter tensor.
        p_j: Second parameter tensor.
        model: Neural network.
        loss_func: Loss function.
        X: Input tensor.
        y: Target tensor.

    Returns:
        The cvec empirical Fisher block for p_i and p_j.
    """
    return vec_empfisher(
        model,
        loss_func,
        X,
        y,
        (p_i, p_j),
        vec="cvec",
    )


def rvec_empfisher(
    p_i: Tensor,
    p_j: Tensor,
    model: Sequential,
    loss_func: MSELoss,
    X: Tensor,
    y: Tensor,
) -> Tensor:
    """Calculate empirical Fisher block in rvec convention.

    Args:
        p_i: First parameter tensor.
        p_j: Second parameter tensor.
        model: Neural network.
        loss_func: Loss function.
        X: Input tensor.
        y: Target tensor.

    Returns:
        The rvec empirical Fisher block for p_i and p_j.
    """
    return vec_empfisher(
        model,
        loss_func,
        X,
        y,
        (p_i, p_j),
        vec="rvec",
    )


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

    curvature_funcs = {
        "cvec_ggn": partial(
            cvec_ggn, loss=loss, output=output
        ),
        "rvec_ggn": partial(
            rvec_ggn, loss=loss, output=output
        ),
        "cvec_mcfisher_1": partial(
            cvec_mcfisher,
            model=model,
            loss_func=loss_func,
            X=X,
            y=y,
            num_samples=1,
            max_samples=max_num_mc_samples,
        ),
        "rvec_mcfisher_1": partial(
            rvec_mcfisher,
            model=model,
            loss_func=loss_func,
            X=X,
            y=y,
            num_samples=1,
            max_samples=max_num_mc_samples,
        ),
        "cvec_mcfisher_100": partial(
            cvec_mcfisher,
            model=model,
            loss_func=loss_func,
            X=X,
            y=y,
            num_samples=100,
            max_samples=max_num_mc_samples,
        ),
        "rvec_mcfisher_100": partial(
            rvec_mcfisher,
            model=model,
            loss_func=loss_func,
            X=X,
            y=y,
            num_samples=100,
            max_samples=max_num_mc_samples,
        ),
        "cvec_empfisher": partial(
            cvec_empfisher,
            model=model,
            loss_func=loss_func,
            X=X,
            y=y,
        ),
        "rvec_empfisher": partial(
            rvec_empfisher,
            model=model,
            loss_func=loss_func,
            X=X,
            y=y,
        ),
    }

    for name, func in curvature_funcs.items():
        bda_options = (
            [False, True] if "ggn" in name else [False]
        )
        for bda in bda_options:
            # compute curvature matrix tiles and combine them
            C = curv_mat(func, model, bda)

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
            highlight_blocks(ax, param_dims)
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

    rvec_diff_spec_norms = []
    mc_sample_list = [1, 10, 100, 1000]
    mc_sample_list = [
        elem
        for elem in mc_sample_list
        if elem <= max_num_mc_samples
    ]
    num_trials = 3

    rvec_G = curv_mat(
        partial(rvec_ggn, loss=loss, output=output),
        model,
        bda=False,
    )

    for mc_samples in mc_sample_list:
        rvec_curr_diff_spec_norms = []
        for trial_idx in range(num_trials):
            rvec_F = curv_mat(
                partial(
                    rvec_mcfisher,
                    model=model,
                    loss_func=loss_func,
                    X=X,
                    y=y,
                    num_samples=mc_samples,
                    max_samples=float("inf"),
                    seed=trial_idx,
                ),
                model,
                bda=False,
            )
            rvec_diff_spec_norm = matrix_norm(
                rvec_G - rvec_F, ord=2
            )

            rvec_curr_diff_spec_norms.append(
                rvec_diff_spec_norm
            )

        rvec_diff_spec_norms.append(
            rvec_curr_diff_spec_norms
        )

    rvec_diff_spec_norms = tensor(rvec_diff_spec_norms)

    plot_diff_spec_norms(
        rvec_diff_spec_norms, mc_sample_list, name="rvec"
    )
