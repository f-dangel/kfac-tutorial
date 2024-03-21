"""Visualize the Rosenbrock function and is partially linearized version."""

from os import path

import matplotlib.pyplot as plt
from torch import (
    Tensor,
    cat,
    linspace,
    log10,
    meshgrid,
    sqrt,
    zeros,
)
from tueplots import bundles

from kfs.ggn_rosenbrock import (
    g,
    l,
    rosenbrock,
    rosenbrock_partially_linearized,
)
from kfs.plots import SAVEDIR

if __name__ == "__main__":
    x0_min, x0_max = -1.25, 1.25
    x1_min, x1_max = -0.75, 1.75
    x0_resolution, x1_resolution = 128, 128

    x0 = linspace(x0_min, x0_max, x0_resolution)
    x1 = linspace(x1_min, x1_max, x1_resolution)

    anchor = Tensor([-0.5, -0.05]).requires_grad_(True)

    alpha = 10

    x1_grid, x0_grid = meshgrid(x1, x0, indexing="ij")
    xs = cat([x0_grid.unsqueeze(0), x1_grid.unsqueeze(0)])

    grid = zeros(x1_resolution, x0_resolution)
    grid_linearized = zeros(x1_resolution, x0_resolution)
    for i in range(x0_resolution):
        for j in range(x1_resolution):
            x = xs[:, j, i]
            grid[j, i], _ = rosenbrock(x, alpha)
            grid_linearized[j, i], _ = (
                rosenbrock_partially_linearized(
                    x, anchor, alpha
                )
            )

    with plt.rc_context(bundles.icml2024()):
        fig, ax = plt.subplots()
        ax.contour(x0, x1, grid.pow(1 / 2.5), levels=15)
        ax.contour(
            x0,
            x1,
            grid_linearized.detach().pow(1 / 2.5),
            levels=7,
            linestyles="dashed",
            alpha=0.5,
            cmap="gray",
        )

        ax.plot(*anchor.detach(), "*", color="red")
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        plt.savefig(
            path.join(SAVEDIR, "linearized_rosenbrock.pdf"),
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig)
