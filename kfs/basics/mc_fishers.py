"""MC-approximated Type-I Fisher tensor and matrices."""

from typing import Tuple, Union

from torch import (
    Tensor,
    arange,
    unravel_index,
    zeros,
    zeros_like,
)
from torch.nn import Module

from kfs.basics.flattening import cvec, rvec
from kfs.basics.mc_fisher_product import mcfishervp


def mcfisher(
    model: Module,
    loss_func: Module,
    inputs: Tensor,
    labels: Tensor,
    params: Union[Tensor, Tuple[Tensor, Tensor]],
    mc_samples: int,
    seed: int = 42,
) -> Tensor:
    r"""Compute the MC approximation of model's Fisher.

    The MC (Type-I) Fisher corresponds to a Monte Carlo
    approximation of the Fisher Information Matrix of
    model w.r.t. loss_func using mc_samples samples.

    See $\text{\Cref{def:mc_fisher}}$.

    Args:
        model: The model whose Fisher is approximated.
        loss_func: The (scaled) negative log-likelihood
            function.
        inputs: The inputs over which the Fisher is
            calculated.
        labels: The labels corresponding to inputs.
        params: The parameters of model w.r.t. which
            we are calculating the Fisher.
        mc_samples: The number of MC samples used for
            the approximation.
        seed: Optional seed for reproducibility.

    Returns:
        The Monte Carlo approximated Fisher. Has shape
        params.shape + params.shape if
        params is a Tensor, and
        params[0].shape + params[1].shape if
        params is a tuple.
    """
    params1, params2 = (
        (params, params)
        if isinstance(params, Tensor)
        else params
    )
    (dtype,) = {params1.dtype, params2.dtype}
    (device,) = {params1.device, params2.device}
    F = zeros(
        params1.shape + params2.shape,
        dtype=dtype,
        device=device,
    )

    for d in arange(params1.numel()):
        d_unraveled = unravel_index(d, params1.shape)
        one_hot_d = zeros_like(params1)
        one_hot_d[d_unraveled] = 1.0
        F[d_unraveled] = mcfishervp(
            model,
            loss_func,
            inputs,
            labels,
            (params2, params1),
            one_hot_d,
            mc_samples,
            seed,
            retain_graph=True,
        )
        # mcfishervp is defined in
        # $\text{\Cref{basics/mc_fisher_product}}$

    return F


def vec_mcfisher(
    model: Module,
    loss_func: Module,
    inputs: Tensor,
    labels: Tensor,
    params: Union[Tensor, Tuple[Tensor, Tensor]],
    mc_samples: int,
    vec: str,
    seed: int = 42,
) -> Tensor:
    r"""Compute the vectorized MC Fisher matrix.

    Args:
        model: The model whose Fisher is approximated.
        loss_func: The (scaled) negative log-likelihood
            function.
        inputs: The inputs over which the Fisher is
            calculated.
        labels: The labels corresponding to inputs.
        params: The parameters of model w.r.t. which
            we are calculating the Fisher.
        mc_samples: The number of MC samples used for
            the approximation.
        vec: Name of the flattening scheme. Must be
            either 'rvec' or 'cvec'.
        seed: Optional seed for reproducibility.


    Returns:
        The rvec- or cvec-MC-Fisher matrix of model
        w.r.t. loss_func.
        Has shape (params.numel(), params.numel()) if
        params is a Tensor, and
        (params[0].numel(), params[1].numel()) if
        params is a tuple.
    """
    vec = {"cvec": cvec, "rvec": rvec}[vec]
    F = mcfisher(
        model,
        loss_func,
        inputs,
        labels,
        params,
        mc_samples,
        seed,
    )
    # flatten row indices
    row_ndim = (
        params.ndim
        if isinstance(params, Tensor)
        else params[0].ndim
    )
    F = vec(F, end_dim=row_ndim - 1)
    # flatten column indices
    return vec(F, start_dim=1)
