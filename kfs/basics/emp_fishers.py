"""Empirical Fisher tensor and matrices."""

from torch import (
    Tensor,
    arange,
    unravel_index,
    zeros,
    zeros_like,
)

from typing import Union, Tuple

from kfs.basics.flattening import cvec, rvec
from kfs.basics.emp_fisher_product import empfishervp
from torch.nn import Module


def empfisher(
    model: Module,
    loss_func: Module,
    inputs: Tensor,
    labels: Tensor,
    params: Union[Tensor, Tuple[Tensor, Tensor]],
) -> Tensor:
    r"""Compute the model's empirical Fisher.

    The empirical Fisher is an approximation of the
    Fisher Information Matrix of model w.r.t. loss_func
    where the ground-truth labels are used instead of
    the model's predictions.

    See $\text{\Cref{def:emp_fisher}}$.

    Args:
        model: The model whose empirical Fisher is needed.
        loss_func: The (scaled) negative log-likelihood
            function.
        inputs: The inputs over which the empirical Fisher
            is calculated.
        labels: The labels over which the empirical Fisher
            is calculated.
        params: The parameters of model w.r.t. which
            we are calculating the Fisher.

    Returns:
        The empirical Fisher. Has shape
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
    F = zeros(params1.shape + params2.shape, dtype=dtype)

    for d in arange(params1.numel()):
        d_unraveled = unravel_index(d, params1.shape)
        one_hot_d = zeros_like(params1)
        one_hot_d[d_unraveled] = 1.0
        F[d_unraveled] = empfishervp(
            model,
            loss_func,
            inputs,
            labels,
            (params2, params1),
            one_hot_d,
            retain_graph=True,
        )  # empfishervp is defined in $\text{\Cref{basics/emp_fisher_product}}$

    return F


def vec_empfisher(
    model: Module,
    loss_func: Module,
    inputs: Tensor,
    labels: Tensor,
    params: Union[Tensor, Tuple[Tensor, Tensor]],
    vec: str,
) -> Tensor:
    r"""Compute the vectorized empirical Fisher matrix.

    Args:
        model: The model whose empirical Fisher is needed.
        loss_func: The (scaled) negative log-likelihood
            function.
        inputs: The inputs over which the empirical Fisher
            is calculated.
        labels: The labels over which the empirical Fisher
            is calculated.
        params: The parameters of model w.r.t. which
            we are calculating the empirical Fisher.
        vec: Name of the flattening scheme. Must be
            either 'rvec' or 'cvec'.

    Returns:
        The rvec- or cvec-Empirical-Fisher matrix of model
        w.r.t. loss_func.
        Has shape (params.numel(), params.numel()) if
        params is a Tensor, and
        (params[0].numel(), params[1].numel()) if
        params is a tuple.
    """
    vec = {"cvec": cvec, "rvec": rvec}[vec]
    F = empfisher(
        model,
        loss_func,
        inputs,
        labels,
        params,
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
