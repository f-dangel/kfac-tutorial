"""Empirical-Fisher-vector multiplication."""

from typing import Tuple, Union

from torch import Tensor, zeros_like
from torch.autograd import grad
from torch.nn import CrossEntropyLoss, Module, MSELoss

from kfs.basics.reduction_factors import (
    CrossEntropyLoss_criterion,
    MSELoss_criterion,
    get_reduction_factor,
)


def empfishervp(
    model: Module,
    loss_func: Union[MSELoss, CrossEntropyLoss],
    inputs: Tensor,
    labels: Tensor,
    params: Union[Tensor, Tuple[Tensor, Tensor]],
    v: Tensor,
    retain_graph: bool = False,
) -> Tensor:
    r"""Multiply v with the empirical Fisher of model.

    The Empirical Fisher is an approximation of the
    Fisher Information Matrix of model w.r.t. loss_func
    where the ground-truth labels are used instead of
    the model's predictions.

    See $\text{\Cref{sec:emp_fisher}}$.

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
        v: The vector that is multiplied with the empirical
            Fisher. Has same shape as params.
        retain_graph: Optional argument to retain the
            computation graph. Defaults to False.

    Returns:
        The product of the empirical Fisher
        with v. Has same shape as params if params is a
        Tensor, and params[0] if params is a tuple.
    """
    params = (
        (params, params)
        if isinstance(params, Tensor)
        else params
    )

    result = zeros_like(params[0])

    c_func = {
        MSELoss: MSELoss_criterion,
        CrossEntropyLoss: CrossEntropyLoss_criterion,
    }[type(loss_func)]

    reduction_factor = get_reduction_factor(
        loss_func, labels
    )

    for x, y in zip(inputs, labels):
        pred_n = model(x)

        c_n = c_func(pred_n, y)

        g_n1, g_n2 = grad(
            c_n, params, retain_graph=retain_graph
        )
        g_n1, g_n2 = g_n1.detach(), g_n2.detach()

        result.add_(g_n1 * (g_n2 * v).sum())

    result.mul_(reduction_factor)

    return result
