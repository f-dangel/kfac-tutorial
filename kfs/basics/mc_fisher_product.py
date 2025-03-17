"""MC-approximated Type-I Fisher-vector multiplication."""

from typing import Tuple, Union

from torch import Generator, Tensor, zeros_like
from torch.autograd import grad
from torch.nn import CrossEntropyLoss, Module, MSELoss

from kfs.basics.label_sampling import (
    draw_label_CrossEntropyLoss,
    draw_label_MSELoss,
)
from kfs.basics.reduction_factors import (
    CrossEntropyLoss_criterion,
    MSELoss_criterion,
    get_reduction_factor,
)


def mcfishervp(
    model: Module,
    loss_func: Union[MSELoss, CrossEntropyLoss],
    inputs: Tensor,
    labels: Tensor,
    params: Union[Tensor, Tuple[Tensor, Tensor]],
    v: Tensor,
    mc_samples: int,
    seed: int = 42,
    retain_graph: bool = False,
) -> Tensor:
    r"""Multiply v with the MC Fisher of model.

    The MC (Type-I) Fisher corresponds to a Monte Carlo
    approximation of the Fisher Information Matrix of
    model w.r.t. loss_func using mc_samples samples.

    See $\text{\Cref{sec:fisher}}$.

    Args:
        model: The model whose Fisher is approximated.
        loss_func: The (scaled) negative log-likelihood
            function.
        inputs: The inputs over which the Fisher is
            calculated.
        labels: The labels corresponding to inputs.
        params: The parameters of model w.r.t. which
            we are calculating the Fisher.
        v: The vector that is multiplied with the MC
            Fisher. Has same shape as params.
        mc_samples: The number of MC samples used for
            the approximation.
        seed: Optional seed for reproducibility.
            Default: 42.
        retain_graph: Whether to retain the computation
            graph of f for future differentiation.
            Default: False.

    Returns:
        The product of the Monte Carlo approximated Fisher
        with v. Has same shape as params if params is a
        Tensor, and params[0] if params is a tuple.
    """
    params = (
        (params, params)
        if isinstance(params, Tensor)
        else params
    )

    generator = Generator()
    generator.manual_seed(seed)

    result = zeros_like(params[0])

    c_func = {
        MSELoss: MSELoss_criterion,
        CrossEntropyLoss: CrossEntropyLoss_criterion,
    }[type(loss_func)]
    sample_label_func = {
        MSELoss: draw_label_MSELoss,
        CrossEntropyLoss: draw_label_CrossEntropyLoss,
    }[type(loss_func)]

    reduction_factor = get_reduction_factor(
        loss_func, labels
    )

    for x in inputs.split(1):
        pred_n = model(x)

        for _ in range(mc_samples):
            y_tilde_nm = sample_label_func(
                pred_n, generator
            )
            c_nm = c_func(pred_n, y_tilde_nm)

            g_nm1, g_nm2 = grad(
                c_nm,
                params,
                retain_graph=retain_graph,
            )
            g_nm1, g_nm2 = g_nm1.detach(), g_nm2.detach()

            result.add_(g_nm1 * (g_nm2 * v).sum())

    result.mul_(reduction_factor / mc_samples)

    return result
