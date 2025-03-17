"""Efficient MC Fisher tensor and matrices.

Good for tests that require instantiating Fisher matrices
but scales poorly to high-dimensional parameter spaces.
"""

from typing import Tuple, Union

from torch import (
    Generator,
    Tensor,
    outer,
    zeros,
)
from torch.autograd import grad
from torch.nn import CrossEntropyLoss, Module, MSELoss

from kfs.basics.flattening import cvec, rvec
from kfs.basics.label_sampling import (
    draw_label_CrossEntropyLoss,
    draw_label_MSELoss,
)
from kfs.basics.reduction_factors import (
    CrossEntropyLoss_criterion,
    MSELoss_criterion,
    get_reduction_factor,
)


def mcfisher_quick(
    model: Module,
    loss_func: Union[MSELoss, CrossEntropyLoss],
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

    generator = Generator()
    generator.manual_seed(seed)

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
                (params1, params2),
                retain_graph=True,
            )
            g_nm1, g_nm2 = g_nm1.detach(), g_nm2.detach()
            F_flat = outer(g_nm1.flatten(), g_nm2.flatten())
            F.add_(F_flat.reshape_as(F))

    F.mul_(reduction_factor / mc_samples)
    return F


def vec_mcfisher_quick(
    model: Module,
    loss_func: Union[MSELoss, CrossEntropyLoss],
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
    F = mcfisher_quick(
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
