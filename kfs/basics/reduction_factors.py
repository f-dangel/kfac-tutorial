"""Determine reduction factors of common loss functions."""

from math import isclose
from typing import Union

from torch import (
    Tensor,
    manual_seed,
    rand,
    rand_like,
    randint,
)
from torch.nn import CrossEntropyLoss, MSELoss


def get_reduction_factor(
    loss_func: Union[MSELoss, CrossEntropyLoss],
    labels: Tensor,
) -> float:
    r"""Compute the reduction factor for a PyTorch loss layer.

    Args:
        loss_func: PyTorch loss layer.
        labels: Ground truth targets.

    Returns:
        Reduction factor for the given loss layer and targets
        ($R$ from $\text{\Cref{eq:empirical_risk}}$).

    Raises:
        NotImplementedError: If the loss is unsupported.
    """
    batch_size = labels.shape[0]
    dim_Y = labels.shape[1:].numel()
    reduction = loss_func.reduction

    if isinstance(loss_func, MSELoss):
        return {
            "sum": 2,
            "mean": 2 / (batch_size * dim_Y),
        }[reduction]
    elif isinstance(loss_func, CrossEntropyLoss):
        return {
            "sum": 1,
            "mean": 1 / (batch_size * dim_Y),
        }[reduction]
    else:
        raise NotImplementedError(
            f"Unknown loss: {loss_func}."
        )


def MSELoss_criterion(
    prediction: Tensor, target: Tensor
) -> Tensor:
    r"""Criterion function for PyTorch's square loss.

    Implements $c(\mathbf{f}, \mathbf{y})$ from
    $\text{\Cref{ex:square_loss}}$.

    Args:
        prediction: Prediction for one datum.
        target: Ground truth for one datum.

    Returns:
        The evaluated criterion.
    """
    return 0.5 * ((prediction - target) ** 2).sum()


def CrossEntropyLoss_criterion(
    prediction: Tensor, target: Tensor
) -> Tensor:
    r"""Criterion function for PyTorch's cross-entropy loss.

    Implements $c(\mathbf{f}, \mathbf{y})$ from
    $\text{\Cref{ex:cross_entropy_loss}}$.

    Args:
        prediction: Prediction for one datum.
        target: Ground truth for one datum.

    Returns:
        The evaluated criterion.
    """
    # TODO Figure out if one can do this without flattening
    log_softmax = prediction.log_softmax(dim=0)
    log_softmax_flat = (
        log_softmax.unsqueeze(-1)
        if log_softmax.ndim == 1
        else log_softmax.flatten(start_dim=1)
    )
    target_flat = target.flatten()
    return -sum(
        log_softmax_flat[y_i, i]
        for i, y_i in enumerate(target_flat)
    )


if __name__ == "__main__":
    manual_seed(0)

    batch_size = 10
    dims_Y = (5, 4, 3)

    # regression (tensor case)
    f = rand(batch_size, *dims_Y)
    y = rand_like(f)

    for reduction in ["sum", "mean"]:
        loss_func = MSELoss(reduction=reduction)
        R_truth = loss_func(f, y) / sum(
            MSELoss_criterion(f_n, y_n)
            for f_n, y_n in zip(f, y)
        )
        R = get_reduction_factor(loss_func, y)
        assert isclose(R, R_truth, rel_tol=1e-7)

    # classification (tensor case)
    num_classes = 6
    f = rand(batch_size, num_classes, *dims_Y)
    y = randint(
        low=0, high=num_classes, size=(batch_size, *dims_Y)
    )

    for reduction in ["sum", "mean"]:
        loss_func = CrossEntropyLoss(reduction=reduction)
        R_truth = loss_func(f, y) / sum(
            CrossEntropyLoss_criterion(f_n, y_n)
            for f_n, y_n in zip(f, y)
        )
        R = get_reduction_factor(loss_func, y)
        assert isclose(R, R_truth, rel_tol=1e-7)
