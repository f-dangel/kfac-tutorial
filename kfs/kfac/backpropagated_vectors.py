"""Compute vectors for backpropagation in KFAC."""

from math import sqrt
from typing import List

from torch import Tensor, stack
from torch.autograd import grad
from torch.nn import CrossEntropyLoss, Module, MSELoss

from kfs.basics.hessian_factorizations import (
    symmetric_factorization_CrossEntropyLoss,
    symmetric_factorization_MSELoss,
)
from kfs.basics.label_sampling import (
    draw_label_CrossEntropyLoss,
    draw_label_MSELoss,
)
from kfs.basics.reduction_factors import (
    CrossEntropyLoss_criterion,
    MSELoss_criterion,
)


def compute_backpropagated_vectors(
    loss_func: Module,
    fisher_type: str,
    predictions: Tensor,
    labels: Tensor,
) -> List[Tensor]:
    """Compute backpropagated vectors for KFAC's `B` matrix.

    Args:
        loss_func: The loss function.
        fisher_type: The type of Fisher approximation.
            Can be `'type-2'`, `'mc=1'` (with an arbitrary
            number instead of `1`), `'empirical'` or `'input_only'`.
        predictions: A batch of model predictions.
        labels: A batch of labels.

    Returns:
        A list of backpropagated vectors. Each vector has
        the same shape as `predictions`.

    Raises:
        ValueError: For invalid values of `fisher_type`.
    """
    if fisher_type == "type-2":
        return backpropagated_vectors_type2(
            loss_func, predictions, labels
        )
    elif fisher_type.startswith("mc="):
        mc_samples = int(fisher_type.replace("mc=", ""))
        return backpropagated_vectors_mc(
            loss_func, predictions, labels, mc_samples
        )
    elif fisher_type == "empirical":
        return backpropagated_vectors_empirical(
            loss_func, predictions, labels
        )
    elif fisher_type == "input_only":
        return []
    else:
        raise ValueError(
            f"Unknown Fisher type: {fisher_type}."
        )


def backpropagated_vectors_type2(
    loss_func: Module, predictions: Tensor, labels: Tensor
) -> List[Tensor]:
    """Compute backpropagated vectors for KFAC type-II.

    Args:
        loss_func: The loss function.
        predictions: A batch of model predictions.
        labels: A batch of labels.

    Returns:
        A list of backpropagated vectors. Each vector has
        the same shape as `predictions` and the number of
        vectors is equal `predictions.shape[1:].numel()`,
        i.e. dim(F). Vectors contain columns of the loss
        function's symmetric Hessian decomposition.

    Raises:
        NotImplementedError: For unsupported loss functions.
    """
    if isinstance(loss_func, MSELoss):
        S_func = symmetric_factorization_MSELoss
    elif isinstance(loss_func, CrossEntropyLoss):
        S_func = symmetric_factorization_CrossEntropyLoss
    else:
        raise NotImplementedError(
            f"Unknown loss function: {type(loss_func)}."
        )

    S = []
    for pred_n, y_n in zip(
        predictions.split(1), labels.split(1)
    ):
        pred_n, y_n = pred_n.squeeze(0), y_n.squeeze(0)
        S_n = S_func(pred_n, y_n)
        # flatten the column indices
        S_n = S_n.flatten(start_dim=S_n.ndim // 2)
        # move them to the leading axis
        S_n = S_n.moveaxis(-1, 0)
        S.append(S_n)
    # concatenate over all data points
    S = stack(S, dim=1)
    # convert into list, ith entry contains the ith columns of
    # all symmetric Hessian decompositions
    return [s.squeeze(0) for s in S.split(1)]


def backpropagated_vectors_mc(
    loss_func: Module,
    predictions: Tensor,
    labels: Tensor,
    mc_samples: int,
) -> List[Tensor]:
    """Compute backpropagated vectors for KFAC type-I MC.

    Args:
        loss_func: The loss function.
        predictions: A batch of model predictions.
        labels: A batch of labels.
        mc_samples: The number of Monte Carlo samples.

    Returns:
        A list of backpropagated vectors. Each vector has
        the same shape as `predictions` and the number of
        vectors is equal to `mc_samples`. Vectors contain
        a would-be gradient of the negative log-likelihood
        w.r.t. the model's predictions on a sampled label.

    Raises:
        NotImplementedError: For unsupported loss functions.
    """
    if isinstance(loss_func, MSELoss):
        c_func = MSELoss_criterion
        sample_label_func = draw_label_MSELoss
    elif isinstance(loss_func, CrossEntropyLoss):
        c_func = CrossEntropyLoss_criterion
        sample_label_func = draw_label_CrossEntropyLoss
    else:
        raise NotImplementedError(
            f"Unknown loss function: {type(loss_func)}."
        )

    S = []
    for pred_n, y_n in zip(
        predictions.split(1), labels.split(1)
    ):
        pred_n, y_n = pred_n.squeeze(0), y_n.squeeze(0)
        S_n = []
        for _ in range(mc_samples):
            y_tilde_nm = sample_label_func(pred_n)
            c_nm = c_func(pred_n, y_tilde_nm)
            S_n.append(grad(c_nm, pred_n)[0].detach())

        # concatenate over MC samples
        S_n = stack(S_n)
        S.append(S_n)

    # concatenate over data points
    S = stack(S)
    # move column dimension (MC samples) to leading
    S = S.moveaxis(1, 0)
    # incorporate normalization
    S = S / sqrt(mc_samples)

    # convert into list,
    # ith entry contains the ith sampled gradient
    return [s.squeeze(0) for s in S.split(1)]


def backpropagated_vectors_empirical(
    loss_func: Module,
    predictions: Tensor,
    labels: Tensor,
) -> List[Tensor]:
    """Compute backpropagated vectors for KFAC empirical.

    Args:
        loss_func: The loss function.
        predictions: A batch of model predictions.
        labels: A batch of labels.

    Returns:
        A list of backpropagated vectors. Each vector is
        equivalent to the gradient of the empirical risk
        which are computed during the normal backward pass.
    """
    c_func = {
        MSELoss: MSELoss_criterion,
        CrossEntropyLoss: CrossEntropyLoss_criterion,
    }[type(loss_func)]

    S = []
    for pred_n, y_n in zip(predictions, labels):
        c_n = c_func(pred_n, y_n)
        S.append(grad(c_n, pred_n)[0].detach())

    # concatenate over data points
    S = stack(S)

    # convert into list of single vector
    return [S]
