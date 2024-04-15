from math import sqrt
from typing import List

from torch import Tensor, stack
from torch.autograd import grad
from torch.nn import CrossEntropyLoss, Module, MSELoss

from kfs.hessian_factorizations import (
    symmetric_factorization_CrossEntropyLoss,
    symmetric_factorization_MSELoss,
)
from kfs.label_sampling import (
    draw_label_CrossEntropyLoss,
    draw_label_MSELoss,
)
from kfs.reduction_factors import (
    CrossEntropyLoss_criterion,
    MSELoss_criterion,
)


def compute_backpropagated_vectors(
    loss_func, fisher_type, predictions, labels
) -> List[Tensor]:
    if fisher_type == "type-2":
        return backpropagated_vectors_type2(
            loss_func, predictions, labels
        )
    elif fisher_type.startswith("mc="):
        mc_samples = int(fisher_type.replace("mc=", ""))
        return backpropagated_vectors_mc(
            loss_func, predictions, labels, mc_samples
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
    if isinstance(loss_func, MSELoss):
        S_func = symmetric_factorization_MSELoss
    elif isinstance(loss_func, CrossEntropyLoss):
        S_func = symmetric_factorization_CrossEntropyLoss
    else:
        raise ValueError(
            f"Unknown loss function: {type(loss_func)}."
        )

    dim_Y = labels.ndim - 1
    S = []
    for pred_n, y_n in zip(
        predictions.split(1), labels.split(1)
    ):
        pred_n, y_n = pred_n.squeeze(0), y_n.squeeze(0)
        S_n = S_func(pred_n, y_n)
        # flatten the column dimension
        S_n = S_n.flatten(end_dim=dim_Y - 1)
        S.append(S_n)
    # concatenate over all data points
    S = stack(S)
    # move the column dimension to leading axis
    S = S.moveaxis(-1, 0)
    # convert into list, ith entry contains the ith columns of all
    # symmetric Hessian decompositions
    return [s.squeeze(0) for s in S.split(1)]


def backpropagated_vectors_mc(
    loss_func: Module,
    predictions: Tensor,
    labels: Tensor,
    mc_samples: int,
) -> List[Tensor]:
    if isinstance(loss_func, MSELoss):
        c_func = MSELoss_criterion
        sample_label_func = draw_label_MSELoss
    elif isinstance(loss_func, CrossEntropyLoss):
        c_func = CrossEntropyLoss_criterion
        sample_label_func = draw_label_CrossEntropyLoss
    else:
        raise ValueError(
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
