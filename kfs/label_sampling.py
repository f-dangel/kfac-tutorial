"""Sampling labels from a model's likelihood."""

from torch import Tensor, normal


def draw_label_MSELoss(prediction: Tensor) -> Tensor:
    """Sample a label from the likelihood implied by MSELoss.

    Args:
        prediction: The model's prediction for one datum.
    """
    return normal(prediction, std=1.0)


def draw_label_CrossEntropyLoss(prediction: Tensor) -> Tensor:
    """Sample a label from the likelihood implied be CELoss.

    Args:
        prediction: The model's prediction for one datum.
    """
    dim_Y = prediction.shape[1:]
    p = prediction.softmax(dim=0)

    # multinomial must act on a matrix whose rows are probability densities
    p_mat = p.flatten(start_dim=1).T
    y = p_mat.multinomial(num_samples=1, replacement=True)

    return y.reshape(dim_Y)
