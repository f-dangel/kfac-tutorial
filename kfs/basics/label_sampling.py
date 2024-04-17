"""Sampling labels from a model's likelihood."""

from torch import Tensor, normal


def draw_label_MSELoss(prediction: Tensor) -> Tensor:
    """Sample a label from the likelihood implied by MSELoss.

    Args:
        prediction: The model's prediction for one datum.

    Returns:
        A sample from the likelihood implied by MSELoss.
        Has same shape as `prediction`.
    """
    return normal(prediction, std=1.0)


def draw_label_CrossEntropyLoss(
    prediction: Tensor,
) -> Tensor:
    """Sample a label from the likelihood implied be CELoss.

    Args:
        prediction: The model's prediction for one datum.
            Has shape `(num_classes, *dims_Y)`.

    Returns:
        A sample from the likelihood implied by CELoss.
        Has shape `(*dims_Y)`.
    """
    dim_Y = prediction.shape[1:]
    p = prediction.softmax(dim=0)

    # multinomial takes a matrix whose rows are probabilities
    p_mat = p.flatten(start_dim=1).T
    y = p_mat.multinomial(num_samples=1, replacement=True)

    return y.reshape(dim_Y)
