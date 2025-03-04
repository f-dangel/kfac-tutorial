"""Sampling labels from a model's likelihood."""

from torch import Tensor, normal, Generator


def draw_label_MSELoss(
    prediction: Tensor, generator: Generator | None = None
) -> Tensor:
    """Sample a label from the likelihood implied by MSELoss.

    Args:
        prediction: The model's prediction for one datum.
        rng: Optional random number generator.

    Returns:
        A sample from the likelihood implied by MSELoss.
        Has same shape as `prediction`.
    """
    return normal(prediction, std=1.0, generator=generator)


def draw_label_CrossEntropyLoss(
    prediction: Tensor, generator: Generator | None = None
) -> Tensor:
    """Sample a label from the likelihood implied be CELoss.

    Args:
        prediction: The model's prediction for one datum.
            Has shape `(num_classes, *dims_Y)`.
        rng: Optional random number generator.

    Returns:
        A sample from the likelihood implied by CELoss.
        Has shape `(*dims_Y)`.
    """
    dim_Y = prediction.shape[1:]
    p = prediction.softmax(dim=0)

    # multinomial takes a matrix whose rows are probabilities
    p = p.unsqueeze(-1) if p.ndim == 1 else p
    p_mat = p.flatten(start_dim=1).T
    y = p_mat.multinomial(
        num_samples=1, replacement=True, generator=generator
    )

    return y.reshape(dim_Y)
