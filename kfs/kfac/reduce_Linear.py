"""Kronecker factor computation for Linear layer."""

from einops import reduce
from torch import Tensor, cat
from torch.nn import Linear

from kfs.kfac.scaffold import KFAC


def input_based_factor(
    x_in: Tensor, layer: Linear
) -> Tensor:
    """Compute the input-based Kronecker factor `A`.

    Args:
        x_in: The batched input tensor to the layer.
            Has shape `(batch_size, *, in_features)`.
        layer: The linear layer for which the Kronecker
            factor is computed.

    Returns:
        The input-based Kronecker factor `A`. Has shape
        `(in_features + 1, in_features + 1)` if the layer
        has a bias, otherwise `(in_features, in_features)`.
    """
    if layer.bias is not None:
        x_in = cat(
            [x_in, x_in.new_ones(*x_in.shape[:-1], 1)],
            dim=-1,
        )
    x_in = reduce(x_in, "n ... d_in -> n d_in", "sum")
    return x_in.T @ x_in


def grad_output_based_factor(g_out: Tensor) -> Tensor:
    """Compute the gradient-based Kronecker factor `B`.

    Args:
        g_out: The batched layer output gradient tensor.
            Has shape `(batch_size, *, out_features)`.

    Returns:
        The gradient-based Kronecker factor `B`.
        Has shape `(out_features, out_features)`.
    """
    g_out = reduce(g_out, "n ... d_out -> n d_out", "mean")
    num_outer_products = g_out.shape[0]
    return (g_out.T @ g_out) / num_outer_products


# install both methods in the KFAC scaffold
setting = (Linear, "reduce")
KFAC.COMPUTE_INPUT_BASED_FACTOR[setting] = (
    input_based_factor
)
KFAC.COMPUTE_GRAD_OUTPUT_BASED_FACTOR[setting] = (
    grad_output_based_factor
)
