"""Kronecker factor computation for ConvNd layer."""

from einops import rearrange
from torch import Tensor, cat
from torch.nn import Conv2d
from torch.nn.functional import unfold

from kfs.kfac.scaffold import KFAC


def input_based_factor(
    x_in: Tensor, layer: Conv2d
) -> Tensor:
    """Compute the input-based Kronecker factor `A`.

    Args:
        x_in: The batched input tensor to the layer. Has
            shape `(batch_size, C_in, I_1, I_2, ... I_N)`
            where `I_j` it the `j`-th spatial dimension.
        layer: The convolution layer for which the Kronecker
            factor is computed.

    Returns:
        The input-based Kronecker factor `A`. Let
        `K = K_1 * K_2 * ... * K_N` be the total number of
        kernel elements. Then, `A` has shape
        `(C_in * K + 1, C_in * K + 1` if the layer has a
        bias, otherwise `(C_in * K, C_in * K)`.

    Raises:
        NotImplementedError: If the input tensor does not
            have 2 spatial dimensions.
    """
    conv_dim = x_in.dim() - 2
    if conv_dim != 2:
        raise NotImplementedError(
            f"Conv{conv_dim}d is not supported (only 2d)."
        )
    x_in = unfold(
        x_in,
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        stride=layer.stride,
        padding=layer.padding,
    )
    if layer.bias is not None:
        # unfolded input has axes `(N, C_in * K, num_patches)`
        N, _, num_patches = x_in.shape
        x_in = cat(
            [x_in, x_in.new_ones(N, 1, num_patches)],
            dim=1,
        )
    # treat patch dimension like batch dimension
    x_in = rearrange(
        x_in, "n c_in_k patch -> (n patch) c_in_k"
    )
    return x_in.T @ x_in


def grad_output_based_factor(g_out: Tensor) -> Tensor:
    """Compute the gradient-based Kronecker factor `B`.

    Args:
        g_out: The batched layer output gradient tensor.
            shape `(batch_size, C_out, O_1, O_2, ... O_N)`
            where `O_j` it the `j`-th spatial dimension.

    Returns:
        The gradient-based Kronecker factor `B`.
        Has shape `(C_out, C_out)`.
    """
    # treat all shared dimensions like batch dimension
    g_out = rearrange(g_out, "n c_out ... -> (n ...) c_out")
    num_outer_products = g_out.shape[0]
    return (g_out.T @ g_out) / num_outer_products


# install both methods in the KFAC scaffold
setting = (Conv2d, "expand")
KFAC.COMPUTE_INPUT_BASED_FACTOR[setting] = (
    input_based_factor
)
KFAC.COMPUTE_GRAD_OUTPUT_BASED_FACTOR[setting] = (
    grad_output_based_factor
)
