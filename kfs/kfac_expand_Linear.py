"""Kronecker factor computation for Linear layer."""

from einops import rearrange
from torch import Tensor
from torch.nn import Linear

from kfs.kfac import KFAC


def input_based_factor(x_in: Tensor) -> Tensor:
    batch_size = x_in.shape[0]
    x_in = rearrange(x_in, "n ... d_in -> (n ...) d_in")
    return (x_in.T @ x_in) / batch_size


KFAC.COMPUTE_INPUT_BASED_FACTOR[(Linear, "expand")] = (
    input_based_factor
)


def grad_output_based_factor(g_out: Tensor) -> Tensor:
    g_out = rearrange(g_out, "n ... d_out -> (n ...) d_out")
    return g_out.T @ g_out


KFAC.COMPUTE_GRAD_OUTPUT_BASED_FACTOR[
    (Linear, "expand")
] = grad_output_based_factor
