from typing import List

from torch import Tensor
from torch.nn import CrossEntropyLoss, Module, MSELoss

from kfs.kfac import KFAC


def compute_backpropagated_vectors_input_only(
    output: Tensor, y: Tensor, loss_func: Module
) -> List[Tensor]:
    """Generate vectors to be backpropagated for KFAC.

    Args:
        output: The model's output.
        y: The empirical labels.
        loss_func: The loss function.
        fisher_type: The Fisher type to be approximated.

    Returns:
        A list of vectors to be backpropagated. Each vector
        has the same shape as `output`.
    """
    return []


# NOTE For any new supported loss function with 'input_only', we need to
# register this function
KFAC.COMPUTE_BACKPROPAGATED_VECTORS[
    (MSELoss, "input_only")
] = compute_backpropagated_vectors_input_only
KFAC.COMPUTE_BACKPROPAGATED_VECTORS[
    (CrossEntropyLoss, "input_only")
] = compute_backpropagated_vectors_input_only
