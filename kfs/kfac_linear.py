"""Compute KFAC factors for Linear layers."""

from collections import OrderedDict
from typing import Dict, List, Union

from einops import einsum
from torch import Tensor, allclose, kron, manual_seed, rand
from torch.autograd import grad
from torch.nn import (
    CrossEntropyLoss,
    Linear,
    Module,
    MSELoss,
    Sequential,
)

from kfs.forward_pass import output_and_intermediates
from kfs.ggns import vec_ggn
from kfs.kfac import KFAC


def compute_A_Linear_expand(
    layer: Linear, inputs: Tensor
) -> Tensor:
    """Compute input-based Kronecker factor for a Linear layer.

    Args:
        layer: The Linear layer.
        inputs: The input to the layer.

    Returns:
        The input-based Kronecker factor.
    """
    assert layer.bias is None
    return einsum(
        inputs,
        inputs,
        "batch ... i1, batch ... i2 -> i1 i2",
    )


# register new functionality
KFAC.COMPUTE_INPUT_BASED_FACTOR[(Linear, "expand")] = (
    compute_A_Linear_expand
)


if __name__ == "__main__":
    from kfs import (
        backpropagate_vectors_input_only,  # registers required functionality
    )

    manual_seed(0)

    X, y = rand(1, 3), rand(1, 2)
    layers = OrderedDict(
        {"dense": Linear(3, 2, bias=False)}
    )
    model = Sequential(layers)
    loss_func = MSELoss()

    kfacs = KFAC.compute(
        model,
        loss_func,
        (X, y),
        "input_only",
        "expand",
        None,
    )
    assert set(kfacs.keys()) == {"dense"}

    # GGN computation
    rvec_ggns = {}
    cvec_ggns = {}

    output = model(X)
    loss = loss_func(output, y)

    for name in kfacs:
        layer = model.get_submodule(name)
        (p,) = layer.parameters()
        rvec_ggns[name] = vec_ggn(loss, p, output, "rvec")
        cvec_ggns[name] = vec_ggn(loss, p, output, "cvec")

    for name, (kfac_A, kfac_B) in kfacs.items():
        assert allclose(
            kron(kfac_A, kfac_B), cvec_ggns[name]
        )
        assert allclose(
            kron(kfac_B, kfac_A), rvec_ggns[name]
        )
