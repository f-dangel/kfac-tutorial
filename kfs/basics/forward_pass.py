"""Forward pass helpers that simplify computing KFAC."""

from collections import OrderedDict
from functools import partial
from typing import Dict, List, Set, Tuple

from torch import Tensor, allclose, manual_seed, rand
from torch.nn import Linear, Module, ReLU, Sequential
from torch.utils.hooks import RemovableHandle


def output_and_intermediates(
    model: Module, x: Tensor, names: Set[str]
) -> Dict[str, Tensor]:
    """Feed data through the net, store specified in/outputs.

    Args:
        x: The input to the first layer.
        model: The model whose in/outputs are stored.
        names: Names of layers whose in/outputs are stored.

    Returns:
        A dictionary with the stored in/outputs. Each key is
        the layer name prefixed by either 'in' or 'out'.
    """
    intermediates = {}

    # install hooks on modules whose in/outputs are stored
    handles: List[RemovableHandle] = []

    for name in names:
        hook = partial(
            store_io, name=name, destination=intermediates
        )
        layer = model.get_submodule(name)
        handles.append(layer.register_forward_hook(hook))

    output = model(x)

    # remove hooks
    for handle in handles:
        handle.remove()

    return output, intermediates


def store_io(
    module: Module,
    inputs: Tuple[Tensor],
    output: Tensor,
    name: str,
    destination: Dict[str, Tensor],
):
    """Store the in/output of a module in a dictionary.

    This function can be partially evaluated and then used
    as forward hook.

    Args:
        module: The layer whose in/output is stored.
        inputs: The inputs to the layer.
        output: The output of the layer.
        name: The name under which in/outputs are stored.
            Inputs/Outputs are suffixed with '_in/_out'.
        destination: Dictionary where in/outputs are stored.
    """
    destination[f"{name}_in"] = inputs[0]
    destination[f"{name}_out"] = output


if __name__ == "__main__":
    # Demonstrate functionality of `output_and_intermediates`
    manual_seed(0)  # make deterministic
    batch_size, D_in, D_hidden, D_out = 10, 5, 4, 3
    x = rand(batch_size, D_in)

    # layers of the sequential net
    f1 = Linear(D_in, D_hidden)
    f2 = ReLU()
    f3 = Linear(D_hidden, D_out)

    # forward pass and store ReLU layer inputs/outputs
    layers = OrderedDict({"f1": f1, "f2": f2, "f3": f3})
    net = Sequential(layers)
    f, intermediates = output_and_intermediates(
        net, x, {"f2"}
    )

    # compare with manual forward pass
    assert allclose(f, f3(f2(f1(x))))
    assert allclose(intermediates["f2_in"], f1(x))
    assert allclose(intermediates["f2_out"], f2(f1(x)))
