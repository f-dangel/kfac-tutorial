"""Create a deep linear network."""

from collections import OrderedDict
from typing import List, Optional

from einops.layers.torch import Reduce
from torch.nn import Linear, Sequential


def create_deep_linear_net(
    dims: List[int],
    reduce_shared: Optional[str] = None,
    reduction_pos: Optional[int] = None,
    bias: bool = False,
) -> Sequential:
    L = len(dims) - 1
    reduction_pos = (
        L - 1 if reduction_pos is None else reduction_pos
    )

    layers = OrderedDict({})
    for i in range(L):
        d_in, d_out = dims[i], dims[i + 1]
        layers[f"f_{i}"] = Linear(d_in, d_out, bias=bias)

        # add reduction layer
        if i + 1 == reduction_pos:
            if reduce_shared in {"sum", "mean"}:
                layers["rho"] = Reduce(
                    "batch ... d_out -> batch d_out",
                    reduction=reduce_shared,
                )
            elif reduce_shared is not None:
                raise ValueError(
                    f"Unknown reduction: {reduce_shared}."
                )

    return Sequential(layers)
