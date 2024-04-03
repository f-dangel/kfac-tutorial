"""Kronecker-factored approximate curvature."""

from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Set,
    Tuple,
    Type,
    Union,
)

from torch import Tensor, eye
from torch.autograd import grad
from torch.nn import CrossEntropyLoss, Module, MSELoss

from kfs.forward_pass import output_and_intermediates


class KFAC:
    """Class for computing the KFAC approximation.

    This class defines a scaffold for computing KFAC.
    We will register develop the functionality to support many
    different KFAC flavours in the following chapters and
    register it in this class.

    Attributes:
        SUPPORTED_LOSS_FUNCS: The loss functions KFAC supports.
            New loss functions can be added by extending this set.
        SUPPORTED_MODULES: The layers KFAC supports.
            New layers can be added by extending this set.
        SUPPORTED_FISHER_TYPES: The Fisher types KFAC supports.
            New Fisher types can be added by extending this set.
        SUPPORTED_KFAC_APPROX: The KFAC approximations KFAC supports.
            New approximations can be added by extending this set.
        SUPPORTED_LOSS_AVERAGE: The loss averaging methods KFAC supports.
            New loss averaging methods can be added by extending this set.
    """

    # we will register supported loss functions, layers,
    # Fisher types
    COMPUTE_BACKPROPAGATED_VECTORS: Dict[
        Tuple[Type[Module], str], Callable
    ] = {}
    COMPUTE_INPUT_BASED_FACTOR: Dict[
        Tuple[Type[Module], str], Callable[[Tensor], Tensor]
    ] = {}

    @classmethod
    def compute(
        cls,
        model: Module,
        loss_func: Union[MSELoss, CrossEntropyLoss],
        data: Tuple[Tensor, Tensor],
        fisher_type: str,
        kfac_approx: str,
        loss_average: Union[None, str],
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """Compute KFAC for all supported layers.

        Args:
            model: The model whose KFAC factors are computed.

        Returns:
            A dictionary whose keys are the layer names and
            whose values are tuples of the input-based and
            grad-output-based Kronecker factors of KFAC.
        """
        # determine layers that support the KFAC computation
        supported = {
            layer
            for layer, approx in cls.COMPUTE_INPUT_BASED_FACTOR
            if approx == kfac_approx
        }

        layers = {  # find supported layers
            name: layer
            for name, layer in model.named_modules()
            if isinstance(layer, tuple(supported))
        }
        As: Dict[str, Tensor] = {}  # input-based factors
        Bs: Dict[str, Tensor] = (
            {}
        )  # grad-output based factors

        # forward pass, storing layer inputs and outputs
        X, y = data
        output, intermediates = output_and_intermediates(
            model, X, layers.keys()
        )

        # compute input-based Kronecker factors
        for name, layer in layers.items():
            compute_A = cls.COMPUTE_INPUT_BASED_FACTOR[
                (type(layer), kfac_approx)
            ]
            inputs = intermediates.pop(f"{name}_in")
            As[name] = compute_A(layer, inputs)

        # generate vectors to be backpropagated
        layer_outputs = [
            intermediates.pop(f"{name}_out")
            for name in layers
        ]

        backpropagated_vectors = (
            cls.COMPUTE_BACKPROPAGATED_VECTORS[
                (type(loss_func), fisher_type)
            ](output, y, loss_func)
        )

        for v in backpropagated_vectors:
            grad_outputs = grad(
                output, layer_outputs, grad_output=v
            )

            # fix scale from reduction
            R = ...
            for g in grad_outputs:
                g.mul_(R)

            for (name, layer), g_out in zip(
                layers.items(), grad_outputs
            ):
                # compute grad-output based Kronecker factor
                compute_B = (
                    cls.COMPUTE_GRAD_OUTPUT_BASED_FACTOR[
                        type(layer)
                    ]
                )
                B = compute_B(g_out)
                Bs[name] = (
                    B if name not in Bs else B + Bs[name]
                )

        # if there were no backpropagated vectors, set B to identity
        if not Bs:
            for name, layer in layers.items():
                weight = layer.weight
                # This holds for the layers of this tutorial, but may not hold in general!
                dim_B = weight.shape[0]
                Bs[name] = eye(
                    dim_B,
                    device=weight.device,
                    dtype=weight.dtype,
                )

        return {
            name: (As[name], Bs[name]) for name in layers
        }
