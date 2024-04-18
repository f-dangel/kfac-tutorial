"""Kronecker-factored approximate curvature."""

from typing import Callable, Dict, Tuple, Type, Union

from torch import Tensor, eye
from torch.autograd import grad
from torch.nn import CrossEntropyLoss, Module, MSELoss

from kfs.basics.forward_pass import output_and_intermediates
from kfs.basics.reduction_factors import (
    get_reduction_factor,
)
from kfs.kfac.backpropagated_vectors import (
    compute_backpropagated_vectors,
)


class KFAC:
    """Class for computing the KFAC approximation.

    Note:
        This class defines a scaffold for computing KFAC.
        We will register develop and register the
        functionality to support many different flavours
        in the following chapters. Import this class via
        `from kfs.kfac import KFAC` to use it with full
        functionality.

    Attributes:
        COMPUTE_INPUT_BASED_FACTOR: A dictionary that maps
            layer types and KFAC flavours to functions that
            compute the input-based Kronecker factor.
        COMPUTE_GRAD_OUTPUT_BASED_FACTOR: A dictionary that
            maps layer types and KFAC flavours to functions
            that compute the gradient-based Kronecker factor.
    """

    COMPUTE_INPUT_BASED_FACTOR: Dict[
        Tuple[Type[Module], str],
        Callable[[Tensor, bool], Tensor],
    ] = {}
    COMPUTE_GRAD_OUTPUT_BASED_FACTOR: Dict[
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
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """Compute KFAC for all supported layers.

        Args:
            model: The model whose KFAC factors are computed.
            loss_func: The loss function.
            data: A batch of inputs and labels.
            fisher_type: The type of Fisher approximation.
            kfac_approx: The type of KFAC approximation.

        Returns:
            A dictionary whose keys are the layer names and
            whose values are tuples of the input-based and
            grad-output-based Kronecker factors of KFAC.
        """
        # determine layers that support the KFAC computation
        supported = {
            mod
            for mod, approx in cls.COMPUTE_INPUT_BASED_FACTOR
            if approx == kfac_approx
        }

        layers = {  # find supported layers
            name: mod
            for name, mod in model.named_modules()
            if isinstance(mod, tuple(supported))
        }
        As: Dict[str, Tensor] = {}  # input-based factors
        Bs: Dict[str, Tensor] = (
            {}
        )  # grad-output based factors

        # forward pass, storing layer inputs and outputs
        X, y = data
        prediction, intermediates = (
            output_and_intermediates(
                model, X, layers.keys()
            )
        )

        # compute input-based Kronecker factors
        for name, layer in layers.items():
            compute_A = cls.COMPUTE_INPUT_BASED_FACTOR[
                (type(layer), kfac_approx)
            ]
            inputs = intermediates.pop(f"{name}_in")
            bias_augment = layer.bias is not None
            As[name] = compute_A(inputs, bias_augment)

        # factor in loss reduction
        R = get_reduction_factor(loss_func, y)
        for A in As.values():
            A.mul_(R)

        # generate vectors to be backpropagated
        backpropagated_vectors = (
            compute_backpropagated_vectors(
                loss_func, fisher_type, prediction, y
            )
        )
        layer_outputs = [
            intermediates.pop(f"{name}_out")
            for name in layers
        ]

        for v in backpropagated_vectors:
            grad_outputs = grad(
                prediction,
                layer_outputs,
                grad_outputs=v,
                retain_graph=True,
            )

            for (name, layer), g_out in zip(
                layers.items(), grad_outputs
            ):
                # compute grad-output based Kronecker factor
                compute_B = (
                    cls.COMPUTE_GRAD_OUTPUT_BASED_FACTOR[
                        (type(layer), kfac_approx)
                    ]
                )
                B = compute_B(g_out)
                Bs[name] = (
                    B if name not in Bs else B + Bs[name]
                )

        # if there were no backpropagated vectors, set B=I
        if not backpropagated_vectors:
            for name, layer in layers.items():
                W = layer.weight
                # This holds for the layers of this tutorial,
                # but may not hold in general!
                dim_B = W.shape[0]
                Bs[name] = eye(
                    dim_B, device=W.device, dtype=W.dtype
                )

        return {
            name: (As[name], Bs[name]) for name in layers
        }
