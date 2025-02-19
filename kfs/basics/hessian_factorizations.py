"""Symmetric factorization of criterion function Hessians."""

from einops import einsum
from torch import (
    Tensor,
    allclose,
    arange,
    manual_seed,
    outer,
    rand,
    randint,
    unravel_index,
    zeros,
)

from kfs.basics.hessians import hess
from kfs.basics.reduction_factors import (
    CrossEntropyLoss_criterion,
    MSELoss_criterion,
)


def symmetric_factorization_MSELoss(
    prediction: Tensor, target: Tensor
) -> Tensor:
    r"""Compute symmetric factorization of MSELoss Hessian.

    Implements $\mathrm{S}$ from
    $\text{\Cref{ex:mseloss_hessian_factorization}}$
    in tensor convention.

    Args:
        prediction: A single model prediction.
        target: A single label.

    Returns:
        The symmetric loss Hessian factorization.
        Has shape `(*prediction.shape, *prediction.shape)`.
    """
    dims_F = prediction.shape

    S = zeros(
        *dims_F,
        *dims_F,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    for d_raveled in arange(dims_F.numel()):
        d = unravel_index(d_raveled, dims_F)
        S[d][d] = 1.0

    return S


def symmetric_factorization_CrossEntropyLoss(
    prediction: Tensor, target: Tensor
) -> Tensor:
    r"""Compute symmetric factorization of CELoss Hessian.

    Implements $\mathrm{S}$ from
    $\text{\Cref{ex:crossentropyloss_hessian_factorization}}$
    in tensor convention.

    Args:
        prediction: A single model prediction (logits).
        target: A single label.

    Returns:
        The symmetric loss Hessian factorization.
        Has shape `(*prediction.shape, *prediction.shape)`.
    """
    num_classes = prediction.shape[0]
    dims_Y = prediction.shape[1:]

    # NOTE To simplify indexing, we move the class dimensions
    # to the end & later move them to their correct position
    p = prediction.softmax(0).movedim(0, -1)
    p_sqrt = p.sqrt()
    S = zeros(
        *dims_Y,
        *dims_Y,
        num_classes,
        num_classes,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    # *dims_Y, *dims_Y, num_classes, num_classes

    for d_raveled in arange(dims_Y.numel()):
        d = unravel_index(d_raveled, dims_Y)
        S[d][d] = p_sqrt[d].diag() - outer(p[d], p_sqrt[d])

    # move class dimensions to their correct position
    S = S.movedim(-2, 0).movedim(-1, len(dims_Y) + 1)
    # num_classes, *dims_Y, num_classes, *dims_Y

    return S


def tensor_outer(S: Tensor) -> Tensor:
    """Helper to compute the tensor outer product `S S.T`.

    Args:
        S: The symmetric factorization.

    Returns:
        The Hessian. Has same shape as `S`.
    """
    ndim_F = S.ndim // 2  # sum(dims_Y) + num_classes
    d_in = " ".join(f"d_in_{i}" for i in range(ndim_F))
    d_sum = " ".join(f"d_sum_{i}" for i in range(ndim_F))
    d_out = " ".join(f"d_out_{i}" for i in range(ndim_F))
    equation = (
        f"{d_in} {d_sum}, {d_out} {d_sum} -> {d_in} {d_out}"
    )
    return einsum(S, S, equation)


if __name__ == "__main__":
    manual_seed(0)
    dims_Y = (2, 3, 4)

    # regression
    dims_F = dims_Y
    f = rand(*dims_F, requires_grad=True)
    y = rand(*dims_F)

    S = symmetric_factorization_MSELoss(f, y)
    assert S.shape == (*dims_F, *dims_F)
    S_ST = tensor_outer(S)
    # compare with autodiff Hessian
    H = hess(MSELoss_criterion(f, y), f)
    assert allclose(H, S_ST)

    # classification
    num_classes = 5
    dims_F = (num_classes, *dims_Y)
    f = rand(*dims_F, requires_grad=True)
    y = randint(low=0, high=num_classes, size=dims_Y)

    S = symmetric_factorization_CrossEntropyLoss(f, y)
    assert S.shape == (*dims_F, *dims_F)
    S_ST = tensor_outer(S)
    # compare with autodiff Hessian
    H = hess(CrossEntropyLoss_criterion(f, y), f)
    assert allclose(H, S_ST)
