"""Flattening functions."""

from einops import rearrange
from torch import Tensor, allclose


def rvec(
    t: Tensor, start_dim: int = 0, end_dim: int = -1
) -> Tensor:
    """Flatten a tensor in last-varies-fastest fashion.

    See $\text{\Cref{def:rvec}}$.
    For a matrix, this corresponds to row-stacking.
    This is the common flattening scheme in code.

    Args:
        t: A tensor.
        start_dim: At which dimension to start flattening.
            Default is 0.
        end_dim: The last dimension to flatten. Default is -1.

    Returns:
        The flattened tensor.
    """
    return t.flatten(start_dim=start_dim, end_dim=end_dim)


def cvec(
    t: Tensor, start_dim: int = 0, end_dim: int = -1
) -> Tensor:
    """Flatten a tensor in first-varies-fastest fashion.

    See $\text{\Cref{def:cvec}}$.
    For a matrix, this corresponds to column-stacking.
    This is the common flattening scheme in literature.

    Args:
        t: A tensor.
        start_dim: At which dimension to start flattening.
            Default is 0.
        end_dim: The last dimension to flatten. Default is -1.

    Returns:
        The flattened tensor.
    """
    end_dim = end_dim if end_dim >= 0 else end_dim + t.ndim
    # flip index order, then flatten last-varies-fastest
    before = [f"s{i}" for i in range(start_dim)]
    active = [f"a{i}" for i in range(start_dim, end_dim + 1)]
    after = [f"s{i}" for i in range(end_dim + 1, t.ndim)]
    flipped = active[::-1]

    # build equation, e.g. "s0 a0 a1 s2 -> s0 a1 a0 s2"
    in_equation = " ".join(before + active + after)
    out_equation = " ".join(before + flipped + after)
    equation = f"{in_equation} -> {out_equation}"

    return rvec(
        rearrange(t, equation),
        start_dim=start_dim,
        end_dim=end_dim,
    )


if __name__ == "__main__":
    A = Tensor(
        [
            [1, 2],
            [3, 4],
        ]
    )
    assert allclose(rvec(A), Tensor([1, 2, 3, 4]))
    assert allclose(cvec(A), Tensor([1, 3, 2, 4]))
