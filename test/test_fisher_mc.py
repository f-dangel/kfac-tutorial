from torch import allclose, manual_seed
from torch.nn import MSELoss

from kfs.basics.mc_fishers import vec_mcfisher
from kfs.basics.mc_fishers_quick import vec_mcfisher_quick
from kfs.plots.synthetic_fisher import (
    curv_mat,
    get_model,
    get_rand_data,
)


def test_mcfisher_vp_equals_matrix():
    manual_seed(0)
    N = 100
    D_in, D_hidden, D_out = 5, 4, 3
    X, y = get_rand_data(N, D_in, D_out)
    model = get_model(D_in, D_hidden, D_out)

    loss_func = MSELoss()
    rvec_mcfisher_quick = (
        lambda p_i, p_j: vec_mcfisher_quick(
            model,
            loss_func,
            X,
            y,
            (p_i, p_j),
            1,
            vec="rvec",
            seed=42,
        )
    )
    rvec_mcfisher_slow = lambda p_i, p_j: vec_mcfisher(
        model,
        loss_func,
        X,
        y,
        (p_i, p_j),
        1,
        vec="rvec",
        seed=42,
    )

    curv_mat_quick = curv_mat(
        rvec_mcfisher_quick, model, bda=False
    )
    curv_mat_slow = curv_mat(
        rvec_mcfisher_slow, model, bda=False
    )

    assert allclose(curv_mat_quick, curv_mat_slow)
