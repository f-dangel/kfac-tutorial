"""Output-weight Jacobian of a linear layer."""

from torch import allclose, eye, kron, manual_seed, rand

from kfs.basics.jacobians import vec_jac

if __name__ == "__main__":
    manual_seed(0)  # make deterministic

    # set up combined parameters and augmented input
    D_in, D_out = 3, 2
    W_tilde = rand(D_out, D_in + 1, requires_grad=True)
    x_tilde = rand(D_in + 1)
    x_tilde[-1] = 1.0

    # set up the computation graph
    W, b = W_tilde[:, :-1], W_tilde[:, -1]
    x = x_tilde[:-1]
    z = W @ x + b

    # cvec Jacobian computation & comparison
    J_cvec = vec_jac(z, W_tilde, vec="cvec")
    J_cvec_manual = kron(x_tilde.unsqueeze(0), eye(D_out))
    assert allclose(J_cvec, J_cvec_manual)

    # rvec Jacobian computation & comparison
    J_rvec = vec_jac(z, W_tilde, vec="rvec")
    J_rvec_manual = kron(eye(D_out), x_tilde.unsqueeze(0))
    assert allclose(J_rvec, J_rvec_manual)
