"""Output-weight Jacobian of a linear layer with weight sharing."""

from torch import allclose, eye, kron, manual_seed, rand

from kfs.jacobians import cvec_jac, rvec_jac

if __name__ == "__main__":
    manual_seed(0)  # make deterministic

    # set up combined parameters and augmented input
    D_in, D_out = 3, 2
    S = 5
    W_tilde = rand(D_out, D_in + 1, requires_grad=True)
    X_tilde = rand(D_in + 1, S)
    X_tilde[-1, :] = 1.0

    # set up the computation graph
    W, b = W_tilde[:, :-1], W_tilde[:, -1]
    X = X_tilde[:-1, :]
    Z = W @ X + b.unsqueeze(-1).repeat(1, S)

    # cvec Jacobian computation & comparison
    J_cvec = cvec_jac(Z, W_tilde)
    J_cvec_manual = kron(X_tilde.T.contiguous(), eye(D_out))
    assert allclose(J_cvec, J_cvec_manual)

    # rvec Jacobian computation & comparison
    J_rvec = rvec_jac(Z, W_tilde)
    J_rvec_manual = kron(eye(D_out), X_tilde.T.contiguous())
    assert allclose(J_rvec, J_rvec_manual)
