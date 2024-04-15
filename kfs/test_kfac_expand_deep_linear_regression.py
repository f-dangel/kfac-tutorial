"""Test KFAC for square loss with a deep MLP."""

from collections import OrderedDict

from torch import allclose, kron, manual_seed, rand
from torch.nn import Linear, MSELoss, Sequential

from kfs import KFAC
from kfs.ggns import vec_ggn

manual_seed(0)

batch_size = 5
D_in, D_hidden, D_out = 4, 3, 2
# use double-precision for exact comparison
X = rand(batch_size, D_in).double()
y = rand(batch_size, D_out).double()

layers = OrderedDict(
    {
        "linear1": Linear(D_in, D_hidden, bias=False),
        "linear2": Linear(D_hidden, D_hidden, bias=False),
        "linear3": Linear(D_hidden, D_out, bias=False),
    }
)
model = Sequential(layers).double()
loss_func = MSELoss(reduction="sum").double()

# compute GGN with autodiff
output = model(X)
loss = loss_func(output, y)
ggn_blocks = [
    vec_ggn(loss, p, output, "rvec")
    for p in model.parameters()
]

# compute KFAC type-2
kfac_blocks = KFAC.compute(
    model, loss_func, (X, y), "type-2", "expand", None
)

# compare
assert len(ggn_blocks) == len(kfac_blocks)
for G, (A, B) in zip(ggn_blocks, kfac_blocks.values()):
    G_kfac = kron(B, A)
    assert allclose(G, G_kfac)

# compute KFAC-MC with a large number of samples
kfac_blocks = KFAC.compute(
    model, loss_func, (X, y), "mc=25_000", "expand", None
)
for G, (A, B) in zip(ggn_blocks, kfac_blocks.values()):
    G_kfac = kron(B, A)
    assert allclose(G, G_kfac, rtol=5e-2, atol=5e-4)
