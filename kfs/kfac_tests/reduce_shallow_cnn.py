from einops.layers.torch import Reduce
from torch import kron, manual_seed, rand
from torch.nn import Conv2d, MSELoss, Sequential

from kfs.basics.ggns import vec_ggn
from kfs.kfac import KFAC
from kfs.utils import report_nonclose

manual_seed(0)
batch_size = 8
C_in, C_out = 2, 3
spatial_in = (32, 40)
kernel_size = (4, 5)
stride = (2, 3)
padding = (2, 1)

model = Sequential(
    # Conv2d(C_in, C_out, kernel_size, bias=False),
    Conv2d(C_in, 3, 3, padding=1, bias=False),
    Conv2d(3, C_out, 3, padding=1, bias=False),
    Reduce(
        "batch c_out o1 o2 -> batch c_out", reduction="sum"
    ),
)
loss_func = MSELoss(reduction="sum")

X = rand(batch_size, C_in, *spatial_in)
y = rand(batch_size, C_out)

# compute the GGN with autodiff
prediction = model(X)
loss = loss_func(prediction, y)
rvec_ggns = [
    vec_ggn(loss, p, prediction, "rvec")
    for p in model.parameters()
]
cvec_ggns = [
    vec_ggn(loss, p, prediction, "cvec")
    for p in model.parameters()
]

# compute KFAC-reduce-type-2 and compare
kfacs = KFAC.compute(
    model, loss_func, (X, y), "type-2", "reduce"
)
assert len(rvec_ggns) == len(kfacs) == len(cvec_ggns)
for idx, (G, (A, B)) in enumerate(
    zip(rvec_ggns, kfacs.values())
):
    if idx == 0:
        continue
    report_nonclose(G, kron(B, A))

# for G, (A, B) in zip(cvec_ggns, kfacs.values()):
#     print(G.shape, A.shape, B.shape)
#     print(B)
#     report_nonclose(G, kron(A, B))
