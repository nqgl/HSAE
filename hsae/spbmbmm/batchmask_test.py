import torch
from nqgl.matmul_with_batchmask.matmul_mask import MatMultWithBatchMask
import baseline
# %%
ex = baseline.generate_example(8, 10, 12, 16, xlist=False)

x = ex.x
M = ex.W_enc
gate = ex.gate
x = x.unsqueeze(-2).unsqueeze(-2)
M = M.unsqueeze(0)
gate = gate
MatMultWithBatchMask.apply(x, M, gate.to_sparse())
# %%
