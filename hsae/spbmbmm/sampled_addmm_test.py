#%%
import torch
from torch import sampled_addmm

import baseline
# %%
ex = baseline.generate_example(8, 10, 12, 16, xlist=False)

x = ex.x
M = ex.W_enc
gate = ex.gate
x = x.unsqueeze(-2).unsqueeze(-2)
M = M.unsqueeze(0)
gate = gate.unsqueeze(-1).unsqueeze(-1)
x.shape, M.shape, gate.shape
(x @ M).shape

sampled_addmm(input=gate.to_sparse(), mat2=M, mat1=x)
# %%
