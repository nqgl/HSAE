# %%

import torch
from torch.sparse import _triton_ops as sto
import einops

sto.bsr_scatter_mm
from baseline import generate_example

# if options.sub_b_dec:
#     x_cent = x_cent - b_dec.float()
o = generate_example(8, 10, 12, 16, xlist=False)
x = o.x
W_enc = o.W_enc
gate = o.gate
b_enc = o.b_enc
b_dec = o.b_dec
W_dec = o.W_dec

x = x.unsqueeze(-2)


# m = x_cent @ W_enc
print("x", x.shape)
print("W_enc", W_enc.shape)
print("gate", gate.shape)
# x_rest = einops.rearrange(x, "b d_data -> b d_data")
W_enc_rest = einops.rearrange(W_enc, "n_sae d_data d_dict -> d_data (n_sae d_dict)")
n_sae = W_enc.shape[0]
test = torch.zeros(64, 64, device="cuda")
test[0, 0] = 1
test[1, 1] = 1
tg = test.to_sparse_bsr((16, 16))

gate_values = torch.ones(1, n_sae, device=gate.device)

A = torch.ones(64, 20, device="cuda")
B = torch.ones(20, 64, device="cuda", requires_grad=True)
# o = sto.bsr_dense_addmm(input=tg, mat2=W_enc, mat1=x,
# o = sto.sampled_addmm(input=tg, mat2=B, mat1=A)
o
# m = sto.sampled_addmm(input=gate.to_sparse(), mat2=W_enc, mat1=x)
# o = torch.sparse.sampled_addmm(input=tg, mat2=B, mat1=A)
# o.sum().backward()
# B.grad

# %%
vecsparse = torch.zeros(4, 64, 64, device="cuda")
for i in range(64):
    if i in (0, 33, 43):
        vecsparse[
            torch.arange(0, 4, device="cuda"), torch.arange(i, i + 4, device="cuda"), :
        ] = torch.rand(64, device="cuda")
# vecsparse = vecsparse.to_sparse(1)
vecsparse = vecsparse.to_sparse_csr()

print(vecsparse.shape, B.shape, A.shape)
o = torch.sparse.sampled_addmm(input=vecsparse, mat2=B, mat1=A)

# o.sum().backward()
vecsparse
o

torch.sparse_compressed_tensor()

# %%

o2 = o @ A
o2.sum().backward()

# %%

d_dict = W_enc.shape[-1]
gate_rest1 = einops.repeat(gate, "batch n_sae -> batch (n_sae d_dict)", d_dict=d_dict)
gate_rest1 = gate_rest1.to_sparse_csr()


gate_rest2 = einops.repeat(gate, "batch n_sae -> batch (d_dict n_sae)", d_dict=d_dict)
gate_rest2 = gate_rest2.to_sparse_csr()


gate_rest3 = einops.repeat(gate, "batch n_sae -> (n_sae d_dict) batch", d_dict=d_dict)
gate_rest3 = gate_rest3.to_sparse_csr()


gate_rest4 = einops.repeat(gate, "batch n_sae -> (d_dict n_sae) batch", d_dict=d_dict)
gate_rest4 = gate_rest4.to_sparse_csr()

# %%
print(gate_rest1.crow_indices().size())
print(gate_rest1.col_indices().size())
print(gate_rest1.values().size())
print(gate_rest2.crow_indices().size())
print(gate_rest1.col_indices().size())
print(gate_rest2.values().size())
print(gate_rest3.crow_indices().size())
print(gate_rest1.col_indices().size())
print(gate_rest3.values().size())
print(gate_rest4.crow_indices().size())
print(gate_rest1.col_indices().size())
print(gate_rest4.values().size())
# %%
gate_rest_vecs = einops.repeat(gate, "batch n_sae -> batch n_sae d_dict", d_dict=d_dict)
gate_rest_vecs = gate_rest_vecs.to_sparse_csr(1)

# %%
gate_rest_vecs
# %%
