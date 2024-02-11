# %%

import torch
from torch.sparse import _triton_ops as sto
import einops

sto.bsr_scatter_mm
from nqgl.sae.hsae.spbmbmm.baseline import generate_example
import torch.nn.functional as F

# %%
# if options.sub_b_dec:
#     x_cent = x_cent - b_dec.float()
o = generate_example(32, 256, 512, 32, xlist=False)
x = o.x
W_enc = o.W_enc
gate = o.gate
b_enc = o.b_enc
b_dec = o.b_dec
W_dec = o.W_dec

# x = x.unsqueeze(-2)


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

A = torch.ones(4, 64, 20, device="cuda")
B = torch.ones(4, 20, 64, device="cuda", requires_grad=True)
# o = sto.bsr_dense_addmm(input=tg, mat2=W_enc, mat1=x,
o = sto.sampled_addmm(input=tg, mat2=B, mat1=A)
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
vecsparse = vecsparse.to_sparse_coo()
# vecsparse = torch.sparse_compressed_tensor(
#     compressed_indices=vecsparse.crow_indices(),
#     plain_indices=vecsparse.col_indices(),
#     values=vecsparse.values(),
#     layout=torch.sparse_csr,
# )


print(vecsparse.shape, B.shape, A.shape)
o = torch.sparse.sampled_addmm(input=vecsparse, mat2=B, mat1=A)
torch.sparse.a
# o.sum().backward()
vecsparse
o

# torch.sparse_compressed_tensor()

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

batch = x.shape[0]
n_sae, d_data, d_dict = W_enc.shape


#     | if options.sub_b_dec:
#     |     x_cent = x_cent - b_dec.float()
#     | to do this, do b_dec @ W_enc then make it gate-patterned and negative, then use as input to sampled_addmm
# m = x_cent @ W_enc
W_enc_rest = einops.rearrange(W_enc, "n_sae d_data d_dict -> d_data n_sae d_dict")
# gate_rest = gate.repeat(1, d_dict)  # maybe wrong order here
gate_rest = einops.repeat(gate, "batch n_sae -> batch n_sae d_dict", d_dict=d_dict)
gate_rest = gate_rest.to_sparse_csr(1)

# %%


class SampledAddmmBSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mat2, mat1):
        ctx.save_for_backward(input, mat2, mat1)
        return sto.sampled_addmm(input, mat2, mat1)

    @staticmethod
    def backward(ctx, grad_output):
        input, mat2, mat1 = ctx.saved_tensors
        grad_input, grad_mat2, grad_mat1 = sampled_addmm_backward(
            grad_output, input, mat2, mat1
        )
        print("backward")
        return grad_input.to_dense(), grad_mat2, grad_mat1


def sampled_addmm_backward(grad_output, input, mat2, mat1):
    print("grad output:", grad_output)
    print("input:", input)
    mask = torch.sparse_bsr_tensor(
        crow_indices=input.crow_indices(),
        col_indices=input.col_indices(),
        values=torch.ones_like(input.values()),
    )
    print("go", grad_output.layout)
    gradmasked = grad_output * mask
    print("gm", gradmasked.layout)
    print(input.layout, mat2.layout, mat1.layout)
    grad_mat1 = mat2.t() @ ()
    grad_mat2 = (grad_output * mask) @ mat1.t()
    grad_input = grad_output

    return grad_input, grad_mat2, grad_mat1


# %%


def test_sampled_addmm(A, B, C, sampled_addmm=torch.sparse.sampled_addmm):

    A = torch.nn.Parameter(A, requires_grad=True)
    B = torch.nn.Parameter(B, requires_grad=True)
    C = torch.nn.Parameter(C, requires_grad=True)

    o = sampled_addmm(C, B, A)
    print("oshape", o.shape)
    o.sum().backward()
    agrad = A.grad.clone().detach()
    bgrad = B.grad.clone().detach()
    cgrad = C.grad.clone().detach()
    return o, agrad, bgrad, cgrad


B = torch.rand(4, 64, 32, device="cuda", requires_grad=True)
A = torch.rand(4, 32, 128, device="cuda", requires_grad=True)
C = torch.rand(4, 64, 128, device="cuda", requires_grad=False)
C[:, 16:, :32] = 0

C_bsr = C.to_sparse_bsr((16, 16))
# C_bsr = torch.nn.Parameter(C_bsr, requires_grad=True)
o1 = test_sampled_addmm(
    A.clone(), B.clone(), C_bsr, sampled_addmm=SampledAddmmBSR.apply
)

C_csr = C.to_sparse_csr()

o2 = test_sampled_addmm(
    A.clone(),
    B.clone(),
    C_csr,
    sampled_addmm=torch.sparse.sampled_addmm,
)
# o2 = test_sampled_addmm(
#     B.clone(), A.clone(), C_csr, sampled_addmm=torch.sparse.sampled_addmm
# )
# o2 = (o2[0], o2[2], o2[1], o2[3])


# %%
for i in range(4):
    print(o1[i].shape, o2[i].shape)
    print((o1[i].to_dense() - o2[i].to_dense()).abs().max())

# %%

C_ = torch.rand(4, 64, 128, device="cuda", requires_grad=False)
mask = F.dropout(torch.ones_like(C[:, :, 0]), p=0.95, training=True)
C = C_ * mask.unsqueeze(-1)
C_csr1 = C.to_sparse_csr(1)
# C_bsr1 = C.to_sparse_bsr((16, 16))
# %%
# C_bsr1
# %%
# sto.sampled_addmm(C_csr1, B, A)
torch.sparse.sampled_addmm(C_csr1, B, A)
# %%
C_csr1

# %%
C_csr1.values().shape
mask_csr = mask.unsqueeze(-1).repeat(1, 1, 128).to_sparse_csr(1)
C2 = C_ * mask_csr
# %%
C2.values() == C_csr1.values()
# %%
mask_csr


# %%
csrgate = gate.to_sparse_csr()
csrgate
# %%
# gate


# def csr_W(gate, W):
#     gate = gate.unsqueeze(-1).unsqueeze(-1)
#     return (W * gate).to_sparse_csr(1)
# batch = gate.shape[0]
# fullshape = (batch,) + W_enc.shape
# bcsr_gate = gate.unsqueeze(-1).unsqueeze(-1).expand(*fullshape).to_sparse_csr(2)


def csr2_W(gate, W):
    batch = gate.shape[0]
    fullshape = (batch,) + W.shape
    b_csr_gate = gate.unsqueeze(-1).unsqueeze(-1).expand(*fullshape).to_sparse_csr(2)
    return W.unsqueeze(0).expand(*fullshape) * b_csr_gate


def csr2_x(gate, x):
    batch = gate.shape[0]
    fullshape = gate.shape + (1, x.shape[1])
    b_csr2_gate = gate.unsqueeze(-1).unsqueeze(-1).expand(*fullshape).to_sparse_csr(2)
    return x.unsqueeze(-2).unsqueeze(-2).expand(*fullshape) * b_csr2_gate


x.requires_grad = True
coo2x = coo2_x(gate, x)
coo2W_enc = coo2_W(gate, W_enc)
csr2x = csr2_x(gate, x)
csr2W_enc = csr2_W(gate, W_enc)
# %%
coo2x.values().shape
# coo2_acts = coo2x.values() @ coo2W_enc.values()
# coo2_acts
coo2W_enc.values().shape
csr2W_enc.values().shape


# %%
coo2W_enc
torch.all(coo2W_enc.indices() == coo2x.indices())


# %%

import torch


def coo2_W(gate, W):
    batch = gate.shape[0]
    fullshape = (batch,) + W.shape
    b_coo2_gate = goodgate(gate, fullshape)
    return mult_spd(b_coo2_gate, W.unsqueeze(0).expand(*fullshape))


def goodgate(gate, goodshape):
    spgate = gate.unsqueeze(-1).unsqueeze(-1).to_sparse(2)
    values = spgate.values().expand(spgate.values().shape[:1] + goodshape[2:])
    return index_clone(
        spgate,
        values,
    )


def coo2_x(gate, x):
    fullshape = gate.shape + (1, x.shape[1])
    b_coo2_gate = gate.unsqueeze(-1).unsqueeze(-1).expand(*fullshape).to_sparse(2)
    return x.unsqueeze(-2).unsqueeze(-2).expand(*fullshape) * b_coo2_gate


def coo2_b(gate, b):
    fullshape = gate.shape + (1, b.shape[1])
    b_coo2_gate = gate.unsqueeze(-1).unsqueeze(-1).expand(*fullshape).to_sparse(2)
    return b.unsqueeze(-2).unsqueeze(0).expand(*fullshape) * b_coo2_gate


def coo2transpose(mat):
    mat = mat.coalesce()
    shape = mat.shape[:-2] + (mat.shape[-1], mat.shape[-2])
    return torch.sparse_coo_tensor(
        mat.indices(),
        mat.values().transpose(-2, -1),
        shape,
        device=mat.device,
    )


def _coo2matmul(mat2, mat1):
    mat1 = mat1.coalesce()
    mat2 = mat2.coalesce()
    assert mat2.layout == torch.sparse_coo == mat1.layout
    assert mat2.shape[-1] == mat1.shape[-2]
    assert torch.all(mat2.indices() == mat1.indices())
    shape = mat2.shape[:-1] + mat1.shape[-1:]
    out = torch.sparse_coo_tensor(
        mat2.indices(),
        mat2.values() @ mat1.values(),
        shape,
        device=mat2.device,
    )
    return out


def _coo2matmul_spd(mat2, mat1):
    mat1 = mat1.coalesce()
    mat2 = mat2.coalesce()
    assert mat2.layout == torch.sparse_coo
    assert mat1.layout == torch.strided
    assert mat2.shape[-1] == mat1.shape[-2]
    shape = mat2.shape[:-1] + mat1.shape[-1:]
    out = torch.sparse_coo_tensor(
        mat2.indices(),
        mat2.values() @ mat1[mat2.indices()[0], mat2.indices()[1]],
        shape,
        device=mat2.device,
    )
    return out


class BCOOd2_MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat2, mat1):
        mat2 = mat2.detach()
        mat1 = mat1.detach()
        # ctx.save_for_backward(mat2.detach(), mat1.detach())
        # ctx.save_for_backward(mat2.detach(), mat1.detach())
        ctx.save_for_backward(mat2, mat1)
        out = _coo2matmul(mat2, mat1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.to_dense()
        mat2, mat1 = ctx.saved_tensors
        print(mat2.shape, mat1.shape, grad_output.shape)
        grad_mat2 = coo2transpose(_coo2matmul(mat1, coo2transpose(grad_output)))
        grad_mat1 = _coo2matmul(coo2transpose(mat2), grad_output)
        return grad_mat2.to_dense(), grad_mat1.to_dense()


def coo2matmul(mat2, mat1):
    return BCOOd2_MM.apply(mat2, mat1)


class IndexClone(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spinput, values):
        ctx.save_for_backward(values, spinput)
        return torch.sparse_coo_tensor(
            indices=spinput.indices(),
            values=values,
            size=spinput.shape[:-2] + values.shape[-2:],
            device=spinput.device,
        )

    @staticmethod
    def backward(ctx, grad_output):
        print("grad_output", grad_output.shape)
        values, spinput = ctx.saved_tensors
        return None, grad_output[spinput.indices()[0], spinput.indices()[1]]


def index_clone(spinput, values):
    # return IndexClone.apply(spinput, values)
    return torch.sparse_coo_tensor(
        indices=spinput.indices(),
        values=values,
        size=spinput.shape[:-2] + values.shape[-2:],
        device=spinput.device,
    )


class COO2mult_spd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat2, mat1):
        ctx.save_for_backward(mat2, mat1)
        values = mat2.values() * mat1[mat2.indices()[0], mat2.indices()[1]]


class COO2mult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat2, mat1):
        ctx.save_for_backward(mat2, mat1)


def mult_spd(mat2, mat1):
    mat2 = mat2.coalesce()
    values = mat2.values() * mat1[mat2.indices()[0], mat2.indices()[1]]
    return torch.sparse_coo_tensor(
        indices=mat2.indices(),
        values=values,
        size=mat2.shape,
        device=mat2.device,
    )


# %%
o = generate_example(8, 10, 12, 16, xlist=False)
x = o.x
W_enc = o.W_enc
gate = o.gate
b_enc = o.b_enc
b_dec = o.b_dec
W_dec = o.W_dec
x.requires_grad = True

coo2x = coo2_x(gate, x)
coo2W_enc = coo2_W(gate, W_enc)


coo2x = torch.nn.Parameter(coo2x, requires_grad=True)
coo2W_enc = torch.nn.Parameter(coo2W_enc, requires_grad=True)
coo2_acts = BCOOd2_MM.apply(coo2x, coo2W_enc)
coo2_acts.to_dense().sum().backward()


# %%
# %%
# torch.sspaddmm()
coo2_b_enc = coo2_b(gate, b_enc)
torch.sparse.addmm(coo2_b_enc, coo2x, coo2W_enc.to_dense())

# %%
W = W_enc
batch = gate.shape[0]
fullshape = (batch,) + W.shape
gate = gate * torch.rand_like(gate)
b_csr_gate = gate.unsqueeze(-1).unsqueeze(-1).expand(*fullshape).to_sparse(2)
test = gate.unsqueeze(-1).unsqueeze(-1).to_sparse(2)

# %%
test.indices() == b_csr_gate.indices()
# %%

b_csr_gate.values()
# %%
torch.all(b_csr_gate.values() == test.values().expand(10, 10, 8))
# %%
W_enc.requires_grad = True
coo2W_enc = coo2_W(gate, W_enc)
# %%
coo2W_enc.sum().backward()

# %%
gate.requires_grad = True
fullshape = gate.shape + (1, W_enc.shape[1])
gg = goodgate(gate, fullshape)
gg.sum().backward()
# %%
gate.grad
# %%
W_enc.grad
# %%
coo2W_enc = coo2W_enc.coalesce()
coo2W_enc.shape
a, b = coo2W_enc.indices()
# %%


sto.scatter_mm(
    o.W_dec,
)

# %%
acts = acts.coalesce()
ai = acts.indices()
acts.shape
# %%
d_data = 8


target = saes_out.to_dense().sum(1).squeeze(-2)
target.shape

# %%

bids, sids = saes_out.indices()
out = torch.index_add(torch.zeros_like(x).unsqueeze(-2), 0, bids, saes_out.values())

out.squeeze() == target
# %%
torch.rand(4, 4)
# torch.sparse_coo_tensor(indices=())
from nqgl.mlutils.time_gpu import TimedFunc, timedfunc_wrapper

gate.shape
from nqgl.sae.hsae.spbmbmm.coo2 import index_clone, _index_make
def _index_make(index, values, size, device=None):
    device = device or values.device
    return torch.sparse_coo_tensor(
        indices=index,
        values=values,
        size=size,
        device=device,
    )


# %%
@timedfunc_wrapper(print=True)
def gate_indices(gate):
    return gate.nonzero().t()


@timedfunc_wrapper(print=True)
def swenc(gate, W_enc):
    gi = gate_indices(gate)
    v = []
    for i in gi:
        v.append(W_enc[i[0]])
    return _index_make(
        gi,
        values = v,
        size=gate.shape[:2] + W_enc.shape[1:],
        device="cuda"
    )


w = swenc(gate, W_enc)
# %%
