from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn.functional as F
import einops
from nqgl.sae.hsae.spbmbmm.fwd_params import CacheBoxes, ForwardOptions
from nqgl.sae.hsae.spbmbmm.coo2 import coo2_W, coo2_x, coo2matmul, coo2_b
from torch.sparse import _triton_ops as sto
from nqgl.mlutils.time_gpu import (
    TimedFunc,
    timedfunc_wrapper,
    profilefunc_wrapper,
    ProfileFunc,
)


def coo2_hsae_spop(
    x: Float[Tensor, "batch d_data"],
    gate: Float[Tensor, "batch n_sae"],
    W_enc: Float[Tensor, "n_sae d_data d_dict"],
    b_enc: Float[Tensor, "n_sae d_dict"],
    W_dec: Float[Tensor, "n_sae d_dict d_data"],
    b_dec: Float[Tensor, "n_sae d_data"],
    cache: CacheBoxes,
    options: ForwardOptions,
):
    batch = x.shape[0]
    n_sae, d_data, d_dict = W_enc.shape
    # gate = gate.unsqueeze(-2).unsqueeze(-1)
    # gate = gate.expand(
    #     batch,
    #     n_sae,
    # )

    #     | if options.sub_b_dec:
    #     |     x_cent = x_cent - b_dec.float()
    #     | to do this, do b_dec @ W_enc then make it gate-patterned and negative, then use as input to sampled_addmm
    # m = x_cent @ W_enc
    # %%
    x_dense = x
    W_enc = coo2_W(gate, W_enc)
    x = coo2_x(gate, x)
    b_enc = coo2_b(gate, b_enc)
    W_dec = coo2_W(gate, W_dec)
    b_dec = coo2_b(gate, b_dec)
    # gate_rest = gate.repeat(1, d_dict)  # maybe wrong order here
    # x = x.unsqueeze(-2).unsqueeze(-2).expand(batch, n_sae, -1)
    # W_enc = W_enc.unsqueeze(0).expand(batch, n_sae, d_data, d_dict)
    # print(gate_rest.shape, W_enc_rest.shape, x.shape)
    m = coo2matmul(x, W_enc)
    # %%

    # print(m)
    acts = F.relu(m + b_enc)
    # acts = m
    # print(acts)

    #     | if cache.acts:
    #     |     cache.acts << acts.squeeze(-2)  # * (gate.squeeze(-2) > 0)
    #     | if options.scale_acts:
    #     |     if options.norm_gate_before_scaling_acts:
    #     |         gate = gate / gate.norm(dim=1, keepdim=True)
    #     |         gate = torch.where(torch.isnan(gate), 0, gate)
    #     |     acts = acts * gate
    #     | else:
    #     |     acts = acts * (gate > 0)

    # b_dec = b_dec * gate

    saes_out = coo2matmul(acts, W_dec) + b_dec
    saes_out = saes_out.coalesce()
    bids, sids = saes_out.indices()
    out = torch.index_add(
        torch.zeros_like(x_dense),
        0,
        bids,
        saes_out.values().squeeze(-2),
    )

    # out = x_dense * saes_out.sum()
    return out

    out_indices = out.indices()
    acc = torch.zeros(out_indices.shape[1], out.shape[-1])
    out

    # acts: batch (nsae d_dict)
    W_dec_rest = einops.rearrange(W_dec, "n_sae d_dict d_data -> (n_sae d_dict) d_data")
    # print(acts)
    saes_out_nb = torch.sparse.mm(acts, W_dec_rest)
    # acts @ W_dec_rest
    saes_out = saes_out_nb.to_dense() + b_dec
    # print(saes_out.shape)
    return saes_out
    pass


def main():
    from baseline import generate_example, test_sparse_forward_validity, test_speed

    torch.set_default_dtype(torch.bfloat16)
    o = generate_example(8, 10, 12, 16, xlist=False)
    x = o.x
    W_enc = o.W_enc
    gate = o.gate
    b_enc = o.b_enc
    b_dec = o.b_dec
    W_dec = o.W_dec
    # coo2_hsae_spop(x, gate, W_enc, b_enc, W_dec, b_dec, None, None)
    # exit()
    # test_speed(coo2_hsae_spop)
    # with torch.cuda.amp.autocast():
    test_speed(coo2_hsae_spop)
    # test_sparse_forward_validity(coo2_hsae_spop, caching=False)


if __name__ == "__main__":
    main()

# %%
