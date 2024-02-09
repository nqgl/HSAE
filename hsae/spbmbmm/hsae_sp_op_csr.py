from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn.functional as F
import einops
from nqgl.sae.hsae.spbmbmm.fwd_params import CacheBoxes, ForwardOptions


def csr_hsae_spop(
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
    W_enc_rest = einops.rearrange(W_enc, "n_sae d_data d_dict -> d_data (n_sae d_dict)")
    # gate_rest = gate.repeat(1, d_dict)  # maybe wrong order here
    gate_rest = einops.repeat(
        gate, "batch n_sae -> batch (n_sae d_dict)", d_dict=d_dict
    )
    gate_rest = gate_rest.to_sparse_csr()

    # x = x.unsqueeze(-2).unsqueeze(-2).expand(batch, n_sae, -1)
    W_enc = W_enc.unsqueeze(0).expand(batch, n_sae, d_data, d_dict)
    # print(gate_rest.shape, W_enc_rest.shape, x.shape)
    m = torch.sparse.sampled_addmm(input=gate_rest, mat2=W_enc_rest, mat1=x) + (
        -1 * gate_rest
    )
    b_enc = b_enc.unsqueeze(0) * (gate > 0).unsqueeze(-1)

    b_enc_rest = (
        einops.rearrange(b_enc, "batch n_sae d_dict -> batch (n_sae d_dict)")
        .expand(batch, -1)
        .to_sparse_csr()
    )
    # this would fail but smn like it could be more efficient
    #    | b_enc_rest = (
    #    |     einops.rearrange(b_enc, "n_sae d_dict -> (n_sae d_dict)")
    #    |     .to_sparse_csr()
    #    |     .unsqueeze(0)
    #    |     .expand(batch, -1)
    #    | )

    # print(m)
    acts = F.relu(m + b_enc_rest)
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
    b_dec = b_dec.unsqueeze(0) * gate.unsqueeze(-1)
    b_dec = b_dec.sum(-2)
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

    o = generate_example(8, 10, 12, 16, xlist=False)
    x = o.x
    W_enc = o.W_enc
    gate = o.gate
    b_enc = o.b_enc
    b_dec = o.b_dec
    W_dec = o.W_dec
    csr_hsae_spop(x, gate, W_enc, b_enc, W_dec, b_dec, None, None)
    # exit()
    test_sparse_forward_validity(csr_hsae_spop, caching=False)
    test_speed(csr_hsae_spop)


if __name__ == "__main__":
    main()
