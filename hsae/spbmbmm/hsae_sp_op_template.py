from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn.functional as F
import einops
from nqgl.sae.hsae.spbmbmm.fwd_params import CacheBoxes, ForwardOptions


def template_spop(
    x: Float[Tensor, "batch d_data"],
    gate: Float[Tensor, "batch n_sae"],
    W_enc: Float[Tensor, "n_sae d_data d_dict"],
    b_enc: Float[Tensor, "n_sae d_dict"],
    W_dec: Float[Tensor, "n_sae d_dict d_data"],
    b_dec: Float[Tensor, "n_sae d_data"],
    cache: CacheBoxes,
    options: ForwardOptions,
):

    #     | if options.sub_b_dec:
    #     |     x_cent = x_cent - b_dec.float()

    # m = x_cent @ W_enc

    # acts = F.relu(m + b_enc)

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

    # saes_out = acts @ W_dec + b_dec
    # return saes_out
    pass


def main():
    from baseline import generate_example, test_sparse_forward_validity

    o = generate_example(8, 10, 12, 16, xlist=False)
    x = o.x
    W_enc = o.W_enc
    gate = o.gate
    b_enc = o.b_enc
    b_dec = o.b_dec
    W_dec = o.W_dec
    template_spop(x, gate, W_enc, b_enc, W_dec, b_dec, None, None)

    test_sparse_forward_validity(template_spop)


if __name__ == "__main__":
    main()
