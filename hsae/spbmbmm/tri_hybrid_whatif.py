import torch
from hsae.spbmbmm.fwd_params import CacheBoxes, ForwardOptions
from nqgl.bmask_bmm.cust_tri_matmuls.sparse_batch_masked.spmask_spoutnogate import (
    masked_matmul,
    batchmatmul_basic_kernel,
)
from nqgl.mlutils.time_gpu import TimedFunc, timedfunc_wrapper

import einops


def sparse_forward_hybrid(
    x: torch.Tensor,
    gate: torch.Tensor,
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    W_dec: torch.Tensor,
    b_dec: torch.Tensor,
    cache: CacheBoxes,
    options: ForwardOptions,
    **cache_kwargs
):
    # x: (batches, d_data)
    # gate: (batches, n_sae)
    assert x.ndim == 2
    batches = x.shape[0]
    d_data = x.shape[1]
    device = x.device
    n_sae = options.n_sae

    gate_spm = gate.unsqueeze(0)
    flat_indices = (
        (gate_spm > 0).transpose(-1, -2).nonzero(as_tuple=False).transpose(0, 1)
    )
    # print("flat:", flat_indices.shape)
    x_spm = einops.rearrange(
        x, "batch d_data        -> 1      1 batch     d_data 1      "
    )
    W_enc_spm = einops.rearrange(
        W_enc, "n_sae d_data d_dict -> 1           n_sae 1  d_dict d_data"
    )
    pre_acts = masked_matmul(
        W_enc_spm,
        x_spm,
        mask_indices=flat_indices,
        sparse_out=True,
    ).transpose(-1, -2)

    gate = gate.transpose(0, 1)
    flat_indices = (gate > 0).to_sparse().indices()  # n_sae batches 1 1
    gate = gate.unsqueeze(-1).unsqueeze(-1)  # n_sae batches 1 1
    # (gate > 0).to_sparse(2)
    # x_s = (x.unsqueeze(-2) * (gate > 0)).to_sparse(2)

    # x_flat = x_s.values()
    # x_flat = (
    #     x.unsqueeze(-2)
    #     .unsqueeze(0)
    #     .expand(n_sae, batches, 1, d_data)[flat_indices[0], flat_indices[1]]
    # )
    # ###
    batch_idxs = flat_indices[1]
    sae_idxs = flat_indices[0]
    # W_enc_flat = W_enc[sae_idxs]
    # b_enc_flat = b_enc[sae_idxs].unsqueeze(-2)
    W_dec_flat = W_dec[sae_idxs]
    b_dec_flat = b_dec[sae_idxs].unsqueeze(-2)
    flat_acts = options.nonlinearity(pre_acts + b_enc[sae_idxs].unsqueeze(-2))

    # x_cent = x_flat.float()  # - b_dec
    # if options.sub_b_dec:
    #     x_cent = x_cent - b_dec_flat

    # m = x_cent @ W_enc_flat
    # flat_acts = options.nonlinearity(m + b_enc_flat)
    assert not torch.any(torch.isinf(flat_acts))
    assert not torch.any(torch.isnan(flat_acts))

    # if cache.acts:
    #     cache.acts << (
    #         full_acts(flat_acts, flat_indices, batches, options=options).float()
    #     )
    # if cache.flat_acts:
    #     cache.flat_acts << flat_acts

    if options.scale_acts:
        if options.norm_gate_before_scaling_acts:
            gate = gate / gate.norm(dim=0, keepdim=True)
        flat_acts = flat_acts * gate[sae_idxs, batch_idxs]
    else:
        flat_acts = flat_acts * (gate[sae_idxs, batch_idxs] > 0)
    # self.cache(acts=acts, flat_acts=flat_acts, gate=gate, **cache_kwargs)
    m = (flat_acts) @ W_dec_flat
    saes_out_flat = m + b_dec_flat

    flatsize = saes_out_flat.shape[0]
    # z = torch.zeros(batches, d_data, device=device)
    # bids = batch_idxs.reshape(flatsize, 1).expand(-1, d_data)
    # sae_re = saes_out_flat.reshape(flatsize, d_data)

    x_out = torch.scatter_add(
        torch.zeros(batches, d_data, device=device),
        0,
        batch_idxs.reshape(flatsize, 1).expand(-1, d_data),
        saes_out_flat.reshape(flatsize, d_data),
    )
    return x_out


from baseline import test_sparse_forward_validity, test_speed_forward


def main():
    TimedFunc(sparse_forward_hybrid, name="sparse_forward_hybrid", print=True)
    test_speed_forward(sparse_forward_hybrid)


if __name__ == "__main__":
    main()
