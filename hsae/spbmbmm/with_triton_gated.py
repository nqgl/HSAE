import torch
from baseline import test_sparse_forward_validity, test_speed_forward
import einops
from hsae.spbmbmm.fwd_params import CacheBoxes, ForwardOptions
from nqgl.bmask_bmm.cust_tri_matmuls.sparse_batch_masked.spmask_spoutnogate import (
    masked_matmul,
)


def sparse_forward_triton(
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
    # gate = gate.transpose(0, 1)

    gate = gate.unsqueeze(0)  # (1 batch, n_sae)
    flat_indices = (gate > 0).to_sparse().indices()
    # gate = gate.unsqueeze(-1).unsqueeze(-1)
    ###
    # batch_idxs = flat_indices[1]
    # sae_idxs = flat_indices[0]
    # W_enc_flat = W_enc[sae_idxs]
    # b_enc_flat = b_enc[sae_idxs].unsqueeze(-2)
    # W_dec_flat = W_dec[sae_idxs]
    # b_dec_flat = b_dec[sae_idxs].unsqueeze(-2)

    # x_cent = x_flat.float() #- b_dec
    # if options.sub_b_dec:
    #     x_cent = x_cent - b_dec_flat

    # x to 1, b, 1, 1, d
    b_enc_spm = einops.rearrange(
        b_enc, "n_sae d_dict        -> 1       1     n_sae 1      d_dict"
    )
    x_spm = einops.rearrange(
        x, "batch d_data        -> 1       batch 1     1      d_data"
    )
    W_enc_spm = einops.rearrange(
        W_enc, "n_sae d_data d_dict -> 1       1     n_sae d_data d_dict"
    )
    W_dec_spm = einops.rearrange(
        W_dec, "n_sae d_dict d_data -> n_sae   1     1     d_dict d_data"
    )
    b_dec_spm = einops.rearrange(
        b_dec, "n_sae d_data        -> n_sae   1     1     1      d_data"
    )

    pre_acts = masked_matmul(x_spm, W_enc_spm, mask_indices=flat_indices)
    acts = options.nonlinearity(
        pre_acts + b_enc_spm * (gate > 0).unsqueeze(-1).unsqueeze(-1)
    )

    # if cache.acts:
    #     cache.acts << full_acts(flat_acts, flat_indices, batches, options=options).float()
    # if cache.flat_acts:
    #     cache.flat_acts << flat_acts

    # if options.scale_acts:
    #     if options.norm_gate_before_scaling_acts:
    #         gate = gate / gate.norm(dim=0, keepdim=True)
    #     flat_acts = flat_acts * gate[sae_idxs, batch_idxs]
    # else:
    #     flat_acts = flat_acts * (gate[sae_idxs, batch_idxs] > 0)

    acts_spm = einops.rearrange(
        acts, "1 batch n_sae 1 d_dict -> n_sae batch 1 1 d_dict"
    )
    m = masked_matmul(acts_spm, W_dec_spm, mask_indices=torch.flip(flat_indices, (0,)))
    saes_out = m + b_dec_spm * (gate > 0).transpose(0, -1).unsqueeze(-1).unsqueeze(-1)
    # x_out = torch.scatter_add(
    #     torch.zeros(batches, d_data, device=device),
    #     0,
    #     batch_idxs.reshape(flatsize, 1).expand(-1, d_data),
    #     saes_out_flat.reshape(flatsize, d_data),
    # )
    x_out = einops.rearrange(
        saes_out, "n_sae batch 1 1 d_data -> batch n_sae d_data"
    ).sum(1)
    return x_out


from nqgl.mlutils.time_gpu import TimedFunc, timedfunc_wrapper


@timedfunc_wrapper()
def sparse_forward_triton_sp_internal(
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

    gate = gate.unsqueeze(0)  # (1 batch, n_sae)
    flat_indices = (gate > 0).to_sparse().indices()
    # gate = gate.unsqueeze(-1).unsqueeze(-1)
    ###
    sae_idxs = flat_indices[2]
    batch_idxs = flat_indices[1]
    flatsize = flat_indices.shape[1]

    # x to 1, b, 1, 1, d
    b_enc_spm = einops.rearrange(
        b_enc, "n_sae d_dict        -> 1       1     n_sae 1      d_dict"
    )
    x_spm = einops.rearrange(
        x, "batch d_data        -> 1       batch 1     1      d_data"
    )
    W_enc_spm = einops.rearrange(
        W_enc, "n_sae d_data d_dict -> 1       1     n_sae d_data d_dict"
    )
    W_dec_spm = einops.rearrange(
        W_dec, "n_sae d_dict d_data -> n_sae       1     1 d_dict d_data"
    )
    b_dec_spm = einops.rearrange(
        b_dec, "n_sae d_data        -> n_sae       1     1 1      d_data"
    )
    masked_mm = TimedFunc(masked_matmul)
    pre_acts = masked_mm(
        x_spm,
        W_enc_spm,
        mask_indices=flat_indices,
        sparse_out=True,
    )
    # print("b_enc[sae_idxs].unsqueeze(-2)", b_enc[sae_idxs].unsqueeze(-2).shape)
    acts = options.nonlinearity(pre_acts + b_enc[sae_idxs].unsqueeze(-2))
    assert acts.shape[0] == flatsize
    # print("pre_acts", pre_acts.shape)
    # print("acts", acts.shape)

    m = masked_mm(
        acts,
        W_dec_spm,
        mask_indices=flat_indices.flip(0),  # 1 batch n_sae
        flat_p_B1_dim=batches,
        sparse_out=True,
    )
    saes_out_flat = m + b_dec[sae_idxs].unsqueeze(-2)
    # print(m.shape, b_dec[sae_idxs].shape)
    # print(saes_out_flat.shape, flatsize)
    scatter_add = TimedFunc(torch.scatter_add)
    x_out = scatter_add(
        torch.zeros(batches, d_data, device=device),
        0,
        batch_idxs.reshape(flatsize, 1).expand(flatsize, d_data),
        saes_out_flat.reshape(flatsize, d_data),
    )
    return x_out


def main():
    mask = torch.zeros(2, 2, 2)

    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 1
    mask[1, 0, 0] = 1
    mask[1, 1, 1] = 1

    # p[:, :, :, :, :5] = torch.eye(5) * (4 + 1)
    # p[0,0,0, :, :5] = torch.eye(5) * (69)
    # p[0,1,0, :, :5] = torch.eye(5) * (1 + 1)
    # p[1, 0, 0, :, :5] = torch.eye(5) * (2 + 1)
    # p[1, 1, 0, :, :5] = torch.eye(5) * (3 + 1)
    # p[:, :, :, :, :5] = torch.eye(5) * (5 + 1)

    # q[:, :, :, :4] = torch.eye(4)
    q = torch.randn(2, 1, 2, 6, 4)
    p = torch.randn(2, 2, 1, 5, 6)

    p = p.cuda()
    q = q.cuda()
    mask = mask.cuda()

    mask_indices = mask.to_sparse().indices()
    out = masked_matmul(p, q, mask_indices=mask_indices, sparse_out=True)
    o = torch.matmul(p.clone(), q.clone()) * (mask.unsqueeze(-1).unsqueeze(-1))
    print(out)
    print(o.shape, out.shape)
    print(mask.unsqueeze(-1).unsqueeze(-1).shape)
    out_full = torch.zeros_like(o)
    out_full[mask_indices[0], mask_indices[1], mask_indices[2]] = out
    print(o)
    print(out_full - o)

    # test_sparse_forward_validity(sparse_forward_triton_sp_internal, caching=False)
    # test_speed_forward(sparse_forward_triton)
    test_speed_forward(sparse_forward_triton_sp_internal)


if __name__ == "__main__":
    main()
