import torch
import torch.nn as nn
import torch.nn.functional as F
from unpythonic import box
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FowardOptions:
    batches: int
    d_data: int
    d_dict: int
    n_sae: int
    scale_acts: bool = True
    scale_b_dec: bool = True
    scale_b_enc: bool = True
    cache_acts: bool = True
    norm_acts: bool = True
    sub_b_dec: bool = True
    norm_gate_before_scaling_acts = False
    device: str = "cuda"
    nonlinearity: nn.Module = torch.relu

@dataclass
class CacheBoxes:
    acts :Optional[box] = None
    flat_acts :Optional[box] = None
    l1 :box = field(default_factory=box)
    l0 :box = field(default_factory=box)
b = CacheBoxes()




def encode_flat(
    x,
    W_enc,
    b_dec,
    b_enc,
    nonlinearity = torch.relu
):
    # batch, n_sae, d_data
    # x_cent = x_cent.unsqueeze(-2)
    # logging.info("\nmult", x_cent.shape, W_enc.shape)
    # input()
    x_cent = x.float() #- b_dec
    m = x_cent @ W_enc
    acts = nonlinearity(m + b_enc)

    # logging.info("x_cent", x_cent.shape)
    # logging.info("m", m.shape)
    # logging.info("acts", acts.shape)
    return acts

def decode_flat(acts, W_dec, b_dec):
    m = acts @ W_dec
    o = m + b_dec
    return o



def full_acts(flat_acts, flat_indices, batches, options: FowardOptions):
    acts = torch.zeros(
        batches, options.n_sae, options.d_dict, device=options.device
    )
    acts[flat_indices[1], flat_indices[0]] = flat_acts.squeeze(-2)
    return acts



def sparse_forward(
        x: torch.Tensor, 
        gate: torch.Tensor, 
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        cache: CacheBoxes,
        options: FowardOptions,
        **cache_kwargs
    ):
    # x: (batches, d_data)
    # gate: (batches, n_sae)
    assert x.ndim == 2
    batches = x.shape[0]
    d_data = x.shape[1]
    device = x.device

    # print(gate)
    # input()
    gate = gate.unsqueeze(-1).unsqueeze(-1).transpose(0, 1)  # n_sae batches 1 1
    bgate = gate != 0

    x_s = (x.unsqueeze(-2) * bgate).to_sparse(2)
    flat_indices = x_s.indices()
    batch_idxs = flat_indices[1]
    sae_idxs = flat_indices[0]
    assert torch.sum(gate[sae_idxs, batch_idxs] == 0) == 0

    x_flat = x_s.values()

    W_enc_flat = W_enc[sae_idxs]
    b_enc_flat = b_enc[sae_idxs].unsqueeze(-2)
    W_dec_flat = W_dec[sae_idxs]
    b_dec_flat = b_dec[sae_idxs].unsqueeze(-2)

    x_cent = x_flat.float() #- b_dec
    m = x_cent @ W_enc_flat
    flat_acts = options.nonlinearity(m + b_enc_flat)
    assert torch.all(~torch.isinf(flat_acts))
    assert torch.all(~torch.isnan(flat_acts))

    if cache.acts:
        cache.acts << full_acts(flat_acts, flat_indices, batches, options=options).float()
    if cache.flat_acts:
        cache.flat_acts << flat_acts
    # self.cache(acts=acts, flat_acts=flat_acts, gate=gate, **cache_kwargs)
    m = gate[sae_idxs, batch_idxs] * flat_acts @ W_dec_flat
    saes_out_flat = m + b_dec_flat

    flatsize = saes_out_flat.shape[0]
    z = torch.zeros(batches, d_data, device=device)
    bids = batch_idxs.reshape(flatsize, 1).expand(-1, d_data)
    sae_re = saes_out_flat.reshape(flatsize, d_data)

    x_out = torch.scatter_add(
        torch.zeros(batches, d_data, device=device),
        0,
        batch_idxs.reshape(flatsize, 1).expand(-1, d_data),
        saes_out_flat.reshape(flatsize, d_data),
    )
    return x_out



def sparse_bgate_forward(
        x: torch.Tensor, 
        gate: torch.Tensor, 
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        cache: CacheBoxes,
        options: FowardOptions,
        **cache_kwargs
    ):
    # x: (batches, d_data)
    # gate: (batches, n_sae)
    assert x.ndim == 2
    batches = x.shape[0]
    d_data = x.shape[1]
    device = x.device
    n_sae = options.n_sae
    d_dict = options.d_dict

    # print(gate)
    # input()
    gate = gate.unsqueeze(-1).unsqueeze(-1).transpose(0, 1)  # n_sae batches 1 1
    bgate = (gate != 0).squeeze(-1).squeeze(-1)


    x_bg = x.unsqueeze(-2).unsqueeze(0).expand(n_sae, batches, 1, d_data)[bgate]

    W_enc_bg = W_enc.unsqueeze(1).expand(n_sae, batches, d_data, d_dict)[bgate]
    b_enc_bg = b_enc.unsqueeze(-2).unsqueeze(1).expand(n_sae, batches, 1, d_dict)[bgate]
    W_dec_bg = W_dec.unsqueeze(1).expand(n_sae, batches, d_dict, d_data)[bgate]
    b_dec_bg = b_dec.unsqueeze(-2).unsqueeze(1).expand(n_sae, batches, 1, d_data)[bgate]
    
    # b_dec_flat = b_dec_flat * gate[flat_indices[0], flat_indices[1]]

    # flat_acts = encode_flat(
    #     x=x_flat, W_enc=W_enc_flat, b_dec=b_dec_flat, b_enc=b_enc_flat
    # )
    x_cent = x_bg.float() #- b_dec
    m = x_cent @ W_enc_bg
    flat_acts = options.nonlinearity(m + b_enc_bg)
    assert torch.all(~torch.isinf(flat_acts))
    assert torch.all(~torch.isnan(flat_acts))

    # if cache.acts:
        # cache.acts << full_acts(flat_acts, flat_indices, batches, options=options).float()
    if cache.flat_acts:
        cache.flat_acts << flat_acts
    # self.cache(acts=acts, flat_acts=flat_acts, gate=gate, **cache_kwargs)
    m = (
        gate[bgate] 
        * 
        flat_acts @ W_dec_bg
    )
    saes_out_bg = m + b_dec_bg

    x_out = torch.zeros(options.n_sae, batches, 1, d_data, device=device)
    x_out[bgate] = saes_out_bg
    return x_out.sum(0).squeeze(-2)    



def dense(
    x: torch.Tensor, 
    gate: torch.Tensor, 
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    W_dec: torch.Tensor,
    b_dec: torch.Tensor,
    cache: CacheBoxes,
    options: FowardOptions,
):
    x = x.unsqueeze(-2).unsqueeze(-2)
    gate = gate.unsqueeze(-1).unsqueeze(-1)
    x_cent = x.float()
    b_dec = b_dec.unsqueeze(-2)
    b_enc = b_enc.unsqueeze(-2)
    m = x_cent @ W_enc
    acts = F.relu(m + b_enc)
    if cache.acts:
        cache.acts << acts.squeeze(-2)# * (gate.squeeze(-2) > 0)
    # acts = acts * gate

    # saes_out = (acts @ W_dec + b_dec * (gate > 0))
    acts = F.relu(m + b_enc)
    # print(torch.count_nonzero(gate > 0))
    saes_out = (gate > 0)  * (acts @ W_dec + b_dec)
    return saes_out.sum(dim=1).squeeze(-2)

class O:
    pass

def generate_example(d_dict, d_data, n_sae, batches, p_sparse = 0.95, xlist=100, requires_grad=False, device="cuda"):
    o = O()
    o.W_enc = torch.randn(n_sae, d_data, d_dict, device=device, requires_grad=requires_grad)
    o.W_dec = torch.randn(n_sae, d_dict, d_data, device=device, requires_grad=requires_grad)
    o.b_enc = torch.randn(n_sae, d_dict, device=device, requires_grad=requires_grad)
    o.b_dec = torch.randn(n_sae, d_data, device=device, requires_grad=requires_grad)
    o.srcgate = torch.rand(batches, n_sae, device=device, requires_grad=requires_grad)
    if xlist:
        o.X = []
        o.gmasks = []
        for i in range(xlist):
            o.X.append(torch.randn(batches, d_data, device=device))
            o.gmasks.append(F.dropout(torch.ones_like(o.srcgate), p=p_sparse, training=True) * (1 - p_sparse))
    else:
        gmask = F.dropout(torch.ones_like(o.srcgate), p=p_sparse, training=True) * (1 - p_sparse)
        o.gate = o.srcgate * gmask.detach()
        o.x = torch.randn(batches, d_data, device=device)

    # o.gate = o.gate * gmask.detach()
    o.options = FowardOptions(
        batches = batches,
        d_data = d_data,
        d_dict = d_dict,
        n_sae = n_sae,
        device = device
    )

    return o

def test_sparse_forward_validity():
    d_dict = 20
    d_data = 21
    n_sae = 22
    batches = 40
    device = "cuda"
    cs = CacheBoxes(acts=box())
    cd = CacheBoxes(acts=box())
    o = generate_example(d_dict, d_data, n_sae, batches, xlist=False, device=device)
    s = sparse_forward(o.x, o.gate, o.W_enc, o.b_enc, o.W_dec, o.b_dec, cache=cs, options=o.options)
    d = dense(o.x, o.gate, o.W_enc, o.b_enc, o.W_dec, o.b_dec, cache=cd, options=o.options)
    print(s.shape, d.shape) 
    print("output eq?", torch.allclose(s, d, rtol=1e-4))
    print(cs.acts.x.shape, cd.acts.x.shape)
    print("acts eq?", torch.allclose(cs.acts.x, cd.acts.x, rtol=1e-4, atol=1e-6))


def comptime(o, *F, cache=True, backward = False, optim = False):
    if cache:
        caches = [CacheBoxes(box(), box()) for _ in F]
    else:
        caches = [CacheBoxes() for _ in F]
    for f, c in list(zip(F,caches)) * 1:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        print(f.__name__, end=" ")
        if optim:
            optim = torch.optim.SGD([o.W_enc, o.b_enc, o.W_dec, o.b_dec, o.srcgate], lr = 1e-6)
            optim.zero_grad()
        start.record()
        for x, gmask in zip(o.X, o.gmasks):
            y = f(
                x = x,
                gate = torch.abs(o.srcgate) * gmask.clone().detach(),
                W_enc = o.W_enc,
                b_enc = o.b_enc,
                W_dec = o.W_dec,
                b_dec = o.b_dec,
                options=o.options, cache=c)
            if backward:
                (x.float()- y.float()).pow(2).sum().backward()
                if optim:
                    optim.step()
                    optim.zero_grad()
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        torch.cuda.empty_cache()
    

def test_speed():
    d_dict = 32
    d_data = 256
    n_sae = 128
    batches = 256   
    device = "cuda"
    p_sparse = 0.99
    
    o = generate_example(d_dict, d_data, n_sae, batches, p_sparse, device=device, requires_grad=True)

    comptime(o, sparse_forward, dense, sparse_bgate_forward, cache=True, backward = False)
    # comptime(o, sparse_forward, dense, cache=True, backward = True)
    # print("sparse first")
    # comptime(o, sparse_forward, sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward, cache=False, backward = True, optim=True)
    # print("dense first")
    for i in range(1):
        comptime(o, dense, sparse_forward, sparse_bgate_forward, cache=True, backward = True, optim=True)
        comptime(o, dense, sparse_forward, sparse_bgate_forward, cache=True, backward = True, optim=True)
        comptime(o, dense, sparse_forward, cache=True, backward = True, optim=True)
        comptime(o, dense, sparse_forward, cache=True, backward = True, optim=True)
    # comptime(o, sparse_forward, sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward, cache=False, backward = True, optim=True)


def main():
    test_sparse_forward_validity()
    test_speed()
    
if __name__ == "__main__":
    main()
