import torch
import torch.nn as nn
import torch.nn.functional as F
from unpythonic import box
from dataclasses import dataclass, field
from typing import Optional
from jaxtyping import Float
from torch import Tensor
from torch import jit

@dataclass
class ForwardOptions:
    batches: jit.Final[int]
    d_data: jit.Final[int]
    d_dict: jit.Final[int]
    n_sae: jit.Final[int]
    sub_b_dec: jit.Final[bool] = False
    # scale_b_dec: jit.Final[bool] = True
    scale_acts: jit.Final[bool] = False
    # scale_b_enc: jit.Final[bool] = True
    # cache_acts: jit.Final[bool] = True
    # norm_acts: jit.Final[bool] = True
    norm_gate_before_scaling_acts: jit.Final[bool] = False
    device: str = "cuda"
    nonlinearity: nn.Module = torch.relu

@dataclass
class CacheBoxes:
    acts :Optional[box] = None
    flat_acts :Optional[box] = None
    l1 :box = field(default_factory=box) #TODO implement caching these
    l0 :box = field(default_factory=box) #TODO implement caching these
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



def full_acts(flat_acts, flat_indices, batches, options: ForwardOptions):
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

    # print(gate)
    # input()
    ###
    # gate = gate.unsqueeze(-1).unsqueeze(-1).transpose(0, 1)  # n_sae batches 1 1

    # x_s = (x.unsqueeze(-2) * (gate > 0)).to_sparse(2)
    # flat_indices = x_s.indices()
    # assert torch.sum(gate[sae_idxs, batch_idxs] == 0) == 0

    # x_flat = x_s.values()
    ###
    gate = gate.transpose(0, 1)
    flat_indices = (gate > 0).to_sparse().indices()  # n_sae batches 1 1
    gate = gate.unsqueeze(-1).unsqueeze(-1)  # n_sae batches 1 1
    # (gate > 0).to_sparse(2)
    # x_s = (x.unsqueeze(-2) * (gate > 0)).to_sparse(2)

    # x_flat = x_s.values()
    x_flat = (
        x.unsqueeze(-2).unsqueeze(0).expand(
            n_sae, batches, 1, d_data
        )[flat_indices[0], flat_indices[1]]
    )
    ###
    batch_idxs = flat_indices[1]
    sae_idxs = flat_indices[0]
    W_enc_flat = W_enc[sae_idxs]
    b_enc_flat = b_enc[sae_idxs].unsqueeze(-2)
    W_dec_flat = W_dec[sae_idxs]
    b_dec_flat = b_dec[sae_idxs].unsqueeze(-2)

    x_cent = x_flat.float() #- b_dec
    if options.sub_b_dec:
        x_cent = x_cent - b_dec_flat

    m = x_cent @ W_enc_flat
    flat_acts = options.nonlinearity(m + b_enc_flat)
    assert not torch.any(torch.isinf(flat_acts))
    assert not torch.any(torch.isnan(flat_acts))

    if cache.acts:
        cache.acts << full_acts(flat_acts, flat_indices, batches, options=options).float()
    if cache.flat_acts:
        cache.flat_acts << flat_acts

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

def sparse_jit_forward(options):

    @torch.compile
    def sparse_forward_script(
            x: torch.Tensor, 
            gate: torch.Tensor,
            flat_indices: torch.Tensor,
            W_enc: torch.Tensor,
            b_enc: torch.Tensor,
            W_dec: torch.Tensor,
            b_dec: torch.Tensor,
            cache: CacheBoxes,
        ):
        # x: (batches, d_data)
        # gate: (batches, n_sae)
        batches = x.shape[0]
        d_data = x.shape[1]
        device = x.device
        n_sae = options.n_sae

        # print(gate)
        # input()
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # n_sae batches 1 1
        # (gate > 0).to_sparse(2)
        # x_s = (x.unsqueeze(-2) * (gate > 0)).to_sparse(2)
    
        # x_flat = x_s.values()
        x_flat = (
            x.unsqueeze(-2).unsqueeze(0).expand(
                n_sae, batches, 1, d_data
            )[flat_indices[0], flat_indices[1]]
        )
        batch_idxs = flat_indices[1]
        sae_idxs = flat_indices[0]


        W_enc_flat = W_enc[sae_idxs]
        b_enc_flat = b_enc[sae_idxs].unsqueeze(-2)
        W_dec_flat = W_dec[sae_idxs]
        b_dec_flat = b_dec[sae_idxs].unsqueeze(-2)

        x_cent = x_flat.float() #- b_dec
        if options.sub_b_dec:
            x_cent = x_cent - b_dec_flat

        m = x_cent @ W_enc_flat
        flat_acts = options.nonlinearity(m + b_enc_flat)

        # if cache.acts:
        #     cache.acts << full_acts(flat_acts, flat_indices, batches, options=options).float()
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


    def sparse_forward_prep(
            x: torch.Tensor, 
            gate: torch.Tensor,
            W_enc: torch.Tensor,
            b_enc: torch.Tensor,
            W_dec: torch.Tensor,
            b_dec: torch.Tensor,
            cache: CacheBoxes,
        ):
            gate = gate.transpose(0, 1)
            flat_indices = (gate > 0).to_sparse().indices()  # n_sae batches 1 1
            return sparse_forward_script(
                x=x,
                gate=gate,
                flat_indices=flat_indices,
                W_enc=W_enc,
                b_enc=b_enc,
                W_dec=W_dec,
                b_dec=b_dec,
                cache=cache                
            )
    





    @torch.compile
    def densecompiled(
        x: Float[Tensor, "batch d_data"], 
        gate: Float[Tensor, "batch n_sae"], 
        W_enc: Float[Tensor, "n_sae d_data d_dict"],
        b_enc: Float[Tensor, "n_sae d_dict"],
        W_dec: Float[Tensor, "n_sae d_dict d_data"],
        b_dec: Float[Tensor, "n_sae d_data"],
        cache: CacheBoxes,
    ):
        x = x.unsqueeze(-2).unsqueeze(-2)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        b_dec = b_dec.unsqueeze(-2)
        b_enc = b_enc.unsqueeze(-2)

        x_cent = x.float()
        if options.sub_b_dec:
            x_cent = x_cent - b_dec.float()
        
        m = x_cent @ W_enc
        acts = F.relu(m + b_enc)
        if cache.acts:
            cache.acts << acts.squeeze(-2)# * (gate.squeeze(-2) > 0)
        # acts = F.relu(m + b_enc)
        if options.scale_acts:
            if options.norm_gate_before_scaling_acts:
                gate = gate / gate.norm(dim=1, keepdim=True)
                gate = torch.where(
                    torch.isnan(gate),
                    0,
                    gate
                )
            acts = acts * gate
        else:
            acts = acts * (gate > 0)
        # saes_out = (acts @ W_dec + b_dec * (gate > 0))

        saes_out = (gate > 0) * (acts @ W_dec + b_dec)
        return saes_out.sum(dim=1).squeeze(-2)
    return sparse_forward_prep, densecompiled



def sparse_bgate_forward(
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
    assert not torch.any(torch.isinf(flat_acts))
    assert not torch.any(torch.isnan(flat_acts))
    if cache.flat_acts:
        cache.flat_acts << flat_acts
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
    x: Float[Tensor, "batch d_data"], 
    gate: Float[Tensor, "batch n_sae"], 
    W_enc: Float[Tensor, "n_sae d_data d_dict"],
    b_enc: Float[Tensor, "n_sae d_dict"],
    W_dec: Float[Tensor, "n_sae d_dict d_data"],
    b_dec: Float[Tensor, "n_sae d_data"],
    cache: CacheBoxes,
    options: ForwardOptions,
):
    x = x.unsqueeze(-2).unsqueeze(-2)
    gate = gate.unsqueeze(-1).unsqueeze(-1)
    b_dec = b_dec.unsqueeze(-2)
    b_enc = b_enc.unsqueeze(-2)

    x_cent = x.float()
    if options.sub_b_dec:
        x_cent = x_cent - b_dec.float()
    
    m = x_cent @ W_enc
    acts = F.relu(m + b_enc)
    if cache.acts:
        cache.acts << acts.squeeze(-2)# * (gate.squeeze(-2) > 0)
    # acts = F.relu(m + b_enc)
    if options.scale_acts:
        if options.norm_gate_before_scaling_acts:
            gate = gate / gate.norm(dim=1, keepdim=True)
            gate = torch.where(
                torch.isnan(gate),
                0,
                gate
            )
        acts = acts * gate
    else:
        acts = acts * (gate > 0)
    # saes_out = (acts @ W_dec + b_dec * (gate > 0))

    saes_out = (gate > 0) * (acts @ W_dec + b_dec)
    return saes_out.sum(dim=1).squeeze(-2)

class O:
    pass

def generate_example(d_dict, d_data, n_sae, batches, p_sparse = 0.95, xlist=100, requires_grad=False, device="cuda", optionsdict={}):
    o = O()
    o.W_enc = torch.randn(n_sae, d_data, d_dict, device=device, requires_grad=requires_grad)
    o.W_dec = torch.randn(n_sae, d_dict, d_data, device=device, requires_grad=requires_grad)
    o.b_enc = torch.randn(n_sae, d_dict, device=device, requires_grad=requires_grad)
    o.b_dec = torch.randn(n_sae, d_data, device=device, requires_grad=requires_grad)
    o.srcgate = torch.ones(batches, n_sae, device=device, requires_grad=requires_grad)
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
    o.options = ForwardOptions(
        batches = batches,
        d_data = d_data,
        d_dict = d_dict,
        n_sae = n_sae,
        device = device,
        **optionsdict
    )

    return o

def test_sparse_forward_validity(testforward=sparse_forward, caching=True):
    d_dict = 32
    d_data = 32
    n_sae = 32
    batches = 32
    device = "cuda"

    binary_options = [
        "sub_b_dec",
        "scale_acts",
        "norm_gate_before_scaling_acts",
    ]
    for i in range(2 ** len(binary_options)):
        optionsdict = {
            k: bool(i & (1 << j)) for j, k in enumerate(binary_options)
        }
        cs = CacheBoxes(acts=box())
        cd = CacheBoxes(acts=box())
        o = generate_example(d_dict, d_data, n_sae, batches, xlist=False, device=device, optionsdict=optionsdict)
        s = testforward(o.x.clone(), o.gate.clone(), o.W_enc.clone(), o.b_enc.clone(), o.W_dec.clone(), o.b_dec.clone(), cache=cs, options=o.options)
        d = dense(o.x.clone(), o.gate.clone(), o.W_enc.clone(), o.b_enc.clone(), o.W_dec.clone(), o.b_dec.clone(), cache=cd, options=o.options)
        print(optionsdict)
        print(s.shape, d.shape) 
        print("output eq?", torch.allclose(s, d, rtol=1e-0, atol=1e-0))
        if caching:
            print(cs.acts.x.shape, cd.acts.x.shape)
            print("acts eq?", torch.allclose(cs.acts.x, cd.acts.x, rtol=1e-0, atol=1e-0))

def comptime(o, *F, cache=True, backward = False, optim = False, skipoptions=False):
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
            if skipoptions:
                opts = {}
            else:
                opts = {"options": o.options}
            y = f(
                x = x,
                gate = torch.abs(o.srcgate) * gmask.clone().detach().clone(),
                W_enc = o.W_enc.clone(),
                b_enc = o.b_enc.clone(),
                W_dec = o.W_dec.clone(),
                b_dec = o.b_dec.clone(),
                cache = c,
                **opts
            )
            if backward:
                (x.float()- y.float()).pow(2).sum().backward()
                if optim:
                    optim.step()
                    optim.zero_grad()
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        torch.cuda.empty_cache()
    

def test_speed(afunc=sparse_forward):
    d_dict = 32
    d_data = 256
    n_sae = 4096
    batches = 4
    device = "cuda"
    p_sparse = 0.99
    
    o = generate_example(d_dict, d_data, n_sae, batches, p_sparse, device=device, requires_grad=True)

    comptime(o, sparse_forward, dense, sparse_bgate_forward, cache=True, backward = False)
    # comptime(o, sparse_forward, dense, cache=True, backward = True)
    # print("sparse first")
    # comptime(o, sparse_forward, sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward, cache=False, backward = True, optim=True)
    # print("dense first")
    reps1 = 1
    reps2 = 10
    jitforward, jitdense = sparse_jit_forward(o.options)
    for _ in range(reps1):
        for _ in range(reps2):
            comptime(o, jitdense, skipoptions=True, cache=True, backward = True, optim=True)

        for _ in range(reps2):
            comptime(o, jitdense, skipoptions=True, cache=True, backward = True, optim=True)
        for _ in range(reps2):
            comptime(o, jitforward, skipoptions=True, cache=True, backward = True, optim=True)
        for _ in range(reps2):
            comptime(o, sparse_forward, cache=True, backward = True, optim=True)
        for _ in range(reps2):
            comptime(o, dense, cache=True, backward = True, optim=True)
        for _ in range(reps2):
            comptime(o, sparse_bgate_forward, cache=True, backward = True, optim=True)

        # comptime(o, dense, sparse_forward, sparse_bgate_forward, cache=True, backward = True, optim=True)
        # comptime(o, dense, sparse_forward, cache=True, backward = True, optim=True)
        # comptime(o, dense, sparse_forward, cache=True, backward = True, optim=True)
    # comptime(o, sparse_forward, sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward, cache=False, backward = True, optim=True)

def test_speed2(afunc=sparse_forward):
    d_dict = 32
    d_data = 512
    n_sae = 1024
    batches = 32
    device = "cuda"
    p_sparse = 0.995
    
    o = generate_example(d_dict, d_data, n_sae, batches, p_sparse, device=device, requires_grad=True)

    # comptime(o, sparse_forward, dense, sparse_bgate_forward, cache=True, backward = False)
    # comptime(o, sparse_forward, dense, cache=True, backward = True)
    # print("sparse first")
    # comptime(o, sparse_forward, sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward,sparse_forward, cache=False, backward = True, optim=True)
    # print("dense first")
    reps1 = 1
    reps2 = 10
    jitforward, jitdense = sparse_jit_forward(o.options)
    for _ in range(reps1):
        for _ in range(reps2):
            comptime(o, afunc, skipoptions=False, cache=False, backward = False, optim=False)
        for _ in range(reps2):
            comptime(o, sparse_forward, cache=False, backward = False, optim=False)
        for _ in range(reps2):
            comptime(o, dense, cache=False, backward = False, optim=False)
        for _ in range(reps2):
            comptime(o, sparse_bgate_forward, cache=False, backward = False, optim=False)


def main():
    test_sparse_forward_validity()
    test_speed()
    
if __name__ == "__main__":
    main()
