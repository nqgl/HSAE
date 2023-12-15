import novel_nonlinearities

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from argparse import ArgumentParser
import pprint
from functools import partial
from collections import namedtuple
from dataclasses import asdict
from typing import Tuple, Callable, Optional
from sae import AutoEncoder, AutoEncoderConfig
from hsae_config import HierarchicalAutoEncoderConfig, HierarchicalAutoEncoderLayerConfig
from setup_utils import SAVE_DIR, DTYPES


class HierarchicalAutoEncoder(nn.Module):
    def __init__(self, cfg :HierarchicalAutoEncoderConfig, sae0 :Optional[AutoEncoder] = None):
        super().__init__()
        self.sae_0 = AutoEncoder(cfg) if sae0 is None else sae0
        self.saes = nn.ModuleList(
                HierarchicalAutoEncoderLayer(cfg = layer_cfg, cfg_0=cfg)
            for layer_cfg in cfg.sublayer_cfgs)
        self.cfg = cfg

    def forward(self, x, rescaling=False):
        if rescaling:
            self.sae_0.update_scaling(x)
        x = self.sae_0.scale(x)


        cache_kwargs = dict(cache_acts = True, cache_l0 = True)
        x_0 = self.sae_0(x, **cache_kwargs)
        x_n = x_0
        acts = self.sae_0.cached_acts
        self.cached_l1_loss = self.sae_0.cached_l1_loss
        for sae in self.saes:
            x_ = x if not self.cfg.sublayers_train_on_error else x - x_n
            gate = self.gate(acts)
            x_n += sae(x_, gate) #we should just store acts to a class var at default prob
            acts = sae.cached_acts
            self.cached_l1_loss += sae.cached_l1_loss
        self.cached_l2_loss = (x - x_n) ** 2
        return self.sae_0.unscale(x_n)

    def gate(self, acts):
        if self.cfg.gate_mode == "binary":
            return (acts > 0).float()
        elif self.cfg.gate_mode == "norm":
            return acts / (acts.norm(dim=-1, keepdim=True) + 1e-9)
        elif self.cfg.gate_mode == "acts":
            return acts
        
    def make_decoder_weights_and_grad_unit_norm(self):
        self.sae_0.make_decoder_weights_and_grad_unit_norm()
        for sae in self.saes:
            sae.make_decoder_weights_and_grad_unit_norm()



class HierarchicalAutoEncoderLayer(AutoEncoder, nn.Module):
    def __init__(self, cfg :HierarchicalAutoEncoderLayerConfig, cfg_0 :HierarchicalAutoEncoderConfig):
        super().__init__(cfg_0)
        self.cfg = cfg

        dtype = DTYPES[cfg_0.enc_dtype]
        
        self.b_dec = nn.Parameter(
            torch.zeros(
                self.cfg.n_sae, 
                self.cfg0.d_data, 
                dtype=dtype
            )
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.n_sae, 
                    self.cfg0.d_data, 
                    self.cfg.d_dict, 
                    dtype=dtype
                )
            )
        )

        self.b_enc = nn.Parameter(
            torch.zeros(
                self.cfg.n_sae,
                self.cfg.d_dict,
                dtype=dtype
            )
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.n_sae,
                    self.cfg.d_dict, 
                    self.cfg0.d_data, 
                    dtype=dtype
                )
            )
            )
        
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to(self.cfg0.device)




    def encode_flat(self, x, W_enc, b_dec, b_enc, cache_acts=False, cache_l0=False, record_activation_frequency=False):
        # batch, n_sae, d_data
        # x = x.reshape(-1, 1, self.cfg0.d_data)
        x_cent = x - b_dec
        # x_cent = x_cent.unsqueeze(-2)
        m = x_cent @ W_enc
        acts = self.nonlinearity(m + b_enc)

        # print("x_cent", x_cent.shape)
        # print("m", m.shape)
        # print("acts", acts.shape)
        return acts

    def decode_flat(self, acts, W_dec, b_dec):
        return acts @ W_dec + b_dec
            





    def forward(self, x: torch.Tensor, gate: torch.Tensor):
        # x: (batches, d_data)
        # gate: (batches, n_sae)
        batches = x.shape[0]

        # x = self.scale(x)
        dgate = gate
        gate = gate.to_sparse()
        flat_indices = gate.indices().to(self.cfg0.device)
        # print("flat_indices", flat_indices.shape)
        batch_idxs = flat_indices[0]
        sae_idxs = flat_indices[1]
        x_flat = x[batch_idxs].unsqueeze(-2)

        W_enc = self.W_enc[sae_idxs]
        b_enc = self.b_enc[sae_idxs].unsqueeze(-2)
        W_dec = self.W_dec[sae_idxs]
        b_dec = self.b_dec[sae_idxs].unsqueeze(-2)
        # print(W_enc.shape, b_enc.shape, W_dec.shape, b_dec.shape)
        # print(self.W_enc.shape, self.b_enc.shape, self.W_dec.shape, self.b_dec.shape)
        # print("x_flat", x_flat.shape)
        flat_acts = self.encode_flat(x=x_flat, W_enc=W_enc, b_dec=b_dec, b_enc=b_enc)
        # print("flat_acts", flat_acts.shape)

        acts = torch.zeros(batches, self.cfg.n_sae, self.cfg.d_dict, device=self.cfg0.device)
        # acts = acts.scatter_add(
        #     0, 
        #     flat_indices.unsqueeze(-1).unsqueeze(-1).expand(2, -1, self.cfg.n_sae, self.cfg.d_dict), 
        #     flat_acts.reshape(-1, 1, self.cfg.d_dict)
        # )
        acts[flat_indices] = flat_acts
        # print(acts.shape, flat_acts.shape)
        self.cache(acts)
        # flat_acts = flat_acts * dgate[flat_indices]
        saes_out_flat = self.decode_flat(flat_acts, W_dec=W_dec, b_dec=b_dec)
        # print("saes_out_flat", saes_out_flat.shape)

        x_out = torch.scatter_add(
            torch.zeros(batches, self.cfg0.d_data, device=self.cfg0.device),
            0, 
            batch_idxs.unsqueeze(-1).expand(-1, self.cfg0.d_data),
            saes_out_flat.reshape(-1, self.cfg0.d_data)
        )

        # x_reconstruct = self.unscale(x_out)
        return x_out
        





    def cache(self, acts, cache_l0 = True, cache_acts = True):
        self.cached_l1_loss = acts.float().abs().sum(dim=-1).mean()
        self.cached_l0_norm = torch.count_nonzero(acts, dim=-1).float().mean() if cache_l0 else None
        self.cached_acts = acts if cache_acts else None
        record_activation_frequency = False
        if record_activation_frequency:
            activated = torch.mean((acts > 0).float(), dim=0)
            activated = torch.count_nonzero(acts, dim=0) / acts.shape[0]
            self.neuron_activation_frequency = activated + self.neuron_activation_frequency.detach()
            self.steps_since_activation_frequency_reset += 1
        return acts



    @torch.no_grad()
    def update_scaling(self, x :torch.Tensor):
        if self.cfg0.sublayers_train_on_error:
            x_cent = x - x.mean(dim=0)
            # var = (x_cent ** 2).sum(dim=-1)
            # std = torch.sqrt(var).mean()
            std = x_cent.norm(dim=-1).mean()
            self.std_dev_accumulation += std #x_cent.std(dim=0).mean() is p diferent I believe
            self.std_dev_accumulation_steps += 1
            self.scaling_factor = self.std_dev_accumulation / self.std_dev_accumulation_steps



    @torch.no_grad()
    def scale(self, x):
        raise Exception("lower-level layer getting scaled")
        return x / self.scaling_factor

    @torch.no_grad()
    def unscale(self, x):
        raise Exception("lower-level layer getting scaled")
        return x * self.scaling_factor

@torch.no_grad()
def make_ll_self_similar(hsae :HierarchicalAutoEncoder, layer_index :int):
    sae0 = hsae.sae_0
    sae = hsae.saes[layer_index]
    for i in range(hsae.cfg.d_data):
        sae.W_enc[i, :, :] = sae0.W_enc[:, :]
        sae.W_dec[i, :, :] = sae0.W_dec[:, :]

def main():
    d = 32
    cfg = HierarchicalAutoEncoderConfig(dict_mult=1, d_data = d)
    hsae = HierarchicalAutoEncoder(cfg)
    # make_ll_self_similar(hsae, 0)
    torch.seed()
    opt = torch.optim.SGD(hsae.parameters(), lr=1e-4)
    features = torch.randn(d * 4, d).cuda()
    
    for i in range(6000):
        v = torch.randn(100, d * 4).cuda()
        v = F.dropout(v, p=0.95, training=True) * 0.05
        x = v @ features
        y = hsae(x)
        l = (x - y).pow(2).mean()
        l += hsae.sae_0.cached_l1_loss.mean() * 0.01
        l.backward()
        opt.step()
        opt.zero_grad()
        print(l)
    print("l0", hsae.sae_0.cached_l0_norm)
    # x = torch.randn(2, d).cuda()
    # y = hsae(x)
    # y0 = hsae.sae_0(x)
    # print(x)
    # print(y)
    # print(y.shape, y0.shape)
    # print(y0)
    # print(y / y0)
    # gate = torch.zeros(2, d).cuda()
    # gate[0, 4] = 1
    # y1 = hsae.saes[0](x, gate)
    # print(y1)





if __name__ == "__main__":
    main()
    print("Done __main__")