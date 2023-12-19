from nqgl.sae.hsae.hsae_layer import HierarchicalAutoEncoderLayer
from nqgl.sae.hsae.config import HierarchicalAutoEncoderConfig
from setup_utils import SAVE_DIR

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pprint
from typing import Tuple, Callable, Optional, List
from sae.model import AutoEncoder, AutoEncoderConfig
# import torch_sparse
# import torchsparsegradutils as tsgu

class HierarchicalAutoEncoder(nn.Module):
    def __init__(self, cfg :HierarchicalAutoEncoderConfig, sae0 :Optional[AutoEncoder] = None):
        super().__init__()
        self.sae_0 = AutoEncoder(cfg) if sae0 is None else sae0
        self.saes :List["HierarchicalAutoEncoderLayer"] = nn.ModuleList(
                HierarchicalAutoEncoderLayer(cfg = layer_cfg, cfg_0=cfg)
            for layer_cfg in cfg.sublayer_cfgs)
        self.cfg = cfg
        self.neurons_to_be_reset = None

    def forward(self, x, rescaling=False, record_activation_frequency=False, dense=True):
        if rescaling:
            self.sae_0.update_scaling(x)
        x = self.sae_0.scale(x)


        cache_kwargs = dict(cache_acts = True, cache_l0 = True)
        x_0 = self.sae_0(x, **cache_kwargs)
        x_n = x_0
        acts = self.sae_0.cached_acts
        self.cached_l1_loss = self.sae_0.cached_l1_loss
        # print("x_0", x_0.shape)
        prev = self.sae_0
        for sae in self.saes:
            x_ = x if not self.cfg.sublayers_train_on_error else x - x_n
            gate = self.gate(acts)
            x_next = sae(x_, gate, dense=dense, prev_sae = prev)
            # print("x_next", x_next.shape)
            # print("x_n", x_n.shape)
            print("x_n", x_n.shape)
            print("x_next", x_next.shape)

            x_n = x_n + x_next #we should just store acts to a class var at default prob
            acts = sae.cached_acts
            # self.cached_l1_loss += sae.cached_l1_loss
            prev = sae
        self.cached_l2_loss = (x - x_n) ** 2
        self.cached_l1_loss = self.sae_0.cached_l1_loss
        self.cached_l0_norm = self.sae_0.cached_l0_norm
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

    def get_loss(self):
        l = self.cached_l2_loss.mean()
        l += (self.sae_0.cached_l1_loss).sum(-1).mean() * self.sae_0.cfg.l1_coeff
        for sae in self.saes:
            l += sae.get_loss()
        return l


    def loss_contributions(self, x):
        saes = [sae for sae in self.saes]
        x_0 = self.sae_0(x, cache_acts = True)
        gate = self.gate(self.sae_0.cached_acts)
        x_i = [x_0]
        mses = []
        for i, sae in enumerate(saes):
            x_i.append(sae(x, gate))
        x_re = torch.sum(x)
        mse = (x - x_re).pow(2).mean()
        x_re_i = [x_re - xi for xi in x_i]
        mses = torch.tensor([(x - xrei).pow(2).mean() for xrei in x_re_i], device=self.cfg.device)
        contrib = (mses - mse)
        d = {}
        for i in range(len(x_i)):
            d[f"sae{i} mse contrib"] = contrib[i]/sum(contrib)
        return d






@torch.no_grad()
def make_ll_self_similar(hsae :HierarchicalAutoEncoder, layer_index :int):
    sae0 = hsae.sae_0
    sae = hsae.saes[layer_index]
    for i in range(sae.cfg.n_sae):
        sae.W_enc[i, :, :] = sae0.W_enc[:, :]
        sae.W_dec[i, :, :] = sae0.W_dec[:, :]
        sae.b_dec[i, :] = sae0.b_dec[:]
        sae.b_enc[i, :] = sae0.b_enc[:]

def main():
    d = 32
    cfg = HierarchicalAutoEncoderConfig(dict_mult=1, d_data = d)
    hsae = HierarchicalAutoEncoder(cfg)
    # make_ll_self_similar(hsae, 0)
    torch.seed()
    opt = torch.optim.SGD(hsae.parameters(), lr=1e-4)
    features = torch.randn(d * 4, d).cuda()
    
    for i in range(60):
        v = torch.randn(10, d * 4).cuda()
        v = F.dropout(v, p=0.95, training=True) * 0.05
        x = v @ features
        y = hsae(x)
        l = (x - y).pow(2).mean()
        l += hsae.sae_0.cached_l1_loss.mean() * 0.01
        l.backward()
        opt.step()
        opt.zero_grad()
        # print(l)
    print("l0", hsae.sae_0.cached_l0_norm)
    bs = 5
    make_ll_self_similar(hsae, 0)
    x = torch.randn(bs, d).cuda()
    y = hsae(x)
    y0 = hsae.sae_0(x)
    print(x)
    print(y)
    print(y.shape, y0.shape)
    print(y0)
    print(y / y0)
    print("L0", hsae.sae_0.cached_l0_norm)
    print("L0_1", hsae.saes[0].cached_l0_norm)
    print("L1", hsae.sae_0.cached_l0_norm)
    gate = torch.zeros(bs, d).cuda()
    gate[0, 4] = 1
    y1 = hsae.saes[0](x, gate)
    print(y1)





if __name__ == "__main__":
    main()
    # print("Done __main__")