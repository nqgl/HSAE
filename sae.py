import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, List
from gradcool_functions import undying_relu
import os
import json
from dataclasses import dataclass

def ratio_nice_str(f :float):
    r = f.as_integer_ratio()
    while r[1] > 100 or r[0] > 100:
        r = (r[0] // 2, r[1] // 2)
    
    return f"{r[0]}-{r[1]}"

@dataclass
class AutoEncoderConfig:
    lr :Any[float, torch.Tensor]
    d_act :int
    d_dict :int
    l1_coeff :Optional[Any[float, torch.Tensor]]




class AutoEncoder(nn.Module):
    def __init__(self, cfg, optimizer_class = torch.optim.SGD, optimizer_kwargs = {}):
        super(AutoEncoder, self).__init__()
        self.project_in = nn.Linear(cfg.d_act, cfg.d_act)
        self.encoder = nn.Linear(cfg.d_act, cfg.d_dict)
        self.decoder = nn.Linear(cfg.d_dict, cfg.d_act)
        self.project_out = nn.Linear(cfg.d_act, cfg.d_act)
        self.cfg :AutoEncoderConfig = cfg
        self.alive_neurons = torch.zeros(cfg.d_dict, dtype=torch.bool)
        self.optim = None

    def forward(self, x):
        return self.decode(self.encode(x))
    
    def encode(self, x):
        x = self.project_in(x)
        x = self.encoder(x)
        x = undying_relu(x)
        return x
    
    def encode_relu(self, x):
        x = self.project_in(x)
        x = self.encoder(x)
        x = F.relu(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        x = self.project_out(x)
        return x
    
    def norm_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=-1)


    def get_l1_loss(self, acts):
        l1 = torch.sum(torch.abs(acts), dim=-2)
        assert l1.shape[-1] == self.cfg.d_act

        return l1

    def get_l0_loss(self, x):
        acts = self.encode_relu(x)
        l0 = (acts != 0)
        return l0
    
    def get_l0l1_loss(self, x, n=0):


    def trainstep_backward(self, x):
        r_acts = self.encode_relu(x)
        acts = self.encode(x)
        l1 = self.get_l1_loss(r_acts)
        l2 = self.get_l2_loss(acts)
        loss = l2 + l1 * self.cfg.l1_coeff
        loss.backward()
        self.

    def do_train_step(self, x, optim=None, y=None):
        if y is None:
            y = x
        if optim is None:
            optim = self.optim


    @classmethod
    def open_saved(cls, directory :str, id :Optional[int] = None, iter = None):
        folder_str = f"sae_{ratio_nice_str(l1_coeff)}_{ratio_nice_str(lr)}_{version}"
        modelfolder = os.path.join(directory, folder_str)
        if iter is None:
            iter = max([int(f.split("_")[1]) for f in os.listdir(modelfolder) if f.startswith("model")])
        cfg = json.load(os.path.join(modelfolder, f"config_{iter}.json"))
        cfg = AutoEncoderConfig(**cfg)
        model = torch.load(os.path.join(modelfolder, f"model_{iter}.pt"))


    def save(self, directory, version, stats = None):
        l1_coeff = self.cfg.l1_coeff
        lr = self.cfg.lr
        folder_str = f"sae_id_{ratio_nice_str(l1_coeff)}_{ratio_nice_str(lr)}_{str(version)}"
        if not os.path.exists(os.path.join(directory, folder_str)):
            os.makedirs(os.path.join(directory, folder_str))
        modelfolder = os.path.join(directory, folder_str)
        iter = max([int(f.split("_")[1]) for f in os.listdir(modelfolder) if f.startswith("model")])
        model_name = os.path.join(modelfolder, f"model_{iter}.pt")
        config_name = os.path.join(modelfolder, f"config_{iter}.json")
        torch.save(self, model_name)
        with open(config_name, "w") as f:
            json.dump(self.cfg.__dict__, f)
        if stats is not None:
            stats_name = os.path.join(modelfolder, f"stats_{iter}.json")
            with open(stats_name, "w") as f:
                if isinstance(stats, str):
                    f.write(stats)
                else:
                    json.dump(stats, f)
        return iter