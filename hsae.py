# pair programming by Glen Taggart and Kieth Wynroe based mostly off of 
# this work https://colab.research.google.com/drive/1MjF_5-msnSe5F9Qy4kEGSeqyYPE9_D2p?authuser=1#scrollTo=7WXAjU3mRak6
# which I think was made by Bart Bussman, based off Neel Nanda's code.
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
        self.sae_0 = AutoEncoder(cfg) if sae0 is None else sae0
        self.saes = nn.ModuleList(
                HierarchicalAutoEncoderLayer(cfg = layer_cfg, cfg0=cfg) 
            for layer_cfg in cfg.sublayer_cfgs)
        self.cfg = cfg

    def __forward__(self, x, ):
        x_0 = self.sae_0(x)
        x_n = x_0
        for sae in self.saes:
            x_ = x if not self.cfg0.sublayers_train_on_error else x - x_n
            x_n += sae(x_)
        return x_n

        




class HierarchicalAutoEncoderLayer(AutoEncoder, nn.Module):
    def __init__(self, cfg :HierarchicalAutoEncoderLayerConfig, cfg0 :HierarchicalAutoEncoderConfig):
        self.cfg = cfg0
        self.cfg_i = cfg

        
        if len(child_widths) > 0:
            if isinstance(child_widths, int):
                child_width = child_widths
            self.next_layer = SparseHierarchicalAutoencoder(
                n_input_features=n_input_features,
                n_learned_features=child_widths[0],
                geometric_median_dataset=geometric_median_dataset,
                child_widths=child_widths[1:],
                layer_width=n_learned_features,
            )
        

    def forward(self, x, gate):


    def encode(self, x):

    def decode(self, x, W_dec):




    @torch.no_grad()
    def update_scaling(self, x :torch.Tensor):
        if self.cfg.sublayers_train_on_error:
            x_cent = x - x.mean(dim=0)
            # var = (x_cent ** 2).sum(dim=-1)
            # std = torch.sqrt(var).mean()
            std = x_cent.norm(dim=-1).mean()
            self.std_dev_accumulation += std #x_cent.std(dim=0).mean() is p diferent I believe
            self.std_dev_accumulation_steps += 1
            self.scaling_factor = self.std_dev_accumulation / self.std_dev_accumulation_steps
        else:



    @torch.no_grad()
    def scale(self, x):
        return x / self.scaling_factor

    @torch.no_grad()
    def unscale(self, x):
        return x * self.scaling_factor
