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
from typing import Tuple, Callable
from sae_config import AutoEncoderConfig
from setup_utils import SAVE_DIR, DTYPES

class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_dict = cfg.dict_size
        l1_coeff = cfg.l1_coeff
        dtype = DTYPES[cfg.enc_dtype]
        torch.manual_seed(cfg.seed)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.act_size, d_dict, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_dict, cfg.act_size, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_dict, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.act_size, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.step_num = 0
        self.d_dict = d_dict
        self.l1_coeff = l1_coeff
        self.acts_cached = None
        self.l2_loss_cached = None
        self.l1_loss_cached = None
        self.l0_norm_cached = None
        self.cfg = cfg      
        self.to(cfg.device)
        self.cached_acts = None
        self.nonlinearity = novel_nonlinearities.cfg_to_nonlinearity(cfg)
        self.activation_frequency = torch.zeros(self.d_dict, dtype=torch.float32).to(cfg.device)
        self.steps_since_activation_frequency_reset = 0
        self.to_be_reset = None
        self.scaling_factor = cfg.data_rescale
        self.std_dev_accumulation = 0
        self.std_dev_accumulation_steps = 0

    def encode(self, x, cache_acts = False, cache_l0 = False, record_activation_frequency = False, rescaling = False):
        x = x * self.cfg.data_rescale
        if rescaling:
            self.update_scaling(x)
        x = self.scale(x)
        x_cent = x - self.b_dec
        # print(x_cent.dtype, x.dtype, self.W_dec.dtype, self.b_dec.dtype)
        acts = self.nonlinearity(x_cent @ self.W_enc + self.b_enc)
        return acts

    def forward(self, x, cache_l0 = True, cache_acts = False, record_activation_frequency = False, rescaling = False):
        x = x * self.cfg.data_rescale
        if rescaling:
            self.update_scaling(x)
        x = self.scale(x)
        x_cent = x - self.b_dec
        # print(x_cent.dtype, x.dtype, self.W_dec.dtype, self.b_dec.dtype)
        acts = self.nonlinearity(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        # self.l2_loss_cached = (x_reconstruct.float() - x.float()).pow(2).mean(-1).mean(0)
        x_diff = x_reconstruct.float() - x.float()
        # self.re_init_neurons(x_diff)
        self.l1_loss_cached = acts.float().abs().mean(dim=(-2))
        self.l2_loss_cached = (x_diff).pow(2).mean(-1).mean(0)

        if cache_l0:
            self.l0_norm_cached = (acts > 0).float().sum(dim=-1).mean()
        else:
            self.l0_norm_cached = None
        if cache_acts:
            self.cached_acts = acts
        else:
            self.cached_acts = None
        if record_activation_frequency:
            # print(acts.shape)
            activated = torch.mean((acts > 0).float(), dim=0)
            # print("activated shape", activated.shape)
            # print("freq shape", self.activation_frequency.shape)
            self.activation_frequency = activated + self.activation_frequency.detach()
            self.steps_since_activation_frequency_reset += 1
        return self.unscale(x_reconstruct) / self.cfg.data_rescale
    

    def get_loss(self):
        self.step_num += 1
        if self.cfg.cosine_l1 is None:
            l1_coeff = self.l1_coeff
        else:
            c_period, c_range = self.cfg.cosine_l1["period"], self.cfg.cosine_l1["range"]
            l1_coeff = self.l1_coeff * (1 + c_range * torch.cos(torch.tensor(2 * torch.pi * self.step_num / c_period).detach()))

        l2 = torch.mean(self.l2_loss_cached)
        l1 = torch.sum(l1_coeff * self.l1_loss_cached)
        return l1 + l2

    @torch.no_grad()
    def update_scaling(self, x :torch.Tensor):
        x_cent = x - x.mean(dim=0)
        # var = (x_cent ** 2).sum(dim=-1)
        # std = torch.sqrt(var).mean()
        std = x_cent.norm(dim=-1).mean()
        self.std_dev_accumulation += std #x_cent.std(dim=0).mean() is p diferent I believe
        self.std_dev_accumulation_steps += 1
        self.scaling_factor = self.std_dev_accumulation / self.std_dev_accumulation_steps

    @torch.no_grad()
    def scale(self, x):
        return x / self.scaling_factor

    @torch.no_grad()
    def unscale(self, x):
        return x * self.scaling_factor


    @torch.no_grad()
    def neurons_to_reset(self, to_be_reset :torch.Tensor):
        if to_be_reset.sum() > 0:
            self.to_be_reset = torch.argwhere(to_be_reset).squeeze(1)
            w_enc_norms = self.W_enc[:, ~ to_be_reset].norm(dim=0)
            # print("w_enc_norms", w_enc_norms.shape)
            # print("to_be_reset", self.to_be_reset.sum())
            self.alive_norm_along_feature_axis = torch.mean(torch.mean(w_enc_norms))
        else:
            self.to_be_reset = None
    
    @torch.no_grad()
    def re_init_neurons(self, x_diff):
        self.re_init_neurons_gram_shmidt_precise_topk(x_diff)

    @torch.no_grad()
    def re_init_neurons_gram_shmidt_precise_topk(self, x_diff):
        t = self.cfg.gram_shmidt_trail
        n_reset = min(x_diff.shape[0], self.cfg.act_size // 2, self.cfg.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        # print(x_diff.shape)
        # v_orth[0] = F.normalize(x_diff[0], dim=-1)
        magnitudes = x_diff.norm(dim=-1)
        indices = torch.topk(magnitudes, n_reset).indices
        x_diff = x_diff[indices]
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            v_orth[i] = x_diff[i]
            for j in range(max(0, i - t), i):
                v_orth[i] -= torch.dot(v_orth[j], v_orth[i]) * v_orth[j] / torch.dot(v_orth[j], v_orth[j])
            if v_orth[i].norm() < 1e-6:
                n_succesfully_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # print(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])


    
    @torch.no_grad()
    def re_init_neurons_gram_shmidt_precise(self, x_diff):
        t = self.cfg.gram_shmidt_trail
        n_reset = min(x_diff.shape[0], self.cfg.act_size // 2, self.cfg.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        # print(x_diff.shape)
        # v_orth[0] = F.normalize(x_diff[0], dim=-1)
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            v_orth[i] = x_diff[i]
            for j in range(max(0, i - t), i):
                v_orth[i] -= torch.dot(v_orth[j], v_orth[i]) * v_orth[j] / torch.dot(v_orth[j], v_orth[j])
            if v_orth[i].norm() < 1e-6:
                n_succesfully_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # print(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])

    @torch.no_grad()
    def reset_neurons(self, new_directions :torch.Tensor, norm_encoder_proportional_to_alive = True):
        if new_directions.shape[0] > self.to_be_reset.shape[0]:
            new_directions = new_directions[:self.to_be_reset.shape[0]]
        num_resets = new_directions.shape[0]
        to_reset = self.to_be_reset[:num_resets]
        self.to_be_reset = self.to_be_reset[num_resets:]
        if self.to_be_reset.shape[0] == 0:
            self.to_be_reset = None
        new_directions = new_directions / new_directions.norm(dim=-1, keepdim=True)
        print(f"to_reset shape", to_reset.shape)
        print(f"new_directions shape", new_directions.shape)
        print(f"self.W_enc shape", self.W_enc.shape)
        if norm_encoder_proportional_to_alive:
            self.W_enc.data[:, to_reset] = new_directions.T * self.alive_norm_along_feature_axis * 0.2
        else:
            self.W_enc.data[:, to_reset] = new_directions.T
        self.W_dec.data[to_reset, :] = new_directions
        self.b_enc.data[to_reset] = 0



    @torch.no_grad()
    def reset_activation_frequencies(self):
        self.activation_frequency[:] = 0
        self.steps_since_activation_frequency_reset = 0


    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    @staticmethod
    def get_version():
        version_list = [int(file.name.split("_")[0]) for file in list(SAVE_DIR.iterdir()) if "_cfg.json" in str(file)]
        if len(version_list):
            return 1 + max(version_list)
        else:
            return 0

    def save(self, name=""):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR/(str(version)+ "_" + name + ".pt"))
        with open(SAVE_DIR/(str(version)+ "_" + name + "_cfg.json"), "w") as f:
            json.dump(asdict(self.cfg), f)
        print("Saved as version", version)

    @classmethod
    def load(cls, version, cfg = None, save_dir = None):
        save_dir = SAVE_DIR if save_dir is None else Path(save_dir)
        # get correct name with globbing
        import glob
        if cfg is None:
            cfg_name = glob.glob(str(save_dir/(str(version)+"*_cfg.json")))
            cfg = json.load(open(cfg_name[0]))
            cfg = AutoEncoderConfig(**cfg)
        pt_name = glob.glob(str(save_dir/(str(version)+"*.pt")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(pt_name[0]))
        return self

    @classmethod
    def load_latest(cls, new_cfg = None):
        version = cls.get_version() - 1
        ae = cls.load(version, new_cfg)
        return ae




