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
    def __init__(self, cfg :AutoEncoderConfig):
        super().__init__()
        dtype = DTYPES[cfg.enc_dtype]
        torch.manual_seed(cfg.seed)
        
        self.cfg0 = cfg      

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_data, dtype=dtype))
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_data, cfg.d_dict, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_dict, dtype=dtype))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_dict, cfg.d_data, dtype=dtype)))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.nonlinearity = novel_nonlinearities.cfg_to_nonlinearity(cfg)
    

        
        
        # this is useful for implementing non-const l1 currently
        self.l1_coeff = cfg.l1_coeff

        # cached data fields
        self.cached_l2_loss = None
        self.cached_l1_loss = None
        self.cached_l0_norm = None
        self.cached_acts = None

        # neuron reset fields
        self.neuron_activation_frequency = torch.zeros(self.cfg0.d_dict, dtype=torch.float32).to(cfg.device)
        self.steps_since_activation_frequency_reset = 0
        self.neurons_to_be_reset = None
        
        # rescaling fields
        self.scaling_factor = nn.Parameter(torch.tensor(self.cfg0.data_rescale), requires_grad=False)
        self.std_dev_accumulation = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.std_dev_accumulation_steps = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.to(cfg.device)
        
        self.step_num = 0

    def encode(self, x, cache_acts = False, cache_l0 = False, record_activation_frequency = False):
        if self.cfg0.scale_in_forward:
            x = self.scale(x)
        x_cent = x - self.b_dec
        acts = self.nonlinearity(x_cent @ self.W_enc + self.b_enc)

        self.cached_l1_loss = acts.float().abs().mean(dim=(-2))
        self.cached_l0_norm = (acts > 0).float().sum(dim=-1).mean() if cache_l0 else None
        self.cached_l0_norm = torch.count_nonzero(acts, dim=-1).float().mean() if cache_l0 else None
        self.cached_acts = acts if cache_acts else None

        if record_activation_frequency:
            activated = torch.mean((acts > 0).float(), dim=0)
            activated = torch.count_nonzero(acts, dim=0) / acts.shape[0]
            self.neuron_activation_frequency = activated + self.neuron_activation_frequency.detach()
            self.steps_since_activation_frequency_reset += 1

        return acts
    
    def decode(self, acts):
        
        x_reconstruct = acts @ self.W_dec + self.b_dec
        if self.cfg0.scale_in_forward:
            return self.unscale(x_reconstruct)
        else:
            return x_reconstruct


    def forward(self, x, cache_l0 = True, cache_acts = False, record_activation_frequency = False, rescaling = False):
        if rescaling:
            self.update_scaling(x)
    
        acts = self.encode(x, cache_acts=cache_acts, cache_l0=cache_l0, record_activation_frequency=record_activation_frequency)

        x_reconstruct = self.decode(acts)

        x_diff = x_reconstruct.float() - x.float()

        self.cached_l2_loss = (self.scale(x_diff)).pow(2).mean(-1).mean(0)

        return x_reconstruct
    


    def get_loss(self):
        if self.cfg0.cosine_l1 is None:
            l1_coeff = self.l1_coeff
        else:
            self.step_num += 1
            c_period, c_range = self.cfg0.cosine_l1["period"], self.cfg0.cosine_l1["range"]
            l1_coeff = self.l1_coeff * (1 + c_range * torch.cos(torch.tensor(2 * torch.pi * self.step_num / c_period).detach()))

        l2 = torch.mean(self.cached_l2_loss)
        l1 = torch.sum(l1_coeff * self.cached_l1_loss)
        return l1 + l2

    @torch.no_grad()
    def update_scaling(self, x :torch.Tensor):
        x_cent = x - x.mean(dim=0)
        var = x_cent.norm(dim=-1).pow(2).mean()
        std = torch.sqrt(var)
        self.std_dev_accumulation += std #x_cent.std(dim=0).mean() is p diferent I believe
        self.std_dev_accumulation_steps += 1
        self.scaling_factor.data[:] = self.std_dev_accumulation / self.std_dev_accumulation_steps / self.cfg0.data_rescale

    @torch.no_grad()
    def scale(self, x):
        return x / self.scaling_factor

    @torch.no_grad()
    def unscale(self, x):
        return x * self.scaling_factor


    @torch.no_grad()
    def neurons_to_reset(self, to_be_reset :torch.Tensor):
        if to_be_reset.sum() > 0:
            self.neurons_to_be_reset = torch.argwhere(to_be_reset).squeeze(1)
            w_enc_norms = self.W_enc[:, ~ to_be_reset].norm(dim=0)
            # print("w_enc_norms", w_enc_norms.shape)
            # print("to_be_reset", self.to_be_reset.sum())
            self.alive_norm_along_feature_axis = torch.mean(torch.mean(w_enc_norms))
        else:
            self.neurons_to_be_reset = None
    
    @torch.no_grad()
    def re_init_neurons(self, x_diff):
        self.re_init_neurons_gram_shmidt_precise_iterative_argmax(x_diff)

    @torch.no_grad()
    def re_init_neurons_gram_shmidt_precise_iterative_argmax(self, x_diff):
        n_reset = min(x_diff.shape[0], self.cfg0.d_data // 2, self.cfg0.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            magnitudes = x_diff.norm(dim=-1)
            i_max = torch.argmax(magnitudes)
            v_orth[i] = x_diff[i_max]
            for j in range(max(0, i - self.cfg0.gram_shmidt_trail), i):
                v_orth[i] -= torch.dot(v_orth[j], v_orth[i]) * v_orth[j] / torch.dot(v_orth[j], v_orth[j])
            if v_orth[i].norm() < 1e-6:
                n_succesfully_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            x_diff -= (x_diff @ v_orth[i]).unsqueeze(1) * v_orth[i] / torch.dot(v_orth[i], v_orth[i])
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # print(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])


    @torch.no_grad()
    def re_init_neurons_gram_shmidt_precise_topk(self, x_diff):
        t = self.cfg0.gram_shmidt_trail
        n_reset = min(x_diff.shape[0], self.cfg0.d_data // 2, self.cfg0.num_to_resample)
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
        t = self.cfg0.gram_shmidt_trail
        n_reset = min(x_diff.shape[0], self.cfg0.d_data // 2, self.cfg0.num_to_resample)
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
        if new_directions.shape[0] > self.neurons_to_be_reset.shape[0]:
            new_directions = new_directions[:self.neurons_to_be_reset.shape[0]]
        num_resets = new_directions.shape[0]
        to_reset = self.neurons_to_be_reset[:num_resets]
        self.neurons_to_be_reset = self.neurons_to_be_reset[num_resets:]
        if self.neurons_to_be_reset.shape[0] == 0:
            self.neurons_to_be_reset = None
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
        self.neuron_activation_frequency[:] = 0
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
            json.dump(asdict(self.cfg0), f)
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




