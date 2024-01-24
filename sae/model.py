# pair programming by Glen Taggart and Kieth Wynroe based mostly off of
# this work https://colab.research.google.com/drive/1MjF_5-msnSe5F9Qy4kEGSeqyYPE9_D2p?authuser=1#scrollTo=7WXAjU3mRak6
# which I think was made by Bart Bussman, based off Neel Nanda's code.
from nqgl.sae import novel_nonlinearities
from nqgl.sae.sae.config import AutoEncoderConfig
from nqgl.sae.setup_utils import SAVE_DIR, DTYPES
from nqgl.sae.sae.base import BaseSAE
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


class AutoEncoder(BaseSAE, nn.Module):
    sae_type = "SAE"
    CONFIG = AutoEncoderConfig

    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        dtype = DTYPES[cfg.enc_dtype]
        torch.manual_seed(cfg.seed)

        self.cfg = cfg

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_data, dtype=dtype))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg.d_data, cfg.d_dict, dtype=dtype)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_dict, dtype=dtype))
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg.d_dict, cfg.d_data, dtype=dtype)
            )
        )
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
        self.neuron_activation_frequency = torch.zeros(
            self.cfg.d_dict, dtype=torch.float32
        ).to(cfg.device)

        # rescaling fields
        self.scaling_factor = nn.Parameter(torch.ones(1), requires_grad=False)
        self.std_dev_accumulation = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.std_dev_accumulation_steps = nn.Parameter(
            torch.zeros(1), requires_grad=False
        )
        self.to(cfg.device)

        self.step_num = 0

    def encode(
        self, x, cache_acts=False, cache_l0=False, record_activation_frequency=False
    ):
        if self.cfg.scale_in_forward:
            x = self.scale(x)
        x_cent = x - self.b_dec
        acts = self.nonlinearity(x_cent @ self.W_enc + self.b_enc)

        self.cached_l1_loss = acts.float().abs().mean(dim=(-2))
        self.cached_l0_norm = (
            torch.count_nonzero(acts, dim=-1).float().mean() if cache_l0 else None
        )
        self.cached_acts = acts if cache_acts else None

        if record_activation_frequency:
            activated = torch.mean((acts > 0).float(), dim=0)
            activated = torch.count_nonzero(acts, dim=0) / acts.shape[0]
            self.neuron_activation_frequency = (
                activated + self.neuron_activation_frequency.detach()
            )
            self.steps_since_activation_frequency_reset += 1

        return acts

    def decode(self, acts):
        x_reconstruct = acts @ self.W_dec + self.b_dec
        if self.cfg.scale_in_forward:
            return self.unscale(x_reconstruct)
        return x_reconstruct

    def forward(
        self,
        x,
        cache_l0=True,
        cache_acts=False,
        record_activation_frequency=False,
        rescaling=False,
    ):
        if rescaling:
            self.update_scaling(x)

        acts = self.encode(
            x,
            cache_acts=cache_acts,
            cache_l0=cache_l0,
            record_activation_frequency=record_activation_frequency,
        )

        x_reconstruct = self.decode(acts)

        x_diff = x_reconstruct.float() - x.float()

        self.cached_l2_loss = (self.scale(x_diff)).pow(2).mean(-1).mean(0)

        return x_reconstruct

    def get_loss(self):
        if self.cfg.cosine_l1 is None:
            l1_coeff = self.l1_coeff
        else:
            self.step_num += 1
            c_period, c_range = (
                self.cfg.cosine_l1["period"],
                self.cfg.cosine_l1["range"],
            )
            l1_coeff = self.l1_coeff * (
                1
                + c_range
                * torch.cos(
                    torch.tensor(2 * torch.pi * self.step_num / c_period).detach()
                )
            )

        l2 = torch.mean(self.cached_l2_loss)
        l1 = torch.sum(l1_coeff * self.cached_l1_loss)
        return l1 + l2

    @torch.no_grad()
    def update_scaling(self, x: torch.Tensor):
        x_cent = x - x.mean(dim=0)
        var = x_cent.norm(dim=-1).pow(2).mean()
        std = torch.sqrt(var)
        self.std_dev_accumulation += (
            std  # x_cent.std(dim=0).mean() is p diferent I believe
        )
        self.std_dev_accumulation_steps += 1
        self.scaling_factor.data[:] = (
            self.std_dev_accumulation
            / self.std_dev_accumulation_steps
            / self.cfg.data_rescale
        )

    # @torch.no_grad()
    def scale(self, x):
        return x / self.scaling_factor

    # @torch.no_grad()
    def unscale(self, x):
        return x * self.scaling_factor

    # @torch.no_grad()
    # def reset_neurons(
    #     self, new_directions: torch.Tensor, norm_encoder_proportional_to_alive=True
    # ):
    #     if new_directions.shape[0] > self.neurons_to_be_reset.shape[0]:
    #         new_directions = new_directions[: self.neurons_to_be_reset.shape[0]]
    #     num_resets = new_directions.shape[0]
    #     to_reset = self.neurons_to_be_reset[:num_resets]
    #     self.neurons_to_be_reset = self.neurons_to_be_reset[num_resets:]
    #     if self.neurons_to_be_reset.shape[0] == 0:
    #         self.neurons_to_be_reset = None
    #     new_directions = new_directions / new_directions.norm(dim=-1, keepdim=True)
    #     print(f"to_reset shape", to_reset.shape)
    #     print(f"new_directions shape", new_directions.shape)
    #     print(f"self.W_enc shape", self.W_enc.shape)
    #     if norm_encoder_proportional_to_alive:
    #         self.W_enc.data[:, to_reset] = (
    #             new_directions.T * self.alive_norm_along_feature_axis * 0.2
    #         )
    #     else:
    #         self.W_enc.data[:, to_reset] = new_directions.T
    #     self.W_dec.data[to_reset, :] = new_directions
    #     self.b_enc.data[to_reset] = 0

    # @torch.no_grad()
    # def reset_activation_frequencies(self):
    #     self.neuron_activation_frequency[:] = 0
    #     self.steps_since_activation_frequency_reset = torch.zeros(1)
